use std::collections::{hash_map::Entry, HashMap, VecDeque};

use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::config::KvReuseMode;
use crate::runtime::metrics::CacheSource;
use crate::runtime::numeric::saturating_usize_to_u64;
use crate::runtime::{llama_seq_id, llama_token};

use super::prefix_state_cache::PendingPrefixSnapshot;
use super::{PrefixCachePolicy, PrefixStateCache};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SequenceMirror {
    pub current_kv_tokens: Vec<llama_token>,
    pub n_past: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheCandidate {
    #[default]
    None,
    Live,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CachePreparation {
    pub source: CacheSource,
    pub cache_hits: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KvCacheAdmission {
    pub seq_id: llama_seq_id,
    pub generation: u64,
    pub mirror: SequenceMirror,
    pub candidate: CacheCandidate,
    pub requires_kv_clear: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SnapshotRestore {
    pub token_count: usize,
    pub prefix_tokens: Vec<llama_token>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResidentRef {
    seq_id: llama_seq_id,
    generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct SessionRecord {
    resident: Option<ResidentRef>,
    in_flight: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PhysicalSequence {
    generation: u64,
    state: SeqState,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
enum SeqState {
    #[default]
    Free,
    Idle {
        mirror: SequenceMirror,
    },
    Leased,
}

#[derive(Debug, Clone)]
pub struct KvCacheManager {
    sessions: HashMap<String, SessionRecord>,
    idle_lru: VecDeque<String>,
    physical: Vec<PhysicalSequence>,
    prefix_state_cache: PrefixStateCache,
    prefix_cache_policy: PrefixCachePolicy,
}

impl KvCacheManager {
    pub fn new(max_sequences: usize) -> Self {
        Self::with_prefix_cache(max_sequences, 32, 256 * crate::defaults::BYTES_PER_MIB, 128)
    }

    pub fn with_prefix_cache(
        max_sequences: usize,
        max_prefix_cache_entries: usize,
        max_prefix_cache_bytes: usize,
        prefix_cache_interval_tokens: usize,
    ) -> Self {
        let max_sequences = max_sequences.clamp(1, max_representable_sequences());
        Self {
            sessions: HashMap::with_capacity(max_sequences),
            idle_lru: VecDeque::with_capacity(max_sequences),
            physical: (0..max_sequences)
                .map(|_| PhysicalSequence {
                    generation: 0,
                    state: SeqState::Free,
                })
                .collect(),
            prefix_state_cache: PrefixStateCache::new(
                max_prefix_cache_entries,
                max_prefix_cache_bytes,
            ),
            prefix_cache_policy: PrefixCachePolicy::new(prefix_cache_interval_tokens),
        }
    }

    pub fn can_admit(&mut self, context_key: &str) -> bool {
        self.prune_stale_idle_sessions();
        if self
            .sessions
            .get(context_key)
            .is_some_and(|record| record.in_flight)
        {
            return false;
        }
        if self.sessions.contains_key(context_key) {
            return self.has_available_sequence();
        }
        if self.sessions.len() >= self.physical.len()
            && self.first_valid_idle_session_key().is_none()
        {
            return false;
        }
        self.has_available_sequence()
    }

    pub fn admit(
        &mut self,
        context_key: &str,
        mode: KvReuseMode,
        bypass_cache: bool,
    ) -> Option<KvCacheAdmission> {
        if !self.can_admit(context_key) {
            return None;
        }

        if !bypass_cache && live_reuse_enabled(mode) {
            if let Some(admission) = self.try_admit_warm(context_key) {
                return Some(admission);
            }
        }

        self.admit_cold(context_key)
    }

    pub fn finalize_slot(
        &mut self,
        context_key: &str,
        seq_id: llama_seq_id,
        generation: u64,
        mirror: SequenceMirror,
        completed: bool,
        mode: KvReuseMode,
    ) {
        if !self.sequence_generation_matches(seq_id, generation) {
            return;
        }

        if completed && live_reuse_enabled(mode) {
            if let Some(index) = seq_index(seq_id, self.physical.len()) {
                self.physical[index].state = SeqState::Idle { mirror };
                self.set_session_idle(context_key, ResidentRef { seq_id, generation });
            }
        } else {
            self.evict_sequence(seq_id);
            self.clear_session(context_key);
        }
    }

    pub fn release_slot_for_reset(
        &mut self,
        context_key: &str,
        seq_id: llama_seq_id,
        generation: u64,
    ) {
        if !self.sequence_generation_matches(seq_id, generation) {
            return;
        }
        self.evict_sequence(seq_id);
        self.clear_session(context_key);
    }

    pub fn evict_all_active_and_idle(&mut self) {
        for index in 0..self.physical.len() {
            self.physical[index].state = SeqState::Free;
            self.physical[index].generation = self.physical[index].generation.saturating_add(1);
        }
        self.sessions.clear();
        self.idle_lru.clear();
        self.prefix_state_cache.clear_pending_snapshots();
    }

    pub(crate) fn restore_best_snapshot_prefix(
        &mut self,
        native_runtime: &mut NativeRuntimeHandle,
        seq_id: llama_seq_id,
        model_fingerprint: u64,
        snapshot_scope: &str,
        prompt_tokens: &[llama_token],
        minimum_token_count: usize,
    ) -> Option<SnapshotRestore> {
        let handle = self.prefix_state_cache.find_best_prefix_handle(
            model_fingerprint,
            snapshot_scope,
            prompt_tokens,
            &mut self.prefix_cache_policy,
        )?;
        if handle.token_count <= minimum_token_count
            || !self
                .prefix_state_cache
                .restore_by_handle(native_runtime, seq_id, handle)
        {
            return None;
        }

        let entry = self.prefix_state_cache.entries.get(handle.index)?;
        Some(SnapshotRestore {
            token_count: entry.token_count,
            prefix_tokens: entry.prefix_tokens.clone(),
        })
    }

    pub(crate) fn capture_prefix_snapshot(
        &mut self,
        native_runtime: &NativeRuntimeHandle,
        model_fingerprint: u64,
        snapshot_scope: &str,
        seq_id: llama_seq_id,
        tokens: &[llama_token],
        terminal_token_count: usize,
    ) -> bool {
        let token_count = tokens.len();
        if !self
            .prefix_cache_policy
            .should_store_boundary(token_count, terminal_token_count)
        {
            return false;
        }

        let captured = self.prefix_state_cache.capture_prefix_state(
            native_runtime,
            seq_id,
            model_fingerprint,
            snapshot_scope,
            tokens,
            token_count,
            self.prefix_cache_policy.hash_prefix(tokens, token_count),
            saturating_usize_to_u64(token_count),
        );
        if !captured {
            return false;
        }
        self.prefix_cache_policy.record_store(token_count);
        true
    }

    pub(crate) fn queue_prefix_snapshot(
        &mut self,
        model_fingerprint: u64,
        snapshot_scope: &str,
        seq_id: llama_seq_id,
        generation: u64,
        tokens: &[llama_token],
        terminal_token_count: usize,
    ) -> bool {
        let token_count = tokens.len();
        if !self
            .prefix_cache_policy
            .should_store_boundary(token_count, terminal_token_count)
        {
            return false;
        }

        self.prefix_state_cache
            .enqueue_pending_snapshot(PendingPrefixSnapshot {
                seq_id,
                generation,
                model_fingerprint,
                snapshot_scope: snapshot_scope.to_string(),
                token_count,
                prefix_hash: self.prefix_cache_policy.hash_prefix(tokens, token_count),
                retention_priority: saturating_usize_to_u64(token_count),
                prefix_tokens: tokens[..token_count].to_vec(),
            });
        true
    }

    pub(crate) fn drain_pending_prefix_snapshots(
        &mut self,
        native_runtime: &NativeRuntimeHandle,
        max_to_drain: usize,
    ) -> usize {
        let generations_by_seq: Vec<u64> = self
            .physical
            .iter()
            .map(|sequence| sequence.generation)
            .collect();
        let prefix_cache_policy = &mut self.prefix_cache_policy;
        self.prefix_state_cache.drain_pending_snapshots(
            native_runtime,
            max_to_drain,
            |seq_id, generation| {
                seq_index(seq_id, generations_by_seq.len())
                    .is_some_and(|index| generations_by_seq[index] == generation)
            },
            |token_count| prefix_cache_policy.record_store(token_count),
        )
    }

    #[cfg(test)]
    pub(crate) fn pending_prefix_snapshot_count(&self) -> usize {
        self.prefix_state_cache.pending_snapshot_count()
    }

    fn try_admit_warm(&mut self, context_key: &str) -> Option<KvCacheAdmission> {
        let resident = self.sessions.get(context_key)?.resident?;
        let index = self.valid_resident_index(resident)?;
        let SeqState::Idle { mirror } =
            std::mem::replace(&mut self.physical[index].state, SeqState::Leased)
        else {
            return None;
        };
        self.remove_lru_key(context_key);
        let record = self.sessions.get_mut(context_key)?;
        record.resident = None;
        record.in_flight = true;
        Some(KvCacheAdmission {
            seq_id: resident.seq_id,
            generation: resident.generation,
            mirror,
            candidate: CacheCandidate::Live,
            requires_kv_clear: false,
        })
    }

    fn admit_cold(&mut self, context_key: &str) -> Option<KvCacheAdmission> {
        let (seq_id, requires_kv_clear) = self.select_cold_target(context_key)?;
        let index = seq_index(seq_id, self.physical.len())?;
        self.prefix_state_cache
            .drop_pending_snapshots_for_seq(seq_id);
        self.physical[index].state = SeqState::Free;
        self.physical[index].generation = self.physical[index].generation.saturating_add(1);
        self.physical[index].state = SeqState::Leased;
        self.remove_lru_key(context_key);
        match self.sessions.entry(context_key.to_string()) {
            Entry::Occupied(mut occupied) => {
                let record = occupied.get_mut();
                record.resident = None;
                record.in_flight = true;
            }
            Entry::Vacant(vacant) => {
                vacant.insert(SessionRecord {
                    resident: None,
                    in_flight: true,
                });
            }
        }
        Some(KvCacheAdmission {
            seq_id,
            generation: self.physical[index].generation,
            mirror: SequenceMirror::default(),
            candidate: CacheCandidate::None,
            requires_kv_clear,
        })
    }

    fn select_cold_target(&mut self, context_key: &str) -> Option<(llama_seq_id, bool)> {
        if let Some(seq_id) = self.take_own_resident_target(context_key) {
            return Some((seq_id, true));
        }
        if self.sessions.contains_key(context_key) || self.sessions.len() < self.physical.len() {
            if let Some(target) = self.first_free_sequence_target() {
                return Some(target);
            }
        }
        if let Some(seq_id) = self.evict_lru_idle_session() {
            return Some((seq_id, true));
        }
        self.first_free_sequence_target()
    }

    fn take_own_resident_target(&mut self, context_key: &str) -> Option<llama_seq_id> {
        let resident = self.sessions.get(context_key)?.resident?;
        let index = self.valid_resident_index(resident)?;
        self.remove_lru_key(context_key);
        if let Some(record) = self.sessions.get_mut(context_key) {
            record.resident = None;
        }
        self.physical[index].state = SeqState::Free;
        Some(resident.seq_id)
    }

    fn evict_lru_idle_session(&mut self) -> Option<llama_seq_id> {
        while let Some(context_key) = self.idle_lru.pop_front() {
            let Some(record) = self.sessions.get(&context_key).cloned() else {
                continue;
            };
            if record.in_flight {
                continue;
            }
            let Some(resident) = record.resident else {
                self.sessions.remove(&context_key);
                continue;
            };
            let Some(index) = self.valid_resident_index(resident) else {
                self.sessions.remove(&context_key);
                continue;
            };
            self.sessions.remove(&context_key);
            self.physical[index].state = SeqState::Free;
            return Some(resident.seq_id);
        }
        None
    }

    fn set_session_idle(&mut self, context_key: &str, resident: ResidentRef) {
        let record = self
            .sessions
            .entry(context_key.to_string())
            .or_insert_with(SessionRecord::default);
        record.resident = Some(resident);
        record.in_flight = false;
        self.refresh_lru_key(context_key);
    }

    fn clear_session(&mut self, context_key: &str) {
        self.remove_lru_key(context_key);
        self.sessions.remove(context_key);
    }

    fn evict_sequence(&mut self, seq_id: llama_seq_id) {
        let Some(index) = seq_index(seq_id, self.physical.len()) else {
            return;
        };
        self.prefix_state_cache
            .drop_pending_snapshots_for_seq(seq_id);
        self.physical[index].state = SeqState::Free;
        self.physical[index].generation = self.physical[index].generation.saturating_add(1);
    }

    fn prune_stale_idle_sessions(&mut self) {
        let mut retained = VecDeque::with_capacity(self.idle_lru.capacity());
        while let Some(context_key) = self.idle_lru.pop_front() {
            let Some(record) = self.sessions.get(&context_key).cloned() else {
                continue;
            };
            if record.in_flight {
                continue;
            }
            let valid = record
                .resident
                .and_then(|resident| self.valid_resident_index(resident))
                .is_some();
            if valid {
                retained.push_back(context_key);
            } else {
                self.sessions.remove(&context_key);
            }
        }
        self.idle_lru = retained;
    }

    fn first_valid_idle_session_key(&self) -> Option<&str> {
        self.idle_lru.iter().find_map(|context_key| {
            let record = self.sessions.get(context_key)?;
            if record.in_flight {
                return None;
            }
            record
                .resident
                .and_then(|resident| self.valid_resident_index(resident))
                .map(|_| context_key.as_str())
        })
    }

    fn has_available_sequence(&self) -> bool {
        self.first_free_sequence_target().is_some() || self.first_valid_idle_session_key().is_some()
    }

    fn first_free_sequence_target(&self) -> Option<(llama_seq_id, bool)> {
        self.physical
            .iter()
            .enumerate()
            .find(|(_, sequence)| matches!(sequence.state, SeqState::Free))
            .and_then(|(index, sequence)| {
                let seq_id = llama_seq_id::try_from(index).ok()?;
                Some((seq_id, free_sequence_requires_clear(sequence)))
            })
    }

    fn valid_resident_index(&self, resident: ResidentRef) -> Option<usize> {
        let index = seq_index(resident.seq_id, self.physical.len())?;
        let sequence = &self.physical[index];
        (sequence.generation == resident.generation
            && matches!(sequence.state, SeqState::Idle { .. }))
        .then_some(index)
    }

    fn sequence_generation_matches(&self, seq_id: llama_seq_id, generation: u64) -> bool {
        seq_index(seq_id, self.physical.len())
            .is_some_and(|index| self.physical[index].generation == generation)
    }

    fn refresh_lru_key(&mut self, context_key: &str) {
        self.remove_lru_key(context_key);
        self.idle_lru.push_back(context_key.to_string());
    }

    fn remove_lru_key(&mut self, context_key: &str) {
        if let Some(index) = self.idle_lru.iter().position(|key| key == context_key) {
            self.idle_lru.remove(index);
        }
    }
}

impl Default for KvCacheManager {
    fn default() -> Self {
        Self::new(1)
    }
}

fn live_reuse_enabled(mode: KvReuseMode) -> bool {
    matches!(
        mode,
        KvReuseMode::LiveSlotPrefix | KvReuseMode::LiveSlotAndSnapshot
    )
}

fn seq_index(seq_id: llama_seq_id, len: usize) -> Option<usize> {
    let index = usize::try_from(seq_id).ok()?;
    (index < len).then_some(index)
}

fn max_representable_sequences() -> usize {
    usize::try_from(llama_seq_id::MAX)
        .ok()
        .and_then(|value| value.checked_add(1))
        .unwrap_or(usize::MAX)
}

fn free_sequence_requires_clear(sequence: &PhysicalSequence) -> bool {
    sequence.generation > 0
}

#[cfg(test)]
#[path = "../../../tests/runtime/session/kv_cache_manager_tests.rs"]
mod kv_cache_manager_tests;
