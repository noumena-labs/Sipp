//! Snapshot prefix-cache: LRU+priority store of llama.cpp state buffers keyed by (model, scope, prefix-hash).

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::defaults::BYTES_PER_MIB;
use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::{llama_seq_id, llama_token};

use super::PrefixCachePolicy;

mod state_io;
mod storage;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCacheEntry {
    pub model_fingerprint: u64,
    pub snapshot_scope: String,
    pub token_count: usize,
    pub prefix_hash: u64,
    pub retention_priority: u64,
    pub hit_count: u64,
    pub approx_bytes: usize,
    pub prefix_tokens: Vec<llama_token>,
    pub state_bytes: Vec<u8>,
    pub last_used: Instant,
}

pub(super) struct PrefixStateStoreRequest<'a> {
    state_bytes: Vec<u8>,
    seq_id: llama_seq_id,
    model_fingerprint: u64,
    snapshot_scope: &'a str,
    tokens: &'a [llama_token],
    token_count: usize,
    prefix_hash: u64,
    retention_priority: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::runtime::session) struct PendingPrefixSnapshot {
    pub seq_id: llama_seq_id,
    pub generation: u64,
    pub model_fingerprint: u64,
    pub snapshot_scope: String,
    pub token_count: usize,
    pub prefix_hash: u64,
    pub retention_priority: u64,
    pub prefix_tokens: Vec<llama_token>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PrefixCacheLookupKey {
    pub model_fingerprint: u64,
    pub snapshot_scope: String,
    pub token_count: usize,
    pub prefix_hash: u64,
}

impl PrefixCacheLookupKey {
    pub(super) fn new(
        model_fingerprint: u64,
        snapshot_scope: &str,
        token_count: usize,
        prefix_hash: u64,
    ) -> Self {
        Self {
            model_fingerprint,
            snapshot_scope: snapshot_scope.to_string(),
            token_count,
            prefix_hash,
        }
    }

    pub(super) fn for_entry(entry: &PrefixCacheEntry) -> Self {
        Self::new(
            entry.model_fingerprint,
            &entry.snapshot_scope,
            entry.token_count,
            entry.prefix_hash,
        )
    }
}

/// Opaque handle returned by [`PrefixStateCache::find_best_prefix_handle`].
///
/// Carrying the index rather than `&PrefixCacheEntry` lets callers release the
/// outstanding borrow before invoking other cache methods (e.g.
/// `restore_by_handle`) without having to clone the entry's `state_bytes`
/// payload, which can be hundreds of megabytes for real models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCacheHandle {
    pub(crate) index: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone)]
pub struct PrefixStateCache {
    pub(crate) entries: Vec<PrefixCacheEntry>,
    pub(super) lookup_buckets: HashMap<PrefixCacheLookupKey, Vec<usize>>,
    pub(super) pending_snapshots: VecDeque<PendingPrefixSnapshot>,
    pub(super) max_entries: usize,
    pub(super) max_total_bytes: usize,
    pub(super) total_approx_bytes: usize,
}

impl PrefixStateCache {
    pub fn new(max_entries: usize, max_total_bytes: usize) -> Self {
        let max_entries = max_entries.max(1);
        Self {
            entries: Vec::with_capacity(max_entries),
            lookup_buckets: HashMap::with_capacity(max_entries),
            pending_snapshots: VecDeque::new(),
            max_entries,
            max_total_bytes: max_total_bytes.max(1),
            total_approx_bytes: 0,
        }
    }

    #[cfg(test)]
    pub(super) fn find_best_prefix(
        &mut self,
        model_fingerprint: u64,
        snapshot_scope: &str,
        prompt_tokens: &[llama_token],
        prefix_cache_policy: &mut PrefixCachePolicy,
    ) -> Option<&PrefixCacheEntry> {
        let handle = self.find_best_prefix_handle(
            model_fingerprint,
            snapshot_scope,
            prompt_tokens,
            prefix_cache_policy,
        )?;
        self.entries.get(handle.index)
    }

    /// Returns a handle to the best matching prefix without holding a borrow
    /// on the entry payload. Callers then use [`Self::restore_by_handle`] and
    /// [`Self::entry_by_handle`] to read individual fields, which lets them
    /// skip cloning the entry's potentially huge `state_bytes` Vec.
    pub fn find_best_prefix_handle(
        &mut self,
        model_fingerprint: u64,
        snapshot_scope: &str,
        prompt_tokens: &[llama_token],
        prefix_cache_policy: &mut PrefixCachePolicy,
    ) -> Option<PrefixCacheHandle> {
        prefix_cache_policy.record_lookup();
        let best_index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                prefix_entry_matches(entry, model_fingerprint, snapshot_scope, prompt_tokens)
            })
            .max_by(|(_, left), (_, right)| {
                left.token_count
                    .cmp(&right.token_count)
                    .then(left.retention_priority.cmp(&right.retention_priority))
                    .then(left.last_used.cmp(&right.last_used))
            })
            .map(|(index, _)| index)?;

        let token_count = self.entries[best_index].token_count;
        self.entries[best_index].hit_count = self.entries[best_index].hit_count.saturating_add(1);
        self.entries[best_index].last_used = Instant::now();
        prefix_cache_policy.record_hit(token_count);
        Some(PrefixCacheHandle {
            index: best_index,
            token_count,
        })
    }

    /// Restores a cached prefix into `seq_id` without exposing the entry's
    /// `state_bytes` to the caller. Returns `false` when the handle is stale
    /// or the underlying llama call fails.
    pub(crate) fn restore_by_handle(
        &self,
        runtime: &mut NativeRuntimeHandle,
        seq_id: llama_seq_id,
        handle: PrefixCacheHandle,
    ) -> bool {
        let Some(entry) = self.entries.get(handle.index) else {
            return false;
        };
        self.restore_prefix_state(runtime, seq_id, entry)
    }
}

fn prefix_entry_matches(
    entry: &PrefixCacheEntry,
    model_fingerprint: u64,
    snapshot_scope: &str,
    prompt_tokens: &[llama_token],
) -> bool {
    entry.model_fingerprint == model_fingerprint
        && entry.snapshot_scope == snapshot_scope
        && entry.token_count <= prompt_tokens.len()
        && entry.prefix_tokens.len() == entry.token_count
        && entry.prefix_tokens.as_slice() == &prompt_tokens[..entry.token_count]
}

impl Default for PrefixStateCache {
    fn default() -> Self {
        Self::new(32, 256 * BYTES_PER_MIB)
    }
}

pub(super) fn prefix_entry_approx_bytes(
    state_byte_len: usize,
    token_count: usize,
) -> Option<usize> {
    let token_bytes = token_count.checked_mul(std::mem::size_of::<llama_token>())?;
    state_byte_len.checked_add(token_bytes)
}

#[cfg(test)]
#[path = "../../../tests/runtime/session/prefix_state_cache_tests.rs"]
mod prefix_state_cache_tests;
