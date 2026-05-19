//! Snapshot prefix-cache: LRU+priority store of llama.cpp state buffers keyed by (model, context_key, prefix-hash).

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::time::Instant;

use cogentlm_sys as ffi;

use crate::runtime::{llama_seq_id, llama_token};

use super::{PrefixCachePolicy, PrefixCachePolicyStats};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCacheEntry {
    pub model_fingerprint: u64,
    pub context_key: String,
    pub token_count: usize,
    pub prefix_hash: u64,
    pub retention_priority: u64,
    pub hit_count: u64,
    pub approx_bytes: usize,
    pub prefix_tokens: Vec<llama_token>,
    pub state_bytes: Vec<u8>,
    pub last_used: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingPrefixSnapshot {
    pub model_fingerprint: u64,
    pub context_key: String,
    pub seq_id: llama_seq_id,
    pub token_count: usize,
    pub prefix_hash: u64,
    pub retention_priority: u64,
    pub prefix_tokens: Vec<llama_token>,
}

struct PrefixStateStoreRequest<'a> {
    context: *mut ffi::llama_context,
    seq_id: llama_seq_id,
    model_fingerprint: u64,
    context_key: &'a str,
    tokens: &'a [llama_token],
    token_count: usize,
    prefix_hash: u64,
    retention_priority: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PrefixCacheLookupKey {
    pub model_fingerprint: u64,
    pub token_count: usize,
    pub prefix_hash: u64,
}

/// Opaque handle returned by [`PrefixStateCache::find_best_prefix_handle`].
///
/// Carrying the index rather than `&PrefixCacheEntry` lets callers release the
/// outstanding borrow before invoking other cache methods (e.g.
/// `restore_by_handle`) without having to clone the entry's `state_bytes`
/// payload, which can be hundreds of megabytes for real models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCacheHandle {
    index: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone)]
pub struct PrefixStateCache {
    entries: Vec<PrefixCacheEntry>,
    lookup_buckets: HashMap<PrefixCacheLookupKey, Vec<usize>>,
    pending_snapshots: VecDeque<PendingPrefixSnapshot>,
    max_entries: usize,
    max_total_bytes: usize,
    total_approx_bytes: usize,
}

impl PrefixStateCache {
    pub fn new(max_entries: usize, max_total_bytes: usize) -> Self {
        let max_entries = max_entries.max(1);
        Self {
            entries: Vec::with_capacity(max_entries),
            lookup_buckets: HashMap::with_capacity(max_entries),
            pending_snapshots: VecDeque::with_capacity(max_entries),
            max_entries,
            max_total_bytes: max_total_bytes.max(1),
            total_approx_bytes: 0,
        }
    }

    pub fn find_best_prefix(
        &mut self,
        model_fingerprint: u64,
        context_key: &str,
        prompt_tokens: &[llama_token],
        prefix_cache_policy: &mut PrefixCachePolicy,
    ) -> Option<&PrefixCacheEntry> {
        let handle = self.find_best_prefix_handle(
            model_fingerprint,
            context_key,
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
        context_key: &str,
        prompt_tokens: &[llama_token],
        prefix_cache_policy: &mut PrefixCachePolicy,
    ) -> Option<PrefixCacheHandle> {
        prefix_cache_policy.record_lookup();
        let candidates = prefix_cache_policy.build_candidate_boundaries(prompt_tokens);
        if candidates.is_empty() {
            return None;
        }

        for candidate in candidates {
            let lookup_key = PrefixCacheLookupKey {
                model_fingerprint,
                token_count: candidate.token_count,
                prefix_hash: candidate.prefix_hash,
            };
            let Some(bucket) = self.lookup_buckets.get(&lookup_key) else {
                continue;
            };

            let mut best_index: Option<usize> = None;
            for &entry_index in bucket {
                let Some(entry) = self.entries.get(entry_index) else {
                    continue;
                };
                if entry.prefix_tokens.len() != candidate.token_count {
                    continue;
                }
                if entry.prefix_tokens.as_slice() != &prompt_tokens[..candidate.token_count] {
                    continue;
                }

                let prefer_entry = best_index
                    .and_then(|index| self.entries.get(index))
                    .is_none_or(|best_entry: &PrefixCacheEntry| {
                        (entry.context_key == context_key && best_entry.context_key != context_key)
                            || (entry.context_key == best_entry.context_key
                                && entry.retention_priority > best_entry.retention_priority)
                            || (entry.context_key == best_entry.context_key
                                && entry.retention_priority == best_entry.retention_priority
                                && entry.last_used > best_entry.last_used)
                    });
                if prefer_entry {
                    best_index = Some(entry_index);
                }
            }

            if let Some(best_index) = best_index {
                let token_count = self.entries[best_index].token_count;
                self.entries[best_index].hit_count =
                    self.entries[best_index].hit_count.saturating_add(1);
                self.entries[best_index].last_used = Instant::now();
                prefix_cache_policy.record_hit(token_count);
                return Some(PrefixCacheHandle {
                    index: best_index,
                    token_count,
                });
            }
        }

        None
    }

    pub fn entry_by_handle(&self, handle: PrefixCacheHandle) -> Option<&PrefixCacheEntry> {
        self.entries.get(handle.index)
    }

    /// Restores a cached prefix into `seq_id` without exposing the entry's
    /// `state_bytes` to the caller. Returns `false` when the handle is stale
    /// or the underlying llama call fails.
    pub(crate) fn restore_by_handle(
        &self,
        context: *mut ffi::llama_context,
        seq_id: llama_seq_id,
        handle: PrefixCacheHandle,
    ) -> bool {
        let Some(entry) = self.entries.get(handle.index) else {
            return false;
        };
        self.restore_prefix_state(context, seq_id, entry)
    }

    fn store_prefix_state(&mut self, request: PrefixStateStoreRequest<'_>) -> bool {
        if request.context.is_null()
            || request.seq_id < 0
            || request.token_count == 0
            || request.token_count > request.tokens.len()
        {
            return false;
        }

        let mut data_ptr: *mut u8 = std::ptr::null_mut();
        let mut prefix_state_size = 0_usize;
        let ok = unsafe {
            ffi::cogent_llama_state_seq_get_data_ext_alloc(
                request.context,
                request.seq_id,
                ffi::LLAMA_STATE_SEQ_FLAGS_NONE,
                &mut data_ptr,
                &mut prefix_state_size,
            )
        };
        if !ok || data_ptr.is_null() || prefix_state_size == 0 {
            return false;
        }

        let Some(approx_bytes) = prefix_entry_approx_bytes(prefix_state_size, request.token_count)
        else {
            unsafe {
                ffi::cogent_free_buffer(data_ptr.cast::<c_void>());
            }
            return false;
        };
        let state_bytes =
            unsafe { std::slice::from_raw_parts(data_ptr, prefix_state_size) }.to_vec();
        unsafe {
            ffi::cogent_free_buffer(data_ptr.cast::<c_void>());
        }

        self.insert_or_update_entry(PrefixCacheEntry {
            model_fingerprint: request.model_fingerprint,
            context_key: request.context_key.to_string(),
            token_count: request.token_count,
            prefix_hash: request.prefix_hash,
            retention_priority: request.retention_priority,
            hit_count: 0,
            approx_bytes,
            prefix_tokens: request.tokens[..request.token_count].to_vec(),
            state_bytes,
            last_used: Instant::now(),
        });
        true
    }

    pub fn insert_test_entry(&mut self, mut entry: PrefixCacheEntry) {
        let Some(approx_bytes) =
            prefix_entry_approx_bytes(entry.state_bytes.len(), entry.token_count)
        else {
            return;
        };
        entry.approx_bytes = approx_bytes;
        self.insert_or_update_entry(entry);
    }

    pub(crate) fn restore_prefix_state(
        &self,
        context: *mut ffi::llama_context,
        seq_id: llama_seq_id,
        entry: &PrefixCacheEntry,
    ) -> bool {
        if context.is_null() || seq_id < 0 || entry.state_bytes.is_empty() {
            return false;
        }
        unsafe {
            ffi::cogent_llama_state_seq_set_data_ext(
                context,
                seq_id,
                ffi::LLAMA_STATE_SEQ_FLAGS_NONE,
                entry.state_bytes.as_ptr(),
                entry.state_bytes.len(),
            )
        }
    }

    pub fn enqueue_pending_snapshot(&mut self, snapshot: PendingPrefixSnapshot) {
        if snapshot.seq_id < 0
            || snapshot.token_count == 0
            || snapshot.prefix_tokens.len() < snapshot.token_count
        {
            return;
        }

        if let Some(existing) = self.pending_snapshots.iter_mut().find(|pending| {
            pending.seq_id == snapshot.seq_id
                && pending.context_key == snapshot.context_key
                && pending.model_fingerprint == snapshot.model_fingerprint
                && pending.token_count == snapshot.token_count
        }) {
            *existing = snapshot;
            return;
        }

        self.pending_snapshots.push_back(snapshot);
    }

    pub fn pending_snapshot_count(&self) -> usize {
        self.pending_snapshots.len()
    }

    pub fn drain_pending_snapshots(
        &mut self,
        context: *mut ffi::llama_context,
        max_to_drain: usize,
    ) -> usize {
        if context.is_null() || self.pending_snapshots.is_empty() {
            return 0;
        }

        let budget = if max_to_drain == 0 {
            self.pending_snapshots.len()
        } else {
            max_to_drain.min(self.pending_snapshots.len())
        };
        let mut drained = 0;
        while drained < budget {
            let Some(pending) = self.pending_snapshots.pop_front() else {
                break;
            };
            self.store_prefix_state(PrefixStateStoreRequest {
                context,
                seq_id: pending.seq_id,
                model_fingerprint: pending.model_fingerprint,
                context_key: &pending.context_key,
                tokens: &pending.prefix_tokens,
                token_count: pending.token_count,
                prefix_hash: pending.prefix_hash,
                retention_priority: pending.retention_priority,
            });
            drained += 1;
        }
        drained
    }

    pub fn drain_pending_snapshots_for_seq(
        &mut self,
        context: *mut ffi::llama_context,
        seq_id: llama_seq_id,
        max_to_drain: usize,
    ) -> usize {
        if context.is_null() || seq_id < 0 || self.pending_snapshots.is_empty() {
            return 0;
        }

        let mut retained = VecDeque::with_capacity(self.pending_snapshots.len());
        let mut drained = 0;
        while let Some(pending) = self.pending_snapshots.pop_front() {
            let budget_remaining = max_to_drain == 0 || drained < max_to_drain;
            if pending.seq_id == seq_id && budget_remaining {
                self.store_prefix_state(PrefixStateStoreRequest {
                    context,
                    seq_id: pending.seq_id,
                    model_fingerprint: pending.model_fingerprint,
                    context_key: &pending.context_key,
                    tokens: &pending.prefix_tokens,
                    token_count: pending.token_count,
                    prefix_hash: pending.prefix_hash,
                    retention_priority: pending.retention_priority,
                });
                drained += 1;
            } else {
                retained.push_back(pending);
            }
        }
        self.pending_snapshots = retained;
        drained
    }

    pub fn drain_best_pending_snapshot_for_seq(
        &mut self,
        context: *mut ffi::llama_context,
        seq_id: llama_seq_id,
    ) -> usize {
        if context.is_null() || seq_id < 0 || self.pending_snapshots.is_empty() {
            return 0;
        }

        let mut best_index = None;
        let mut max_tokens = 0;
        for (index, pending) in self.pending_snapshots.iter().enumerate() {
            if pending.seq_id == seq_id && pending.token_count > max_tokens {
                max_tokens = pending.token_count;
                best_index = Some(index);
            }
        }

        let mut drained = 0;
        if let Some(best_idx) = best_index {
            if let Some(pending) = self.pending_snapshots.remove(best_idx) {
                self.store_prefix_state(PrefixStateStoreRequest {
                    context,
                    seq_id: pending.seq_id,
                    model_fingerprint: pending.model_fingerprint,
                    context_key: &pending.context_key,
                    tokens: &pending.prefix_tokens,
                    token_count: pending.token_count,
                    prefix_hash: pending.prefix_hash,
                    retention_priority: pending.retention_priority,
                });
                drained = 1;
            }
        }

        self.pending_snapshots.retain(|snapshot| snapshot.seq_id != seq_id);
        drained
    }

    pub fn drop_pending_snapshots_for_seq(&mut self, seq_id: llama_seq_id) {
        if seq_id < 0 {
            return;
        }
        self.pending_snapshots
            .retain(|snapshot| snapshot.seq_id != seq_id);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lookup_buckets.clear();
        self.pending_snapshots.clear();
        self.total_approx_bytes = 0;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn total_approx_bytes(&self) -> usize {
        self.total_approx_bytes
    }

    pub fn policy_stats(policy: &PrefixCachePolicy) -> PrefixCachePolicyStats {
        policy.stats()
    }

    fn insert_or_update_entry(&mut self, entry: PrefixCacheEntry) {
        if entry.token_count == 0 || entry.token_count > entry.prefix_tokens.len() {
            return;
        }

        if let Some(existing_index) = self.find_existing_entry_index(
            entry.model_fingerprint,
            &entry.context_key,
            &entry.prefix_tokens,
            entry.token_count,
            entry.prefix_hash,
        ) {
            // Replacing an existing entry keeps the bucket key identical; just update byte accounting.
            let Some(total_without_existing) = self
                .total_approx_bytes
                .checked_sub(self.entries[existing_index].approx_bytes)
            else {
                return;
            };
            let next_total = total_without_existing.checked_add(entry.approx_bytes);
            let Some(next_total) = next_total else {
                return;
            };
            self.entries[existing_index] = entry;
            self.total_approx_bytes = next_total;
        } else {
            let new_index = self.entries.len();
            let bucket_key = PrefixCacheLookupKey {
                model_fingerprint: entry.model_fingerprint,
                token_count: entry.token_count,
                prefix_hash: entry.prefix_hash,
            };
            let Some(next_total) = self.total_approx_bytes.checked_add(entry.approx_bytes) else {
                return;
            };
            self.entries.push(entry);
            self.total_approx_bytes = next_total;
            self.lookup_buckets
                .entry(bucket_key)
                .or_default()
                .push(new_index);
        }
        self.enforce_limit();
    }

    fn find_existing_entry_index(
        &self,
        model_fingerprint: u64,
        context_key: &str,
        tokens: &[llama_token],
        token_count: usize,
        prefix_hash: u64,
    ) -> Option<usize> {
        let lookup_key = PrefixCacheLookupKey {
            model_fingerprint,
            token_count,
            prefix_hash,
        };
        self.lookup_buckets.get(&lookup_key).and_then(|bucket| {
            bucket.iter().copied().find(|&entry_index| {
                self.entries.get(entry_index).is_some_and(|entry| {
                    entry.context_key == context_key
                        && entry.prefix_tokens.len() == token_count
                        && token_count <= tokens.len()
                        && entry.prefix_tokens.as_slice() == &tokens[..token_count]
                })
            })
        })
    }

    fn enforce_limit(&mut self) {
        while self.entries.len() > self.max_entries
            || self.total_approx_bytes > self.max_total_bytes
        {
            let Some(evict_index) = self
                .entries
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    left.retention_priority
                        .cmp(&right.retention_priority)
                        .then(left.hit_count.cmp(&right.hit_count))
                        .then(left.last_used.cmp(&right.last_used))
                })
                .map(|(index, _)| index)
            else {
                break;
            };
            self.remove_entry_at(evict_index);
        }
    }

    /// `Vec::remove` shifts every later element down by one, so every bucket
    /// that points at an index > `evict_index` needs that index decremented.
    /// We also delete the bucket entry that pointed at the removed slot.
    fn remove_entry_at(&mut self, evict_index: usize) {
        let len = self.entries.len();
        if evict_index >= len {
            return;
        }
        let last_index = len - 1;
        let removed = self.entries.swap_remove(evict_index);
        debug_assert!(removed.approx_bytes <= self.total_approx_bytes);
        self.total_approx_bytes = self.total_approx_bytes.saturating_sub(removed.approx_bytes);
        let removed_key = PrefixCacheLookupKey {
            model_fingerprint: removed.model_fingerprint,
            token_count: removed.token_count,
            prefix_hash: removed.prefix_hash,
        };
        if let Some(bucket) = self.lookup_buckets.get_mut(&removed_key) {
            bucket.retain(|index| *index != evict_index);
            if bucket.is_empty() {
                self.lookup_buckets.remove(&removed_key);
            }
        }
        if evict_index < last_index {
            let moved = &self.entries[evict_index];
            let moved_key = PrefixCacheLookupKey {
                model_fingerprint: moved.model_fingerprint,
                token_count: moved.token_count,
                prefix_hash: moved.prefix_hash,
            };
            if let Some(bucket) = self.lookup_buckets.get_mut(&moved_key) {
                for index in bucket.iter_mut() {
                    if *index == last_index {
                        *index = evict_index;
                        break;
                    }
                }
            }
        }
    }
}

impl Default for PrefixStateCache {
    fn default() -> Self {
        Self::new(32, 256 * 1024 * 1024)
    }
}

fn prefix_entry_approx_bytes(state_byte_len: usize, token_count: usize) -> Option<usize> {
    let token_bytes = token_count.checked_mul(std::mem::size_of::<llama_token>())?;
    state_byte_len.checked_add(token_bytes)
}

#[cfg(test)]
mod tests;
