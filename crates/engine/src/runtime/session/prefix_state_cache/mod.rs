//! Snapshot prefix-cache: LRU+priority store of llama.cpp state buffers keyed by (model, context_key, prefix-hash).

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use cogentlm_sys as ffi;

use crate::defaults::BYTES_PER_MIB;
use crate::runtime::{llama_seq_id, llama_token};

use super::PrefixCachePolicy;

mod snapshots;
mod state_io;
mod storage;

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

pub(super) struct PrefixStateStoreRequest<'a> {
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

impl PrefixCacheLookupKey {
    pub(super) fn new(model_fingerprint: u64, token_count: usize, prefix_hash: u64) -> Self {
        Self {
            model_fingerprint,
            token_count,
            prefix_hash,
        }
    }

    pub(super) fn for_entry(entry: &PrefixCacheEntry) -> Self {
        Self::new(
            entry.model_fingerprint,
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
            let lookup_key = PrefixCacheLookupKey::new(
                model_fingerprint,
                candidate.token_count,
                candidate.prefix_hash,
            );
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

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lookup_buckets.clear();
        self.pending_snapshots.clear();
        self.total_approx_bytes = 0;
    }
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
mod tests {
    mod prefix_state_cache_tests;
}
