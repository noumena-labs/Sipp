use crate::runtime::llama_token;

#[cfg(test)]
use super::prefix_entry_approx_bytes;
use super::{PrefixCacheEntry, PrefixCacheLookupKey, PrefixStateCache};

impl PrefixStateCache {
    #[cfg(test)]
    pub(super) fn insert_test_entry(&mut self, mut entry: PrefixCacheEntry) {
        let Some(approx_bytes) =
            prefix_entry_approx_bytes(entry.state_bytes.len(), entry.token_count)
        else {
            return;
        };
        entry.approx_bytes = approx_bytes;
        self.insert_or_update_entry(entry);
    }

    pub(super) fn insert_or_update_entry(&mut self, entry: PrefixCacheEntry) {
        if entry.token_count == 0 || entry.token_count != entry.prefix_tokens.len() {
            return;
        }

        if let Some(existing_index) = self.find_existing_entry_index(
            entry.model_fingerprint,
            &entry.snapshot_scope,
            &entry.prefix_tokens,
            entry.token_count,
            entry.prefix_hash,
        ) {
            let Some(total_without_existing) = self
                .total_approx_bytes
                .checked_sub(self.entries[existing_index].approx_bytes)
            else {
                return;
            };
            let Some(next_total) = total_without_existing.checked_add(entry.approx_bytes) else {
                return;
            };
            self.entries[existing_index] = entry;
            self.total_approx_bytes = next_total;
        } else {
            let new_index = self.entries.len();
            let bucket_key = PrefixCacheLookupKey::for_entry(&entry);
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
        snapshot_scope: &str,
        tokens: &[llama_token],
        token_count: usize,
        prefix_hash: u64,
    ) -> Option<usize> {
        let lookup_key =
            PrefixCacheLookupKey::new(model_fingerprint, snapshot_scope, token_count, prefix_hash);
        self.lookup_buckets.get(&lookup_key).and_then(|bucket| {
            bucket.iter().copied().find(|&entry_index| {
                self.entries.get(entry_index).is_some_and(|entry| {
                    entry.snapshot_scope == snapshot_scope
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

    fn remove_entry_at(&mut self, evict_index: usize) {
        let len = self.entries.len();
        if evict_index >= len {
            return;
        }

        let last_index = len - 1;
        let removed = self.entries.swap_remove(evict_index);
        debug_assert!(removed.approx_bytes <= self.total_approx_bytes);
        self.total_approx_bytes = self.total_approx_bytes.saturating_sub(removed.approx_bytes);
        self.remove_bucket_index(PrefixCacheLookupKey::for_entry(&removed), evict_index);

        if evict_index < last_index {
            self.repoint_moved_entry_bucket(evict_index, last_index);
        }
    }

    fn remove_bucket_index(&mut self, key: PrefixCacheLookupKey, removed_index: usize) {
        if let Some(bucket) = self.lookup_buckets.get_mut(&key) {
            bucket.retain(|index| *index != removed_index);
            if bucket.is_empty() {
                self.lookup_buckets.remove(&key);
            }
        }
    }

    fn repoint_moved_entry_bucket(&mut self, moved_index: usize, previous_index: usize) {
        let moved = &self.entries[moved_index];
        let moved_key = PrefixCacheLookupKey::for_entry(moved);
        if let Some(bucket) = self.lookup_buckets.get_mut(&moved_key) {
            for index in bucket.iter_mut() {
                if *index == previous_index {
                    *index = moved_index;
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/runtime/session/prefix_state_cache/storage_tests.rs"]
mod storage_tests;
