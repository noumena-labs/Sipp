use crate::runtime::llama_token;

use super::{prefix_entry_approx_bytes, PrefixCacheEntry, PrefixCacheLookupKey, PrefixStateCache};

impl PrefixStateCache {
    pub fn insert_test_entry(&mut self, mut entry: PrefixCacheEntry) {
        let Some(approx_bytes) =
            prefix_entry_approx_bytes(entry.state_bytes.len(), entry.token_count)
        else {
            return;
        };
        entry.approx_bytes = approx_bytes;
        self.insert_or_update_entry(entry);
    }

    pub(super) fn insert_or_update_entry(&mut self, entry: PrefixCacheEntry) {
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
        context_key: &str,
        tokens: &[llama_token],
        token_count: usize,
        prefix_hash: u64,
    ) -> Option<usize> {
        let lookup_key = PrefixCacheLookupKey::new(model_fingerprint, token_count, prefix_hash);
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
mod tests {
    use std::time::Instant;

    use crate::runtime::session::PrefixCachePolicy;

    use super::*;

    fn entry(context_key: &str, tokens: Vec<llama_token>, priority: u64) -> PrefixCacheEntry {
        let policy = PrefixCachePolicy::new(2);
        let prefix_hash = policy.hash_prefix(&tokens, tokens.len());
        PrefixCacheEntry {
            model_fingerprint: 7,
            context_key: context_key.to_string(),
            token_count: tokens.len(),
            prefix_hash,
            retention_priority: priority,
            hit_count: 0,
            approx_bytes: 0,
            prefix_tokens: tokens,
            state_bytes: vec![1, 2, 3],
            last_used: Instant::now(),
        }
    }

    #[test]
    fn enforces_entry_limit_by_priority_then_hits() {
        let mut cache = PrefixStateCache::new(2, usize::MAX);
        cache.insert_test_entry(entry("low", vec![1, 2], 0));
        cache.insert_test_entry(entry("high", vec![1, 3], 10));
        cache.insert_test_entry(entry("mid", vec![1, 4], 5));

        assert_eq!(cache.len(), 2);
        let keys: Vec<_> = cache
            .entries
            .iter()
            .map(|entry| entry.context_key.as_str())
            .collect();
        assert!(!keys.contains(&"low"));
        assert!(keys.contains(&"high"));
        assert!(keys.contains(&"mid"));
    }

    #[test]
    fn rejects_invalid_test_entry_without_indexing_panic() {
        let mut cache = PrefixStateCache::default();
        let mut invalid = entry("bad", vec![1, 2], 1);
        invalid.token_count = 3;

        cache.insert_test_entry(invalid);

        assert!(cache.is_empty());
        assert_eq!(cache.total_approx_bytes(), 0);
    }

    #[test]
    fn rejects_entries_with_overflowing_approx_byte_count() {
        let mut cache = PrefixStateCache::default();
        let mut overflowing = entry("too-large", vec![1, 2], 1);
        overflowing.token_count = usize::MAX / std::mem::size_of::<llama_token>() + 1;

        cache.insert_test_entry(overflowing);

        assert!(cache.is_empty());
        assert_eq!(cache.total_approx_bytes(), 0);
        assert_eq!(prefix_entry_approx_bytes(usize::MAX, 1), None);
    }

    #[test]
    fn rejects_entries_when_total_approx_bytes_would_overflow() {
        let mut cache = PrefixStateCache::new(8, usize::MAX);
        cache.total_approx_bytes = usize::MAX - 1;

        cache.insert_test_entry(entry("overflow", vec![1, 2], 1));

        assert!(cache.is_empty());
        assert_eq!(cache.total_approx_bytes(), usize::MAX - 1);
    }

    #[test]
    fn replacement_rejects_total_approx_bytes_overflow_without_mutating_entry() {
        let mut cache = PrefixStateCache::new(8, usize::MAX);
        cache.insert_test_entry(entry("ctx", vec![1, 2], 1));
        cache.total_approx_bytes = usize::MAX;
        let original = cache.entries[0].clone();
        let mut replacement = entry("ctx", vec![1, 2], 99);
        replacement.state_bytes = vec![1, 2, 3, 4];

        cache.insert_test_entry(replacement);

        assert_eq!(cache.entries[0], original);
        assert_eq!(cache.total_approx_bytes(), usize::MAX);
    }
}
