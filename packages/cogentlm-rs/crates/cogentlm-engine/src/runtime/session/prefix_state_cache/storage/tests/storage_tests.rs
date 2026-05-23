//! Unit tests for the parent module.

use std::time::Instant;

use crate::runtime::session::PrefixCachePolicy;

use super::super::*;

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

    assert_eq!(cache.entries.len(), 2);
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

    assert!(cache.entries.is_empty());
    assert_eq!(cache.total_approx_bytes, 0);
}

#[test]
fn rejects_entries_with_overflowing_approx_byte_count() {
    let mut cache = PrefixStateCache::default();
    let mut overflowing = entry("too-large", vec![1, 2], 1);
    overflowing.token_count = usize::MAX / std::mem::size_of::<llama_token>() + 1;

    cache.insert_test_entry(overflowing);

    assert!(cache.entries.is_empty());
    assert_eq!(cache.total_approx_bytes, 0);
    assert_eq!(prefix_entry_approx_bytes(usize::MAX, 1), None);
}

#[test]
fn rejects_entries_when_total_approx_bytes_would_overflow() {
    let mut cache = PrefixStateCache::new(8, usize::MAX);
    cache.total_approx_bytes = usize::MAX - 1;

    cache.insert_test_entry(entry("overflow", vec![1, 2], 1));

    assert!(cache.entries.is_empty());
    assert_eq!(cache.total_approx_bytes, usize::MAX - 1);
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
    assert_eq!(cache.total_approx_bytes, usize::MAX);
}
