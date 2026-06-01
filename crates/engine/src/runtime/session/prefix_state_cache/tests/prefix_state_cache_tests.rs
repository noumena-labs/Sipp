//! Unit tests for the parent module.

use super::super::*;
use crate::runtime::session::PrefixCachePolicy;

fn entry(snapshot_scope: &str, tokens: Vec<i32>, priority: u64) -> PrefixCacheEntry {
    let policy = PrefixCachePolicy::new(2);
    let prefix_hash = policy.hash_prefix(&tokens, tokens.len());
    PrefixCacheEntry {
        model_fingerprint: 7,
        snapshot_scope: snapshot_scope.to_string(),
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
fn finds_longest_matching_prefix_and_records_hit() {
    let mut cache = PrefixStateCache::default();
    cache.insert_test_entry(entry("a", vec![1, 2], 0));
    cache.insert_test_entry(entry("a", vec![1, 2, 3, 4], 0));
    let mut policy = PrefixCachePolicy::new(2);

    let found = cache
        .find_best_prefix(7, "a", &[1, 2, 3, 4, 9], &mut policy)
        .expect("prefix");

    assert_eq!(found.token_count, 4);
    assert_eq!(policy.stats.hit_count, 1);
    assert_eq!(policy.stats.restored_token_count, 4);
}

#[test]
fn matching_prefix_hit_count_saturates() {
    let mut cache = PrefixStateCache::default();
    let mut entry = entry("a", vec![1, 2], 0);
    entry.hit_count = u64::MAX;
    cache.insert_test_entry(entry);
    let mut policy = PrefixCachePolicy::new(2);

    let found = cache
        .find_best_prefix(7, "a", &[1, 2, 3], &mut policy)
        .expect("prefix");

    assert_eq!(found.hit_count, u64::MAX);
}

#[test]
fn isolates_by_snapshot_scope_then_prefers_priority() {
    let mut cache = PrefixStateCache::default();
    cache.insert_test_entry(entry("other", vec![1, 2], 99));
    cache.insert_test_entry(entry("target", vec![1, 2], 1));
    cache.insert_test_entry(entry("target", vec![1, 2], 2));
    let mut policy = PrefixCachePolicy::new(2);

    let found = cache
        .find_best_prefix(7, "target", &[1, 2, 3], &mut policy)
        .expect("prefix");

    assert_eq!(found.snapshot_scope, "target");
    assert_eq!(found.retention_priority, 2);
}

#[test]
fn finds_prompt_seed_prefix_below_interval_minimum() {
    let mut cache = PrefixStateCache::default();
    let tokens: Vec<_> = (0..18).collect();
    cache.insert_test_entry(entry("ctx", tokens, 0));
    let prompt: Vec<_> = (0..19).collect();
    let mut policy = PrefixCachePolicy::new(128);

    let found = cache
        .find_best_prefix(7, "ctx", &prompt, &mut policy)
        .expect("prompt seed prefix");

    assert_eq!(found.token_count, 18);
    assert_eq!(policy.stats.hit_count, 1);
    assert_eq!(policy.stats.restored_token_count, 18);
}

#[test]
fn prefix_state_cache_presizes_bounded_collections() {
    let cache = PrefixStateCache::new(5, 100);

    assert!(cache.entries.capacity() >= 5);
    assert!(cache.lookup_buckets.capacity() >= 5);
}
