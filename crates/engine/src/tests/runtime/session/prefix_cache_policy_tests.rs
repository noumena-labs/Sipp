//! Tests the `runtime::session::prefix_cache_policy` module in `cogentlm-engine`.
//!
//! Covers runtime support modules with deterministic in-memory fixtures and no native model execution.

use super::*;

#[test]
fn interval_zero_stores_only_terminal_boundaries() {
    let policy = PrefixCachePolicy::new(0);
    assert!(!policy.should_store_boundary(31, 64));
    assert!(!policy.should_store_boundary(32, 64));
    assert!(policy.should_store_boundary(64, 64));
}

#[test]
fn terminal_boundary_can_be_below_interval_minimum() {
    let policy = PrefixCachePolicy::new(128);

    assert!(policy.should_store_boundary(18, 18));
    assert!(!policy.should_store_boundary(18, 19));
}

#[test]
fn hash_prefix_clamps_to_available_tokens() {
    let policy = PrefixCachePolicy::new(4);
    let tokens = vec![1, 2, 3];
    assert_eq!(
        policy.hash_prefix(&tokens, 99),
        policy.hash_prefix(&tokens, 3)
    );
    assert_ne!(policy.hash_prefix(&tokens, 2), 0);
}

#[test]
fn stats_counters_saturate_at_u64_max() {
    let mut policy = PrefixCachePolicy {
        prefix_cache_interval_tokens: 4,
        minimum_prefix_cache_tokens: 4,
        stats: PrefixCachePolicyStats {
            lookup_count: u64::MAX,
            hit_count: u64::MAX,
            store_count: u64::MAX,
            restored_token_count: u64::MAX - 1,
            stored_token_count: u64::MAX - 1,
        },
    };

    policy.record_lookup();
    policy.record_hit(10);
    policy.record_store(10);

    assert_eq!(
        policy.stats,
        PrefixCachePolicyStats {
            lookup_count: u64::MAX,
            hit_count: u64::MAX,
            store_count: u64::MAX,
            restored_token_count: u64::MAX,
            stored_token_count: u64::MAX,
        }
    );
}

#[test]
fn negative_tokens_hash_by_stable_bit_pattern() {
    assert_eq!(
        mix_prefix_hash_token(PREFIX_HASH_SEED, -1),
        (PREFIX_HASH_SEED ^ u64::from(u32::MAX)).wrapping_mul(PREFIX_HASH_PRIME)
    );
}

#[test]
fn boundary_helpers_preserve_interval_and_terminal_rules() {
    assert_eq!(
        minimum_prefix_cache_tokens(0),
        MAX_MINIMUM_PREFIX_CACHE_TOKENS
    );
    assert_eq!(minimum_prefix_cache_tokens(4), 4);
    assert_eq!(
        minimum_prefix_cache_tokens(MAX_MINIMUM_PREFIX_CACHE_TOKENS + 1),
        MAX_MINIMUM_PREFIX_CACHE_TOKENS
    );
    assert!(is_terminal_boundary(10, 10));
    assert!(!is_terminal_boundary(8, 10));
    assert!(is_interval_boundary(8, 4));
    assert!(!is_interval_boundary(8, 0));
}
