//! Integration tests for the `cogentlm-shard` crate-level public_api surface.
//!
//! Covers shard and GGUF inspection helpers with deterministic byte fixtures and filesystem-free value checks where possible.

use cogentlm_shard::{BrowserCacheLayout, BrowserCachePolicy};

#[test]
fn default_cache_policy_keeps_small_models_as_single_files() {
    let policy = BrowserCachePolicy::default();

    assert_eq!(
        policy.resolve_layout(Some(policy.direct_load_max_bytes)),
        BrowserCacheLayout::SingleFile
    );
    assert_eq!(
        policy.resolve_layout(Some(policy.direct_load_max_bytes + 1)),
        BrowserCacheLayout::SplitGguf
    );
}

#[test]
fn unknown_source_size_uses_split_layout() {
    assert_eq!(
        BrowserCachePolicy::default().resolve_layout(None),
        BrowserCacheLayout::SplitGguf
    );
}
