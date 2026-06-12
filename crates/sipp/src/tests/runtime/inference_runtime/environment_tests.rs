//! Tests the `runtime::inference_runtime::environment` module in
//! `sipp`.
//!
//! Covers runtime environment helper decisions with empty native handles and
//! pure runtime config values; residency acquisition is intentionally skipped.

use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::config::{KvReuseMode, NativeRuntimeConfig};

use super::*;

#[test]
fn batch_token_budget_prefers_config_and_clamps_native_fallback() {
    let runtime = NativeRuntimeHandle::empty_for_tests();
    let mut config = NativeRuntimeConfig::default();

    assert_eq!(resolve_batch_token_budget(&runtime, &config), 1);

    config.context.n_batch = Some(8);
    assert_eq!(resolve_batch_token_budget(&runtime, &config), 8);

    config.context.n_batch = Some(-8);
    assert_eq!(resolve_batch_token_budget(&runtime, &config), 1);
}

#[test]
fn prefix_cache_environment_helpers_follow_reuse_mode() {
    let cases = [
        (KvReuseMode::Disabled, false, false, 0),
        (KvReuseMode::LiveSlotPrefix, true, false, 12),
        (KvReuseMode::StateSnapshot, false, true, 0),
        (KvReuseMode::LiveSlotAndSnapshot, true, true, 12),
    ];

    for (mode, live_enabled, snapshot_enabled, retained) in cases {
        let mut config = NativeRuntimeConfig::default();
        config.cache.mode = mode;
        config.cache.retained_prefix_tokens = 12;

        assert_eq!(snapshot_prefix_cache_enabled(mode), snapshot_enabled);
        assert_eq!(live_retained_prefix_tokens(&config), retained);
        assert_eq!(live_retained_prefix_tokens(&config) > 0, live_enabled);
    }
}
