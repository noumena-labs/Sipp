use crate::backend::backend_observability_json;
use crate::error::Result;
use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::config::{KvReuseMode, NativeRuntimeConfig};
use crate::runtime::residency::{acquire_residency_lease, ResidencyLease};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/environment_tests.rs"]
mod environment_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(super) fn resolve_batch_token_budget(
    native_runtime: &NativeRuntimeHandle,
    config: &NativeRuntimeConfig,
) -> i32 {
    config
        .context
        .n_batch
        .unwrap_or_else(|| native_runtime.n_batch())
        .max(1)
}

fn live_prefix_reuse_enabled(mode: KvReuseMode) -> bool {
    matches!(
        mode,
        KvReuseMode::LiveSlotPrefix | KvReuseMode::LiveSlotAndSnapshot
    )
}

pub(super) fn snapshot_prefix_cache_enabled(mode: KvReuseMode) -> bool {
    matches!(
        mode,
        KvReuseMode::StateSnapshot | KvReuseMode::LiveSlotAndSnapshot
    )
}

pub(super) fn live_retained_prefix_tokens(config: &NativeRuntimeConfig) -> i32 {
    if live_prefix_reuse_enabled(config.cache.mode) {
        config.cache.retained_prefix_tokens
    } else {
        0
    }
}

pub(super) fn admit_runtime_residency(
    config: &NativeRuntimeConfig,
) -> Result<Option<ResidencyLease>> {
    let raw = backend_observability_json(true)?;
    acquire_residency_lease(config, &raw)
}
