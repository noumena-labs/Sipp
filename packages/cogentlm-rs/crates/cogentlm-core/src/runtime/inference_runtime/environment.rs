use cogentlm_sys as ffi;

use crate::backend::backend_observability_json;
use crate::error::Result;
use crate::runtime::config::{KvReuseMode, NativeRuntimeConfig};
use crate::runtime::residency::{acquire_residency_lease, ResidencyLease};

pub(super) fn resolve_batch_token_budget(
    shared_context: *mut ffi::llama_context,
    config: &NativeRuntimeConfig,
) -> i32 {
    if !shared_context.is_null() {
        return i32::try_from(unsafe { ffi::llama_n_batch(shared_context) })
            .unwrap_or(i32::MAX)
            .max(1);
    }
    config.context.n_batch.unwrap_or(1).max(1)
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
