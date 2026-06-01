use std::time::Instant;

use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::scheduler::{SamplerCacheKey, SamplerHandle, SlotState};

use super::super::duration_ms;
use super::super::sampler::{attach_backend_sampler, create_sampler};
use crate::native_bridge::NativeRuntimeHandle;

pub(super) fn ensure_slot_sampler(
    slot: &mut SlotState,
    native_runtime: &mut NativeRuntimeHandle,
    config: &NativeRuntimeConfig,
    sampler_pool: &mut std::collections::HashMap<SamplerCacheKey, Vec<SamplerHandle>>,
) -> bool {
    let (grammar, json_schema, sampling) = slot
        .request()
        .map(|request| {
            (
                request.grammar.clone(),
                request.json_schema.clone(),
                request.sampling.clone(),
            )
        })
        .unwrap_or_default();

    let sampling_json = match config.try_sampling_json_with_override(sampling.as_ref()) {
        Ok(sampling_json) => sampling_json,
        Err(error) => {
            slot.fail(format!(
                "Failed to serialize sampler configuration: {error}"
            ));
            return false;
        }
    };
    let key = SamplerCacheKey {
        sampling_json,
        grammar: grammar.clone(),
        json_schema: json_schema.clone(),
    };

    if let Some(sampler) = sampler_pool.get_mut(&key).and_then(|vec| vec.pop()) {
        slot.set_sampler(sampler);
        slot.sampler_key = Some(key);
        record_backend_sampler_attach(slot, native_runtime);
        return true;
    }

    match create_sampler(
        native_runtime,
        config,
        sampling.as_ref(),
        Some(&grammar),
        Some(&json_schema),
    ) {
        Ok(sampler) => {
            slot.set_sampler(sampler);
            slot.sampler_key = Some(key);
            record_backend_sampler_attach(slot, native_runtime);
            true
        }
        Err(_) => {
            let message = if grammar.is_empty() {
                "Failed to create per-slot sampler."
            } else {
                "Failed to create per-slot grammar sampler."
            };
            slot.fail(message);
            false
        }
    }
}

fn record_backend_sampler_attach(slot: &mut SlotState, native_runtime: &mut NativeRuntimeHandle) {
    let attach_start = Instant::now();
    let attached = attach_backend_sampler(native_runtime, slot);
    let attach_ms = duration_ms(attach_start, Instant::now());
    if let Some(request) = slot.request_mut() {
        request.debug_metrics_backend_sampler_attach_attempts = request
            .debug_metrics_backend_sampler_attach_attempts
            .saturating_add(1);
        request.debug_metrics_backend_sampler_attach_ms += attach_ms;
        if !attached {
            request.debug_metrics_backend_sampler_attach_failures = request
                .debug_metrics_backend_sampler_attach_failures
                .saturating_add(1);
        }
    }
}
