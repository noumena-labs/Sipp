use std::ptr::NonNull;

use cogentlm_sys as ffi;

use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::scheduler::{SamplerCacheKey, SlotState};

use super::super::sampler::{attach_backend_sampler, create_sampler};

pub(super) fn ensure_slot_sampler(
    slot: &mut SlotState,
    common_init: *mut ffi::cogent_common_init,
    shared_context: *mut ffi::llama_context,
    config: &NativeRuntimeConfig,
    sampler_pool: &mut std::collections::HashMap<
        SamplerCacheKey,
        Vec<NonNull<ffi::cogent_common_sampler>>,
    >,
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
        slot.set_sampler(sampler.as_ptr());
        slot.sampler_key = Some(key);
        attach_backend_sampler(shared_context, slot);
        return true;
    }

    match create_sampler(
        common_init,
        config,
        sampling.as_ref(),
        Some(&grammar),
        Some(&json_schema),
    ) {
        Ok(sampler) => {
            slot.set_sampler(sampler);
            slot.sampler_key = Some(key);
            attach_backend_sampler(shared_context, slot);
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
