use std::collections::HashMap;

use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::llama_seq_id;
use crate::runtime::scheduler::{SamplerCacheKey, SamplerHandle, SlotState};

use super::super::sampler::{attach_backend_sampler, create_sampler, ResidentBackendSampler};

pub(super) fn ensure_slot_sampler(
    slot: &mut SlotState,
    native_runtime: &mut NativeRuntimeHandle,
    config: &NativeRuntimeConfig,
    sampler_pool: &mut HashMap<SamplerCacheKey, Vec<SamplerHandle>>,
    resident_backend_samplers: &mut HashMap<llama_seq_id, ResidentBackendSampler>,
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

    if let Some(resident) = resident_backend_samplers.remove(&slot.seq_id) {
        if resident.key == key {
            slot.set_sampler(resident.sampler);
            slot.sampler_key = Some(key);
            slot.backend_sampler_attached = true;
            return true;
        }

        if slot.seq_id >= 0 {
            native_runtime.detach_sampler(slot.seq_id);
        }
    }

    if let Some(sampler) = sampler_pool.get_mut(&key).and_then(|vec| vec.pop()) {
        slot.set_sampler(sampler);
        slot.sampler_key = Some(key);
        attach_backend_sampler(native_runtime, slot);
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
            attach_backend_sampler(native_runtime, slot);
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
