//! Per-slot sampler lifecycle: create from runtime config, attach/detach to
//! the shared llama context for backend-side sampling.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::config::{NativeRuntimeConfig, SamplingRuntimeOverride};
use crate::runtime::llama_seq_id;
use crate::runtime::scheduler::{SamplerCacheKey, SamplerHandle, SlotPhase, SlotState};

use super::InferenceRuntime;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/sampler_tests.rs"]
mod sampler_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Backend sampler ownership parked on the physical llama sequence.
#[derive(Debug)]
pub(super) struct ResidentBackendSampler {
    pub(super) key: SamplerCacheKey,
    pub(super) sampler: SamplerHandle,
}

/// Hands the slot's CPU sampler to the backend so it can sample inside the
/// decode kernel. Returns `false` if the slot is not eligible or the FFI call
/// rejected the handoff.
pub(super) fn attach_backend_sampler(
    native_runtime: &mut NativeRuntimeHandle,
    slot: &mut SlotState,
) -> bool {
    if slot.seq_id < 0 || slot.backend_sampler_attached {
        return false;
    }

    let Some(sampler) = slot.sampler.as_mut() else {
        return false;
    };
    if !sampler.backend_sampling() {
        return false;
    }

    let attached = native_runtime.attach_sampler(sampler, slot.seq_id);
    if attached {
        slot.backend_sampler_attached = true;
    }
    attached
}

/// Reverses `attach_backend_sampler`. Safe to call when nothing is attached.
pub(super) fn detach_backend_sampler(
    native_runtime: &mut NativeRuntimeHandle,
    slot: &mut SlotState,
) {
    if !slot.backend_sampler_attached {
        return;
    }

    if slot.seq_id >= 0 {
        native_runtime.detach_sampler(slot.seq_id);
    }
    slot.backend_sampler_attached = false;
}

/// Builds a sampler from the runtime's sampling JSON plus optional grammar /
/// JSON-schema constraints. Returns the raw shim pointer on success.
pub(super) fn create_sampler(
    native_runtime: &NativeRuntimeHandle,
    config: &NativeRuntimeConfig,
    sampling_override: Option<&SamplingRuntimeOverride>,
    grammar: Option<&str>,
    json_schema: Option<&str>,
) -> Result<SamplerHandle> {
    let sampling_json = config
        .try_sampling_json_with_override(sampling_override)
        .map_err(|error| {
            Error::RuntimeCommand(format!(
                "failed to serialize sampler configuration: {error}"
            ))
        })?;
    native_runtime.create_sampler(
        &sampling_json,
        grammar.unwrap_or(""),
        json_schema.unwrap_or(""),
    )
}

impl InferenceRuntime {
    pub(super) fn settle_terminal_samplers_locked(&mut self) {
        settle_terminal_samplers(
            &mut self.slot_scheduler,
            &mut self.native_runtime,
            &mut self.sampler_pool,
            &mut self.resident_backend_samplers,
        );
    }

    pub(super) fn detach_all_backend_samplers_locked(&mut self) {
        for slot in &mut self.slot_scheduler.slots {
            detach_backend_sampler(&mut self.native_runtime, slot);
        }
        for seq_id in std::mem::take(&mut self.resident_backend_samplers).into_keys() {
            if seq_id >= 0 {
                self.native_runtime.detach_sampler(seq_id);
            }
        }
    }
}

fn settle_terminal_samplers(
    slot_scheduler: &mut crate::runtime::scheduler::SlotScheduler,
    native_runtime: &mut NativeRuntimeHandle,
    sampler_pool: &mut HashMap<SamplerCacheKey, Vec<SamplerHandle>>,
    resident_backend_samplers: &mut HashMap<llama_seq_id, ResidentBackendSampler>,
) {
    for slot in &mut slot_scheduler.slots {
        if !matches!(slot.phase, SlotPhase::Completed | SlotPhase::Failed) {
            continue;
        }

        if slot.backend_sampler_attached {
            settle_terminal_backend_sampler(slot, native_runtime, resident_backend_samplers);
            continue;
        }

        let cached_key = slot.sampler_key.take();
        let Some(sampler) = slot.take_sampler() else {
            continue;
        };
        // The native reset rewinds the sampling chain but not grammar state,
        // so grammar/schema-constrained samplers are dropped instead of pooled.
        let reusable_key =
            cached_key.filter(|key| key.grammar.is_empty() && key.json_schema.is_empty());
        if let Some(key) = reusable_key {
            let mut sampler = sampler;
            reset_sampler(&mut sampler);
            sampler_pool.entry(key).or_default().push(sampler);
        }
    }
}

fn settle_terminal_backend_sampler(
    slot: &mut SlotState,
    native_runtime: &mut NativeRuntimeHandle,
    resident_backend_samplers: &mut HashMap<llama_seq_id, ResidentBackendSampler>,
) {
    let completed = slot.phase == SlotPhase::Completed;
    if completed && slot.seq_id >= 0 {
        if let (Some(key), Some(mut sampler)) = (slot.sampler_key.take(), slot.sampler.take()) {
            reset_sampler(&mut sampler);
            slot.backend_sampler_attached = false;
            let replaced = resident_backend_samplers
                .insert(slot.seq_id, ResidentBackendSampler { key, sampler });
            debug_assert!(
                replaced.is_none(),
                "resident backend sampler already existed for completed slot seq_id"
            );
            return;
        }
    }

    detach_backend_sampler(native_runtime, slot);
    slot.sampler_key = None;
    slot.sampler = None;
}

fn reset_sampler(sampler: &mut SamplerHandle) {
    sampler.reset();
}
