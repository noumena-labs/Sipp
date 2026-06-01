//! Per-slot sampler lifecycle: create from runtime config, attach/detach to
//! the shared llama context for backend-side sampling.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::config::{NativeRuntimeConfig, RequestSampling};
use crate::runtime::scheduler::{SamplerCacheKey, SamplerHandle, SlotPhase, SlotState};

use super::InferenceRuntime;

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
    sampling_override: Option<&RequestSampling>,
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
    pub(super) fn detach_terminal_backend_samplers_locked(&mut self) {
        for slot in &mut self.slot_scheduler.slots {
            if matches!(slot.phase, SlotPhase::Completed | SlotPhase::Failed) {
                detach_backend_sampler(&mut self.native_runtime, slot);
            }
        }
    }

    pub(super) fn reclaim_and_pool_samplers_locked(&mut self) {
        reclaim_terminal_samplers(&mut self.slot_scheduler, &mut self.sampler_pool);
    }

    pub(super) fn detach_all_backend_samplers_locked(&mut self) {
        for slot in &mut self.slot_scheduler.slots {
            detach_backend_sampler(&mut self.native_runtime, slot);
        }
    }
}

fn reclaim_terminal_samplers(
    slot_scheduler: &mut crate::runtime::scheduler::SlotScheduler,
    sampler_pool: &mut HashMap<SamplerCacheKey, Vec<SamplerHandle>>,
) {
    for slot in &mut slot_scheduler.slots {
        if !matches!(slot.phase, SlotPhase::Completed | SlotPhase::Failed) {
            continue;
        }
        let Some(sampler) = slot.take_sampler() else {
            continue;
        };
        if let Some(key) = slot.sampler_key.take() {
            let mut sampler = sampler;
            reset_sampler(&mut sampler);
            sampler_pool.entry(key).or_default().push(sampler);
        }
    }
}

fn reset_sampler(sampler: &mut SamplerHandle) {
    sampler.reset();
}
