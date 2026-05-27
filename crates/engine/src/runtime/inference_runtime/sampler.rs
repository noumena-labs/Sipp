//! Per-slot sampler lifecycle: create from runtime config, attach/detach to
//! the shared llama context for backend-side sampling.

use std::collections::HashMap;
use std::ffi::CString;
use std::ptr::NonNull;

use cogentlm_sys as ffi;

use crate::error::{Error, Result};
use crate::runtime::config::{NativeRuntimeConfig, SamplingRuntimeConfig};
use crate::runtime::scheduler::{SamplerCacheKey, SlotPhase, SlotState};

use super::native::runtime_command_from_shim_error;
use super::InferenceRuntime;

/// Hands the slot's CPU sampler to the backend so it can sample inside the
/// decode kernel. Returns `false` if the slot is not eligible or the FFI call
/// rejected the handoff.
pub(super) fn attach_backend_sampler(
    shared_context: *mut ffi::llama_context,
    slot: &mut SlotState,
) -> bool {
    if shared_context.is_null() || slot.seq_id < 0 || slot.backend_sampler_attached {
        return false;
    }

    let Some(sampler) = slot.sampler else {
        return false;
    };
    if !unsafe { ffi::cogent_common_sampler_backend_sampling(sampler.as_ptr()) } {
        return false;
    }
    let raw_sampler = unsafe { ffi::cogent_common_sampler_raw(sampler.as_ptr()) };
    if raw_sampler.is_null() {
        return false;
    }

    let attached =
        unsafe { ffi::cogent_llama_set_sampler(shared_context, slot.seq_id, raw_sampler) };
    if attached {
        slot.backend_sampler_attached = true;
    }
    attached
}

/// Reverses `attach_backend_sampler`. Safe to call when nothing is attached.
pub(super) fn detach_backend_sampler(
    shared_context: *mut ffi::llama_context,
    slot: &mut SlotState,
) {
    if !slot.backend_sampler_attached {
        return;
    }

    if !shared_context.is_null() && slot.seq_id >= 0 {
        unsafe {
            ffi::cogent_llama_set_sampler(shared_context, slot.seq_id, std::ptr::null_mut());
        }
    }
    slot.backend_sampler_attached = false;
}

/// Builds a sampler from the runtime's sampling JSON plus optional grammar /
/// JSON-schema constraints. Returns the raw shim pointer on success.
pub(super) fn create_sampler(
    common_init: *mut ffi::cogent_common_init,
    config: &NativeRuntimeConfig,
    sampling_override: Option<&SamplingRuntimeConfig>,
    grammar: Option<&str>,
    json_schema: Option<&str>,
) -> Result<*mut ffi::cogent_common_sampler> {
    if common_init.is_null() {
        return Err(Error::RuntimeNotReady);
    }
    let sampling_json = config
        .try_sampling_json_with_override(sampling_override)
        .map_err(|error| {
            Error::RuntimeCommand(format!(
                "failed to serialize sampler configuration: {error}"
            ))
        })?;
    let sampling_json = CString::new(sampling_json)?;
    let grammar = CString::new(grammar.unwrap_or(""))?;
    let json_schema = CString::new(json_schema.unwrap_or(""))?;
    let mut error = std::ptr::null_mut();
    let sampler = unsafe {
        ffi::cogent_common_sampler_init_from_json(
            common_init,
            sampling_json.as_ptr(),
            grammar.as_ptr(),
            json_schema.as_ptr(),
            &mut error,
        )
    };
    if sampler.is_null() {
        return Err(runtime_command_from_shim_error(
            error,
            "common sampler initialization failed",
        ));
    }
    Ok(sampler)
}

impl InferenceRuntime {
    pub(super) fn detach_terminal_backend_samplers_locked(&mut self) {
        for slot in &mut self.slot_scheduler.slots {
            if matches!(slot.phase, SlotPhase::Completed | SlotPhase::Failed) {
                detach_backend_sampler(self.shared_context, slot);
            }
        }
    }

    pub(super) fn reclaim_and_pool_samplers_locked(&mut self) {
        reclaim_terminal_samplers(&mut self.slot_scheduler, &mut self.sampler_pool);
    }

    pub(super) fn detach_all_backend_samplers_locked(&mut self) {
        for slot in &mut self.slot_scheduler.slots {
            detach_backend_sampler(self.shared_context, slot);
        }
    }
}

fn reclaim_terminal_samplers(
    slot_scheduler: &mut crate::runtime::scheduler::SlotScheduler,
    sampler_pool: &mut HashMap<SamplerCacheKey, Vec<NonNull<ffi::cogent_common_sampler>>>,
) {
    for slot in &mut slot_scheduler.slots {
        if !matches!(slot.phase, SlotPhase::Completed | SlotPhase::Failed) {
            continue;
        }
        let Some(sampler) = slot.take_sampler() else {
            continue;
        };
        if let Some(key) = slot.sampler_key.take() {
            reset_sampler(sampler);
            sampler_pool.entry(key).or_default().push(sampler);
        } else {
            unsafe {
                ffi::cogent_common_sampler_free(sampler.as_ptr());
            }
        }
    }
}

fn reset_sampler(sampler: NonNull<ffi::cogent_common_sampler>) {
    unsafe {
        let raw = ffi::cogent_common_sampler_raw(sampler.as_ptr());
        if !raw.is_null() {
            ffi::llama_sampler_reset(raw);
        }
    }
}
