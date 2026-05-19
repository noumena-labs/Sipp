use cogentlm_sys as ffi;

use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::request::GenerateRequestLifecycle;
use crate::runtime::request::RequestQueue;
use crate::runtime::scheduler::{SlotPhase, SlotState};
use crate::runtime::session::{PrefixCachePolicy, PrefixStateCache, SessionStore};

use super::environment::{
    live_retained_prefix_tokens, resolve_batch_token_budget, snapshot_prefix_cache_enabled,
};
use super::multimodal::run_multimodal_prefill;
use super::prefill::{ensure_decode_step_context_space, prepare_sequence_for_prompt};
use super::InferenceRuntime;

mod recovery;
mod sampler_attach;

use recovery::normalize_runnable_slot_state;
use sampler_attach::ensure_slot_sampler;

impl InferenceRuntime {
    pub(super) fn normalize_slots_for_tick(&mut self, vocab: *const ffi::llama_vocab) {
        let slot_count = self.slot_scheduler.slots().len();
        for slot_index in 0..slot_count {
            let slot = &mut self.slot_scheduler.mutable_slots()[slot_index];
            if slot.request().is_none() || slot.session.is_none() || slot.seq_id < 0 {
                continue;
            }

            let cancel_requested = slot.request().map(|r| r.cancel_requested).unwrap_or(false);
            if cancel_requested {
                slot.terminal_error_message = "Request cancelled.".to_string();
                slot.phase = SlotPhase::Failed;
                if let Some(request) = slot.request_mut() {
                    request.lifecycle = GenerateRequestLifecycle::Cancelled;
                }
                continue;
            }

            normalize_runnable_slot_state(
                slot,
                self.shared_context,
                self.primary_model,
                live_retained_prefix_tokens(&self.config),
            );

            if slot.sampler().is_none()
                && !ensure_slot_sampler(
                    slot,
                    self.common_init,
                    self.shared_context,
                    &self.config,
                    &mut self.sampler_pool,
                )
            {
                continue;
            }

            if slot.phase == SlotPhase::Prefill && slot.prefill_cursor == 0 {
                if run_initial_prefill(
                    slot,
                    vocab,
                    self.mtmd_context,
                    self.shared_context,
                    self.primary_model,
                    &self.config,
                    self.model_fingerprint,
                    &self.session_store,
                    &mut self.prefix_state_cache,
                    &mut self.prefix_cache_policy,
                    &mut self.request_queue,
                    &mut self.scratch_token_piece,
                ) {
                    continue;
                }
            }

            if slot.phase == SlotPhase::Decode
                && !ensure_decode_step_context_space(
                    self.shared_context,
                    live_retained_prefix_tokens(&self.config),
                    slot,
                )
            {
                slot.terminal_error_message =
                    "Failed to extend decode context headroom.".to_string();
                slot.phase = SlotPhase::Failed;
                if let Some(request) = slot.request_mut() {
                    request.lifecycle = GenerateRequestLifecycle::Failed;
                }
                continue;
            }

            if let Some(request) = slot.request_mut() {
                request.lifecycle = GenerateRequestLifecycle::Running;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_initial_prefill(
    slot: &mut SlotState,
    vocab: *const ffi::llama_vocab,
    mtmd_context: *mut ffi::cogent_mtmd_context,
    shared_context: *mut ffi::llama_context,
    primary_model: *mut ffi::llama_model,
    config: &NativeRuntimeConfig,
    model_fingerprint: u64,
    session_store: &SessionStore,
    prefix_state_cache: &mut PrefixStateCache,
    prefix_cache_policy: &mut PrefixCachePolicy,
    request_queue: &mut RequestQueue,
    scratch_token_piece: &mut Vec<i8>,
) -> bool {
    if slot
        .request()
        .is_some_and(|request| request.is_multimodal_turn)
    {
        let ok = run_multimodal_prefill(
            mtmd_context,
            shared_context,
            vocab,
            resolve_batch_token_budget(shared_context, config),
            request_queue,
            slot,
            scratch_token_piece,
        );
        if !ok {
            if slot.terminal_error_message.is_empty() {
                slot.terminal_error_message = "Failed to evaluate multimodal prompt.".to_string();
            }
            slot.phase = SlotPhase::Failed;
            if let Some(request) = slot.request_mut() {
                request.lifecycle = GenerateRequestLifecycle::Failed;
                request.multimodal = None;
            }
        }
        return true;
    }

    let Some(ref mut request) = slot.request else {
        return false;
    };
    let mut prefill_cursor = 0;
    if let Some(cache_hits) = prepare_sequence_for_prompt(
        shared_context,
        primary_model,
        live_retained_prefix_tokens(config),
        snapshot_prefix_cache_enabled(config.cache.mode),
        config.scheduler.policy.decode_token_reserve,
        model_fingerprint,
        session_store,
        prefix_state_cache,
        prefix_cache_policy,
        &request.context_key,
        &request.prompt_tokens,
        request.max_output_tokens,
        &mut slot.mirror,
        slot.seq_id,
        &mut prefill_cursor,
    ) {
        request.cache_hits = cache_hits;
        if !slot.sampler_prompt_seeded
            && request.grammar.is_empty()
            && request.json_schema.is_empty()
        {
            if let Some(sampler) = slot.sampler {
                for &token in &request.prompt_tokens {
                    if !unsafe { ffi::cogent_common_sampler_accept(sampler.as_ptr(), token, false) }
                    {
                        break;
                    }
                }
                slot.sampler_prompt_seeded = true;
            }
        }
        slot.prefill_cursor = prefill_cursor;
        slot.phase = if slot.prefill_cursor >= request.prompt_tokens.len() {
            SlotPhase::Decode
        } else {
            SlotPhase::Prefill
        };
    } else {
        slot.terminal_error_message = "Failed to prepare sequence for prompt reuse.".to_string();
        slot.phase = SlotPhase::Failed;
        request.lifecycle = GenerateRequestLifecycle::Failed;
    }
    false
}
