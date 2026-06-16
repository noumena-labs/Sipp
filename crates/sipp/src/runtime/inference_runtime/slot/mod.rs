use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::config::{
    NativeRuntimeConfig, RequestSampling, SamplerStage, SamplingRuntimeConfig,
};
use crate::runtime::request::GenerateRequestLifecycle;
use crate::runtime::request::RequestQueue;
use crate::runtime::scheduler::{PrefillKind, SlotPhase, SlotState, TerminalAction};
use crate::runtime::session::KvCacheManager;
use crate::runtime::REQUEST_CANCELLED_MESSAGE;

use super::environment::{live_retained_prefix_tokens, resolve_batch_token_budget};
use super::multimodal::run_multimodal_prefill;
use super::prefill::{ensure_decode_step_context_space, prepare_sequence_for_prompt};
use super::InferenceRuntime;

mod recovery;
mod sampler_attach;

use recovery::normalize_runnable_slot_state;
use sampler_attach::ensure_slot_sampler;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../../tests/runtime/inference_runtime/slot_tests.rs"]
mod slot_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

impl InferenceRuntime {
    pub(super) fn normalize_slots_for_tick(&mut self) {
        let slot_count = self.slot_scheduler.slots.len();
        for slot_index in 0..slot_count {
            let slot = &mut self.slot_scheduler.slots[slot_index];
            if slot.request().is_none() || slot.seq_id < 0 {
                continue;
            }

            let cancel_requested = slot.request().map(|r| r.cancel_requested).unwrap_or(false);
            if cancel_requested {
                slot.cancel(REQUEST_CANCELLED_MESSAGE);
                continue;
            }

            normalize_runnable_slot_state(
                slot,
                &mut self.native_runtime,
                live_retained_prefix_tokens(&self.config),
            );

            // Embedding-only slots have no sampler; any resident sampler for
            // this physical sequence belongs to a previous text request.
            if slot.plan.terminal != TerminalAction::SampleTokens {
                let seq_id = slot.seq_id;
                if seq_id >= 0 && self.resident_backend_samplers.remove(&seq_id).is_some() {
                    self.native_runtime.detach_sampler(seq_id);
                }
            } else if slot.sampler.is_none() {
                if !ensure_slot_sampler(
                    slot,
                    &mut self.native_runtime,
                    &self.config,
                    &mut self.sampler_pool,
                    &mut self.resident_backend_samplers,
                ) {
                    continue;
                }
            }

            if slot.phase == SlotPhase::Prefill && slot.prefill_cursor == 0 {
                if run_initial_prefill(
                    slot,
                    &mut self.native_runtime,
                    &self.config,
                    self.model_fingerprint,
                    &mut self.kv_cache,
                    &mut self.total_cache_hits,
                    &mut self.request_queue,
                    &mut self.scratch_token_piece,
                ) {
                    continue;
                }
            }

            if slot.phase == SlotPhase::Decode
                && !ensure_decode_step_context_space(
                    &mut self.native_runtime,
                    live_retained_prefix_tokens(&self.config),
                    slot,
                )
            {
                slot.fail("Failed to extend decode context headroom.");
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
    native_runtime: &mut NativeRuntimeHandle,
    config: &NativeRuntimeConfig,
    model_fingerprint: u64,
    kv_cache: &mut KvCacheManager,
    total_cache_hits: &mut usize,
    request_queue: &mut RequestQueue,
    scratch_token_piece: &mut Vec<u8>,
) -> bool {
    if slot
        .request()
        .is_some_and(|request| request.is_multimodal_turn)
    {
        let ok = run_multimodal_prefill(
            native_runtime,
            resolve_batch_token_budget(native_runtime, config),
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

    // Encoder-decoder prompts are rewritten to a single decoder-start token by
    // the admission pass. Allowing the prefix cache to LCP-match that token
    // across unrelated source prompts would let an old turn's decoder KV
    // poison the new turn — disable cache reuse and start from a fresh KV
    // state for this slot. Same rule for embedding-context requests, whose
    // outputs are read directly from the encoder pass, not from cached KV.
    let bypass_prefix_cache = slot.plan.prefill == PrefillKind::Encode
        || slot.plan.terminal == TerminalAction::ReadEmbedding;
    let Some(ref mut request) = slot.request else {
        return false;
    };
    let mut prefill_cursor = 0;
    if let Some(cache_preparation) = prepare_sequence_for_prompt(
        native_runtime,
        live_retained_prefix_tokens(config),
        config.cache.mode,
        bypass_prefix_cache,
        config.scheduler.policy.decode_token_reserve,
        model_fingerprint,
        kv_cache,
        slot.cache_candidate,
        &request.context_key,
        &request.prompt_tokens,
        request.max_output_tokens,
        &mut slot.mirror,
        slot.seq_id,
        &mut prefill_cursor,
    ) {
        request.cache_hits = cache_preparation.cache_hits;
        if cache_preparation.cache_hits > 0 {
            *total_cache_hits =
                total_cache_hits.saturating_add(cache_preparation.cache_hits as usize);
        }
        request.cache_source = cache_preparation.source;
        if !slot.sampler_prompt_seeded
            && request.grammar.is_empty()
            && request.json_schema.is_empty()
        {
            if let Some(sampler) = slot.sampler.as_mut() {
                let seed_start = prompt_sampler_seed_start(
                    config,
                    request.sampling.as_ref(),
                    request.prompt_tokens.len(),
                );
                for &token in &request.prompt_tokens[seed_start..] {
                    if !sampler.accept(token, false) {
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

fn prompt_sampler_seed_start(
    config: &NativeRuntimeConfig,
    sampling_override: Option<&RequestSampling>,
    prompt_len: usize,
) -> usize {
    let sampling = match sampling_override {
        Some(RequestSampling::Full(sampling)) => sampling,
        Some(RequestSampling::Patch(_)) | None => &config.sampling,
    };
    let Some(history_len) = finite_prompt_history_len(sampling) else {
        return 0;
    };
    prompt_len.saturating_sub(history_len)
}

fn finite_prompt_history_len(sampling: &SamplingRuntimeConfig) -> Option<usize> {
    if sampling.mirostat.unwrap_or(0) != 0 {
        return Some(0);
    }

    let mut history_len = 0;
    if sampler_stage_enabled(sampling, SamplerStage::Penalties) && penalties_enabled(sampling) {
        update_history_len(
            &mut history_len,
            sampling.repeat_last_n.unwrap_or(CPP_DEFAULT_REPEAT_LAST_N),
        )?;
    }
    if sampler_stage_enabled(sampling, SamplerStage::Dry) && dry_enabled(sampling) {
        update_history_len(
            &mut history_len,
            sampling
                .dry_penalty_last_n
                .unwrap_or(CPP_DEFAULT_DRY_PENALTY_LAST_N),
        )?;
    }
    Some(history_len)
}

fn sampler_stage_enabled(sampling: &SamplingRuntimeConfig, stage: SamplerStage) -> bool {
    sampling.samplers.is_empty() && matches!(stage, SamplerStage::Penalties | SamplerStage::Dry)
        || sampling.samplers.contains(&stage)
}

fn penalties_enabled(sampling: &SamplingRuntimeConfig) -> bool {
    sampling.repeat_last_n.unwrap_or(CPP_DEFAULT_REPEAT_LAST_N) != 0
        && (sampling
            .repeat_penalty
            .unwrap_or(CPP_DEFAULT_REPEAT_PENALTY)
            != CPP_DEFAULT_REPEAT_PENALTY
            || sampling
                .frequency_penalty
                .unwrap_or(CPP_DEFAULT_FREQUENCY_PENALTY)
                != CPP_DEFAULT_FREQUENCY_PENALTY
            || sampling
                .presence_penalty
                .unwrap_or(CPP_DEFAULT_PRESENCE_PENALTY)
                != CPP_DEFAULT_PRESENCE_PENALTY)
}

fn dry_enabled(sampling: &SamplingRuntimeConfig) -> bool {
    sampling
        .dry_multiplier
        .unwrap_or(CPP_DEFAULT_DRY_MULTIPLIER)
        != 0.0
        && sampling.dry_base.unwrap_or(CPP_DEFAULT_DRY_BASE) >= 1.0
        && sampling
            .dry_penalty_last_n
            .unwrap_or(CPP_DEFAULT_DRY_PENALTY_LAST_N)
            != 0
}

fn update_history_len(history_len: &mut usize, last_n: i32) -> Option<()> {
    if last_n < 0 {
        return None;
    }
    *history_len = (*history_len).max(last_n as usize);
    Some(())
}

const CPP_DEFAULT_REPEAT_LAST_N: i32 = 64;
const CPP_DEFAULT_DRY_PENALTY_LAST_N: i32 = -1;
const CPP_DEFAULT_REPEAT_PENALTY: f32 = 1.0;
const CPP_DEFAULT_FREQUENCY_PENALTY: f32 = 0.0;
const CPP_DEFAULT_PRESENCE_PENALTY: f32 = 0.0;
const CPP_DEFAULT_DRY_MULTIPLIER: f32 = 0.0;
const CPP_DEFAULT_DRY_BASE: f32 = 1.75;
