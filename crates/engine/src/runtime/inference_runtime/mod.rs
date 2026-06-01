//! Core inference runtime: owns the llama.cpp context, schedules requests,
//! and drives the prefill/decode loop.
//!
//! Internal helpers live in sibling submodules (e.g. `numeric`) so this file
//! stays focused on the runtime state machine.

use std::collections::HashSet;
use std::time::Instant;

use crate::error::Error;
use crate::native_bridge::{NativeRuntimeHandle, SamplerHandle};
use crate::runtime::config::{NativeRuntimeConfig, ResolvedRuntimeLimits};
use crate::runtime::llama::LlamaBatchBuilder;
use crate::runtime::llama_token;
use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::numeric::duration_ms;
use crate::runtime::request::{GenerateRequestId, RequestQueue, NO_SAMPLED_TOKEN_ID};
use crate::runtime::residency::ResidencyLease;
use crate::runtime::scheduler::{
    BatchPlanner, SamplerCacheKey, SharedBatchPlan, SlotPhase, SlotScheduler,
};
use crate::runtime::session::KvCacheManager;

pub(crate) mod capabilities;
mod decode;
mod diagnostics;
#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/diagnostics_tests.rs"]
mod diagnostics_tests;
mod embedding_read;
mod encoder;
mod environment;
mod lifecycle;
mod multimodal;
mod native;
mod numeric;
mod observability;
mod prefill;
mod prefix_snapshots;
mod request;

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/observability_tests.rs"]
mod observability_tests;

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/prefill_tests.rs"]
mod prefill_tests;

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/runtime_tests.rs"]
pub(crate) mod runtime_tests;

mod sampler;
mod scheduler_api;
#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/scheduler_api_tests.rs"]
mod scheduler_api_tests;
mod slot;
mod text;

use encoder::resolve_request_slot_plan_for_capabilities;
use environment::{resolve_batch_token_budget, snapshot_prefix_cache_enabled};
use numeric::{
    clamp_usize_to_i32, fingerprint_path, nonnegative_i32_to_usize, positive_i32_to_usize,
    saturating_i32_delta, saturating_usize_delta_to_i32, unique_slot_first_use,
};

const DEFAULT_PROMPT_CONTEXT_KEY: &str = "__primary_prompt__";
const LLAMA_SAMPLER_SAMPLE_FAILED: &str = "llama_sampler_sample() failed.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestStepResult {
    Invalid = -1,
    FatalNoProgress = -2,
    #[default]
    Waiting = 0,
    Progressed = 1,
    Terminal = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SchedulerBurstResult {
    pub status: RequestStepResult,
    pub ticks_executed: i32,
    pub progressed_ticks: i32,
    pub completed_response_count: i32,
    pub emitted_token_count: i32,
}

#[derive(Debug)]
struct PendingLogitsContribution {
    slot_index: usize,
    batch_token_index: i32,
    sampled_token: llama_token,
}

pub struct InferenceRuntime {
    config: NativeRuntimeConfig,
    pub(crate) resolved_limits: ResolvedRuntimeLimits,
    pub(crate) capabilities: capabilities::RuntimeModelCapabilities,
    native_runtime: NativeRuntimeHandle,
    // Held for RAII. Field order drops the native runtime before releasing residency.
    _residency_lease: Option<ResidencyLease>,
    last_runtime_observability: RuntimeObservabilityMetrics,
    has_last_runtime_observability: bool,
    kv_cache: KvCacheManager,
    pub request_queue: RequestQueue,
    slot_scheduler: SlotScheduler,
    batch_planner: BatchPlanner,
    shared_batch_builder: LlamaBatchBuilder,
    next_request_id: GenerateRequestId,
    model_fingerprint: u64,
    committed_observability_request_ids: HashSet<GenerateRequestId>,
    scratch_decode_ready_slots: Vec<usize>,
    scratch_prefill_ready_slots: Vec<usize>,
    scratch_logits_contributions: Vec<PendingLogitsContribution>,
    /// Reused across every tick to avoid allocating a fresh ~16 KiB Vec for
    /// the batch contributions each scheduler iteration.
    scratch_plan: SharedBatchPlan,
    /// Reused by `token_to_piece` to avoid a 128-byte Vec allocation per
    /// emitted token. Sized once and cleared per call.
    scratch_token_piece: Vec<u8>,
    total_decode_ms: f64,
    total_prefill_ms: f64,
    total_input_tokens: usize,
    total_output_tokens: usize,
    total_cache_hits: usize,
    total_prefill_tokens: usize,
    sampler_pool: std::collections::HashMap<SamplerCacheKey, Vec<SamplerHandle>>,
}

impl InferenceRuntime {
    pub fn capabilities(&self) -> crate::engine::protocol::ModelCapabilities {
        self.capabilities.to_public()
    }

    pub fn is_ready(&self) -> bool {
        self.native_runtime.is_loaded()
            && (self.config.multimodal.projector_path.is_none() || self.native_runtime.mtmd_ready())
    }

    fn run_scheduler_tick_locked(&mut self) -> RequestStepResult {
        if !self.is_ready() {
            return RequestStepResult::Invalid;
        }

        let completed_before = self.request_queue.completed_responses.len();
        let mut admitted_any = false;
        let capabilities = self.capabilities.clone();
        while let Some(slot_index) = self.slot_scheduler.admit_pending_requests(
            &mut self.request_queue,
            &mut self.kv_cache,
            self.config.cache.mode,
            |request| resolve_request_slot_plan_for_capabilities(&capabilities, request).ok(),
        ) {
            admitted_any = true;
            if let Err(error) = self.run_admission_prefill(slot_index) {
                self.fail_admitted_slot(slot_index, error);
            }
        }

        let tick_executed = self.run_policy_batch_tick_locked();
        self.detach_terminal_backend_samplers_locked();
        self.reclaim_and_pool_samplers_locked();
        self.slot_scheduler.finalize_completed_slots(
            &mut self.request_queue,
            &mut self.kv_cache,
            self.config.cache.mode,
        );
        self.commit_new_completed_responses_observability_locked();

        let completed_after = self.request_queue.completed_responses.len();
        if completed_after > completed_before {
            return RequestStepResult::Progressed;
        }

        if !tick_executed {
            let Some(active_slot_index) = self.slot_scheduler.slots.iter().position(|slot| {
                slot.request().is_some()
                    && slot.phase != SlotPhase::Idle
                    && slot.phase != SlotPhase::Completed
                    && slot.phase != SlotPhase::Failed
            }) else {
                return if admitted_any {
                    RequestStepResult::Progressed
                } else {
                    RequestStepResult::Waiting
                };
            };

            let diagnostic = self.build_no_progress_diagnostic_locked();
            if let Some(active_slot) = self.slot_scheduler.slots.get_mut(active_slot_index) {
                if active_slot.phase != SlotPhase::Failed
                    && active_slot.phase != SlotPhase::Completed
                {
                    active_slot.fail(diagnostic);
                }
            }

            self.detach_terminal_backend_samplers_locked();
            self.reclaim_and_pool_samplers_locked();
            self.slot_scheduler.finalize_completed_slots(
                &mut self.request_queue,
                &mut self.kv_cache,
                self.config.cache.mode,
            );
            self.commit_new_completed_responses_observability_locked();
            if self.request_queue.completed_responses.len() > completed_before {
                return RequestStepResult::Progressed;
            }
            return RequestStepResult::FatalNoProgress;
        }

        if tick_executed || admitted_any {
            RequestStepResult::Progressed
        } else {
            RequestStepResult::Waiting
        }
    }

    fn run_policy_batch_tick_locked(&mut self) -> bool {
        self.normalize_slots_for_tick();

        self.slot_scheduler
            .select_decode_ready_slots_into(&mut self.scratch_decode_ready_slots);
        self.slot_scheduler
            .select_prefill_ready_slots_into(&mut self.scratch_prefill_ready_slots);
        if self.scratch_decode_ready_slots.is_empty() && self.scratch_prefill_ready_slots.is_empty()
        {
            return false;
        }

        let batch_token_budget = resolve_batch_token_budget(&self.native_runtime, &self.config);
        let tick_budget = SlotScheduler::build_tick_budget(
            self.config.scheduler.policy,
            clamp_usize_to_i32(self.scratch_decode_ready_slots.len()),
            clamp_usize_to_i32(self.scratch_prefill_ready_slots.len()),
            batch_token_budget,
            self.config.scheduler.prefill_chunk_size,
        );
        let effective_prefill_chunk_size = self.resolve_prefill_chunk_size_locked(
            tick_budget,
            clamp_usize_to_i32(self.scratch_decode_ready_slots.len()),
            clamp_usize_to_i32(self.scratch_prefill_ready_slots.len()),
        );

        // Move out so we can pass `&plan` alongside `&mut self`; the Vec
        // allocations travel with `plan` and get returned at end of tick.
        let mut plan = std::mem::take(&mut self.scratch_plan);
        let plan_start = Instant::now();
        self.batch_planner.build_policy_batch_into(
            &mut plan,
            &self.slot_scheduler.slots,
            &self.scratch_decode_ready_slots,
            &self.scratch_prefill_ready_slots,
            tick_budget,
            effective_prefill_chunk_size,
        );
        let plan_ms = duration_ms(plan_start, Instant::now());
        if plan.contributions.is_empty() {
            self.scratch_plan = plan;
            return false;
        }

        let batch_build_start = Instant::now();
        if self
            .shared_batch_builder
            .ensure_capacity(batch_token_budget, self.resolved_limits.n_parallel.max(1))
            .is_err()
        {
            self.fail_plan_slots(&plan, "Shared batch allocation failed.");
            return false;
        }
        self.shared_batch_builder.reset();
        self.scratch_logits_contributions.clear();

        let mut batch_token_index = 0_i32;
        for contribution in plan.contributions.iter() {
            let Some(slot) = self.slot_scheduler.slots.get(contribution.slot_index) else {
                continue;
            };
            if slot.seq_id < 0 {
                continue;
            }
            if !self.shared_batch_builder.add_token(
                contribution.token,
                contribution.position,
                slot.seq_id,
                contribution.request_logits,
            ) {
                self.fail_plan_slots(&plan, "Shared batch builder capacity was exceeded.");
                self.scratch_plan = plan;
                return false;
            }
            if contribution.request_logits {
                self.scratch_logits_contributions
                    .push(PendingLogitsContribution {
                        slot_index: contribution.slot_index,
                        batch_token_index,
                        sampled_token: NO_SAMPLED_TOKEN_ID,
                    });
            }
            batch_token_index += 1;
        }
        let batch_build_ms = duration_ms(batch_build_start, Instant::now());

        // Production metrics — always recorded.
        let decode_start = Instant::now();
        let decode_result = self
            .native_runtime
            .decode(self.shared_batch_builder.batch())
            .map_err(|error| Error::RuntimeCommand(error.to_string()));
        let decode_submitted = Instant::now();
        let sync_ok = self.native_runtime.synchronize();
        let decode_end = Instant::now();
        let native_decode_ms = duration_ms(decode_start, decode_submitted);
        let native_sync_ms = duration_ms(decode_submitted, decode_end);
        let decode_status = match decode_result {
            Ok(status) => status,
            Err(error) => {
                let diagnostic = format!(
                    "llama_decode() failed in shared tick ({error}, n_tokens={})",
                    self.shared_batch_builder.n_tokens()
                );
                self.fail_plan_slots(&plan, &diagnostic);
                self.scratch_plan = plan;
                return false;
            }
        };
        if decode_status != 0 {
            let diagnostic = format!(
                "llama_decode() failed in shared tick (status={decode_status}, n_tokens={})",
                self.shared_batch_builder.n_tokens()
            );
            self.fail_plan_slots(&plan, &diagnostic);
            self.scratch_plan = plan;
            return false;
        }
        if !sync_ok {
            self.fail_plan_slots(&plan, "llama_synchronize() failed in shared tick.");
            self.scratch_plan = plan;
            return false;
        }

        let native_logic_ms = plan_ms + batch_build_ms;
        self.apply_bookkeeping_and_emit(&plan, native_decode_ms, native_sync_ms, native_logic_ms);
        self.sample_logits_and_buffer_output();
        for slot in &mut self.slot_scheduler.slots {
            if slot.phase == SlotPhase::EmitBuffered && !slot.buffered_output_text.is_empty() {
                SlotScheduler::emit_buffered_token_piece(&mut self.request_queue, slot);
            }
        }
        if snapshot_prefix_cache_enabled(self.config.cache.mode) {
            self.capture_prefix_snapshots(&plan);
        }
        // Return the plan's allocations to the scratch slot for reuse.
        self.scratch_plan = plan;
        true
    }
}
