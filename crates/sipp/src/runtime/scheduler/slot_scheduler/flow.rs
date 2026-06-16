use std::time::Instant;

use crate::runtime::config::KvReuseMode;
use crate::runtime::request::{
    GenerateRequest, GenerateResponse, GenerateResponseStatus, RequestQueue, ResponseOutput,
};
use crate::runtime::session::KvCacheManager;
use crate::runtime::{
    numeric::saturating_usize_to_i32,
    scheduler::{PrefillKind, SlotExecutionPlan, SlotPhase, TerminalAction},
    REQUEST_CANCELLED_MESSAGE,
};

use super::metrics::metrics_from_request;
use super::{SlotScheduler, SlotState};

const ACQUIRE_HARDWARE_SEQUENCE_FAILED: &str = "Failed to acquire a hardware sequence ID.";
const REQUEST_FAILED: &str = "Request failed.";
const RESOLVE_SLOT_PLAN_FAILED: &str = "Failed to resolve slot execution plan.";

impl SlotScheduler {
    pub fn resize(&mut self, slot_count: usize, kv_cache: &mut KvCacheManager) {
        if slot_count < self.slots.len() {
            for slot in &mut self.slots[slot_count..] {
                release_slot_for_reset(kv_cache, slot);
                slot.reset_to_idle();
            }
        }

        self.slots.resize_with(slot_count, Default::default);
        for (slot_id, slot) in self.slots.iter_mut().enumerate() {
            reset_slot_identity(slot, slot_id);
            if idle_without_request(slot) {
                continue;
            }
            release_slot_for_reset(kv_cache, slot);
            slot.reset_to_idle();
            reset_slot_identity(slot, slot_id);
        }
    }

    pub fn select_decode_ready_slots_into(&self, out: &mut Vec<usize>) {
        self.select_ready_slots_into(out, decode_slot_ready);
    }

    pub fn select_prefill_ready_slots_into(&self, out: &mut Vec<usize>) {
        self.select_ready_slots_into(out, prefill_slot_ready);
    }

    fn select_ready_slots_into(
        &self,
        out: &mut Vec<usize>,
        mut is_ready: impl FnMut(&SlotState) -> bool,
    ) {
        out.clear();
        for (index, slot) in self.slots.iter().enumerate() {
            if is_ready(slot) {
                out.push(index);
            }
        }
    }

    pub fn admit_pending_requests(
        &mut self,
        request_queue: &mut RequestQueue,
        kv_cache: &mut KvCacheManager,
        cache_mode: KvReuseMode,
        mut resolve_plan: impl FnMut(&GenerateRequest) -> Option<SlotExecutionPlan>,
    ) -> Option<usize> {
        let idle_slot_index = self.slots.iter().position(idle_without_request)?;

        let next_request_id = request_queue
            .try_pop_next_admissible(|request| kv_cache.can_admit(&request.context_key))?;

        let queued_request = request_queue.requests.get(&next_request_id)?;

        let context_key = queued_request.context_key.clone();
        let Some(plan) = resolve_plan(queued_request) else {
            complete_failed_admission(request_queue, next_request_id, RESOLVE_SLOT_PLAN_FAILED);
            return None;
        };
        let bypass_cache =
            plan.prefill == PrefillKind::Encode || plan.terminal == TerminalAction::ReadEmbedding;
        let Some(admission) = kv_cache.admit(&context_key, cache_mode, bypass_cache) else {
            complete_failed_admission(
                request_queue,
                next_request_id,
                ACQUIRE_HARDWARE_SEQUENCE_FAILED,
            );
            return None;
        };
        let mut request = request_queue.take_admitted_request(next_request_id)?;

        let slot = &mut self.slots[idle_slot_index];
        request.cache_mode = cache_mode;
        slot.attach_request(request, admission);
        slot.plan = plan;
        slot.phase = SlotPhase::Prefill;
        Some(idle_slot_index)
    }

    pub fn finalize_completed_slots(
        &mut self,
        request_queue: &mut RequestQueue,
        kv_cache: &mut KvCacheManager,
        cache_mode: KvReuseMode,
    ) {
        for slot in &mut self.slots {
            if !is_terminal_phase(slot.phase) {
                continue;
            }

            let request = slot.request.take();
            let queue_cancel_requested = request_queue
                .requests
                .get(&slot.request_id)
                .is_some_and(|request| request.cancel_requested);
            let request_cancel_requested = request
                .as_ref()
                .is_some_and(|request| request.cancel_requested);
            let response_status =
                completed_slot_status(slot.phase, queue_cancel_requested, request_cancel_requested);
            let mut metrics_request: Option<(GenerateRequest, Instant)> = None;

            // Embedding-finalization wins when present: the embedding-read
            // path set `embedding_output` at terminal time and the emit-buffer
            // text buffer is ignored. Otherwise fall back to the text path.
            let output = if let Some(embedding) = slot.embedding_output.take() {
                ResponseOutput::Embedding {
                    values: embedding.values,
                    pooling: embedding.pooling,
                    normalized: embedding.normalized,
                }
            } else {
                ResponseOutput::Text(std::mem::take(&mut slot.output_text))
            };
            let mut response = GenerateResponse::terminal(
                slot.request_id,
                response_status,
                output,
                completed_slot_error_message(
                    response_status,
                    slot.phase,
                    &slot.terminal_error_message,
                ),
            );

            if let Some(mut request_val) = request {
                let completed_at = Instant::now();
                request_val.completed_at = Some(completed_at);
                request_val.output_tokens = saturating_usize_to_i32(slot.generated_tokens.len());
                request_val.emitted_token_count = request_val.output_tokens;

                let should_commit_live = response_status == GenerateResponseStatus::Completed
                    && !request_val.is_multimodal_turn;
                kv_cache.finalize_slot(
                    &request_val.context_key,
                    slot.seq_id,
                    slot.lease_generation,
                    std::mem::take(&mut slot.mirror),
                    should_commit_live,
                    cache_mode,
                );
                metrics_request = Some((request_val, completed_at));
            }

            if slot.seq_id >= 0 {
                slot.seq_id = -1;
            }

            if let Some((request, completed_at)) = metrics_request {
                response.runtime_observability = metrics_from_request(&request, completed_at);
            }

            request_queue.mark_completed(response);
            slot.reset_to_idle();
        }
    }

    pub fn emit_buffered_token_piece(request_queue: &mut RequestQueue, slot: &mut SlotState) {
        if slot.buffered_output_text.is_empty() {
            return;
        }

        let buffered = std::mem::take(&mut slot.buffered_output_text);
        let request_id = slot.request_id;
        let mut should_emit = false;

        if let Some(request) = slot.request_mut() {
            should_emit = request.emit_tokens;
        }

        if should_emit {
            request_queue.append_token_piece(request_id, &buffered);
        }
        slot.output_text.push_str(&buffered);
    }
}

fn idle_without_request(slot: &SlotState) -> bool {
    slot.phase == SlotPhase::Idle && slot.request().is_none()
}

fn reset_slot_identity(slot: &mut SlotState, slot_id: usize) {
    slot.slot_id = slot_id;
    slot.seq_id = -1;
}

fn is_terminal_phase(phase: SlotPhase) -> bool {
    matches!(phase, SlotPhase::Completed | SlotPhase::Failed)
}

fn decode_slot_ready(slot: &SlotState) -> bool {
    let request_ready = slot.request().is_some();
    let slot_ready = slot.phase == SlotPhase::Decode
        && !slot.generated_tokens.is_empty()
        && slot.buffered_output_text.is_empty();
    request_ready && slot_ready
}

fn prefill_slot_ready(slot: &SlotState) -> bool {
    let Some(request) = slot.request() else {
        return false;
    };
    if slot.phase != SlotPhase::Prefill && slot.phase != SlotPhase::Admitted {
        return false;
    }
    if request.is_multimodal_turn && request.multimodal.is_some() {
        return true;
    }
    slot.prefill_cursor < request.prompt_tokens.len()
}

fn release_slot_for_reset(kv_cache: &mut KvCacheManager, slot: &SlotState) {
    if slot.seq_id < 0 {
        return;
    }
    let Some(request) = slot.request() else {
        return;
    };
    kv_cache.release_slot_for_reset(&request.context_key, slot.seq_id, slot.lease_generation);
}

fn completed_slot_status(
    slot_phase: SlotPhase,
    queue_cancel_requested: bool,
    request_cancel_requested: bool,
) -> GenerateResponseStatus {
    if queue_cancel_requested || request_cancel_requested {
        GenerateResponseStatus::Cancelled
    } else if slot_phase == SlotPhase::Completed {
        GenerateResponseStatus::Completed
    } else {
        GenerateResponseStatus::Failed
    }
}

fn completed_slot_error_message(
    response_status: GenerateResponseStatus,
    slot_phase: SlotPhase,
    terminal_error_message: &str,
) -> String {
    if response_status == GenerateResponseStatus::Cancelled {
        REQUEST_CANCELLED_MESSAGE.to_string()
    } else if slot_phase == SlotPhase::Failed {
        if terminal_error_message.is_empty() {
            REQUEST_FAILED.to_string()
        } else {
            terminal_error_message.to_string()
        }
    } else {
        String::new()
    }
}

fn complete_failed_admission(
    request_queue: &mut RequestQueue,
    request_id: u32,
    error_message: &'static str,
) {
    request_queue.mark_completed(GenerateResponse::failed(request_id, error_message));
}
