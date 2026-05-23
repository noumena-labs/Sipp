use std::time::Instant;

use crate::runtime::request::{
    GenerateRequest, GenerateResponse, GenerateResponseStatus, GenerateTokenEmissionMode,
    RequestQueue,
};
use crate::runtime::session::SessionStore;
use crate::runtime::{
    numeric::{duration_ms, saturating_usize_to_i32},
    scheduler::SlotPhase,
    REQUEST_CANCELLED_MESSAGE,
};

use super::metrics::metrics_from_request;
use super::{SlotScheduler, SlotState};

const CREATE_OR_FIND_SESSION_FAILED: &str = "Failed to create or find a session.";
const ACQUIRE_HARDWARE_SEQUENCE_FAILED: &str = "Failed to acquire a hardware sequence ID.";
const PREPARE_SESSION_FOR_ADMISSION_FAILED: &str = "Failed to prepare session for admission.";
const REQUEST_FAILED: &str = "Request failed.";

impl SlotScheduler {
    pub fn resize(&mut self, slot_count: usize) {
        if slot_count < self.slots.len() {
            for slot in &mut self.slots[slot_count..] {
                slot.reset_to_idle();
            }
        }

        self.slots.resize_with(slot_count, Default::default);
        for (slot_id, slot) in self.slots.iter_mut().enumerate() {
            reset_slot_identity(slot, slot_id);
            if idle_without_request(slot) {
                continue;
            }
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
        session_store: &mut SessionStore,
    ) -> bool {
        let debug_metrics_admit_start = Instant::now();
        let Some(idle_slot_index) = self.slots.iter().position(idle_without_request) else {
            return false;
        };

        let has_evictable = session_store.has_evictable_session();
        let Some(next_request_id) = request_queue.try_pop_next_admissible(|request| {
            session_store.can_admit_with_evictable_cached(&request.context_key, has_evictable)
        }) else {
            return false;
        };

        let Some(mut request) = request_queue.requests.get(&next_request_id).cloned() else {
            return false;
        };

        let context_key = request.context_key.clone();
        let sticky_hardware_id = {
            let Some(session) = session_store.get_or_create_session(&context_key) else {
                complete_failed_admission(request_queue, request.id, CREATE_OR_FIND_SESSION_FAILED);
                return false;
            };
            session.hardware_id
        };

        let leased_seq_id = session_store.acquire_seq_id(sticky_hardware_id);
        if leased_seq_id < 0 {
            complete_failed_admission(request_queue, request.id, ACQUIRE_HARDWARE_SEQUENCE_FAILED);
            return false;
        }

        let Some(session_snapshot) =
            session_store.prepare_for_admission(&context_key, leased_seq_id)
        else {
            session_store.release_seq_id(leased_seq_id);
            complete_failed_admission(
                request_queue,
                request.id,
                PREPARE_SESSION_FOR_ADMISSION_FAILED,
            );
            return false;
        };

        session_store.pin(&context_key);
        let slot = &mut self.slots[idle_slot_index];
        request.debug_metrics_admit_ms += duration_ms(debug_metrics_admit_start, Instant::now());
        slot.attach_request(request, session_snapshot);
        slot.seq_id = leased_seq_id;
        slot.phase = SlotPhase::Prefill;
        true
    }

    pub fn finalize_completed_slots(
        &mut self,
        request_queue: &mut RequestQueue,
        session_store: &mut SessionStore,
    ) {
        for slot in &mut self.slots {
            if !is_terminal_phase(slot.phase) {
                continue;
            }

            let debug_metrics_finalize_start = Instant::now();
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

            let mut response = GenerateResponse::terminal(
                slot.request_id,
                response_status,
                std::mem::take(&mut slot.output_text),
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

                if let Some(session) = session_store.find_mut(&request_val.context_key) {
                    session.current_kv_tokens = std::mem::take(&mut slot.mirror.current_kv_tokens);
                    session.n_past = slot.mirror.n_past;
                    session.hardware_id = slot.mirror.hardware_id;
                }

                session_store.unpin(&request_val.context_key);
                if request_val.is_multimodal_turn {
                    session_store.remove(&request_val.context_key);
                }
                metrics_request = Some((request_val, completed_at));
            }

            if slot.seq_id >= 0 {
                session_store.release_seq_id(slot.seq_id);
                slot.seq_id = -1;
            }

            if let Some((mut request, completed_at)) = metrics_request {
                request.debug_metrics_finalize_ms +=
                    duration_ms(debug_metrics_finalize_start, Instant::now());
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
            should_emit = request.token_emission_mode == GenerateTokenEmissionMode::TokenStream;
        }

        if should_emit {
            request_queue.append_streaming_token(request_id, &buffered);
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
    let request_ready = slot.request().is_some() && slot.session.is_some();
    let slot_ready = slot.phase == SlotPhase::Decode
        && !slot.generated_tokens.is_empty()
        && slot.buffered_output_text.is_empty();
    request_ready && slot_ready
}

fn prefill_slot_ready(slot: &SlotState) -> bool {
    let Some(request) = slot.request() else {
        return false;
    };
    if slot.session.is_none() {
        return false;
    }
    if slot.phase != SlotPhase::Prefill && slot.phase != SlotPhase::Admitted {
        return false;
    }
    if request.is_multimodal_turn && request.multimodal.is_some() {
        return true;
    }
    slot.prefill_cursor < request.prompt_tokens.len()
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
