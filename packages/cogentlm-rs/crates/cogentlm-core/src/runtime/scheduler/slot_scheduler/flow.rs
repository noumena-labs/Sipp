use std::time::Instant;

use crate::runtime::request::{
    GenerateRequest, GenerateResponse, GenerateResponseStatus, GenerateTokenEmissionMode,
    RequestQueue,
};
use crate::runtime::session::SessionStore;

use super::metrics::{duration_ms, metrics_from_request, saturating_usize_to_i32};
use super::{SlotPhase, SlotScheduler, SlotState};

impl SlotScheduler {
    pub fn resize(&mut self, slot_count: usize) {
        if slot_count < self.slots.len() {
            for slot in &mut self.slots[slot_count..] {
                slot.reset_to_idle();
            }
        }

        self.slots.resize_with(slot_count, Default::default);
        for (slot_id, slot) in self.slots.iter_mut().enumerate() {
            slot.slot_id = slot_id;
            slot.seq_id = -1;
            if slot.phase == SlotPhase::Idle && slot.request().is_none() {
                continue;
            }
            slot.reset_to_idle();
            slot.slot_id = slot_id;
            slot.seq_id = -1;
        }
    }

    pub fn find_first_active_slot(&self) -> Option<usize> {
        self.slots.iter().position(|slot| {
            slot.request().is_some()
                && slot.phase != SlotPhase::Idle
                && slot.phase != SlotPhase::Completed
                && slot.phase != SlotPhase::Failed
        })
    }

    pub fn select_decode_ready_slots(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.slots.len());
        self.select_decode_ready_slots_into(&mut out);
        out
    }

    pub fn select_decode_ready_slots_into(&self, out: &mut Vec<usize>) {
        out.clear();
        for (index, slot) in self.slots.iter().enumerate() {
            let request_ready = slot.request().is_some() && slot.session.is_some();
            let slot_ready = slot.phase == SlotPhase::Decode
                && !slot.generated_tokens.is_empty()
                && slot.buffered_output_text.is_empty();
            if request_ready && slot_ready {
                out.push(index);
            }
        }
    }

    pub fn select_prefill_ready_slots(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.slots.len());
        self.select_prefill_ready_slots_into(&mut out);
        out
    }

    pub fn select_prefill_ready_slots_into(&self, out: &mut Vec<usize>) {
        out.clear();
        for (index, slot) in self.slots.iter().enumerate() {
            let Some(request) = slot.request() else {
                continue;
            };
            if slot.session.is_none() {
                continue;
            }
            if slot.phase != SlotPhase::Prefill && slot.phase != SlotPhase::Admitted {
                continue;
            }
            if request.is_multimodal_turn && request.multimodal.is_some() {
                out.push(index);
                continue;
            }
            if slot.prefill_cursor < request.prompt_tokens.len() {
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
        let Some(idle_slot_index) = self
            .slots
            .iter()
            .position(|slot| slot.phase == SlotPhase::Idle && slot.request().is_none())
        else {
            return false;
        };

        let has_evictable = session_store.has_evictable_session();
        let Some(next_request_id) = request_queue.try_pop_next_admissible(|request| {
            session_store.can_admit_with_evictable_cached(&request.context_key, has_evictable)
        }) else {
            return false;
        };

        let Some(mut request) = request_queue.find(next_request_id).cloned() else {
            return false;
        };

        let context_key = request.context_key.clone();
        let sticky_hardware_id = {
            let Some(session) = session_store.get_or_create_session(&context_key) else {
                request_queue.mark_completed(GenerateResponse {
                    request_id: request.id,
                    status: GenerateResponseStatus::Failed,
                    error_message: "Failed to create or find a session.".to_string(),
                    ..GenerateResponse::default()
                });
                return false;
            };
            session.hardware_id
        };

        let leased_seq_id = session_store.acquire_seq_id(sticky_hardware_id);
        if leased_seq_id < 0 {
            request_queue.mark_completed(GenerateResponse {
                request_id: request.id,
                status: GenerateResponseStatus::Failed,
                error_message: "Failed to acquire a hardware sequence ID.".to_string(),
                ..GenerateResponse::default()
            });
            return false;
        }

        let Some(session_snapshot) =
            session_store.prepare_for_admission(&context_key, leased_seq_id)
        else {
            session_store.release_seq_id(leased_seq_id);
            request_queue.mark_completed(GenerateResponse {
                request_id: request.id,
                status: GenerateResponseStatus::Failed,
                error_message: "Failed to prepare session for admission.".to_string(),
                ..GenerateResponse::default()
            });
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
            if slot.phase != SlotPhase::Completed && slot.phase != SlotPhase::Failed {
                continue;
            }

            let debug_metrics_finalize_start = Instant::now();
            let request = slot.request.take();
            let queue_cancel_requested = request_queue
                .find(slot.request_id)
                .is_some_and(|request| request.cancel_requested);
            let mut metrics_request: Option<(GenerateRequest, Instant)> = None;

            let mut response = GenerateResponse {
                request_id: slot.request_id,
                status: if queue_cancel_requested
                    || request.as_ref().is_some_and(|r| r.cancel_requested)
                {
                    GenerateResponseStatus::Cancelled
                } else if slot.phase == SlotPhase::Completed {
                    GenerateResponseStatus::Completed
                } else {
                    GenerateResponseStatus::Failed
                },
                output_text: std::mem::take(&mut slot.output_text),
                ..GenerateResponse::default()
            };

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

            if response.status == GenerateResponseStatus::Cancelled {
                response.error_message = "Request cancelled.".to_string();
            } else if slot.phase == SlotPhase::Failed {
                response.error_message = if slot.terminal_error_message.is_empty() {
                    "Request failed.".to_string()
                } else {
                    slot.terminal_error_message.clone()
                };
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
