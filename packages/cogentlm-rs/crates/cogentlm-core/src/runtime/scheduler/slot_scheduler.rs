use std::time::{Duration, Instant};

use crate::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode, SchedulerTickBudget};
use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::request::{
    GenerateRequest, GenerateResponse, GenerateResponseStatus, GenerateTokenEmissionMode,
    RequestQueue,
};
use crate::runtime::session::SessionStore;

use super::{SlotPhase, SlotState};

#[derive(Debug, Default)]
pub struct SlotScheduler {
    slots: Vec<SlotState>,
}

impl SlotScheduler {
    pub fn resize(&mut self, slot_count: usize) {
        if slot_count < self.slots.len() {
            for slot in &mut self.slots[slot_count..] {
                slot.reset_to_idle();
            }
        }

        self.slots.resize_with(slot_count, SlotState::default);
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
        let mut out = Vec::new();
        self.select_decode_ready_slots_into(&mut out);
        out
    }

    /// Fill `out` with the indices of slots ready to decode. Clears the
    /// caller's buffer first so it can be reused across ticks without
    /// reallocating each time.
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
        let mut out = Vec::new();
        self.select_prefill_ready_slots_into(&mut out);
        out
    }

    /// Fill `out` with the indices of slots ready to prefill. See
    /// [`Self::select_decode_ready_slots_into`].
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

    pub fn slots(&self) -> &[SlotState] {
        &self.slots
    }

    pub fn mutable_slots(&mut self) -> &mut [SlotState] {
        &mut self.slots
    }

    pub fn build_tick_budget(
        policy: SchedulerPolicyConfig,
        decode_ready_count: i32,
        prefill_ready_count: i32,
        max_batch_tokens: i32,
        _prefill_chunk_size: i32,
    ) -> SchedulerTickBudget {
        let mut budget = SchedulerTickBudget {
            total_token_budget: max_batch_tokens.max(0),
            decode_first: decode_ready_count > 0,
            ..SchedulerTickBudget::default()
        };

        if budget.total_token_budget <= 0 {
            return budget;
        }

        let clamped_decode_ready = decode_ready_count.max(0);
        let clamped_prefill_ready = prefill_ready_count.max(0);

        if clamped_decode_ready == 0 {
            budget.reserved_decode_tokens = 0;
            budget.reserved_prefill_tokens = budget.total_token_budget;
            return budget;
        }

        if clamped_prefill_ready == 0 {
            budget.reserved_decode_tokens = clamped_decode_ready.min(budget.total_token_budget);
            budget.reserved_prefill_tokens =
                budget.total_token_budget - budget.reserved_decode_tokens;
            return budget;
        }

        let requested_decode_reserve = if policy.decode_token_reserve > 0 {
            policy.decode_token_reserve.min(clamped_decode_ready)
        } else {
            clamped_decode_ready
        };
        let decode_ready_budget = clamped_decode_ready.min(budget.total_token_budget);

        budget.reserved_decode_tokens = match policy.mode {
            SchedulerPolicyMode::LatencyFirst => {
                if policy.decode_token_reserve > 0 {
                    decode_ready_budget.min(requested_decode_reserve)
                } else {
                    decode_ready_budget
                }
            }
            SchedulerPolicyMode::ThroughputFirst => {
                let prefill_floor = if budget.total_token_budget > 1 {
                    ((budget.total_token_budget * 3) / 4).max(1)
                } else {
                    0
                };
                let decode_ceiling = (budget.total_token_budget - prefill_floor).max(1);
                let throughput_reserve = if policy.decode_token_reserve > 0 {
                    requested_decode_reserve
                } else {
                    1
                };
                decode_ready_budget
                    .min(decode_ceiling)
                    .min(throughput_reserve)
            }
            SchedulerPolicyMode::Balanced => {
                let prefill_floor = if budget.total_token_budget > 1 { 1 } else { 0 };
                let decode_ceiling = (budget.total_token_budget - prefill_floor).max(0);
                let mut decode_tokens = decode_ready_budget.min(decode_ceiling);
                if policy.decode_token_reserve > 0 {
                    decode_tokens = decode_tokens.min(requested_decode_reserve);
                }
                decode_tokens
            }
        };

        budget.reserved_prefill_tokens =
            (budget.total_token_budget - budget.reserved_decode_tokens).max(0);
        budget
    }

    pub fn is_idle(&self) -> bool {
        self.slots.iter().all(|slot| slot.phase == SlotPhase::Idle)
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

        let Some(next_request_id) = request_queue
            .try_pop_next_admissible(|request| session_store.can_admit(&request.context_key))
        else {
            return false;
        };

        let Some(mut request) = request_queue.find(next_request_id).cloned() else {
            return false;
        };

        let context_key = request.context_key.clone();
        let sticky_hardware_id = {
            let session = session_store.get_or_create_session(context_key.clone());
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

        let session_snapshot = {
            let session = session_store
                .find_mut(&context_key)
                .expect("session exists after get_or_create_session");
            if leased_seq_id != session.hardware_id {
                session.current_kv_tokens.clear();
                session.n_past = 0;
            }
            session.hardware_id = leased_seq_id;

            // Move out tokens, clone cheap metadata fields, then move tokens into snapshot
            let tokens = std::mem::take(&mut session.current_kv_tokens);
            let mut snap = session.clone();
            snap.current_kv_tokens = tokens;
            snap
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
            let request_snapshot = slot.request().cloned();
            let queue_cancel_requested = request_queue
                .find(slot.request_id)
                .is_some_and(|request| request.cancel_requested);
            let mut metrics_request: Option<(GenerateRequest, Instant)> = None;

            let mut response = GenerateResponse {
                request_id: slot.request_id,
                status: if queue_cancel_requested
                    || request_snapshot
                        .as_ref()
                        .is_some_and(|request| request.cancel_requested)
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

            if let Some(mut request) = request_snapshot {
                let completed_at = Instant::now();
                request.completed_at = Some(completed_at);
                request.output_tokens = slot.generated_tokens.len().min(i32::MAX as usize) as i32;
                request.emitted_token_count = request.output_tokens;

                if let Some(session) = session_store.find_mut(&request.context_key) {
                    // The slot is about to be reset, so take the token Vec
                    // by-move instead of cloning. For long contexts this
                    // saves an O(N) i32 copy on every request finalisation.
                    session.current_kv_tokens = std::mem::take(&mut slot.mirror.current_kv_tokens);
                    session.n_past = slot.mirror.n_past;
                    session.hardware_id = slot.mirror.hardware_id;
                }

                session_store.unpin(&request.context_key);
                if request.is_multimodal_turn {
                    session_store.remove(&request.context_key);
                }
                metrics_request = Some((request, completed_at));
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

fn metrics_from_request(
    request: &GenerateRequest,
    completed_at: Instant,
) -> RuntimeObservabilityMetrics {
    RuntimeObservabilityMetrics {
        ttft_ms: request
            .first_token_at
            .and_then(|first_token_at| {
                request
                    .enqueued_at
                    .map(|enqueued| duration_ms(enqueued, first_token_at))
            })
            .unwrap_or(0.0),
        itl_avg_ms: average_inter_token_ms(request.output_tokens, request.decode_ms),
        itl_p99_ms: request.itl_p99_ms,
        e2e_ms: request
            .enqueued_at
            .map(|enqueued| duration_ms(enqueued, completed_at))
            .unwrap_or(0.0),
        prefill_ms: request.prefill_ms,
        decode_ms: request.decode_ms,
        native_gpu_ms: request.native_gpu_ms,
        native_sync_ms: request.native_sync_ms,
        native_logic_ms: request.native_logic_ms,
        input_tokens: if request.input_tokens > 0 {
            request.input_tokens
        } else {
            request.prompt_tokens.len() as i32
        },
        output_tokens: request.output_tokens,
        cache_hits: request.cache_hits,
        prefill_tokens: request.prefill_tokens,
        debug_metrics_scheduler_ticks: request.debug_metrics_scheduler_ticks,
        debug_metrics_decode_ticks: request.debug_metrics_decode_ticks,
        debug_metrics_prefill_ticks: request.debug_metrics_prefill_ticks,
        debug_metrics_backend_sampler_attach_attempts: request
            .debug_metrics_backend_sampler_attach_attempts,
        debug_metrics_backend_sampler_attach_failures: request
            .debug_metrics_backend_sampler_attach_failures,
        debug_metrics_admit_ms: request.debug_metrics_admit_ms,
        debug_metrics_normalize_ms: request.debug_metrics_normalize_ms,
        debug_metrics_backend_sampler_attach_ms: request.debug_metrics_backend_sampler_attach_ms,
        debug_metrics_select_slots_ms: request.debug_metrics_select_slots_ms,
        debug_metrics_plan_ms: request.debug_metrics_plan_ms,
        debug_metrics_batch_build_ms: request.debug_metrics_batch_build_ms,
        debug_metrics_llama_decode_ms: request.debug_metrics_llama_decode_ms,
        debug_metrics_llama_sync_ms: request.debug_metrics_llama_sync_ms,
        debug_metrics_apply_bookkeeping_ms: request.debug_metrics_apply_bookkeeping_ms,
        debug_metrics_apply_decode_results_ms: request.debug_metrics_apply_decode_results_ms,
        debug_metrics_sample_ms: request.debug_metrics_sample_ms,
        debug_metrics_token_piece_ms: request.debug_metrics_token_piece_ms,
        debug_metrics_emit_ms: request.debug_metrics_emit_ms,
        debug_metrics_prefix_queue_ms: request.debug_metrics_prefix_queue_ms,
        debug_metrics_finalize_ms: request.debug_metrics_finalize_ms,
        debug_metrics_commit_observability_ms: request.debug_metrics_commit_observability_ms,
        debug_metrics_post_decode_ms: request.debug_metrics_post_decode_ms,
    }
}

fn average_inter_token_ms(output_tokens: i32, decode_ms: f64) -> f64 {
    if output_tokens > 1 {
        decode_ms / f64::from(output_tokens - 1)
    } else {
        0.0
    }
}

fn duration_ms(start: Instant, end: Instant) -> f64 {
    duration_as_ms(end.saturating_duration_since(start))
}

fn duration_as_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode};
    use crate::runtime::request::GenerateRequestId;
    use crate::runtime::session::SequenceState;

    fn request(id: GenerateRequestId, context_key: &str) -> GenerateRequest {
        let mut request = GenerateRequest::new(id, context_key);
        request.prompt_tokens = vec![1, 2, 3];
        request
    }

    #[test]
    fn resize_resets_non_idle_slots_and_preserves_slot_ids() {
        let mut scheduler = SlotScheduler::default();
        scheduler.resize(2);
        scheduler.slots[0].attach_request(request(1, "a"), SequenceState::default());
        scheduler.slots[0].phase = SlotPhase::Decode;

        scheduler.resize(2);

        assert_eq!(scheduler.slots[0].slot_id, 0);
        assert_eq!(scheduler.slots[0].phase, SlotPhase::Idle);
        assert_eq!(scheduler.slots[1].slot_id, 1);
    }

    #[test]
    fn selects_decode_ready_slots_without_buffered_text() {
        let mut scheduler = SlotScheduler::default();
        scheduler.resize(3);
        scheduler.slots[0].attach_request(request(1, "a"), SequenceState::default());
        scheduler.slots[0].session = Some(SequenceState::default());
        scheduler.slots[0].phase = SlotPhase::Decode;
        scheduler.slots[0].generated_tokens.push(10);
        scheduler.slots[1].attach_request(request(2, "b"), SequenceState::default());
        scheduler.slots[1].session = Some(SequenceState::default());
        scheduler.slots[1].phase = SlotPhase::Decode;
        scheduler.slots[1].generated_tokens.push(11);
        scheduler.slots[1].buffered_output_text = "wait".to_string();

        assert_eq!(scheduler.select_decode_ready_slots(), vec![0]);
    }

    #[test]
    fn build_tick_budget_matches_balanced_policy_with_prefill_floor() {
        let budget = SlotScheduler::build_tick_budget(
            SchedulerPolicyConfig {
                mode: SchedulerPolicyMode::Balanced,
                decode_token_reserve: 8,
                enable_adaptive_prefill_chunking: false,
            },
            8,
            2,
            4,
            0,
        );

        assert_eq!(budget.reserved_decode_tokens, 3);
        assert_eq!(budget.reserved_prefill_tokens, 1);
    }

    #[test]
    fn admit_pending_request_leases_sequence_and_pins_session() {
        let mut scheduler = SlotScheduler::default();
        scheduler.resize(1);
        let mut queue = RequestQueue::new();
        assert!(queue.push(request(1, "ctx")));
        let mut sessions = SessionStore::new(2, 1);

        assert!(scheduler.admit_pending_requests(&mut queue, &mut sessions));

        let slot = &scheduler.slots[0];
        assert_eq!(slot.request_id, 1);
        assert_eq!(slot.seq_id, 0);
        assert_eq!(slot.phase, SlotPhase::Prefill);
        assert_eq!(
            sessions.find("ctx").map(|session| session.pin_count),
            Some(1)
        );
        assert!(!sessions.can_admit("other"));
    }

    #[test]
    fn finalize_completed_slot_writes_response_and_releases_session() {
        let mut scheduler = SlotScheduler::default();
        scheduler.resize(1);
        let mut queue = RequestQueue::new();
        assert!(queue.push(request(1, "ctx")));
        let mut sessions = SessionStore::new(2, 1);
        assert!(scheduler.admit_pending_requests(&mut queue, &mut sessions));

        let slot = &mut scheduler.slots[0];
        slot.phase = SlotPhase::Completed;
        slot.output_text = "done".to_string();
        slot.mirror.current_kv_tokens = vec![1, 2, 3, 4];
        slot.mirror.n_past = 4;

        scheduler.finalize_completed_slots(&mut queue, &mut sessions);

        let response = queue.peek_completed_response(1).expect("response");
        assert_eq!(response.status, GenerateResponseStatus::Completed);
        assert_eq!(response.output_text, "done");
        assert_eq!(
            sessions.find("ctx").map(|session| session.pin_count),
            Some(0)
        );
        assert_eq!(
            sessions
                .find("ctx")
                .map(|session| session.current_kv_tokens.clone()),
            Some(vec![1, 2, 3, 4])
        );
        assert_eq!(scheduler.slots[0].phase, SlotPhase::Idle);
        assert!(sessions.can_admit("other"));
    }

    #[test]
    fn emit_buffered_piece_appends_output_and_stream_frame_when_enabled() {
        let mut queue = RequestQueue::new();
        let (producer, consumer) = crate::runtime::request::token_byte_ring(1024);
        queue.set_token_ring_producer(Some(producer));
        let mut slot = SlotState::new(0);
        let mut request = request(1, "ctx");
        request.token_emission_mode = GenerateTokenEmissionMode::TokenStream;
        slot.attach_request(request, SequenceState::default());
        slot.buffered_output_text = "tok".to_string();

        SlotScheduler::emit_buffered_token_piece(&mut queue, &mut slot);

        assert_eq!(slot.output_text, "tok");
        assert_eq!(queue.total_emitted_token_count(), 1);
        let drain = consumer.drain_available(16, 1024);
        assert_eq!(drain.frames.len(), 1);
        assert_eq!(drain.frames[0].stream_id, 1);
        assert_eq!(drain.frames[0].bytes, b"tok");
    }
}
