use std::time::Instant;

use cogentlm_sys as ffi;

use crate::runtime::request::GenerateRequestLifecycle;
use crate::runtime::scheduler::{BatchContributionKind, SlotPhase, SlotScheduler, TerminalAction};

use super::text::{append_token_piece_to_slot, apply_stop_sequences_to_slot, flush_pending_utf8};
use super::{
    duration_ms, nonnegative_i32_to_usize, unique_slot_first_use, InferenceRuntime,
    LLAMA_SAMPLER_SAMPLE_FAILED,
};

impl InferenceRuntime {
    pub(super) fn apply_bookkeeping_and_emit(
        &mut self,
        plan: &crate::runtime::scheduler::SharedBatchPlan,
        native_decode_ms: f64,
        native_sync_ms: f64,
        native_logic_ms: f64,
    ) {
        let mut tick_had_prefill = false;
        let mut tick_had_decode = false;
        let tick_ms = native_decode_ms + native_sync_ms + native_logic_ms;
        let mut timed_slots: u64 = 0;
        let mut prefill_timed_slots: u64 = 0;
        let mut decode_timed_slots: u64 = 0;
        let mut emitted_slots: u64 = 0;

        for contribution in &plan.contributions {
            let Some(slot) = self.slot_scheduler.slots.get_mut(contribution.slot_index) else {
                continue;
            };

            let Some(next_n_past) = slot.mirror.n_past.checked_add(1) else {
                slot.fail("KV position overflowed during batch bookkeeping.");
                continue;
            };
            slot.mirror.current_kv_tokens.push(contribution.token);
            slot.mirror.n_past = next_n_past;
            slot.batch_participation_count = slot.batch_participation_count.saturating_add(1);

            let is_prefill = contribution.kind == BatchContributionKind::Prefill;
            if is_prefill {
                let prompt_len = slot.request().map(|r| r.prompt_tokens.len()).unwrap_or(0);
                let Some(next_prefill_cursor) = slot.prefill_cursor.checked_add(1) else {
                    slot.fail("Prefill cursor overflowed during batch bookkeeping.");
                    continue;
                };
                slot.prefill_cursor = next_prefill_cursor;
                slot.phase = if slot.prefill_cursor >= prompt_len {
                    SlotPhase::Decode
                } else {
                    SlotPhase::Prefill
                };
            } else {
                slot.decode_step_count = slot.decode_step_count.saturating_add(1);
            }

            let unique_timed = unique_slot_first_use(&mut timed_slots, contribution.slot_index);
            let unique_prefill_timed = is_prefill
                && unique_slot_first_use(&mut prefill_timed_slots, contribution.slot_index);
            let unique_decode_timed = !is_prefill
                && unique_slot_first_use(&mut decode_timed_slots, contribution.slot_index);
            if let Some(request) = slot.request_mut() {
                if unique_timed {
                    request.native_gpu_ms += native_decode_ms;
                    request.native_sync_ms += native_sync_ms;
                    request.native_logic_ms += native_logic_ms;
                }
                if is_prefill {
                    request.prefill_tokens = request.prefill_tokens.saturating_add(1);
                    if unique_prefill_timed {
                        request.prefill_ms += tick_ms;
                    }
                } else if unique_decode_timed {
                    request.decode_ms += tick_ms;
                }
            }
            if is_prefill {
                self.total_prefill_tokens = self.total_prefill_tokens.saturating_add(1);
                tick_had_prefill = true;
            } else {
                tick_had_decode = true;
            }

            if unique_slot_first_use(&mut emitted_slots, contribution.slot_index)
                && !slot.buffered_output_text.is_empty()
            {
                SlotScheduler::emit_buffered_token_piece(&mut self.request_queue, slot);
            }
        }

        if tick_had_decode {
            self.total_decode_ms += tick_ms;
        }
        if tick_had_prefill {
            self.total_prefill_ms += tick_ms;
        }

        // Decoder-only embedding slots: when prefill just drained the prompt,
        // read the pooled embedding from the context instead of letting the
        // standard sample/decode loop take over. We collect indices in this
        // separate pass so the per-contribution borrow on `slot` above can
        // unwind cleanly before the embedding read, which needs `&mut self`.
        let pending_reads: Vec<usize> = self
            .slot_scheduler
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                let prompt_len = slot.request().map(|r| r.prompt_tokens.len()).unwrap_or(0);
                let ready = slot.phase == SlotPhase::Decode
                    && slot.plan.terminal == TerminalAction::ReadEmbedding
                    && slot.prefill_cursor >= prompt_len
                    && slot.embedding_output.is_none();
                ready.then_some(index)
            })
            .collect();
        for slot_index in pending_reads {
            if let Err(error) = self.read_slot_embedding(slot_index) {
                if let Some(slot) = self.slot_scheduler.slots.get_mut(slot_index) {
                    slot.fail(format!("embedding read failed: {error}"));
                }
            }
        }
    }

    pub(super) fn sample_logits_and_buffer_output(
        &mut self,
        vocab: *const ffi::llama_vocab,
    ) -> (f64, f64) {
        let mut sample_ms = 0.0;
        let mut token_piece_ms = 0.0;
        let now = Instant::now();
        let enable_metrics = self.debug_metrics_enabled;
        for pending_logits in &mut self.scratch_logits_contributions {
            let Some(slot) = self.slot_scheduler.slots.get_mut(pending_logits.slot_index) else {
                continue;
            };
            let Some(sampler) = slot.sampler else {
                continue;
            };

            let sample_start = enable_metrics.then(Instant::now);
            let next_token = unsafe {
                ffi::cogent_common_sampler_sample(
                    sampler.as_ptr(),
                    self.shared_context,
                    pending_logits.batch_token_index,
                )
            };
            if let Some(start) = sample_start {
                sample_ms += duration_ms(start, Instant::now());
            }
            pending_logits.sampled_token = next_token;

            if next_token == ffi::LLAMA_TOKEN_NULL {
                slot.fail(LLAMA_SAMPLER_SAMPLE_FAILED);
                continue;
            }
            unsafe {
                ffi::cogent_common_sampler_accept(sampler.as_ptr(), next_token, true);
            }

            let is_eog = unsafe { ffi::llama_vocab_is_eog(vocab, next_token) };
            if is_eog {
                if let Some(request) = slot.request_mut() {
                    request.first_token_at.get_or_insert(now);
                    request.first_sampled_token_id = next_token;
                    request.lifecycle = GenerateRequestLifecycle::Completed;
                }
                flush_pending_utf8(slot);
                slot.phase = SlotPhase::Completed;
                continue;
            }

            slot.generated_tokens.push(next_token);
            self.total_output_tokens = self.total_output_tokens.saturating_add(1);

            let piece_start = enable_metrics.then(Instant::now);
            append_token_piece_to_slot(vocab, next_token, slot, &mut self.scratch_token_piece);
            if let Some(start) = piece_start {
                token_piece_ms += duration_ms(start, Instant::now());
            }

            let stop_matched = apply_stop_sequences_to_slot(slot);
            let gen_len = slot.generated_tokens.len();
            let mut cancel = false;
            let mut max_output_tokens = 0;
            if let Some(r) = slot.request() {
                cancel = r.cancel_requested;
                max_output_tokens = r.max_output_tokens;
            }
            let should_complete = stop_matched
                || cancel
                || (max_output_tokens > 0
                    && gen_len >= nonnegative_i32_to_usize(max_output_tokens));

            {
                let Some(request) = slot.request_mut() else {
                    continue;
                };
                request.first_token_at.get_or_insert(now);
                request.first_sampled_token_id = next_token;
                request.output_tokens = request.output_tokens.saturating_add(1);
                request.emitted_token_count = request.emitted_token_count.saturating_add(1);
                request.last_token_at = Some(now);
                request.lifecycle = if should_complete {
                    GenerateRequestLifecycle::Completed
                } else {
                    GenerateRequestLifecycle::Decoding
                };
            }

            if should_complete {
                flush_pending_utf8(slot);
                slot.phase = SlotPhase::Completed;
            } else {
                slot.phase = SlotPhase::EmitBuffered;
            }
        }
        (sample_ms, token_piece_ms)
    }
}
