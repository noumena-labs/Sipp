use crate::runtime::request::GenerateRequestLifecycle;
use crate::runtime::scheduler::SlotPhase;

use super::{unique_slot_first_use, InferenceRuntime};

impl InferenceRuntime {
    pub(super) fn build_no_progress_diagnostic_locked(&self) -> String {
        let mut counts = NoProgressCounts::default();

        for slot in self.slot_scheduler.slots() {
            let Some(request) = slot.request() else {
                continue;
            };
            if !matches!(
                slot.phase,
                SlotPhase::Idle | SlotPhase::Completed | SlotPhase::Failed
            ) {
                counts.active += 1;
            }
            if slot.phase == SlotPhase::Decode
                && slot.buffered_output_text.is_empty()
                && !slot.generated_tokens.is_empty()
            {
                counts.decode_ready += 1;
            }
            if slot.phase == SlotPhase::Prefill
                && (request.is_multimodal_turn || slot.prefill_cursor < request.prompt_tokens.len())
            {
                counts.prefill_ready += 1;
            }
            if slot.phase == SlotPhase::Decode && slot.generated_tokens.is_empty() {
                counts.decode_without_seed += 1;
            }
            if slot.phase == SlotPhase::Streaming && slot.buffered_output_text.is_empty() {
                counts.streaming_without_buffer += 1;
            }
        }

        counts.to_message()
    }

    pub(super) fn fail_plan_slots(
        &mut self,
        plan: &crate::runtime::scheduler::SharedBatchPlan,
        message: &str,
    ) {
        let mut failed_slots: u64 = 0;
        for contribution in &plan.contributions {
            if !unique_slot_first_use(&mut failed_slots, contribution.slot_index) {
                continue;
            }
            let Some(slot) = self
                .slot_scheduler
                .mutable_slots()
                .get_mut(contribution.slot_index)
            else {
                continue;
            };
            slot.terminal_error_message = message.to_string();
            slot.phase = SlotPhase::Failed;
            if let Some(request) = slot.request_mut() {
                request.lifecycle = GenerateRequestLifecycle::Failed;
            }
        }
    }
}

#[derive(Default)]
pub(super) struct NoProgressCounts {
    pub(super) active: usize,
    pub(super) decode_ready: usize,
    pub(super) prefill_ready: usize,
    pub(super) decode_without_seed: usize,
    pub(super) streaming_without_buffer: usize,
}

impl NoProgressCounts {
    pub(super) fn to_message(&self) -> String {
        format!(
            "Shared batch tick could not make progress (active={}, decode_ready={}, prefill_ready={}, decode_without_seed={}, streaming_without_buffer={}).",
            self.active,
            self.decode_ready,
            self.prefill_ready,
            self.decode_without_seed,
            self.streaming_without_buffer,
        )
    }
}
