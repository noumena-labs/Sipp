use crate::runtime::scheduler::{SlotPhase, SlotState};

use super::helpers::token_limit_reached;
use super::{BatchContributionKind, SharedBatchPlan};

pub(super) fn apply_decode_results(slots: &mut [SlotState], plan: &SharedBatchPlan) {
    for contribution in &plan.contributions {
        let Some(slot) = slots.get_mut(contribution.slot_index) else {
            continue;
        };
        let Some(request) = slot.request() else {
            continue;
        };
        let request_max_output_tokens = request.max_output_tokens;
        let request_prompt_len = request.prompt_tokens.len();

        slot.batch_participation_count = slot.batch_participation_count.saturating_add(1);

        if contribution.kind == BatchContributionKind::Prefill {
            slot.prefill_cursor = slot.prefill_cursor.saturating_add(1);
            slot.phase = if slot.prefill_cursor >= request_prompt_len {
                SlotPhase::Decode
            } else {
                SlotPhase::Prefill
            };
            continue;
        }

        if contribution.kind != BatchContributionKind::Decode {
            continue;
        }

        slot.decode_step_count = slot.decode_step_count.saturating_add(1);
        slot.phase = if token_limit_reached(slot.generated_tokens.len(), request_max_output_tokens)
        {
            SlotPhase::Completed
        } else if !slot.buffered_output_text.is_empty() {
            SlotPhase::Streaming
        } else {
            SlotPhase::Decode
        };
    }
}
