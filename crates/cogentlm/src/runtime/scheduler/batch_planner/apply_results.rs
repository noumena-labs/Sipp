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
            slot.phase = phase_after_prefill(slot.prefill_cursor, request_prompt_len);
            continue;
        }

        if contribution.kind != BatchContributionKind::Decode {
            continue;
        }

        slot.decode_step_count = slot.decode_step_count.saturating_add(1);
        slot.phase = phase_after_decode(
            slot.generated_tokens.len(),
            request_max_output_tokens,
            !slot.buffered_output_text.is_empty(),
        );
    }
}

fn phase_after_prefill(prefill_cursor: usize, request_prompt_len: usize) -> SlotPhase {
    if prefill_cursor >= request_prompt_len {
        SlotPhase::Decode
    } else {
        SlotPhase::Prefill
    }
}

fn phase_after_decode(
    generated_token_count: usize,
    max_output_tokens: i32,
    has_buffered_output: bool,
) -> SlotPhase {
    if token_limit_reached(generated_token_count, max_output_tokens) {
        SlotPhase::Completed
    } else if has_buffered_output {
        SlotPhase::EmitBuffered
    } else {
        SlotPhase::Decode
    }
}
