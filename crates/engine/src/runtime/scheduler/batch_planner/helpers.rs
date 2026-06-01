use crate::runtime::config::SchedulerTickBudget;
#[cfg(test)]
use crate::runtime::numeric::positive_i32_to_usize;
use crate::runtime::numeric::{positive_fair_share_i32, saturating_usize_to_i32};

const DECODE_PRESSURE_PREFILL_FLOOR: i32 = 8;

pub(super) fn resolve_prefill_slice_cap(
    budget: SchedulerTickBudget,
    configured_prefill_chunk_size: i32,
    remaining_prefill_budget: i32,
    active_prefill_slot_count: usize,
    has_decode_pressure: bool,
) -> i32 {
    if remaining_prefill_budget <= 0 {
        return 0;
    }

    let mut slice_cap = remaining_prefill_budget;
    if configured_prefill_chunk_size > 0 {
        slice_cap = slice_cap.min(configured_prefill_chunk_size);
    }

    if active_prefill_slot_count > 1 {
        let active_prefill_slot_count = saturating_usize_to_i32(active_prefill_slot_count).max(1);
        let fair_share =
            positive_fair_share_i32(remaining_prefill_budget, active_prefill_slot_count);
        slice_cap = slice_cap.min(fair_share);
    }

    if has_decode_pressure {
        let decode_pressure_slice_cap = remaining_prefill_budget.min(
            budget
                .effective_decode_budget()
                .max(DECODE_PRESSURE_PREFILL_FLOOR),
        );
        slice_cap = slice_cap.min(decode_pressure_slice_cap);
    }

    slice_cap.max(1)
}

#[cfg(test)]
pub(super) fn token_limit_reached(generated_token_count: usize, max_output_tokens: i32) -> bool {
    positive_i32_to_usize(max_output_tokens).is_some_and(|limit| generated_token_count >= limit)
}

#[cfg(test)]
mod tests {
    mod helpers_tests;
}
