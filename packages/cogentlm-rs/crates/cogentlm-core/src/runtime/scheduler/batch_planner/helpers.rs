use crate::runtime::config::SchedulerTickBudget;

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
        let fair_share = (remaining_prefill_budget / active_prefill_slot_count).max(1);
        slice_cap = slice_cap.min(fair_share);
    }

    if has_decode_pressure {
        let decode_pressure_slice_cap =
            remaining_prefill_budget.min(budget.effective_decode_budget().max(8));
        slice_cap = slice_cap.min(decode_pressure_slice_cap);
    }

    slice_cap.max(1)
}

pub(super) fn erase_active_slot(
    active_prefill_slots: &mut Vec<usize>,
    in_tick_offset: &mut Vec<usize>,
    index: usize,
) {
    active_prefill_slots.remove(index);
    in_tick_offset.remove(index);
}

pub(super) fn token_limit_reached(generated_token_count: usize, max_output_tokens: i32) -> bool {
    max_output_tokens > 0
        && generated_token_count >= usize::try_from(max_output_tokens).unwrap_or(usize::MAX)
}

pub(super) fn saturating_usize_to_i32(value: usize) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

pub(super) fn saturating_u32_to_i32(value: u32) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

pub(super) fn positive_i32_to_usize(value: i32) -> Option<usize> {
    usize::try_from(value).ok().filter(|value| *value > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_saturate_scheduler_counts() {
        assert_eq!(saturating_usize_to_i32(i32::MAX as usize + 1), i32::MAX);
        assert_eq!(saturating_u32_to_i32(u32::MAX), i32::MAX);
        assert_eq!(positive_i32_to_usize(0), None);
        assert_eq!(positive_i32_to_usize(4), Some(4));
        assert!(token_limit_reached(4, 4));
        assert!(!token_limit_reached(usize::MAX, -1));
    }
}
