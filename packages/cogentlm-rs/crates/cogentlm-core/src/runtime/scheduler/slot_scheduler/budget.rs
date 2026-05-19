use crate::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode, SchedulerTickBudget};

pub(super) fn build_tick_budget(
    policy: SchedulerPolicyConfig,
    decode_ready_count: i32,
    prefill_ready_count: i32,
    max_batch_tokens: i32,
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
        budget.reserved_prefill_tokens = budget.total_token_budget - budget.reserved_decode_tokens;
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
