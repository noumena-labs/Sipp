use crate::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode, SchedulerTickBudget};
use crate::runtime::numeric::{nonnegative_i32, positive_i32};

const THROUGHPUT_PREFILL_FLOOR_NUMERATOR: i32 = 3;
const THROUGHPUT_PREFILL_FLOOR_DENOMINATOR: i32 = 4;
const MIN_THROUGHPUT_DECODE_RESERVE: i32 = 1;
const BALANCED_PREFILL_FLOOR: i32 = 1;

pub(super) fn build_tick_budget(
    policy: SchedulerPolicyConfig,
    decode_ready_count: i32,
    prefill_ready_count: i32,
    max_batch_tokens: i32,
) -> SchedulerTickBudget {
    let mut budget = SchedulerTickBudget {
        total_token_budget: nonnegative_i32(max_batch_tokens),
        decode_first: decode_ready_count > 0,
        ..SchedulerTickBudget::default()
    };

    if budget.total_token_budget <= 0 {
        return budget;
    }

    let clamped_decode_ready = nonnegative_i32(decode_ready_count);
    let clamped_prefill_ready = nonnegative_i32(prefill_ready_count);

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

    let requested_decode_reserve = requested_decode_reserve(policy, clamped_decode_ready);
    let decode_ready_budget = clamped_decode_ready.min(budget.total_token_budget);

    budget.reserved_decode_tokens = match policy.mode {
        SchedulerPolicyMode::LatencyFirst => {
            if has_decode_reserve(policy) {
                decode_ready_budget.min(requested_decode_reserve)
            } else {
                decode_ready_budget
            }
        }
        SchedulerPolicyMode::ThroughputFirst => {
            let prefill_floor = throughput_prefill_floor(budget.total_token_budget);
            let decode_ceiling = positive_i32(budget.total_token_budget - prefill_floor);
            let throughput_reserve = if has_decode_reserve(policy) {
                requested_decode_reserve
            } else {
                MIN_THROUGHPUT_DECODE_RESERVE
            };
            decode_ready_budget
                .min(decode_ceiling)
                .min(throughput_reserve)
        }
        SchedulerPolicyMode::Balanced => {
            let prefill_floor = balanced_prefill_floor(budget.total_token_budget);
            let decode_ceiling = remaining_token_budget(budget.total_token_budget, prefill_floor);
            let mut decode_tokens = decode_ready_budget.min(decode_ceiling);
            if has_decode_reserve(policy) {
                decode_tokens = decode_tokens.min(requested_decode_reserve);
            }
            decode_tokens
        }
    };

    budget.reserved_prefill_tokens =
        remaining_token_budget(budget.total_token_budget, budget.reserved_decode_tokens);
    budget
}

fn has_decode_reserve(policy: SchedulerPolicyConfig) -> bool {
    policy.decode_token_reserve > 0
}

fn requested_decode_reserve(policy: SchedulerPolicyConfig, decode_ready_count: i32) -> i32 {
    if has_decode_reserve(policy) {
        policy.decode_token_reserve.min(decode_ready_count)
    } else {
        decode_ready_count
    }
}

fn throughput_prefill_floor(total_token_budget: i32) -> i32 {
    if total_token_budget > 1 {
        positive_i32(
            (total_token_budget * THROUGHPUT_PREFILL_FLOOR_NUMERATOR)
                / THROUGHPUT_PREFILL_FLOOR_DENOMINATOR,
        )
    } else {
        0
    }
}

fn balanced_prefill_floor(total_token_budget: i32) -> i32 {
    if total_token_budget > 1 {
        BALANCED_PREFILL_FLOOR
    } else {
        0
    }
}

fn remaining_token_budget(total_token_budget: i32, reserved_tokens: i32) -> i32 {
    nonnegative_i32(total_token_budget - reserved_tokens)
}
