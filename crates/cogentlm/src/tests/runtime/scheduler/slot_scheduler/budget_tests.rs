//! Tests the `runtime::scheduler::slot_scheduler::budget` module in `cogentlm`.
//!
//! Covers scheduler planning, budget accounting, slot state, and flow decisions with deterministic in-memory fixtures.

use crate::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode};

use super::budget::build_tick_budget;

#[test]
fn balanced_policy_keeps_prefill_floor() {
    let budget = build_tick_budget(
        SchedulerPolicyConfig {
            mode: SchedulerPolicyMode::Balanced,
            decode_token_reserve: 8,
            enable_adaptive_prefill_chunking: false,
        },
        8,
        2,
        4,
    );

    assert_eq!(budget.reserved_decode_tokens, 3);
    assert_eq!(budget.reserved_prefill_tokens, 1);
}

#[test]
fn throughput_policy_keeps_prefill_floor_and_min_decode_reserve() {
    let budget = build_tick_budget(
        SchedulerPolicyConfig {
            mode: SchedulerPolicyMode::ThroughputFirst,
            decode_token_reserve: 0,
            enable_adaptive_prefill_chunking: false,
        },
        8,
        2,
        8,
    );

    assert_eq!(budget.reserved_decode_tokens, 1);
    assert_eq!(budget.reserved_prefill_tokens, 7);
}

#[test]
fn budget_without_prefill_reserves_decode_up_to_ready_count() {
    let budget = build_tick_budget(
        SchedulerPolicyConfig {
            mode: SchedulerPolicyMode::ThroughputFirst,
            decode_token_reserve: 0,
            enable_adaptive_prefill_chunking: false,
        },
        3,
        0,
        8,
    );

    assert_eq!(budget.reserved_decode_tokens, 3);
    assert_eq!(budget.reserved_prefill_tokens, 5);
}
