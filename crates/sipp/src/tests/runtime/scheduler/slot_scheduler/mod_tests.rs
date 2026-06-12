//! Tests the `runtime::scheduler::slot_scheduler` module in
//! `sipp`.
//!
//! Covers the public scheduler budget wrapper with deterministic policy inputs
//! and no runtime/native state.

use crate::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode};

use super::SlotScheduler;

#[test]
fn slot_scheduler_build_tick_budget_delegates_to_policy_budget() {
    let budget = SlotScheduler::build_tick_budget(
        SchedulerPolicyConfig {
            mode: SchedulerPolicyMode::LatencyFirst,
            decode_token_reserve: 3,
            enable_adaptive_prefill_chunking: true,
        },
        2,
        4,
        6,
        99,
    );

    assert_eq!(budget.total_token_budget, 6);
    assert_eq!(budget.reserved_decode_tokens, 2);
    assert_eq!(budget.reserved_prefill_tokens, 4);
    assert!(budget.decode_first);
}
