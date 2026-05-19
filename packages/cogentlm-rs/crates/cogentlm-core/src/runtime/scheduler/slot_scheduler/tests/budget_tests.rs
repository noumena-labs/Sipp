use crate::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode};

use super::super::budget::build_tick_budget;

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
