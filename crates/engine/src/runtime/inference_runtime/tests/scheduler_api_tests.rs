use std::time::Duration;

use crate::runtime::config::{
    NativeRuntimeConfig, SchedulerPolicyConfig, SchedulerPolicyMode, SchedulerTickBudget,
};

use super::super::scheduler_api::completed_or_waiting;
use super::super::{RequestStepResult, SchedulerBurstResult};

#[test]
fn scheduler_loop_reports_invalid_when_runtime_is_not_ready() {
    let mut runtime = super::runtime_tests::test_runtime(NativeRuntimeConfig::default());

    let result = runtime.run_scheduler_loop(1, 0, 0, Duration::ZERO);

    assert_eq!(result.status, RequestStepResult::Invalid);
    assert_eq!(result.ticks_executed, 0);
}

#[test]
fn completed_or_waiting_reports_progress_for_any_completed_response() {
    let result = SchedulerBurstResult {
        completed_response_count: 1,
        ..SchedulerBurstResult::default()
    };

    assert_eq!(completed_or_waiting(&result), RequestStepResult::Progressed);
}

#[test]
fn adaptive_prefill_chunk_matches_cpp_fair_share() {
    let mut config = NativeRuntimeConfig::default();
    config.scheduler.prefill_chunk_size = 8;
    config.scheduler.policy = SchedulerPolicyConfig {
        mode: SchedulerPolicyMode::Balanced,
        decode_token_reserve: 1,
        enable_adaptive_prefill_chunking: true,
    };
    let runtime = super::runtime_tests::test_runtime(config);

    let chunk = runtime.resolve_prefill_chunk_size_locked(
        SchedulerTickBudget {
            total_token_budget: 9,
            reserved_decode_tokens: 1,
            reserved_prefill_tokens: 8,
            decode_first: true,
        },
        1,
        4,
    );

    assert_eq!(chunk, 2);
}
