//! Tests the `runtime::inference_runtime::scheduler_api` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use std::time::Duration;

use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;

use super::*;

#[test]
fn record_tick_progress_accumulates_completed_emitted_and_progressed_counts() {
    let mut result = SchedulerBurstResult::default();

    record_tick_progress(&mut result, 1, 3, 2, 5, RequestStepResult::Progressed);

    assert_eq!(result.ticks_executed, 1);
    assert_eq!(result.completed_response_count, 2);
    assert_eq!(result.emitted_token_count, 3);
    assert_eq!(result.progressed_ticks, 1);
}

#[test]
fn record_tick_progress_saturates_negative_or_non_progressing_deltas() {
    let mut result = SchedulerBurstResult::default();

    record_tick_progress(&mut result, 3, 1, 5, 2, RequestStepResult::Waiting);

    assert_eq!(result.ticks_executed, 1);
    assert_eq!(result.completed_response_count, 0);
    assert_eq!(result.emitted_token_count, 0);
    assert_eq!(result.progressed_ticks, 0);
}

#[test]
fn completed_or_waiting_treats_terminal_progress_as_progressed() {
    let result = SchedulerBurstResult {
        progressed_ticks: 1,
        ..SchedulerBurstResult::default()
    };

    assert_eq!(completed_or_waiting(&result), RequestStepResult::Progressed);
    assert_eq!(
        completed_or_waiting(&SchedulerBurstResult::default()),
        RequestStepResult::Waiting
    );
}

#[test]
fn scheduler_burst_rejects_invalid_tick_limit() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());

    let result = runtime.run_scheduler_burst(0, 0, 0, Duration::ZERO);

    assert_eq!(result.status, RequestStepResult::Invalid);
    assert_eq!(result.ticks_executed, 0);
}

#[test]
fn scheduler_loop_rejects_not_ready_runtime() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());

    let result = runtime.run_scheduler_loop(1, 0, 0, Duration::ZERO);

    assert_eq!(result.status, RequestStepResult::Invalid);
    assert_eq!(result.ticks_executed, 0);
}
