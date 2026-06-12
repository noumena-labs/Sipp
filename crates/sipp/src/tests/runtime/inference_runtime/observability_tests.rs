//! Tests the `runtime::inference_runtime::observability` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::metrics::{CacheSource, RuntimeObservabilityMetrics};
use crate::runtime::request::{GenerateRequest, GenerateResponse};
use crate::runtime::scheduler::SlotState;

#[test]
fn runtime_observability_clamps_saturated_total_counters() {
    let mut config = NativeRuntimeConfig::default();
    config.observability.runtime_metrics = true;
    let mut runtime = super::runtime_tests::test_runtime(config);
    runtime.total_input_tokens = usize::MAX;
    runtime.total_output_tokens = usize::MAX;
    runtime.total_cache_hits = usize::MAX;
    runtime.total_prefill_tokens = usize::MAX;

    let metrics = runtime
        .try_get_runtime_observability()
        .expect("runtime metrics");

    assert_eq!(metrics.input_tokens, i32::MAX);
    assert_eq!(metrics.output_tokens, i32::MAX);
    assert_eq!(metrics.cache_hits, i32::MAX);
    assert_eq!(metrics.prefill_tokens, i32::MAX);
}

#[test]
fn runtime_observability_returns_none_when_disabled() {
    let runtime = super::runtime_tests::test_runtime(NativeRuntimeConfig::default());

    assert_eq!(runtime.try_get_runtime_observability(), None);
}

#[test]
fn runtime_observability_prefers_active_request_metrics() {
    let mut config = NativeRuntimeConfig::default();
    config.observability.runtime_metrics = true;
    let mut runtime = super::runtime_tests::test_runtime(config);
    runtime.total_input_tokens = 99;
    let mut request = GenerateRequest::new(7, "ctx");
    request.input_tokens = 3;
    request.output_tokens = 2;
    request.cache_source = CacheSource::Live;
    let mut slot = SlotState::new(0);
    slot.request = Some(request);
    runtime.slot_scheduler.slots.push(slot);

    let metrics = runtime
        .try_get_runtime_observability()
        .expect("runtime metrics");

    assert_eq!(metrics.input_tokens, 3);
    assert_eq!(metrics.output_tokens, 2);
    assert_eq!(metrics.cache_source, CacheSource::Live);
}

#[test]
fn runtime_observability_commits_last_completed_response_once() {
    let mut config = NativeRuntimeConfig::default();
    config.observability.runtime_metrics = true;
    let mut runtime = super::runtime_tests::test_runtime(config);
    let response_metrics = RuntimeObservabilityMetrics {
        input_tokens: 11,
        output_tokens: 5,
        cache_hits: 2,
        ..RuntimeObservabilityMetrics::default()
    };
    runtime.request_queue.completed_responses.insert(
        7,
        GenerateResponse {
            request_id: 7,
            runtime_observability: response_metrics,
            ..GenerateResponse::default()
        },
    );

    runtime.commit_new_completed_responses_observability_locked();
    runtime.commit_new_completed_responses_observability_locked();

    assert_eq!(runtime.committed_observability_request_ids.len(), 1);
    assert_eq!(
        runtime.try_get_runtime_observability(),
        Some(response_metrics)
    );
}
