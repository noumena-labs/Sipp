//! Tests the `runtime::inference_runtime::observability` module in `cogentlm-engine`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use crate::runtime::config::NativeRuntimeConfig;

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
