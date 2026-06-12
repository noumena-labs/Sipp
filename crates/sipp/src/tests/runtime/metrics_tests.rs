//! Tests the `runtime::metrics` module in `sipp`.
//!
//! Covers runtime support modules with deterministic in-memory fixtures and no native model execution.

use super::*;
use std::time::Duration;

#[test]
fn request_metrics_include_derived_latency_fields() {
    let enqueued_at = Instant::now();
    let mut request = GenerateRequest::new(1, "ctx");
    request.enqueued_at = Some(enqueued_at);
    request.first_token_at = Some(enqueued_at + Duration::from_millis(12));
    request.output_tokens = 3;
    request.decode_ms = 8.0;

    let metrics = RuntimeObservabilityMetrics::from_request(&request);

    assert_eq!(metrics.ttft_ms, 12.0);
    assert_eq!(metrics.itl_avg_ms, 4.0);
}
