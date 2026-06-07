//! Tests HTTP helper behavior and verifies that the Prometheus surface never
//! exposes request, caller, or provider identifiers as labels.

use crate::{http::constant_time_eq, metrics::GatewayMetrics};

#[test]
fn metrics_do_not_contain_high_cardinality_identifiers() {
    let metrics = GatewayMetrics::new();
    let rendered = metrics.render();

    assert!(!rendered.contains("request_id="));
    assert!(!rendered.contains("caller="));
    assert!(!rendered.contains("provider="));
}

#[test]
fn constant_time_comparison_handles_different_lengths() {
    assert!(constant_time_eq(b"secret", b"secret"));
    assert!(!constant_time_eq(b"secret", b"secret2"));
}
