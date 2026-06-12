use std::time::Instant;

use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::numeric::saturating_usize_to_i32;
use crate::runtime::request::GenerateRequest;

pub(super) fn metrics_from_request(
    request: &GenerateRequest,
    completed_at: Instant,
) -> RuntimeObservabilityMetrics {
    let mut metrics = RuntimeObservabilityMetrics::from_request(request);
    metrics.itl_p99_ms = request.itl_p99_ms;
    metrics.e2e_ms = RuntimeObservabilityMetrics::e2e_ms_from_request(request, completed_at);
    if metrics.input_tokens <= 0 {
        metrics.input_tokens = saturating_usize_to_i32(request.prompt_tokens.len());
    }
    metrics
}
