use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::http::StatusCode;
use cogentlm_gateway::GatewayObservability;
use cogentlm_gateway_core::Operation;

/// Application-owned low-cardinality gateway metrics.
pub struct GatewayMetrics {
    requests: [AtomicU64; 3],
    errors: [AtomicU64; 3],
}

/// Point-in-time metric values for one gateway operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationMetricSnapshot {
    pub operation: &'static str,
    pub requests: u64,
    pub errors: u64,
}

impl GatewayMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self {
            requests: std::array::from_fn(|_| AtomicU64::new(0)),
            errors: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Render Prometheus text exposition.
    pub fn render(&self) -> String {
        let mut output = String::new();
        for operation in [Operation::Query, Operation::Chat, Operation::Embed] {
            let index = operation_index(operation);
            let name = operation_name(operation);
            let _ = writeln!(
                output,
                "cogentlm_gateway_requests_total{{operation=\"{name}\"}} {}",
                self.requests[index].load(Ordering::Relaxed)
            );
            let _ = writeln!(
                output,
                "cogentlm_gateway_errors_total{{operation=\"{name}\"}} {}",
                self.errors[index].load(Ordering::Relaxed)
            );
        }
        output
    }

    /// Return low-cardinality metric counters for dashboard rendering.
    pub fn snapshot(&self) -> [OperationMetricSnapshot; 3] {
        [Operation::Query, Operation::Chat, Operation::Embed].map(|operation| {
            let index = operation_index(operation);
            OperationMetricSnapshot {
                operation: operation_name(operation),
                requests: self.requests[index].load(Ordering::Relaxed),
                errors: self.errors[index].load(Ordering::Relaxed),
            }
        })
    }
}

impl Default for GatewayMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl GatewayObservability for GatewayMetrics {
    fn request_started(&self, operation: Operation, _request_id: Option<&str>) {
        self.requests[operation_index(operation)].fetch_add(1, Ordering::Relaxed);
    }

    fn request_finished(
        &self,
        operation: Operation,
        _request_id: Option<&str>,
        status: StatusCode,
    ) {
        if !status.is_success() {
            self.errors[operation_index(operation)].fetch_add(1, Ordering::Relaxed);
        }
    }
}

const fn operation_index(operation: Operation) -> usize {
    match operation {
        Operation::Query => 0,
        Operation::Chat => 1,
        Operation::Embed => 2,
    }
}

const fn operation_name(operation: Operation) -> &'static str {
    match operation {
        Operation::Query => "query",
        Operation::Chat => "chat",
        Operation::Embed => "embed",
    }
}
