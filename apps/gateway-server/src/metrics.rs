use std::{
    fmt::Write,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use cogentlm_core::TokenUsage;
use cogentlm_gateway::{GatewayCancellationReason, Operation};

const LATENCY_BUCKETS_MS: [u64; 8] = [10, 25, 50, 100, 250, 500, 1_000, 5_000];

/// Replica-local gateway metrics rendered in Prometheus text format.
pub struct GatewayMetrics {
    requests_ok: [AtomicU64; 3],
    requests_error: [AtomicU64; 3],
    latency_buckets: [AtomicU64; LATENCY_BUCKETS_MS.len()],
    latency_count: AtomicU64,
    latency_sum_micros: AtomicU64,
    in_flight: AtomicU64,
    active_streams: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    cancellations: [AtomicU64; 4],
}

impl GatewayMetrics {
    /// Create zeroed metrics.
    pub fn new() -> Self {
        Self {
            requests_ok: std::array::from_fn(|_| AtomicU64::new(0)),
            requests_error: std::array::from_fn(|_| AtomicU64::new(0)),
            latency_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            latency_count: AtomicU64::new(0),
            latency_sum_micros: AtomicU64::new(0),
            in_flight: AtomicU64::new(0),
            active_streams: AtomicU64::new(0),
            input_tokens: AtomicU64::new(0),
            output_tokens: AtomicU64::new(0),
            cancellations: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Increment the active request gauge.
    pub fn request_started(&self) {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
    }

    /// Record request completion and latency.
    pub fn request_finished(&self, operation: Operation, success: bool, elapsed: Duration) {
        let index = operation_index(operation);
        let counter = if success {
            &self.requests_ok[index]
        } else {
            &self.requests_error[index]
        };
        counter.fetch_add(1, Ordering::Relaxed);
        self.in_flight.fetch_sub(1, Ordering::Relaxed);

        let millis = elapsed.as_millis().min(u128::from(u64::MAX)) as u64;
        for (index, bound) in LATENCY_BUCKETS_MS.iter().enumerate() {
            if millis <= *bound {
                self.latency_buckets[index].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.latency_count.fetch_add(1, Ordering::Relaxed);
        self.latency_sum_micros.fetch_add(
            elapsed.as_micros().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
    }

    /// Increment or decrement the active stream gauge.
    pub fn stream_delta(&self, delta: i8) {
        if delta > 0 {
            self.active_streams
                .fetch_add(delta as u64, Ordering::Relaxed);
        } else {
            self.active_streams
                .fetch_sub(delta.unsigned_abs() as u64, Ordering::Relaxed);
        }
    }

    /// Record token usage.
    pub fn usage(&self, usage: TokenUsage) {
        if let Some(tokens) = usage.input_tokens {
            self.input_tokens
                .fetch_add(u64::from(tokens), Ordering::Relaxed);
        }
        if let Some(tokens) = usage.output_tokens {
            self.output_tokens
                .fetch_add(u64::from(tokens), Ordering::Relaxed);
        }
    }

    /// Record one cancellation outcome.
    pub fn cancellation(&self, reason: GatewayCancellationReason) {
        self.cancellations[cancellation_index(reason)].fetch_add(1, Ordering::Relaxed);
    }

    /// Render Prometheus exposition text.
    pub fn render(&self) -> String {
        let mut output = String::new();
        output.push_str("# TYPE cogentlm_gateway_requests_total counter\n");
        for operation in [Operation::Query, Operation::Chat, Operation::Embed] {
            let index = operation_index(operation);
            let name = operation.as_str();
            let _ = writeln!(
                output,
                "cogentlm_gateway_requests_total{{operation=\"{name}\",outcome=\"ok\"}} {}",
                self.requests_ok[index].load(Ordering::Relaxed)
            );
            let _ = writeln!(
                output,
                "cogentlm_gateway_requests_total{{operation=\"{name}\",outcome=\"error\"}} {}",
                self.requests_error[index].load(Ordering::Relaxed)
            );
        }
        output.push_str("# TYPE cogentlm_gateway_request_duration_seconds histogram\n");
        for (index, bound) in LATENCY_BUCKETS_MS.iter().enumerate() {
            let _ = writeln!(
                output,
                "cogentlm_gateway_request_duration_seconds_bucket{{le=\"{}\"}} {}",
                *bound as f64 / 1_000.0,
                self.latency_buckets[index].load(Ordering::Relaxed)
            );
        }
        let count = self.latency_count.load(Ordering::Relaxed);
        let _ = writeln!(
            output,
            "cogentlm_gateway_request_duration_seconds_bucket{{le=\"+Inf\"}} {count}"
        );
        let _ = writeln!(
            output,
            "cogentlm_gateway_request_duration_seconds_sum {}",
            self.latency_sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0
        );
        let _ = writeln!(
            output,
            "cogentlm_gateway_request_duration_seconds_count {count}"
        );
        output.push_str("# TYPE cogentlm_gateway_in_flight_requests gauge\n");
        let _ = writeln!(
            output,
            "cogentlm_gateway_in_flight_requests {}",
            self.in_flight.load(Ordering::Relaxed)
        );
        output.push_str("# TYPE cogentlm_gateway_active_streams gauge\n");
        let _ = writeln!(
            output,
            "cogentlm_gateway_active_streams {}",
            self.active_streams.load(Ordering::Relaxed)
        );
        output.push_str("# TYPE cogentlm_gateway_tokens_total counter\n");
        let _ = writeln!(
            output,
            "cogentlm_gateway_tokens_total{{type=\"input\"}} {}",
            self.input_tokens.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            output,
            "cogentlm_gateway_tokens_total{{type=\"output\"}} {}",
            self.output_tokens.load(Ordering::Relaxed)
        );
        output.push_str("# TYPE cogentlm_gateway_cancellations_total counter\n");
        for reason in [
            GatewayCancellationReason::ClientDisconnected,
            GatewayCancellationReason::ServerShutdown,
            GatewayCancellationReason::CallerCancelled,
            GatewayCancellationReason::DeadlineExceeded,
        ] {
            let _ = writeln!(
                output,
                "cogentlm_gateway_cancellations_total{{reason=\"{}\"}} {}",
                reason.as_str(),
                self.cancellations[cancellation_index(reason)].load(Ordering::Relaxed)
            );
        }
        output
    }
}

impl Default for GatewayMetrics {
    fn default() -> Self {
        Self::new()
    }
}

fn operation_index(operation: Operation) -> usize {
    match operation {
        Operation::Query => 0,
        Operation::Chat => 1,
        Operation::Embed => 2,
    }
}

fn cancellation_index(reason: GatewayCancellationReason) -> usize {
    match reason {
        GatewayCancellationReason::ClientDisconnected => 0,
        GatewayCancellationReason::ServerShutdown => 1,
        GatewayCancellationReason::CallerCancelled => 2,
        GatewayCancellationReason::DeadlineExceeded => 3,
    }
}
