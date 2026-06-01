use std::time::Instant;

use crate::runtime::config::KvReuseMode;
use crate::runtime::numeric::duration_ms;
use crate::runtime::request::GenerateRequest;

/// Cache source used to satisfy a request prefill.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheSource {
    /// No reusable KV state was used.
    #[default]
    None = 0,
    /// A live slot with matching prefix state was reused.
    Live = 1,
    /// A serialized prefix snapshot was restored before prefill.
    Snapshot = 2,
}

/// Cache reuse mode reported in runtime observability metrics.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheMetricMode {
    /// Prefix KV reuse is disabled.
    Disabled = 0,
    /// Only live slot prefix reuse is enabled.
    #[default]
    LiveSlotPrefix = 1,
    /// Only serialized state snapshot reuse is enabled.
    StateSnapshot = 2,
    /// Live slot prefix reuse and serialized snapshots are both enabled.
    LiveSlotAndSnapshot = 3,
}

impl From<KvReuseMode> for CacheMetricMode {
    fn from(mode: KvReuseMode) -> Self {
        match mode {
            KvReuseMode::Disabled => Self::Disabled,
            KvReuseMode::LiveSlotPrefix => Self::LiveSlotPrefix,
            KvReuseMode::StateSnapshot => Self::StateSnapshot,
            KvReuseMode::LiveSlotAndSnapshot => Self::LiveSlotAndSnapshot,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RuntimeObservabilityMetrics {
    pub ttft_ms: f64,
    pub itl_avg_ms: f64,
    pub itl_p99_ms: f64,
    pub e2e_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub native_gpu_ms: f64,
    pub native_sync_ms: f64,
    pub native_logic_ms: f64,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cache_mode: CacheMetricMode,
    pub cache_source: CacheSource,
    pub cache_hits: i32,
    pub prefill_tokens: i32,
}

impl RuntimeObservabilityMetrics {
    pub(crate) fn average_inter_token_ms(output_tokens: i32, decode_ms: f64) -> f64 {
        if output_tokens > 1 {
            decode_ms / f64::from(output_tokens - 1)
        } else {
            0.0
        }
    }

    pub(crate) fn ttft_ms_from_request(request: &GenerateRequest) -> f64 {
        request
            .first_token_at
            .and_then(|first_token_at| {
                request
                    .enqueued_at
                    .map(|enqueued| duration_ms(enqueued, first_token_at))
            })
            .unwrap_or(0.0)
    }

    pub(crate) fn e2e_ms_from_request(request: &GenerateRequest, completed_at: Instant) -> f64 {
        request
            .enqueued_at
            .map(|enqueued| duration_ms(enqueued, completed_at))
            .unwrap_or(0.0)
    }

    pub(crate) fn from_request(request: &GenerateRequest) -> Self {
        Self {
            ttft_ms: Self::ttft_ms_from_request(request),
            itl_avg_ms: Self::average_inter_token_ms(request.output_tokens, request.decode_ms),
            prefill_ms: request.prefill_ms,
            decode_ms: request.decode_ms,
            native_gpu_ms: request.native_gpu_ms,
            native_sync_ms: request.native_sync_ms,
            native_logic_ms: request.native_logic_ms,
            input_tokens: request.input_tokens,
            output_tokens: request.output_tokens,
            cache_mode: request.cache_mode.into(),
            cache_source: request.cache_source,
            cache_hits: request.cache_hits,
            prefill_tokens: request.prefill_tokens,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    mod metrics_tests;
}
