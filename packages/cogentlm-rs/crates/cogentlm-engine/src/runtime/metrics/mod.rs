use std::time::Instant;

use crate::runtime::numeric::duration_ms;
use crate::runtime::request::GenerateRequest;

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
    pub cache_hits: i32,
    pub prefill_tokens: i32,
    pub debug_metrics_scheduler_ticks: i32,
    pub debug_metrics_decode_ticks: i32,
    pub debug_metrics_prefill_ticks: i32,
    pub debug_metrics_backend_sampler_attach_attempts: i32,
    pub debug_metrics_backend_sampler_attach_failures: i32,
    pub debug_metrics_admit_ms: f64,
    pub debug_metrics_normalize_ms: f64,
    pub debug_metrics_backend_sampler_attach_ms: f64,
    pub debug_metrics_select_slots_ms: f64,
    pub debug_metrics_plan_ms: f64,
    pub debug_metrics_batch_build_ms: f64,
    pub debug_metrics_llama_decode_ms: f64,
    pub debug_metrics_llama_sync_ms: f64,
    pub debug_metrics_apply_bookkeeping_ms: f64,
    pub debug_metrics_apply_decode_results_ms: f64,
    pub debug_metrics_sample_ms: f64,
    pub debug_metrics_token_piece_ms: f64,
    pub debug_metrics_emit_ms: f64,
    pub debug_metrics_prefix_queue_ms: f64,
    pub debug_metrics_finalize_ms: f64,
    pub debug_metrics_commit_observability_ms: f64,
    pub debug_metrics_post_decode_ms: f64,
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
            cache_hits: request.cache_hits,
            prefill_tokens: request.prefill_tokens,
            debug_metrics_scheduler_ticks: request.debug_metrics_scheduler_ticks,
            debug_metrics_decode_ticks: request.debug_metrics_decode_ticks,
            debug_metrics_prefill_ticks: request.debug_metrics_prefill_ticks,
            debug_metrics_backend_sampler_attach_attempts: request
                .debug_metrics_backend_sampler_attach_attempts,
            debug_metrics_backend_sampler_attach_failures: request
                .debug_metrics_backend_sampler_attach_failures,
            debug_metrics_admit_ms: request.debug_metrics_admit_ms,
            debug_metrics_normalize_ms: request.debug_metrics_normalize_ms,
            debug_metrics_backend_sampler_attach_ms: request
                .debug_metrics_backend_sampler_attach_ms,
            debug_metrics_select_slots_ms: request.debug_metrics_select_slots_ms,
            debug_metrics_plan_ms: request.debug_metrics_plan_ms,
            debug_metrics_batch_build_ms: request.debug_metrics_batch_build_ms,
            debug_metrics_llama_decode_ms: request.debug_metrics_llama_decode_ms,
            debug_metrics_llama_sync_ms: request.debug_metrics_llama_sync_ms,
            debug_metrics_apply_bookkeeping_ms: request.debug_metrics_apply_bookkeeping_ms,
            debug_metrics_apply_decode_results_ms: request.debug_metrics_apply_decode_results_ms,
            debug_metrics_sample_ms: request.debug_metrics_sample_ms,
            debug_metrics_token_piece_ms: request.debug_metrics_token_piece_ms,
            debug_metrics_emit_ms: request.debug_metrics_emit_ms,
            debug_metrics_prefix_queue_ms: request.debug_metrics_prefix_queue_ms,
            debug_metrics_finalize_ms: request.debug_metrics_finalize_ms,
            debug_metrics_commit_observability_ms: request.debug_metrics_commit_observability_ms,
            debug_metrics_post_decode_ms: request.debug_metrics_post_decode_ms,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
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
}
