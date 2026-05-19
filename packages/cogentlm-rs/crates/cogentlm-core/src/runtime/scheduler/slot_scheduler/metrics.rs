use std::time::{Duration, Instant};

use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::request::GenerateRequest;

pub(super) fn metrics_from_request(
    request: &GenerateRequest,
    completed_at: Instant,
) -> RuntimeObservabilityMetrics {
    RuntimeObservabilityMetrics {
        ttft_ms: request
            .first_token_at
            .and_then(|first_token_at| {
                request
                    .enqueued_at
                    .map(|enqueued| duration_ms(enqueued, first_token_at))
            })
            .unwrap_or(0.0),
        itl_avg_ms: average_inter_token_ms(request.output_tokens, request.decode_ms),
        itl_p99_ms: request.itl_p99_ms,
        e2e_ms: request
            .enqueued_at
            .map(|enqueued| duration_ms(enqueued, completed_at))
            .unwrap_or(0.0),
        prefill_ms: request.prefill_ms,
        decode_ms: request.decode_ms,
        native_gpu_ms: request.native_gpu_ms,
        native_sync_ms: request.native_sync_ms,
        native_logic_ms: request.native_logic_ms,
        input_tokens: if request.input_tokens > 0 {
            request.input_tokens
        } else {
            saturating_usize_to_i32(request.prompt_tokens.len())
        },
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
        debug_metrics_backend_sampler_attach_ms: request.debug_metrics_backend_sampler_attach_ms,
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
    }
}

fn average_inter_token_ms(output_tokens: i32, decode_ms: f64) -> f64 {
    if output_tokens > 1 {
        decode_ms / f64::from(output_tokens - 1)
    } else {
        0.0
    }
}

pub(super) fn saturating_usize_to_i32(value: usize) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

pub(super) fn duration_ms(start: Instant, end: Instant) -> f64 {
    duration_as_ms(end.saturating_duration_since(start))
}

fn duration_as_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
