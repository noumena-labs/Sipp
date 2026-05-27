//! Translation between runtime observability metrics, backend info from the
//! shim, and the public engine/request stats surfaced over the driver API.

use serde_json::Value;

use crate::backend::{
    json_array, json_array_strings, json_str, json_string_or, json_u64, KEY_AVAILABLE_BACKENDS,
    KEY_DEVICES, KEY_DEVICE_ID, KEY_MEMORY_FREE_BYTES, KEY_MEMORY_TOTAL_BYTES, KEY_NAME, KEY_TYPE,
};
use crate::engine::protocol::{
    BackendDevice, BackendInfo, EmbeddingResult, EngineStats, FinishReason, GenerationResult,
    RequestStats,
};
use crate::error::{Error, Result};
use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::numeric::MILLIS_PER_SECOND_F64;
use crate::runtime::request::{GenerateResponse, GenerateResponseStatus, ResponseOutput};

const UNKNOWN_BACKEND: &str = "unknown";

pub(super) fn engine_stats_from_runtime(metrics: RuntimeObservabilityMetrics) -> EngineStats {
    let rates = RuntimeMetricRates::from_metrics(metrics);
    let timings = RuntimeMetricTimings::from_metrics(metrics);

    EngineStats {
        input_tokens: i64::from(metrics.input_tokens),
        output_tokens: i64::from(metrics.output_tokens),
        cache_hits: i64::from(metrics.cache_hits),
        prefill_tokens: i64::from(metrics.prefill_tokens),
        ttft_ms: timings.ttft_ms,
        inter_token_ms: timings.inter_token_ms,
        e2e_ms: timings.e2e_ms,
        tokens_per_second: rates.output_tokens_per_second,
        decode_tokens_per_second: rates.decode_tokens_per_second,
        prefill_tokens_per_second: rates.prefill_tokens_per_second,
        prefill_ms: metrics.prefill_ms,
        decode_ms: metrics.decode_ms,
        backend_ms: metrics.native_gpu_ms,
        sync_ms: metrics.native_sync_ms,
        engine_overhead_ms: metrics.native_logic_ms,
        debug_metrics_scheduler_ticks: i64::from(metrics.debug_metrics_scheduler_ticks),
        debug_metrics_decode_ticks: i64::from(metrics.debug_metrics_decode_ticks),
        debug_metrics_prefill_ticks: i64::from(metrics.debug_metrics_prefill_ticks),
        debug_metrics_backend_sampler_attach_attempts: i64::from(
            metrics.debug_metrics_backend_sampler_attach_attempts,
        ),
        debug_metrics_backend_sampler_attach_failures: i64::from(
            metrics.debug_metrics_backend_sampler_attach_failures,
        ),
        debug_metrics_admit_ms: metrics.debug_metrics_admit_ms,
        debug_metrics_normalize_ms: metrics.debug_metrics_normalize_ms,
        debug_metrics_backend_sampler_attach_ms: metrics.debug_metrics_backend_sampler_attach_ms,
        debug_metrics_select_slots_ms: metrics.debug_metrics_select_slots_ms,
        debug_metrics_plan_ms: metrics.debug_metrics_plan_ms,
        debug_metrics_batch_build_ms: metrics.debug_metrics_batch_build_ms,
        debug_metrics_llama_decode_ms: metrics.debug_metrics_llama_decode_ms,
        debug_metrics_llama_sync_ms: metrics.debug_metrics_llama_sync_ms,
        debug_metrics_apply_bookkeeping_ms: metrics.debug_metrics_apply_bookkeeping_ms,
        debug_metrics_apply_decode_results_ms: metrics.debug_metrics_apply_decode_results_ms,
        debug_metrics_sample_ms: metrics.debug_metrics_sample_ms,
        debug_metrics_token_piece_ms: metrics.debug_metrics_token_piece_ms,
        debug_metrics_emit_ms: metrics.debug_metrics_emit_ms,
        debug_metrics_prefix_queue_ms: metrics.debug_metrics_prefix_queue_ms,
        debug_metrics_finalize_ms: metrics.debug_metrics_finalize_ms,
        debug_metrics_commit_observability_ms: metrics.debug_metrics_commit_observability_ms,
        debug_metrics_post_decode_ms: metrics.debug_metrics_post_decode_ms,
        ..EngineStats::default()
    }
}

pub(super) fn generation_result_from_response(
    response: GenerateResponse,
) -> Result<GenerationResult> {
    let text = match response.output {
        ResponseOutput::Text(text) => text,
        ResponseOutput::Embedding { .. } => {
            return Err(Error::RuntimeCommand(
                "generation request completed with embedding output".to_string(),
            ));
        }
    };
    Ok(GenerationResult {
        id: response.request_id.to_string(),
        text,
        finish_reason: match response.status {
            GenerateResponseStatus::Completed => FinishReason::Stop,
            GenerateResponseStatus::Cancelled => FinishReason::Cancelled,
            GenerateResponseStatus::Failed | GenerateResponseStatus::Pending => FinishReason::Error,
        },
        stats: request_stats_from_runtime(response.runtime_observability),
    })
}

pub(super) fn embedding_result_from_response(
    response: GenerateResponse,
) -> Result<EmbeddingResult> {
    match response.output {
        ResponseOutput::Embedding {
            values,
            pooling,
            normalized,
        } => Ok(EmbeddingResult {
            id: response.request_id.to_string(),
            values,
            pooling,
            normalized,
            stats: request_stats_from_runtime(response.runtime_observability),
        }),
        ResponseOutput::Text(_) => Err(Error::RuntimeCommand(
            "embedding request completed with text output".to_string(),
        )),
    }
}

pub(super) fn request_stats_from_runtime(metrics: RuntimeObservabilityMetrics) -> RequestStats {
    let rates = RuntimeMetricRates::from_metrics(metrics);
    let timings = RuntimeMetricTimings::from_metrics(metrics);

    RequestStats {
        input_tokens: metrics.input_tokens,
        output_tokens: metrics.output_tokens,
        cache_hits: metrics.cache_hits,
        ttft_ms: timings.ttft_ms,
        inter_token_ms: timings.inter_token_ms,
        e2e_ms: timings.e2e_ms,
        tokens_per_second: rates.output_tokens_per_second,
        decode_tokens_per_second: rates.decode_tokens_per_second,
        prefill_ms: metrics.prefill_ms,
        decode_ms: metrics.decode_ms,
        debug_metrics_scheduler_ticks: metrics.debug_metrics_scheduler_ticks,
        debug_metrics_decode_ticks: metrics.debug_metrics_decode_ticks,
        debug_metrics_prefill_ticks: metrics.debug_metrics_prefill_ticks,
        debug_metrics_backend_sampler_attach_attempts: metrics
            .debug_metrics_backend_sampler_attach_attempts,
        debug_metrics_backend_sampler_attach_failures: metrics
            .debug_metrics_backend_sampler_attach_failures,
        debug_metrics_admit_ms: metrics.debug_metrics_admit_ms,
        debug_metrics_normalize_ms: metrics.debug_metrics_normalize_ms,
        debug_metrics_backend_sampler_attach_ms: metrics.debug_metrics_backend_sampler_attach_ms,
        debug_metrics_select_slots_ms: metrics.debug_metrics_select_slots_ms,
        debug_metrics_plan_ms: metrics.debug_metrics_plan_ms,
        debug_metrics_batch_build_ms: metrics.debug_metrics_batch_build_ms,
        debug_metrics_llama_decode_ms: metrics.debug_metrics_llama_decode_ms,
        debug_metrics_llama_sync_ms: metrics.debug_metrics_llama_sync_ms,
        debug_metrics_apply_bookkeeping_ms: metrics.debug_metrics_apply_bookkeeping_ms,
        debug_metrics_apply_decode_results_ms: metrics.debug_metrics_apply_decode_results_ms,
        debug_metrics_sample_ms: metrics.debug_metrics_sample_ms,
        debug_metrics_token_piece_ms: metrics.debug_metrics_token_piece_ms,
        debug_metrics_emit_ms: metrics.debug_metrics_emit_ms,
        debug_metrics_prefix_queue_ms: metrics.debug_metrics_prefix_queue_ms,
        debug_metrics_finalize_ms: metrics.debug_metrics_finalize_ms,
        debug_metrics_commit_observability_ms: metrics.debug_metrics_commit_observability_ms,
        debug_metrics_post_decode_ms: metrics.debug_metrics_post_decode_ms,
    }
}

struct RuntimeMetricTimings {
    ttft_ms: Option<f64>,
    inter_token_ms: Option<f64>,
    e2e_ms: Option<f64>,
}

impl RuntimeMetricTimings {
    fn from_metrics(metrics: RuntimeObservabilityMetrics) -> Self {
        Self {
            ttft_ms: non_zero_metric(metrics.ttft_ms),
            inter_token_ms: non_zero_metric(metrics.itl_avg_ms),
            e2e_ms: non_zero_metric(metrics.e2e_ms),
        }
    }
}

struct RuntimeMetricRates {
    output_tokens_per_second: Option<f64>,
    decode_tokens_per_second: Option<f64>,
    prefill_tokens_per_second: Option<f64>,
}

impl RuntimeMetricRates {
    fn from_metrics(metrics: RuntimeObservabilityMetrics) -> Self {
        Self {
            output_tokens_per_second: tokens_per_second(metrics.output_tokens, metrics.e2e_ms),
            decode_tokens_per_second: tokens_per_second(metrics.output_tokens, metrics.decode_ms),
            prefill_tokens_per_second: tokens_per_second(
                metrics.prefill_tokens,
                metrics.prefill_ms,
            ),
        }
    }
}

pub(super) fn non_zero_metric(value: f64) -> Option<f64> {
    (value > 0.0).then_some(value)
}

fn tokens_per_second(output_tokens: i32, elapsed_ms: f64) -> Option<f64> {
    (output_tokens > 0 && elapsed_ms > 0.0)
        .then(|| f64::from(output_tokens) / (elapsed_ms / MILLIS_PER_SECOND_F64))
}

pub(super) fn read_backend_info() -> BackendInfo {
    let Ok(raw) = crate::backend::backend_observability_json(true) else {
        return unknown_backend_info();
    };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return unknown_backend_info();
    };

    let available = json_array_strings(&value, KEY_AVAILABLE_BACKENDS, KEY_NAME);
    let selected = available
        .first()
        .cloned()
        .unwrap_or_else(|| UNKNOWN_BACKEND.to_string());
    let devices = json_array(&value, KEY_DEVICES).map_or_else(Vec::new, parse_backend_devices);

    BackendInfo {
        selected,
        available,
        devices,
    }
}

fn unknown_backend_info() -> BackendInfo {
    BackendInfo {
        selected: UNKNOWN_BACKEND.to_string(),
        ..BackendInfo::default()
    }
}

fn parse_backend_devices(items: &[Value]) -> Vec<BackendDevice> {
    items.iter().map(parse_backend_device).collect()
}

fn parse_backend_device(value: &Value) -> BackendDevice {
    BackendDevice {
        id: json_str(value, KEY_DEVICE_ID).map(str::to_string),
        name: json_string_or(value, KEY_NAME, ""),
        device_type: json_string_or(value, KEY_TYPE, UNKNOWN_BACKEND),
        memory_total_bytes: json_u64(value, KEY_MEMORY_TOTAL_BYTES),
        memory_free_bytes: json_u64(value, KEY_MEMORY_FREE_BYTES),
    }
}

#[cfg(test)]
mod tests {
    mod stats_tests;
}
