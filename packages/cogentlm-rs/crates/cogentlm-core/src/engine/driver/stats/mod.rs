//! Translation between runtime observability metrics, backend info from the
//! shim, and the public engine/request stats surfaced over the driver API.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::engine::protocol::{
    BackendDevice, BackendInfo, EngineStats, FinishReason, RequestResult, RequestStats,
};
use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::request::{GenerateResponse, GenerateResponseStatus};

pub(super) fn engine_stats_from_runtime(metrics: RuntimeObservabilityMetrics) -> EngineStats {
    let tokens_per_second = if metrics.e2e_ms > 0.0 && metrics.output_tokens > 0 {
        Some(f64::from(metrics.output_tokens) / (metrics.e2e_ms / 1000.0))
    } else {
        None
    };
    let decode_tokens_per_second =
        decode_tokens_per_second(metrics.output_tokens, metrics.decode_ms);
    let prefill_tokens_per_second = if metrics.prefill_ms > 0.0 && metrics.prefill_tokens > 0 {
        Some(f64::from(metrics.prefill_tokens) / (metrics.prefill_ms / 1000.0))
    } else {
        None
    };

    EngineStats {
        input_tokens: i64::from(metrics.input_tokens),
        output_tokens: i64::from(metrics.output_tokens),
        cache_hits: i64::from(metrics.cache_hits),
        prefill_tokens: i64::from(metrics.prefill_tokens),
        ttft_ms: non_zero_metric(metrics.ttft_ms),
        inter_token_ms: non_zero_metric(metrics.itl_avg_ms),
        e2e_ms: non_zero_metric(metrics.e2e_ms),
        tokens_per_second,
        decode_tokens_per_second,
        prefill_tokens_per_second,
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

pub(super) fn request_result_from_response(response: &GenerateResponse) -> RequestResult {
    RequestResult {
        id: response.request_id.to_string(),
        text: response.output_text.clone(),
        finish_reason: match response.status {
            GenerateResponseStatus::Completed => FinishReason::Stop,
            GenerateResponseStatus::Cancelled => FinishReason::Cancelled,
            GenerateResponseStatus::Failed | GenerateResponseStatus::Pending => FinishReason::Error,
        },
        stats: request_stats_from_runtime(response.runtime_observability),
    }
}

pub(super) fn request_stats_from_runtime(metrics: RuntimeObservabilityMetrics) -> RequestStats {
    let tokens_per_second = if metrics.e2e_ms > 0.0 && metrics.output_tokens > 0 {
        Some(f64::from(metrics.output_tokens) / (metrics.e2e_ms / 1000.0))
    } else {
        None
    };
    let decode_tokens_per_second =
        decode_tokens_per_second(metrics.output_tokens, metrics.decode_ms);

    RequestStats {
        input_tokens: metrics.input_tokens,
        output_tokens: metrics.output_tokens,
        cache_hits: metrics.cache_hits,
        ttft_ms: non_zero_metric(metrics.ttft_ms),
        inter_token_ms: non_zero_metric(metrics.itl_avg_ms),
        e2e_ms: non_zero_metric(metrics.e2e_ms),
        tokens_per_second,
        decode_tokens_per_second,
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

pub(super) fn non_zero_metric(value: f64) -> Option<f64> {
    (value > 0.0).then_some(value)
}

pub(super) fn decode_tokens_per_second(output_tokens: i32, decode_ms: f64) -> Option<f64> {
    (output_tokens > 0 && decode_ms > 0.0).then(|| f64::from(output_tokens) / (decode_ms / 1000.0))
}

pub(super) fn read_backend_info() -> BackendInfo {
    let Ok(raw) = crate::backend::backend_observability_json(true) else {
        return BackendInfo {
            selected: "unknown".to_string(),
            ..BackendInfo::default()
        };
    };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return BackendInfo {
            selected: "unknown".to_string(),
            ..BackendInfo::default()
        };
    };

    let available = value
        .get("availableBackends")
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |items| parse_backend_names(items));
    let selected = available
        .first()
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let devices = value
        .get("devices")
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |items| parse_backend_devices(items));

    BackendInfo {
        selected,
        available,
        devices,
    }
}

fn parse_backend_names(items: &[Value]) -> Vec<String> {
    let mut names = Vec::with_capacity(items.len());
    names.extend(
        items
            .iter()
            .filter_map(|item| item.get("name").and_then(Value::as_str).map(str::to_string)),
    );
    names
}

fn parse_backend_devices(items: &[Value]) -> Vec<BackendDevice> {
    let mut devices = Vec::with_capacity(items.len());
    devices.extend(items.iter().map(parse_backend_device));
    devices
}

fn parse_backend_device(value: &Value) -> BackendDevice {
    BackendDevice {
        id: value
            .get("deviceId")
            .and_then(Value::as_str)
            .map(str::to_string),
        name: value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        device_type: value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        memory_total_bytes: value.get("memoryTotalBytes").and_then(Value::as_u64),
        memory_free_bytes: value.get("memoryFreeBytes").and_then(Value::as_u64),
    }
}

pub(super) fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(duration_millis_u64)
        .unwrap_or(0)
}

fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests;
