//! Marshal core types into Python `dict` / `PyAny` objects.

use cogentlm_core::engine::protocol::{
    BackendDevice, BackendInfo, ModelState, RequestState, RequestStats,
};
use cogentlm_core::runtime::metrics::RuntimeObservabilityMetrics;
use cogentlm_core::runtime::request::GenerateResponse;
use cogentlm_core::{
    BackendSelection, EngineEvent, EngineState, EngineStats, LoadedModelInfo, ModelInfo,
    ModelServiceState, RequestResult, ResolvedRuntimeLimits, TokenBatch,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::enums::{
    backend_preference_name, engine_status_name, finish_reason_name, model_modality_name,
    model_source_kind_name, model_status_name, request_status_name, response_status_name,
};

pub(super) fn response_to_dict(py: Python<'_>, response: GenerateResponse) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("request_id", response.request_id)?;
    dict.set_item("status", response_status_name(response.status))?;
    dict.set_item("output_text", response.output_text)?;
    dict.set_item("error_message", response.error_message)?;
    dict.set_item(
        "runtime_observability",
        metrics_to_dict(py, response.runtime_observability)?,
    )?;
    Ok(dict.into_py(py))
}

pub(super) fn request_result_to_dict(py: Python<'_>, result: RequestResult) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", result.id)?;
    dict.set_item("text", result.text)?;
    dict.set_item("finish_reason", finish_reason_name(result.finish_reason))?;
    dict.set_item("stats", request_stats_to_dict(py, &result.stats)?)?;
    Ok(dict.into_py(py))
}

pub(super) fn loaded_model_info_to_dict(py: Python<'_>, loaded: LoadedModelInfo) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("model", model_info_to_dict(py, loaded.model)?)?;
    dict.set_item("backend", backend_selection_to_dict(py, loaded.backend)?)?;
    dict.set_item("runtime_fingerprint", loaded.runtime_fingerprint)?;
    Ok(dict.into_py(py))
}

pub(super) fn model_info_to_dict(py: Python<'_>, model: ModelInfo) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", model.id)?;
    dict.set_item("name", model.name)?;
    dict.set_item("modality", model_modality_name(model.modality))?;
    dict.set_item("status", model_status_name(model.status))?;
    dict.set_item("source", model_source_kind_name(model.source))?;
    dict.set_item("bytes", model.bytes)?;
    dict.set_item("loaded", model.loaded)?;
    dict.set_item("chat_template", model.chat_template)?;
    dict.set_item("bos_text", model.bos_text)?;
    dict.set_item("eos_text", model.eos_text)?;
    dict.set_item("media_marker", model.media_marker)?;
    dict.set_item("created_at_unix_ms", model.created_at_unix_ms)?;
    dict.set_item("updated_at_unix_ms", model.updated_at_unix_ms)?;
    Ok(dict.into_py(py))
}

pub(super) fn backend_selection_to_dict(py: Python<'_>, backend: BackendSelection) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("requested", backend_preference_name(backend.requested))?;
    dict.set_item("selected", backend.selected)?;
    dict.set_item("available", backend.available)?;
    dict.set_item("gpu_offload_expected", backend.gpu_offload_expected)?;
    dict.set_item("reason", backend.reason)?;
    Ok(dict.into_py(py))
}

pub(super) fn model_service_state_to_dict(py: Python<'_>, state: ModelServiceState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("status", engine_status_name(state.status))?;
    if let Some(model) = state.model {
        dict.set_item("model", model_info_to_dict(py, model)?)?;
    } else {
        dict.set_item("model", py.None())?;
    }
    dict.set_item("backend", backend_info_to_dict(py, state.backend)?)?;
    dict.set_item(
        "runtime",
        resolved_runtime_limits_to_dict(py, state.runtime)?,
    )?;
    let requests = PyList::empty_bound(py);
    for request in state.requests {
        requests.append(request_state_to_dict(py, request)?)?;
    }
    dict.set_item("requests", requests)?;
    dict.set_item("stats", engine_stats_to_dict(py, &state.stats)?)?;
    dict.set_item("updated_at_unix_ms", state.updated_at_unix_ms)?;
    Ok(dict.into_py(py))
}

pub(super) fn token_batch_to_dict(py: Python<'_>, batch: TokenBatch) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("request_id", batch.request_id)?;
    dict.set_item("stream_id", batch.stream_id)?;
    dict.set_item("sequence_start", batch.sequence_start)?;
    dict.set_item("text", batch.text)?;
    dict.set_item("frame_count", batch.frame_count)?;
    dict.set_item("byte_count", batch.byte_count)?;
    let stats = PyDict::new_bound(py);
    stats.set_item("frames_sent", batch.stats.frames_sent)?;
    stats.set_item("bytes_sent", batch.stats.bytes_sent)?;
    stats.set_item("frames_dropped", batch.stats.frames_dropped)?;
    stats.set_item("batches_sent", batch.stats.batches_sent)?;
    dict.set_item("stats", stats)?;
    Ok(dict.into_py(py))
}

pub(super) fn engine_state_to_dict(py: Python<'_>, state: EngineState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("status", engine_status_name(state.status))?;
    if let Some(model) = state.model {
        dict.set_item("model", model_state_to_dict(py, model)?)?;
    } else {
        dict.set_item("model", py.None())?;
    }
    dict.set_item("backend", backend_info_to_dict(py, state.backend)?)?;
    dict.set_item(
        "runtime",
        resolved_runtime_limits_to_dict(py, state.runtime)?,
    )?;
    let requests = PyList::empty_bound(py);
    for request in state.requests {
        requests.append(request_state_to_dict(py, request)?)?;
    }
    dict.set_item("requests", requests)?;
    dict.set_item("stats", engine_stats_to_dict(py, &state.stats)?)?;
    dict.set_item("updated_at_unix_ms", state.updated_at_unix_ms)?;
    Ok(dict.into_py(py))
}

pub(super) fn model_state_to_dict(py: Python<'_>, model: ModelState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", model.id)?;
    dict.set_item("name", model.name)?;
    Ok(dict.into_py(py))
}

pub(super) fn request_state_to_dict(py: Python<'_>, request: RequestState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", request.id)?;
    dict.set_item("status", request_status_name(request.status))?;
    dict.set_item("input_tokens", request.input_tokens)?;
    dict.set_item("output_tokens", request.output_tokens)?;
    Ok(dict.into_py(py))
}

pub(super) fn backend_info_to_dict(py: Python<'_>, backend: BackendInfo) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("selected", backend.selected)?;
    dict.set_item("available", backend.available)?;
    let devices = PyList::empty_bound(py);
    for device in backend.devices {
        devices.append(backend_device_to_dict(py, device)?)?;
    }
    dict.set_item("devices", devices)?;
    Ok(dict.into_py(py))
}

pub(super) fn backend_device_to_dict(py: Python<'_>, device: BackendDevice) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", device.id)?;
    dict.set_item("name", device.name)?;
    dict.set_item("type", device.device_type)?;
    dict.set_item("memory_total_bytes", device.memory_total_bytes)?;
    dict.set_item("memory_free_bytes", device.memory_free_bytes)?;
    Ok(dict.into_py(py))
}

pub(super) fn resolved_runtime_limits_to_dict(
    py: Python<'_>,
    runtime: Option<ResolvedRuntimeLimits>,
) -> PyResult<Py<PyAny>> {
    let Some(runtime) = runtime else {
        return Ok(py.None());
    };
    let dict = PyDict::new_bound(py);
    dict.set_item("n_ctx", runtime.n_ctx)?;
    dict.set_item("n_batch", runtime.n_batch)?;
    dict.set_item("n_ubatch", runtime.n_ubatch)?;
    dict.set_item("n_parallel", runtime.n_parallel)?;
    dict.set_item("kv_unified", runtime.kv_unified)?;
    dict.set_item("flash_attention", runtime.flash_attention)?;
    dict.set_item("cache_type_k", runtime.cache_type_k)?;
    dict.set_item("cache_type_v", runtime.cache_type_v)?;
    Ok(dict.into_py(py))
}

pub(super) fn engine_stats_to_dict(py: Python<'_>, stats: &EngineStats) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("requests_running", stats.requests_running)?;
    dict.set_item("requests_queued", stats.requests_queued)?;
    dict.set_item("requests_completed", stats.requests_completed)?;
    dict.set_item("requests_failed", stats.requests_failed)?;
    dict.set_item("input_tokens", stats.input_tokens)?;
    dict.set_item("output_tokens", stats.output_tokens)?;
    dict.set_item("cache_hits", stats.cache_hits)?;
    dict.set_item("prefill_tokens", stats.prefill_tokens)?;
    dict.set_item("ttft_ms", stats.ttft_ms)?;
    dict.set_item("inter_token_ms", stats.inter_token_ms)?;
    dict.set_item("e2e_ms", stats.e2e_ms)?;
    dict.set_item("tokens_per_second", stats.tokens_per_second)?;
    dict.set_item("decode_tokens_per_second", stats.decode_tokens_per_second)?;
    dict.set_item("prefill_tokens_per_second", stats.prefill_tokens_per_second)?;
    dict.set_item("prefill_ms", stats.prefill_ms)?;
    dict.set_item("decode_ms", stats.decode_ms)?;
    dict.set_item("backend_ms", stats.backend_ms)?;
    dict.set_item("sync_ms", stats.sync_ms)?;
    dict.set_item("engine_overhead_ms", stats.engine_overhead_ms)?;
    dict.set_item(
        "debug_metrics_scheduler_ticks",
        stats.debug_metrics_scheduler_ticks,
    )?;
    dict.set_item(
        "debug_metrics_decode_ticks",
        stats.debug_metrics_decode_ticks,
    )?;
    dict.set_item(
        "debug_metrics_prefill_ticks",
        stats.debug_metrics_prefill_ticks,
    )?;
    dict.set_item(
        "debug_metrics_backend_sampler_attach_attempts",
        stats.debug_metrics_backend_sampler_attach_attempts,
    )?;
    dict.set_item(
        "debug_metrics_backend_sampler_attach_failures",
        stats.debug_metrics_backend_sampler_attach_failures,
    )?;
    dict.set_item("debug_metrics_admit_ms", stats.debug_metrics_admit_ms)?;
    dict.set_item(
        "debug_metrics_normalize_ms",
        stats.debug_metrics_normalize_ms,
    )?;
    dict.set_item(
        "debug_metrics_backend_sampler_attach_ms",
        stats.debug_metrics_backend_sampler_attach_ms,
    )?;
    dict.set_item(
        "debug_metrics_select_slots_ms",
        stats.debug_metrics_select_slots_ms,
    )?;
    dict.set_item("debug_metrics_plan_ms", stats.debug_metrics_plan_ms)?;
    dict.set_item(
        "debug_metrics_batch_build_ms",
        stats.debug_metrics_batch_build_ms,
    )?;
    dict.set_item(
        "debug_metrics_llama_decode_ms",
        stats.debug_metrics_llama_decode_ms,
    )?;
    dict.set_item(
        "debug_metrics_llama_sync_ms",
        stats.debug_metrics_llama_sync_ms,
    )?;
    dict.set_item(
        "debug_metrics_apply_bookkeeping_ms",
        stats.debug_metrics_apply_bookkeeping_ms,
    )?;
    dict.set_item(
        "debug_metrics_apply_decode_results_ms",
        stats.debug_metrics_apply_decode_results_ms,
    )?;
    dict.set_item("debug_metrics_sample_ms", stats.debug_metrics_sample_ms)?;
    dict.set_item(
        "debug_metrics_token_piece_ms",
        stats.debug_metrics_token_piece_ms,
    )?;
    dict.set_item("debug_metrics_emit_ms", stats.debug_metrics_emit_ms)?;
    dict.set_item(
        "debug_metrics_prefix_queue_ms",
        stats.debug_metrics_prefix_queue_ms,
    )?;
    dict.set_item("debug_metrics_finalize_ms", stats.debug_metrics_finalize_ms)?;
    dict.set_item(
        "debug_metrics_commit_observability_ms",
        stats.debug_metrics_commit_observability_ms,
    )?;
    dict.set_item(
        "debug_metrics_post_decode_ms",
        stats.debug_metrics_post_decode_ms,
    )?;
    Ok(dict.into_py(py))
}

pub(super) fn request_stats_to_dict(py: Python<'_>, stats: &RequestStats) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("input_tokens", stats.input_tokens)?;
    dict.set_item("output_tokens", stats.output_tokens)?;
    dict.set_item("cache_hits", stats.cache_hits)?;
    dict.set_item("ttft_ms", stats.ttft_ms)?;
    dict.set_item("inter_token_ms", stats.inter_token_ms)?;
    dict.set_item("e2e_ms", stats.e2e_ms)?;
    dict.set_item("tokens_per_second", stats.tokens_per_second)?;
    dict.set_item("decode_tokens_per_second", stats.decode_tokens_per_second)?;
    dict.set_item("prefill_ms", stats.prefill_ms)?;
    dict.set_item("decode_ms", stats.decode_ms)?;
    dict.set_item(
        "debug_metrics_scheduler_ticks",
        stats.debug_metrics_scheduler_ticks,
    )?;
    dict.set_item(
        "debug_metrics_decode_ticks",
        stats.debug_metrics_decode_ticks,
    )?;
    dict.set_item(
        "debug_metrics_prefill_ticks",
        stats.debug_metrics_prefill_ticks,
    )?;
    dict.set_item(
        "debug_metrics_backend_sampler_attach_attempts",
        stats.debug_metrics_backend_sampler_attach_attempts,
    )?;
    dict.set_item(
        "debug_metrics_backend_sampler_attach_failures",
        stats.debug_metrics_backend_sampler_attach_failures,
    )?;
    dict.set_item("debug_metrics_admit_ms", stats.debug_metrics_admit_ms)?;
    dict.set_item(
        "debug_metrics_normalize_ms",
        stats.debug_metrics_normalize_ms,
    )?;
    dict.set_item(
        "debug_metrics_backend_sampler_attach_ms",
        stats.debug_metrics_backend_sampler_attach_ms,
    )?;
    dict.set_item(
        "debug_metrics_select_slots_ms",
        stats.debug_metrics_select_slots_ms,
    )?;
    dict.set_item("debug_metrics_plan_ms", stats.debug_metrics_plan_ms)?;
    dict.set_item(
        "debug_metrics_batch_build_ms",
        stats.debug_metrics_batch_build_ms,
    )?;
    dict.set_item(
        "debug_metrics_llama_decode_ms",
        stats.debug_metrics_llama_decode_ms,
    )?;
    dict.set_item(
        "debug_metrics_llama_sync_ms",
        stats.debug_metrics_llama_sync_ms,
    )?;
    dict.set_item(
        "debug_metrics_apply_bookkeeping_ms",
        stats.debug_metrics_apply_bookkeeping_ms,
    )?;
    dict.set_item(
        "debug_metrics_apply_decode_results_ms",
        stats.debug_metrics_apply_decode_results_ms,
    )?;
    dict.set_item("debug_metrics_sample_ms", stats.debug_metrics_sample_ms)?;
    dict.set_item(
        "debug_metrics_token_piece_ms",
        stats.debug_metrics_token_piece_ms,
    )?;
    dict.set_item("debug_metrics_emit_ms", stats.debug_metrics_emit_ms)?;
    dict.set_item(
        "debug_metrics_prefix_queue_ms",
        stats.debug_metrics_prefix_queue_ms,
    )?;
    dict.set_item("debug_metrics_finalize_ms", stats.debug_metrics_finalize_ms)?;
    dict.set_item(
        "debug_metrics_commit_observability_ms",
        stats.debug_metrics_commit_observability_ms,
    )?;
    dict.set_item(
        "debug_metrics_post_decode_ms",
        stats.debug_metrics_post_decode_ms,
    )?;
    Ok(dict.into_py(py))
}

pub(super) fn engine_event_to_dict(py: Python<'_>, event: EngineEvent) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    match event {
        EngineEvent::State(state) => {
            dict.set_item("type", "state")?;
            dict.set_item("state", engine_state_to_dict(py, (*state).clone())?)?;
        }
        EngineEvent::LoadProgress {
            loaded_bytes,
            total_bytes,
            asset_name,
        } => {
            dict.set_item("type", "load-progress")?;
            dict.set_item("loaded_bytes", loaded_bytes)?;
            dict.set_item("total_bytes", total_bytes)?;
            dict.set_item("asset_name", asset_name)?;
        }
        EngineEvent::RequestStarted {
            request_id,
            stream_id,
        } => {
            dict.set_item("type", "request-started")?;
            dict.set_item("request_id", request_id)?;
            dict.set_item("stream_id", stream_id)?;
        }
        EngineEvent::RequestCompleted { result } => {
            dict.set_item("type", "request-completed")?;
            dict.set_item("result", request_result_to_dict(py, (*result).clone())?)?;
        }
        EngineEvent::RequestFailed { request_id, error } => {
            dict.set_item("type", "request-failed")?;
            dict.set_item("request_id", request_id)?;
            dict.set_item("error", error)?;
        }
        EngineEvent::Closed => {
            dict.set_item("type", "closed")?;
        }
    }
    Ok(dict.into_py(py))
}

pub(super) fn metrics_to_dict(
    py: Python<'_>,
    metrics: RuntimeObservabilityMetrics,
) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("ttft_ms", metrics.ttft_ms)?;
    dict.set_item("itl_avg_ms", metrics.itl_avg_ms)?;
    dict.set_item("itl_p99_ms", metrics.itl_p99_ms)?;
    dict.set_item("e2e_ms", metrics.e2e_ms)?;
    dict.set_item("prefill_ms", metrics.prefill_ms)?;
    dict.set_item("decode_ms", metrics.decode_ms)?;
    dict.set_item("native_gpu_ms", metrics.native_gpu_ms)?;
    dict.set_item("native_sync_ms", metrics.native_sync_ms)?;
    dict.set_item("native_logic_ms", metrics.native_logic_ms)?;
    dict.set_item("input_tokens", metrics.input_tokens)?;
    dict.set_item("output_tokens", metrics.output_tokens)?;
    dict.set_item("cache_hits", metrics.cache_hits)?;
    dict.set_item("prefill_tokens", metrics.prefill_tokens)?;
    Ok(dict)
}
