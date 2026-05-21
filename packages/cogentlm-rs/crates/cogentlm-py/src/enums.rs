//! Bidirectional parsers + name lookups for the enums surfaced by the bindings.

use cogentlm_engine::engine::protocol::{EngineStatus, FinishReason, RequestStatus};
use cogentlm_engine::engine::{
    CacheKeyPolicy, ChatRole, FlashAttentionMode, GpuLayerConfig, KvCacheType, KvReuseMode,
    RopeScaling, SamplerStage, SplitMode,
};
use cogentlm_engine::lifecycle::{
    BackendPreference, ModelModality, ModelSourceKind, ModelStatus, StatsMode,
};
use cogentlm_engine::runtime::config::SchedulerPolicyMode;
use cogentlm_engine::runtime::request::GenerateResponseStatus;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(super) fn parse_backend_preference(value: &str) -> PyResult<BackendPreference> {
    match normalize_choice(value).as_str() {
        "auto" => Ok(BackendPreference::Auto),
        "cpu" => Ok(BackendPreference::Cpu),
        "cuda" => Ok(BackendPreference::Cuda),
        "metal" => Ok(BackendPreference::Metal),
        "vulkan" => Ok(BackendPreference::Vulkan),
        "webgpu" | "web_gpu" => Ok(BackendPreference::WebGpu),
        _ => Err(PyValueError::new_err(
            "backend must be one of: auto, cpu, cuda, metal, vulkan, webgpu",
        )),
    }
}

pub(super) fn parse_stats_mode(value: &str) -> PyResult<StatsMode> {
    match normalize_choice(value).as_str() {
        "off" => Ok(StatsMode::Off),
        "basic" => Ok(StatsMode::Basic),
        "profile" => Ok(StatsMode::Profile),
        _ => Err(PyValueError::new_err(
            "stats must be one of: off, basic, profile",
        )),
    }
}

pub(super) fn parse_gpu_layers(value: &str) -> PyResult<GpuLayerConfig> {
    let normalized = normalize_choice(value);
    match normalized.as_str() {
        "auto" => Ok(GpuLayerConfig::Auto),
        "all" | "full" => Ok(GpuLayerConfig::All),
        _ => normalized
            .parse::<i32>()
            .map(GpuLayerConfig::Count)
            .map_err(|_| {
                PyValueError::new_err("gpu_layers must be one of: auto, all, or an integer count")
            }),
    }
}

pub(super) fn parse_split_mode(value: &str) -> PyResult<SplitMode> {
    match normalize_choice(value).as_str() {
        "none" => Ok(SplitMode::None),
        "layer" => Ok(SplitMode::Layer),
        "row" => Ok(SplitMode::Row),
        "tensor" => Ok(SplitMode::Tensor),
        _ => Err(PyValueError::new_err(
            "split_mode must be one of: none, layer, row, tensor",
        )),
    }
}

pub(super) fn parse_flash_attention(value: &str) -> PyResult<FlashAttentionMode> {
    match normalize_choice(value).as_str() {
        "auto" => Ok(FlashAttentionMode::Auto),
        "enabled" | "enable" | "on" | "true" => Ok(FlashAttentionMode::Enabled),
        "disabled" | "disable" | "off" | "false" => Ok(FlashAttentionMode::Disabled),
        _ => Err(PyValueError::new_err(
            "flash_attention must be one of: auto, enabled, disabled",
        )),
    }
}

pub(super) fn parse_kv_cache_type(value: &str) -> PyResult<KvCacheType> {
    match normalize_choice(value).as_str() {
        "f16" => Ok(KvCacheType::F16),
        "f32" => Ok(KvCacheType::F32),
        "q8_0" => Ok(KvCacheType::Q8_0),
        "q4_0" => Ok(KvCacheType::Q4_0),
        "q4_1" => Ok(KvCacheType::Q4_1),
        "iq4_nl" => Ok(KvCacheType::Iq4Nl),
        "q5_0" => Ok(KvCacheType::Q5_0),
        "q5_1" => Ok(KvCacheType::Q5_1),
        _ => Err(PyValueError::new_err(
            "cache type must be one of: f16, f32, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1",
        )),
    }
}

pub(super) fn parse_rope_scaling(value: &str) -> PyResult<RopeScaling> {
    match normalize_choice(value).as_str() {
        "none" => Ok(RopeScaling::None),
        "linear" => Ok(RopeScaling::Linear),
        "yarn" => Ok(RopeScaling::Yarn),
        _ => Err(PyValueError::new_err(
            "rope_scaling must be one of: none, linear, yarn",
        )),
    }
}

pub(super) fn parse_kv_reuse_mode(value: &str) -> PyResult<KvReuseMode> {
    match normalize_choice(value).as_str() {
        "disabled" | "none" => Ok(KvReuseMode::Disabled),
        "live_slot_prefix" | "live_slot" => Ok(KvReuseMode::LiveSlotPrefix),
        "state_snapshot" | "snapshot" => Ok(KvReuseMode::StateSnapshot),
        "live_slot_and_snapshot" | "both" => Ok(KvReuseMode::LiveSlotAndSnapshot),
        _ => Err(PyValueError::new_err(
            "cache mode must be one of: disabled, live_slot_prefix, state_snapshot, live_slot_and_snapshot",
        )),
    }
}

pub(super) fn parse_cache_key_policy(value: &str) -> PyResult<CacheKeyPolicy> {
    match normalize_choice(value).as_str() {
        "context_key" => Ok(CacheKeyPolicy::ContextKey),
        "prompt_hash" => Ok(CacheKeyPolicy::PromptHash),
        _ => Err(PyValueError::new_err(
            "cache_key_policy must be one of: context_key, prompt_hash",
        )),
    }
}

pub(super) fn parse_sampler_stage(value: &str) -> PyResult<SamplerStage> {
    match normalize_choice(value).as_str() {
        "dry" => Ok(SamplerStage::Dry),
        "top_k" => Ok(SamplerStage::TopK),
        "typical_p" => Ok(SamplerStage::TypicalP),
        "top_p" => Ok(SamplerStage::TopP),
        "top_n_sigma" => Ok(SamplerStage::TopNSigma),
        "min_p" => Ok(SamplerStage::MinP),
        "xtc" => Ok(SamplerStage::Xtc),
        "temperature" | "temp" => Ok(SamplerStage::Temperature),
        "infill" => Ok(SamplerStage::Infill),
        "penalties" => Ok(SamplerStage::Penalties),
        "adaptive_p" => Ok(SamplerStage::AdaptiveP),
        _ => Err(PyValueError::new_err(
            "sampler stage must be one of: dry, top_k, typical_p, top_p, top_n_sigma, min_p, xtc, temperature, infill, penalties, adaptive_p",
        )),
    }
}

pub(super) fn parse_scheduler_policy(value: &str) -> PyResult<SchedulerPolicyMode> {
    match normalize_choice(value).as_str() {
        "latency_first" | "latency" => Ok(SchedulerPolicyMode::LatencyFirst),
        "balanced" | "balance" => Ok(SchedulerPolicyMode::Balanced),
        "throughput_first" | "throughput" => Ok(SchedulerPolicyMode::ThroughputFirst),
        _ => Err(PyValueError::new_err(
            "scheduler_policy must be one of: latency_first, balanced, throughput_first",
        )),
    }
}

pub(super) fn parse_chat_role(value: &str) -> PyResult<ChatRole> {
    match normalize_choice(value).as_str() {
        "system" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        _ => Err(PyValueError::new_err(
            "chat role must be one of: system, user, assistant",
        )),
    }
}

pub(super) fn normalize_choice(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

pub(super) fn response_status_name(status: GenerateResponseStatus) -> &'static str {
    match status {
        GenerateResponseStatus::Pending => "pending",
        GenerateResponseStatus::Completed => "completed",
        GenerateResponseStatus::Cancelled => "cancelled",
        GenerateResponseStatus::Failed => "failed",
    }
}

pub(super) fn engine_status_name(status: EngineStatus) -> &'static str {
    match status {
        EngineStatus::Idle => "idle",
        EngineStatus::Loading => "loading",
        EngineStatus::Ready => "ready",
        EngineStatus::Running => "running",
        EngineStatus::Error => "error",
        EngineStatus::Closed => "closed",
    }
}

pub(super) fn request_status_name(status: RequestStatus) -> &'static str {
    match status {
        RequestStatus::Queued => "queued",
        RequestStatus::Prefill => "prefill",
        RequestStatus::Decode => "decode",
        RequestStatus::Completed => "completed",
        RequestStatus::Failed => "failed",
        RequestStatus::Cancelled => "cancelled",
    }
}

pub(super) fn finish_reason_name(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Cancelled => "cancelled",
        FinishReason::Error => "error",
    }
}

pub(super) fn backend_preference_name(backend: BackendPreference) -> &'static str {
    match backend {
        BackendPreference::Auto => "auto",
        BackendPreference::Cpu => "cpu",
        BackendPreference::Cuda => "cuda",
        BackendPreference::Metal => "metal",
        BackendPreference::Vulkan => "vulkan",
        BackendPreference::WebGpu => "webgpu",
    }
}

pub(super) fn model_modality_name(modality: ModelModality) -> &'static str {
    match modality {
        ModelModality::Text => "text",
        ModelModality::Vision => "vision",
    }
}

pub(super) fn model_status_name(status: ModelStatus) -> &'static str {
    match status {
        ModelStatus::Ready => "ready",
        ModelStatus::NeedsProjector => "needs_projector",
        ModelStatus::Broken => "broken",
    }
}

pub(super) fn model_source_kind_name(source: ModelSourceKind) -> &'static str {
    match source {
        ModelSourceKind::Local => "local",
        ModelSourceKind::Remote => "remote",
    }
}
