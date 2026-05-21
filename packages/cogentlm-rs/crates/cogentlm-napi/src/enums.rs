//! Bidirectional parsers + name lookups for the core enums surfaced over N-API.

use cogentlm_engine::engine::protocol::{
    EngineStatus as CoreEngineStatus, FinishReason as CoreFinishReason,
    RequestStatus as CoreRequestStatus,
};
use cogentlm_engine::runtime::config::SchedulerPolicyMode;
use cogentlm_engine::runtime::request::GenerateResponseStatus;
use cogentlm_engine::engine::{
    CacheKeyPolicy, ChatRole as CoreChatRole, FlashAttentionMode, GpuLayerConfig, KvCacheType,
    KvReuseMode, RopeScaling, SamplerStage, SplitMode,
};
use cogentlm_engine::lifecycle::{
    BackendPreference as CoreBackendPreference, ModelModality as CoreModelModality,
    ModelSourceKind as CoreModelSourceKind, ModelStatus as CoreModelStatus, StatsMode,
};
use napi::Result;

use super::convert::invalid_arg;

pub(super) fn parse_backend_preference(value: &str) -> Result<CoreBackendPreference> {
    match normalize_choice(value).as_str() {
        "auto" => Ok(CoreBackendPreference::Auto),
        "cpu" => Ok(CoreBackendPreference::Cpu),
        "cuda" => Ok(CoreBackendPreference::Cuda),
        "metal" => Ok(CoreBackendPreference::Metal),
        "vulkan" => Ok(CoreBackendPreference::Vulkan),
        "webgpu" | "web_gpu" => Ok(CoreBackendPreference::WebGpu),
        _ => Err(invalid_arg(
            "backend must be one of: auto, cpu, cuda, metal, vulkan, webgpu",
        )),
    }
}

pub(super) fn parse_stats_mode(value: &str) -> Result<StatsMode> {
    match normalize_choice(value).as_str() {
        "off" => Ok(StatsMode::Off),
        "basic" => Ok(StatsMode::Basic),
        "profile" => Ok(StatsMode::Profile),
        _ => Err(invalid_arg("stats must be one of: off, basic, profile")),
    }
}

pub(super) fn parse_gpu_layers(value: &str) -> Result<GpuLayerConfig> {
    match normalize_choice(value).as_str() {
        "auto" => Ok(GpuLayerConfig::Auto),
        "all" => Ok(GpuLayerConfig::All),
        _ => Err(invalid_arg(
            r#"gpu_layers must be "auto", "all", or { count: number }"#,
        )),
    }
}

pub(super) fn parse_split_mode(value: &str) -> Result<SplitMode> {
    match normalize_choice(value).as_str() {
        "none" => Ok(SplitMode::None),
        "layer" => Ok(SplitMode::Layer),
        "row" => Ok(SplitMode::Row),
        "tensor" => Ok(SplitMode::Tensor),
        _ => Err(invalid_arg(
            "split_mode must be one of: none, layer, row, tensor",
        )),
    }
}

pub(super) fn parse_flash_attention(value: &str) -> Result<FlashAttentionMode> {
    match normalize_choice(value).as_str() {
        "auto" => Ok(FlashAttentionMode::Auto),
        "enabled" | "enable" | "on" | "true" => Ok(FlashAttentionMode::Enabled),
        "disabled" | "disable" | "off" | "false" => Ok(FlashAttentionMode::Disabled),
        _ => Err(invalid_arg(
            "flash_attention must be one of: auto, enabled, disabled",
        )),
    }
}

pub(super) fn parse_kv_cache_type(value: &str) -> Result<KvCacheType> {
    match normalize_choice(value).as_str() {
        "f16" => Ok(KvCacheType::F16),
        "f32" => Ok(KvCacheType::F32),
        "q8_0" => Ok(KvCacheType::Q8_0),
        "q4_0" => Ok(KvCacheType::Q4_0),
        "q4_1" => Ok(KvCacheType::Q4_1),
        "iq4_nl" => Ok(KvCacheType::Iq4Nl),
        "q5_0" => Ok(KvCacheType::Q5_0),
        "q5_1" => Ok(KvCacheType::Q5_1),
        _ => Err(invalid_arg(
            "cache type must be one of: f16, f32, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1",
        )),
    }
}

pub(super) fn parse_rope_scaling(value: &str) -> Result<RopeScaling> {
    match normalize_choice(value).as_str() {
        "none" => Ok(RopeScaling::None),
        "linear" => Ok(RopeScaling::Linear),
        "yarn" => Ok(RopeScaling::Yarn),
        _ => Err(invalid_arg(
            "ropeScaling must be one of: none, linear, yarn",
        )),
    }
}

pub(super) fn parse_kv_reuse_mode(value: &str) -> Result<KvReuseMode> {
    match normalize_choice(value).as_str() {
        "disabled" | "none" => Ok(KvReuseMode::Disabled),
        "live_slot_prefix" | "live_slot" => Ok(KvReuseMode::LiveSlotPrefix),
        "state_snapshot" | "snapshot" => Ok(KvReuseMode::StateSnapshot),
        "live_slot_and_snapshot" | "both" => Ok(KvReuseMode::LiveSlotAndSnapshot),
        _ => Err(invalid_arg(
            "cache mode must be one of: disabled, live_slot_prefix, state_snapshot, live_slot_and_snapshot",
        )),
    }
}

pub(super) fn parse_cache_key_policy(value: &str) -> Result<CacheKeyPolicy> {
    match normalize_choice(value).as_str() {
        "context_key" => Ok(CacheKeyPolicy::ContextKey),
        "prompt_hash" => Ok(CacheKeyPolicy::PromptHash),
        _ => Err(invalid_arg(
            "cache_key_policy must be one of: context_key, prompt_hash",
        )),
    }
}

pub(super) fn parse_sampler_stage(value: &str) -> Result<SamplerStage> {
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
        _ => Err(invalid_arg(
            "sampler stage must be one of: dry, top_k, typical_p, top_p, top_n_sigma, min_p, xtc, temperature, infill, penalties, adaptive_p",
        )),
    }
}

pub(super) fn parse_scheduler_policy(value: &str) -> Result<SchedulerPolicyMode> {
    match normalize_choice(value).as_str() {
        "latency_first" | "latency" => Ok(SchedulerPolicyMode::LatencyFirst),
        "balanced" | "balance" => Ok(SchedulerPolicyMode::Balanced),
        "throughput_first" | "throughput" => Ok(SchedulerPolicyMode::ThroughputFirst),
        _ => Err(invalid_arg(
            "scheduler.policy.mode must be one of: latency_first, balanced, throughput_first",
        )),
    }
}

pub(super) fn parse_chat_role(value: &str) -> Result<CoreChatRole> {
    match normalize_choice(value).as_str() {
        "system" => Ok(CoreChatRole::System),
        "user" => Ok(CoreChatRole::User),
        "assistant" => Ok(CoreChatRole::Assistant),
        _ => Err(invalid_arg(
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

pub(super) fn engine_status_name(status: CoreEngineStatus) -> &'static str {
    match status {
        CoreEngineStatus::Idle => "idle",
        CoreEngineStatus::Loading => "loading",
        CoreEngineStatus::Ready => "ready",
        CoreEngineStatus::Running => "running",
        CoreEngineStatus::Error => "error",
        CoreEngineStatus::Closed => "closed",
    }
}

pub(super) fn request_status_name(status: CoreRequestStatus) -> &'static str {
    match status {
        CoreRequestStatus::Queued => "queued",
        CoreRequestStatus::Prefill => "prefill",
        CoreRequestStatus::Decode => "decode",
        CoreRequestStatus::Completed => "completed",
        CoreRequestStatus::Failed => "failed",
        CoreRequestStatus::Cancelled => "cancelled",
    }
}

pub(super) fn finish_reason_name(reason: CoreFinishReason) -> &'static str {
    match reason {
        CoreFinishReason::Stop => "stop",
        CoreFinishReason::Length => "length",
        CoreFinishReason::Cancelled => "cancelled",
        CoreFinishReason::Error => "error",
    }
}

pub(super) fn backend_preference_name(backend: CoreBackendPreference) -> &'static str {
    match backend {
        CoreBackendPreference::Auto => "auto",
        CoreBackendPreference::Cpu => "cpu",
        CoreBackendPreference::Cuda => "cuda",
        CoreBackendPreference::Metal => "metal",
        CoreBackendPreference::Vulkan => "vulkan",
        CoreBackendPreference::WebGpu => "webgpu",
    }
}

pub(super) fn model_modality_name(modality: CoreModelModality) -> &'static str {
    match modality {
        CoreModelModality::Text => "text",
        CoreModelModality::Vision => "vision",
    }
}

pub(super) fn model_status_name(status: CoreModelStatus) -> &'static str {
    match status {
        CoreModelStatus::Ready => "ready",
        CoreModelStatus::NeedsProjector => "needs_projector",
        CoreModelStatus::Broken => "broken",
    }
}

pub(super) fn model_source_kind_name(source: CoreModelSourceKind) -> &'static str {
    match source {
        CoreModelSourceKind::Local => "local",
        CoreModelSourceKind::Remote => "remote",
    }
}
