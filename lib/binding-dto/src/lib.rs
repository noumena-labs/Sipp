//! Host-neutral request/config DTOs and their conversions to the Sipp core
//! types, shared by the Node and Python bindings and unit-tested here without
//! any host runtime.

use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sipp::core::TokenUsage as CoreTokenUsage;
use sipp::endpoint::Local as CoreLocalEndpoint;
use sipp::engine::protocol::{CacheSource as CoreCacheSource, RequestStats as CoreRequestStats};
use sipp::engine::{
    ChatMessage as CoreChatMessage, ChatRole as CoreChatRole, FlashAttentionMode, GpuLayerConfig,
    KvCacheType, KvReuseMode, LogitBias, ModelPlacementConfig as CoreModelPlacementConfig,
    MultimodalRuntimeConfig as CoreMultimodalRuntimeConfig,
    NativeRuntimeConfig as CoreNativeRuntimeConfig,
    ObservabilityRuntimeConfig as CoreObservabilityRuntimeConfig, PoolingType as CorePoolingType,
    ResidencyRuntimeConfig as CoreResidencyRuntimeConfig, RopeScaling, SamplerStage,
    SamplingRuntimeConfig as CoreSamplingRuntimeConfig,
    SamplingRuntimeOverride as CoreSamplingRuntimeOverride,
    SchedulerRuntimeConfig as CoreSchedulerRuntimeConfig, SplitMode,
};
use sipp::runtime::config::{
    SchedulerPolicyConfig as CoreSchedulerPolicyConfig, SchedulerPolicyMode,
};
use sipp::{
    AnthropicProviderConfig as CoreAnthropicProviderConfig, Endpoint as CoreEndpoint,
    EndpointRef as CoreEndpointRef, GatewayAuthentication as CoreGatewayAuthentication,
    GatewayDescriptor as CoreGatewayDescriptor, GatewayRoutes as CoreGatewayRoutes,
    GatewaySecret as CoreGatewaySecret, GatewayTimeoutPolicy as CoreGatewayTimeoutPolicy,
    LocalEmbedOptions as CoreLocalEmbedOptions, LocalTextOptions as CoreLocalTextOptions,
    ManagedModel as CoreManagedModel,
    OpenAiCompatibleProviderConfig as CoreOpenAiCompatibleProviderConfig,
    OpenAiProviderConfig as CoreOpenAiProviderConfig, ProviderAuthConfig as CoreProviderAuthConfig,
    ProviderDescriptor as CoreProviderDescriptor, ProviderSecret as CoreProviderSecret,
    SippChatRequest as CoreChatRequest, SippEmbedRequest as CoreEmbedRequest,
    SippQueryRequest as CoreQueryRequest, SippTextOptions as CoreTextOptions,
};
use thiserror::Error;

#[cfg(test)]
#[path = "tests/root_tests.rs"]
mod root_tests;
#[cfg(test)]
#[path = "tests/stats_tests.rs"]
mod stats_tests;

/// Conversion failure, mapped by each binding to its host error type.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConvertError {
    /// Caller supplied an invalid argument.
    #[error("{0}")]
    InvalidArg(String),
}

pub type Result<T> = std::result::Result<T, ConvertError>;

fn invalid_arg(message: impl Into<String>) -> ConvertError {
    ConvertError::InvalidArg(message.into())
}

/// Per-token logit bias applied during sampling.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogitBiasConfig {
    pub token: i32,
    pub bias: f64,
}

/// Sampling controls used by local text generation.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct SamplingRuntimeConfig {
    pub samplers: Option<Vec<String>>,
    pub seed: Option<i64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub typical_p: Option<f64>,
    pub xtc_probability: Option<f64>,
    pub xtc_threshold: Option<f64>,
    pub top_n_sigma: Option<f64>,
    pub temperature: Option<f64>,
    pub dynatemp_range: Option<f64>,
    pub dynatemp_exponent: Option<f64>,
    pub repeat_last_n: Option<i32>,
    pub repeat_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub dry_multiplier: Option<f64>,
    pub dry_base: Option<f64>,
    pub dry_allowed_length: Option<i32>,
    pub dry_penalty_last_n: Option<i32>,
    pub dry_sequence_breakers: Option<Vec<String>>,
    pub mirostat: Option<i32>,
    pub mirostat_tau: Option<f64>,
    pub mirostat_eta: Option<f64>,
    pub min_keep: Option<i32>,
    pub n_probs: Option<i32>,
    pub logit_bias: Option<Vec<LogitBiasConfig>>,
    pub ignore_eos: Option<bool>,
    pub grammar_lazy: Option<bool>,
    pub preserved_tokens: Option<Vec<i32>>,
    pub backend_sampling: Option<bool>,
}

impl TryFrom<&SamplingRuntimeConfig> for CoreSamplingRuntimeConfig {
    type Error = ConvertError;

    fn try_from(value: &SamplingRuntimeConfig) -> Result<Self> {
        let override_config = CoreSamplingRuntimeOverride::try_from(value)?;
        let mut config = Self::default();
        override_config.apply_to(&mut config);
        Ok(config)
    }
}

impl TryFrom<&SamplingRuntimeConfig> for CoreSamplingRuntimeOverride {
    type Error = ConvertError;

    fn try_from(value: &SamplingRuntimeConfig) -> Result<Self> {
        if value
            .seed
            .is_some_and(|value| value < 0 || value > u32::MAX as i64)
        {
            return Err(invalid_arg("seed must fit in an unsigned 32-bit integer"));
        }
        Ok(Self {
            samplers: value
                .samplers
                .as_ref()
                .map(|samplers| {
                    samplers
                        .iter()
                        .map(|stage| parse_sampler_stage(stage))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?,
            seed: value.seed.map(|value| value as u32),
            top_k: value.top_k,
            top_p: option_f32(value.top_p, "top_p")?,
            min_p: option_f32(value.min_p, "min_p")?,
            typical_p: option_f32(value.typical_p, "typical_p")?,
            xtc_probability: option_f32(value.xtc_probability, "xtc_probability")?,
            xtc_threshold: option_f32(value.xtc_threshold, "xtc_threshold")?,
            top_n_sigma: option_f32(value.top_n_sigma, "top_n_sigma")?,
            temperature: option_f32(value.temperature, "temperature")?,
            dynatemp_range: option_f32(value.dynatemp_range, "dynatemp_range")?,
            dynatemp_exponent: option_f32(value.dynatemp_exponent, "dynatemp_exponent")?,
            repeat_last_n: value.repeat_last_n,
            repeat_penalty: option_f32(value.repeat_penalty, "repeat_penalty")?,
            frequency_penalty: option_f32(value.frequency_penalty, "frequency_penalty")?,
            presence_penalty: option_f32(value.presence_penalty, "presence_penalty")?,
            dry_multiplier: option_f32(value.dry_multiplier, "dry_multiplier")?,
            dry_base: option_f32(value.dry_base, "dry_base")?,
            dry_allowed_length: value.dry_allowed_length,
            dry_penalty_last_n: value.dry_penalty_last_n,
            dry_sequence_breakers: value.dry_sequence_breakers.clone(),
            mirostat: value.mirostat,
            mirostat_tau: option_f32(value.mirostat_tau, "mirostat_tau")?,
            mirostat_eta: option_f32(value.mirostat_eta, "mirostat_eta")?,
            min_keep: value.min_keep,
            n_probs: value.n_probs,
            logit_bias: value
                .logit_bias
                .as_ref()
                .map(|biases| {
                    biases
                        .iter()
                        .map(|bias| {
                            Ok(LogitBias {
                                token: bias.token,
                                bias: finite_f64_to_f32(bias.bias, "logit_bias.bias")?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?,
            ignore_eos: value.ignore_eos,
            grammar_lazy: value.grammar_lazy,
            preserved_tokens: value.preserved_tokens.clone(),
            backend_sampling: value.backend_sampling,
        })
    }
}

/// Numeric GPU layer count for model placement configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GpuLayerCountConfig {
    pub count: i32,
}

/// String preset or explicit count for GPU layer placement.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum GpuLayers {
    Preset(String),
    Count(GpuLayerCountConfig),
}

/// Device placement and memory mapping settings for local model loading.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ModelPlacementConfig {
    pub devices: Option<Vec<String>>,
    pub gpu_layers: Option<GpuLayers>,
    pub split_mode: Option<String>,
    pub main_gpu: Option<i32>,
    pub tensor_split: Option<Vec<f64>>,
    pub use_mmap: Option<bool>,
    pub use_mlock: Option<bool>,
    pub fit_params: Option<bool>,
    pub fit_params_min_ctx: Option<i32>,
    pub fit_params_target_bytes: Option<Vec<f64>>,
    pub check_tensors: Option<bool>,
    pub no_extra_bufts: Option<bool>,
    pub no_host: Option<bool>,
}

impl TryFrom<&ModelPlacementConfig> for CoreModelPlacementConfig {
    type Error = ConvertError;

    fn try_from(value: &ModelPlacementConfig) -> Result<Self> {
        let mut core = Self::default();
        assign_if_some(&mut core.devices, value.devices.clone());
        if let Some(value) = &value.gpu_layers {
            core.gpu_layers = match value {
                GpuLayers::Preset(value) => parse_gpu_layers(value)?,
                GpuLayers::Count(value) => GpuLayerConfig::from_layer_count(value.count),
            };
        }
        if let Some(value) = &value.split_mode {
            core.split_mode = parse_split_mode(value)?;
        }
        core.main_gpu = value.main_gpu;
        if let Some(value) = &value.tensor_split {
            core.tensor_split = value
                .iter()
                .map(|value| finite_f64_to_f32(*value, "tensor_split"))
                .collect::<Result<Vec<_>>>()?;
        }
        assign_if_some(&mut core.use_mmap, value.use_mmap);
        assign_if_some(&mut core.use_mlock, value.use_mlock);
        assign_if_some(&mut core.fit_params, value.fit_params);
        core.fit_params_min_ctx = value.fit_params_min_ctx;
        if let Some(value) = &value.fit_params_target_bytes {
            core.fit_params_target_bytes = value.iter().map(|value| *value as u64).collect();
        }
        assign_if_some(&mut core.check_tensors, value.check_tensors);
        assign_if_some(&mut core.no_extra_bufts, value.no_extra_bufts);
        assign_if_some(&mut core.no_host, value.no_host);
        Ok(core)
    }
}

/// Embedding pooling strategy used by the local runtime.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingType {
    Unspecified,
    None,
    Mean,
    Cls,
    Last,
    Rank,
}

impl From<PoolingType> for CorePoolingType {
    fn from(value: PoolingType) -> Self {
        match value {
            PoolingType::Unspecified => Self::Unspecified,
            PoolingType::None => Self::None,
            PoolingType::Mean => Self::Mean,
            PoolingType::Cls => Self::Cls,
            PoolingType::Last => Self::Last,
            PoolingType::Rank => Self::Rank,
        }
    }
}

impl TryFrom<&str> for PoolingType {
    type Error = ConvertError;

    fn try_from(value: &str) -> Result<Self> {
        parse_choice(
            value,
            "pooling must be one of: unspecified, none, mean, cls, last, rank",
        )
    }
}

/// Context, threading, attention, and embedding settings for local runtime use.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ContextRuntimeConfig {
    pub n_ctx: Option<i32>,
    pub n_batch: Option<i32>,
    pub n_ubatch: Option<i32>,
    pub n_parallel: Option<i32>,
    pub n_threads: Option<i32>,
    pub n_threads_batch: Option<i32>,
    pub flash_attention: Option<String>,
    pub kv_unified: Option<bool>,
    pub cache_type_k: Option<String>,
    pub cache_type_v: Option<String>,
    pub offload_kqv: Option<bool>,
    pub op_offload: Option<bool>,
    pub swa_full: Option<bool>,
    pub warmup: Option<bool>,
    pub rope_scaling: Option<String>,
    pub rope_freq_base: Option<f64>,
    pub rope_freq_scale: Option<f64>,
    pub yarn_orig_ctx: Option<i32>,
    pub yarn_ext_factor: Option<f64>,
    pub yarn_attn_factor: Option<f64>,
    pub yarn_beta_fast: Option<f64>,
    pub yarn_beta_slow: Option<f64>,
    pub embeddings: Option<bool>,
    pub pooling: Option<PoolingType>,
}

impl TryFrom<&ContextRuntimeConfig> for sipp::engine::ContextRuntimeConfig {
    type Error = ConvertError;

    fn try_from(value: &ContextRuntimeConfig) -> Result<Self> {
        let mut core = sipp::engine::ContextRuntimeConfig {
            n_ctx: value.n_ctx,
            n_batch: value.n_batch,
            n_ubatch: value.n_ubatch,
            n_parallel: value.n_parallel,
            n_threads: value.n_threads,
            n_threads_batch: value.n_threads_batch,
            flash_attention: value
                .flash_attention
                .as_deref()
                .map(parse_flash_attention)
                .transpose()?
                .unwrap_or_default(),
            kv_unified: value.kv_unified,
            ..Default::default()
        };
        if let Some(value) = &value.cache_type_k {
            core.cache_type_k = parse_kv_cache_type(value)?;
        }
        if let Some(value) = &value.cache_type_v {
            core.cache_type_v = parse_kv_cache_type(value)?;
        }
        assign_if_some(&mut core.offload_kqv, value.offload_kqv);
        assign_if_some(&mut core.op_offload, value.op_offload);
        assign_if_some(&mut core.swa_full, value.swa_full);
        assign_if_some(&mut core.warmup, value.warmup);
        if let Some(value) = &value.rope_scaling {
            core.rope_scaling = Some(parse_rope_scaling(value)?);
        }
        core.rope_freq_base = option_f32(value.rope_freq_base, "rope_freq_base")?;
        core.rope_freq_scale = option_f32(value.rope_freq_scale, "rope_freq_scale")?;
        core.yarn_orig_ctx = value.yarn_orig_ctx;
        core.yarn_ext_factor = option_f32(value.yarn_ext_factor, "yarn_ext_factor")?;
        core.yarn_attn_factor = option_f32(value.yarn_attn_factor, "yarn_attn_factor")?;
        core.yarn_beta_fast = option_f32(value.yarn_beta_fast, "yarn_beta_fast")?;
        core.yarn_beta_slow = option_f32(value.yarn_beta_slow, "yarn_beta_slow")?;
        core.embeddings = value.embeddings;
        core.pooling = value.pooling.map(CorePoolingType::from);
        Ok(core)
    }
}

/// Scheduler policy knobs for latency, balance, or throughput behavior.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct SchedulerPolicyConfig {
    pub mode: Option<String>,
    pub decode_token_reserve: Option<i32>,
    pub enable_adaptive_prefill_chunking: Option<bool>,
}

impl TryFrom<&SchedulerPolicyConfig> for CoreSchedulerPolicyConfig {
    type Error = ConvertError;

    fn try_from(value: &SchedulerPolicyConfig) -> Result<Self> {
        let mut core = Self::default();
        if let Some(value) = &value.mode {
            core.mode = parse_scheduler_policy(value)?;
        }
        assign_if_some(&mut core.decode_token_reserve, value.decode_token_reserve);
        assign_if_some(
            &mut core.enable_adaptive_prefill_chunking,
            value.enable_adaptive_prefill_chunking,
        );
        Ok(core)
    }
}

/// Request scheduler and continuous batching settings.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct SchedulerRuntimeConfig {
    pub continuous_batching: Option<bool>,
    pub policy: Option<SchedulerPolicyConfig>,
    pub prefill_chunk_size: Option<i32>,
    pub max_running_requests: Option<i32>,
    pub max_queued_requests: Option<i32>,
}

impl TryFrom<&SchedulerRuntimeConfig> for CoreSchedulerRuntimeConfig {
    type Error = ConvertError;

    fn try_from(value: &SchedulerRuntimeConfig) -> Result<Self> {
        let mut core = Self::default();
        assign_if_some(&mut core.continuous_batching, value.continuous_batching);
        if let Some(value) = &value.policy {
            core.policy = CoreSchedulerPolicyConfig::try_from(value)?;
        }
        assign_if_some(&mut core.prefill_chunk_size, value.prefill_chunk_size);
        core.max_running_requests = value.max_running_requests;
        core.max_queued_requests = value.max_queued_requests;
        Ok(core)
    }
}

/// Prefix KV-cache reuse and snapshot settings.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct CacheRuntimeConfig {
    pub mode: Option<String>,
    pub retained_prefix_tokens: Option<i32>,
    pub snapshot_interval_tokens: Option<i32>,
    pub max_snapshot_entries: Option<i32>,
    pub max_snapshot_bytes: Option<f64>,
}

impl TryFrom<&CacheRuntimeConfig> for sipp::engine::CacheRuntimeConfig {
    type Error = ConvertError;

    fn try_from(value: &CacheRuntimeConfig) -> Result<Self> {
        let mut core = Self::default();
        if let Some(value) = &value.mode {
            core.mode = parse_kv_reuse_mode(value)?;
        }
        assign_if_some(
            &mut core.retained_prefix_tokens,
            value.retained_prefix_tokens,
        );
        assign_if_some(
            &mut core.snapshot_interval_tokens,
            value.snapshot_interval_tokens,
        );
        assign_if_some(&mut core.max_snapshot_entries, value.max_snapshot_entries);
        assign_if_some_map(
            &mut core.max_snapshot_bytes,
            value.max_snapshot_bytes,
            |value| value as usize,
        );
        Ok(core)
    }
}

/// Vision projector and image-token settings for multimodal models.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct MultimodalRuntimeConfig {
    pub projector_path: Option<String>,
    pub use_gpu: Option<bool>,
    pub image_min_tokens: Option<i32>,
    pub image_max_tokens: Option<i32>,
}

impl From<&MultimodalRuntimeConfig> for CoreMultimodalRuntimeConfig {
    fn from(value: &MultimodalRuntimeConfig) -> Self {
        Self {
            projector_path: value.projector_path.clone(),
            use_gpu: value.use_gpu,
            image_min_tokens: value.image_min_tokens,
            image_max_tokens: value.image_max_tokens,
        }
    }
}

/// GPU residency limits for concurrently loaded local models.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ResidencyRuntimeConfig {
    pub max_gpu_models_per_device: Option<f64>,
    pub allow_cpu_models_while_gpu_loaded: Option<bool>,
    pub require_gpu_lease: Option<bool>,
    pub gpu_memory_safety_margin_bytes: Option<f64>,
}

impl From<&ResidencyRuntimeConfig> for CoreResidencyRuntimeConfig {
    fn from(value: &ResidencyRuntimeConfig) -> Self {
        let mut core = Self::default();
        assign_if_some_map(
            &mut core.max_gpu_models_per_device,
            value.max_gpu_models_per_device,
            |value| value as usize,
        );
        assign_if_some(
            &mut core.allow_cpu_models_while_gpu_loaded,
            value.allow_cpu_models_while_gpu_loaded,
        );
        assign_if_some(&mut core.require_gpu_lease, value.require_gpu_lease);
        assign_if_some_map(
            &mut core.gpu_memory_safety_margin_bytes,
            value.gpu_memory_safety_margin_bytes,
            |value| value as u64,
        );
        core
    }
}

/// Runtime metrics and backend profiling options.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ObservabilityRuntimeConfig {
    pub runtime_metrics: Option<bool>,
    pub backend_profiling: Option<bool>,
}

impl From<&ObservabilityRuntimeConfig> for CoreObservabilityRuntimeConfig {
    fn from(value: &ObservabilityRuntimeConfig) -> Self {
        Self {
            runtime_metrics: value.runtime_metrics.unwrap_or(false),
            backend_profiling: value.backend_profiling.unwrap_or(false),
        }
    }
}

/// Complete native runtime configuration for local model loading.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct NativeRuntimeConfig {
    pub placement: Option<ModelPlacementConfig>,
    pub context: Option<ContextRuntimeConfig>,
    pub sampling: Option<SamplingRuntimeConfig>,
    pub scheduler: Option<SchedulerRuntimeConfig>,
    pub cache: Option<CacheRuntimeConfig>,
    pub multimodal: Option<MultimodalRuntimeConfig>,
    pub residency: Option<ResidencyRuntimeConfig>,
    pub observability: Option<ObservabilityRuntimeConfig>,
}

impl TryFrom<&NativeRuntimeConfig> for CoreNativeRuntimeConfig {
    type Error = ConvertError;

    fn try_from(value: &NativeRuntimeConfig) -> Result<Self> {
        Ok(Self {
            placement: optional_core_or_default(value.placement.as_ref(), |value| {
                CoreModelPlacementConfig::try_from(value)
            })?,
            context: optional_core_or_default(value.context.as_ref(), |value| {
                sipp::engine::ContextRuntimeConfig::try_from(value)
            })?,
            sampling: optional_core_or_default(value.sampling.as_ref(), |value| {
                CoreSamplingRuntimeConfig::try_from(value)
            })?,
            scheduler: optional_core_or_default(value.scheduler.as_ref(), |value| {
                CoreSchedulerRuntimeConfig::try_from(value)
            })?,
            cache: optional_core_or_default(value.cache.as_ref(), |value| {
                sipp::engine::CacheRuntimeConfig::try_from(value)
            })?,
            multimodal: optional_core_or_default(value.multimodal.as_ref(), |value| {
                Ok(CoreMultimodalRuntimeConfig::from(value))
            })?,
            residency: optional_core_or_default(value.residency.as_ref(), |value| {
                Ok(CoreResidencyRuntimeConfig::from(value))
            })?,
            observability: optional_core_or_default(value.observability.as_ref(), |value| {
                Ok(CoreObservabilityRuntimeConfig::from(value))
            })?,
        })
    }
}

/// Address of a registered inference endpoint.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct EndpointRef {
    pub id: String,
}

impl TryFrom<&EndpointRef> for CoreEndpointRef {
    type Error = ConvertError;

    fn try_from(value: &EndpointRef) -> Result<Self> {
        Ok(Self::from_id(value.id.clone()))
    }
}

impl From<CoreEndpointRef> for EndpointRef {
    fn from(value: CoreEndpointRef) -> Self {
        Self {
            id: value.id().to_string(),
        }
    }
}

/// Shared generation options for text-producing requests.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct SippTextOptions {
    #[serde(alias = "maxTokens")]
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    #[serde(alias = "topP")]
    pub top_p: Option<f64>,
    pub stop: Option<Vec<String>>,
}

impl TryFrom<&SippTextOptions> for CoreTextOptions {
    type Error = ConvertError;

    fn try_from(value: &SippTextOptions) -> Result<Self> {
        Ok(Self {
            max_tokens: value.max_tokens,
            temperature: value
                .temperature
                .map(|value| finite_f64_to_f32(value, "temperature"))
                .transpose()?,
            top_p: value
                .top_p
                .map(|value| finite_f64_to_f32(value, "topP"))
                .transpose()?,
            stop: value.stop.clone().unwrap_or_default(),
        })
    }
}

/// Local-only prompt options such as grammar constraints and image inputs.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct LocalTextOptions {
    #[serde(alias = "contextKey")]
    pub context_key: Option<String>,
    pub grammar: Option<String>,
    #[serde(alias = "jsonSchema")]
    pub json_schema: Option<String>,
    pub sampling: Option<SamplingRuntimeConfig>,
    /// Raw image bytes; attached natively by the binding, never JSON-encoded.
    #[serde(skip)]
    pub media: Vec<Vec<u8>>,
}

impl TryFrom<LocalTextOptions> for CoreLocalTextOptions {
    type Error = ConvertError;

    fn try_from(value: LocalTextOptions) -> Result<Self> {
        Ok(Self {
            context_key: value.context_key,
            grammar: value.grammar,
            json_schema: value.json_schema,
            sampling: value
                .sampling
                .as_ref()
                .map(CoreSamplingRuntimeOverride::try_from)
                .transpose()?,
            media: value.media,
        })
    }
}

impl TryFrom<&LocalTextOptions> for CoreLocalTextOptions {
    type Error = ConvertError;

    fn try_from(value: &LocalTextOptions) -> Result<Self> {
        Self::try_from(value.clone())
    }
}

/// Local-only embedding options for context and vector normalization.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct LocalEmbedOptions {
    #[serde(alias = "contextKey")]
    pub context_key: Option<String>,
    pub normalize: Option<bool>,
}

impl From<&LocalEmbedOptions> for CoreLocalEmbedOptions {
    fn from(value: &LocalEmbedOptions) -> Self {
        Self {
            context_key: value.context_key.clone(),
            normalize: value.normalize,
        }
    }
}

/// Role/content chat message accepted by chat requests.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl TryFrom<&ChatMessage> for CoreChatMessage {
    type Error = ConvertError;

    fn try_from(value: &ChatMessage) -> Result<Self> {
        Ok(Self {
            role: parse_chat_role(&value.role)?,
            content: value.content.clone(),
        })
    }
}

/// Prompt completion request routed to an inference endpoint.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct SippQueryRequest {
    #[serde(alias = "requestId")]
    pub request_id: Option<String>,
    pub endpoint: Option<EndpointRef>,
    pub prompt: String,
    pub options: Option<SippTextOptions>,
    pub local: Option<LocalTextOptions>,
    pub extra: Option<serde_json::Value>,
    #[serde(alias = "emitTokens")]
    pub emit_tokens: Option<bool>,
}

impl TryFrom<SippQueryRequest> for CoreQueryRequest {
    type Error = ConvertError;

    fn try_from(value: SippQueryRequest) -> Result<Self> {
        Ok(Self {
            endpoint: optional_endpoint(value.endpoint.as_ref())?,
            prompt: value.prompt,
            options: optional_core_or_default(value.options.as_ref(), |value| {
                CoreTextOptions::try_from(value)
            })?,
            local: value
                .local
                .map(CoreLocalTextOptions::try_from)
                .transpose()?
                .unwrap_or_default(),
            extra: json_object_or_empty(value.extra, "extra")?,
            emit_tokens: value.emit_tokens.unwrap_or(false),
        })
    }
}

impl TryFrom<&SippQueryRequest> for CoreQueryRequest {
    type Error = ConvertError;

    fn try_from(value: &SippQueryRequest) -> Result<Self> {
        Self::try_from(value.clone())
    }
}

/// Chat completion request routed to an inference endpoint.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct SippChatRequest {
    #[serde(alias = "requestId")]
    pub request_id: Option<String>,
    pub endpoint: Option<EndpointRef>,
    pub messages: Vec<ChatMessage>,
    pub options: Option<SippTextOptions>,
    pub local: Option<LocalTextOptions>,
    pub extra: Option<serde_json::Value>,
    #[serde(alias = "emitTokens")]
    pub emit_tokens: Option<bool>,
}

impl TryFrom<SippChatRequest> for CoreChatRequest {
    type Error = ConvertError;

    fn try_from(value: SippChatRequest) -> Result<Self> {
        Ok(Self {
            endpoint: optional_endpoint(value.endpoint.as_ref())?,
            messages: chat_messages_into_core(value.messages)?,
            options: optional_core_or_default(value.options.as_ref(), |value| {
                CoreTextOptions::try_from(value)
            })?,
            local: value
                .local
                .map(CoreLocalTextOptions::try_from)
                .transpose()?
                .unwrap_or_default(),
            extra: json_object_or_empty(value.extra, "extra")?,
            emit_tokens: value.emit_tokens.unwrap_or(false),
        })
    }
}

impl TryFrom<&SippChatRequest> for CoreChatRequest {
    type Error = ConvertError;

    fn try_from(value: &SippChatRequest) -> Result<Self> {
        Self::try_from(value.clone())
    }
}

/// Embedding request routed to an inference endpoint.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct SippEmbedRequest {
    #[serde(alias = "requestId")]
    pub request_id: Option<String>,
    pub endpoint: Option<EndpointRef>,
    pub input: String,
    pub local: Option<LocalEmbedOptions>,
    pub extra: Option<serde_json::Value>,
}

impl TryFrom<SippEmbedRequest> for CoreEmbedRequest {
    type Error = ConvertError;

    fn try_from(value: SippEmbedRequest) -> Result<Self> {
        Ok(Self {
            endpoint: optional_endpoint(value.endpoint.as_ref())?,
            input: value.input,
            local: value
                .local
                .map(|value| CoreLocalEmbedOptions::from(&value))
                .unwrap_or_default(),
            extra: json_object_or_empty(value.extra, "extra")?,
        })
    }
}

impl TryFrom<&SippEmbedRequest> for CoreEmbedRequest {
    type Error = ConvertError;

    fn try_from(value: &SippEmbedRequest) -> Result<Self> {
        Self::try_from(value.clone())
    }
}

/// Static header entry for a remote endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StaticHeader {
    pub name: String,
    pub value: String,
}

/// Local endpoint input from a managed model.
#[derive(Debug, Clone)]
pub struct LocalEndpointInput {
    pub model: CoreManagedModel,
    pub runtime: NativeRuntimeConfig,
}

/// Authentication for a gateway endpoint.
#[derive(Debug, Clone)]
pub enum GatewayAuthentication {
    None,
    Bearer(String),
    Header { name: String, value: String },
}

/// Gateway authentication values accepted by native language factories.
#[derive(Debug, Clone)]
pub struct GatewayAuthenticationInput {
    pub kind: String,
    pub value: Option<String>,
    pub header_name: Option<String>,
}

impl TryFrom<GatewayAuthenticationInput> for GatewayAuthentication {
    type Error = ConvertError;

    fn try_from(value: GatewayAuthenticationInput) -> Result<Self> {
        match value.kind.as_str() {
            "none" => Ok(Self::None),
            "bearer" => Ok(Self::Bearer(required_value(
                value.value,
                "authentication value",
            )?)),
            "header" => Ok(Self::Header {
                name: required_value(value.header_name, "authentication headerName")?,
                value: required_value(value.value, "authentication value")?,
            }),
            _ => Err(invalid_arg(
                "authentication kind must be none, bearer, or header",
            )),
        }
    }
}

/// Gateway endpoint input.
#[derive(Debug, Clone)]
pub struct GatewayEndpointInput {
    pub target: String,
    pub base_url: String,
    pub authentication: GatewayAuthentication,
    pub static_headers: Vec<StaticHeader>,
    pub timeout_ms: Option<u64>,
    pub query_route: Option<String>,
    pub chat_route: Option<String>,
    pub embed_route: Option<String>,
    pub protocol_options: Option<serde_json::Value>,
}

/// Direct provider endpoint input.
#[derive(Debug, Clone)]
pub enum ProviderEndpointInput {
    OpenAi(OpenAiProviderInput),
    Anthropic(AnthropicProviderInput),
    OpenAiCompatible(OpenAiCompatibleProviderInput),
}

/// Provider selection values accepted by native language factories.
#[derive(Debug, Clone)]
pub struct ProviderSelectionInput {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout_ms: Option<u64>,
    pub version: Option<String>,
    pub auth_header_name: Option<String>,
    pub auth_header_value: Option<String>,
    pub static_headers: Option<Vec<StaticHeader>>,
    pub correlation_header: Option<String>,
}

/// OpenAI provider input.
#[derive(Debug, Clone)]
pub struct OpenAiProviderInput {
    pub model: String,
    pub api_key: String,
    pub base_url: Option<String>,
    pub timeout_ms: Option<u64>,
}

/// Anthropic provider input.
#[derive(Debug, Clone)]
pub struct AnthropicProviderInput {
    pub model: String,
    pub api_key: String,
    pub base_url: Option<String>,
    pub version: Option<String>,
    pub timeout_ms: Option<u64>,
}

/// Authentication for an OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub enum ProviderAuthentication {
    Bearer(String),
    Header { name: String, value: String },
}

/// OpenAI-compatible provider input.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleProviderInput {
    pub model: String,
    pub base_url: String,
    pub authentication: ProviderAuthentication,
    pub static_headers: Vec<StaticHeader>,
    pub correlation_header: Option<String>,
    pub timeout_ms: Option<u64>,
}

impl TryFrom<ProviderSelectionInput> for ProviderEndpointInput {
    type Error = ConvertError;

    fn try_from(value: ProviderSelectionInput) -> Result<Self> {
        let ProviderSelectionInput {
            provider,
            model,
            api_key,
            base_url,
            timeout_ms,
            version,
            auth_header_name,
            auth_header_value,
            static_headers,
            correlation_header,
        } = value;
        match provider.as_str() {
            "openai" => {
                reject_provider_fields(
                    &[
                        ("version", version.is_some()),
                        ("authHeaderName", auth_header_name.is_some()),
                        ("authHeaderValue", auth_header_value.is_some()),
                        ("staticHeaders", static_headers.is_some()),
                        ("correlationHeader", correlation_header.is_some()),
                    ],
                    "OpenAI",
                )?;
                Ok(Self::OpenAi(OpenAiProviderInput {
                    model,
                    api_key: required_value(api_key, "provider apiKey")?,
                    base_url,
                    timeout_ms,
                }))
            }
            "anthropic" => {
                reject_provider_fields(
                    &[
                        ("authHeaderName", auth_header_name.is_some()),
                        ("authHeaderValue", auth_header_value.is_some()),
                        ("staticHeaders", static_headers.is_some()),
                        ("correlationHeader", correlation_header.is_some()),
                    ],
                    "Anthropic",
                )?;
                Ok(Self::Anthropic(AnthropicProviderInput {
                    model,
                    api_key: required_value(api_key, "provider apiKey")?,
                    base_url,
                    version,
                    timeout_ms,
                }))
            }
            "openai_compatible" => {
                reject_provider_fields(&[("version", version.is_some())], "OpenAI-compatible")?;
                Ok(Self::OpenAiCompatible(OpenAiCompatibleProviderInput {
                    model,
                    base_url: required_value(base_url, "provider baseUrl")?,
                    authentication: provider_authentication(
                        api_key,
                        auth_header_name,
                        auth_header_value,
                    )?,
                    static_headers: static_headers.unwrap_or_default(),
                    correlation_header,
                    timeout_ms,
                }))
            }
            _ => Err(invalid_arg(
                "provider must be one of: openai, anthropic, openai_compatible",
            )),
        }
    }
}

fn reject_provider_fields(fields: &[(&'static str, bool)], provider: &'static str) -> Result<()> {
    if let Some((field, _)) = fields.iter().find(|(_, present)| *present) {
        return Err(invalid_arg(format!(
            "{field} is not valid for the {provider} provider"
        )));
    }
    Ok(())
}

fn provider_authentication(
    api_key: Option<String>,
    header_name: Option<String>,
    header_value: Option<String>,
) -> Result<ProviderAuthentication> {
    match (api_key, header_name, header_value) {
        (Some(value), None, None) => Ok(ProviderAuthentication::Bearer(value)),
        (None, Some(name), Some(value)) => Ok(ProviderAuthentication::Header { name, value }),
        _ => Err(invalid_arg(
            "OpenAI-compatible provider requires either apiKey or authHeaderName/authHeaderValue",
        )),
    }
}

fn required_value(value: Option<String>, name: &'static str) -> Result<String> {
    value.ok_or_else(|| invalid_arg(format!("{name} is required")))
}

impl TryFrom<LocalEndpointInput> for CoreEndpoint {
    type Error = ConvertError;

    fn try_from(value: LocalEndpointInput) -> Result<Self> {
        let runtime = CoreNativeRuntimeConfig::try_from(&value.runtime)?;
        Ok(CoreLocalEndpoint::new(&value.model).runtime(runtime).into())
    }
}

impl TryFrom<GatewayEndpointInput> for CoreEndpoint {
    type Error = ConvertError;

    fn try_from(value: GatewayEndpointInput) -> Result<Self> {
        let timeout = endpoint_timeout(value.timeout_ms)?.unwrap_or(Duration::from_secs(60));
        let mut routes = CoreGatewayRoutes::default();
        assign_if_some(&mut routes.query, value.query_route);
        assign_if_some(&mut routes.chat, value.chat_route);
        assign_if_some(&mut routes.embed, value.embed_route);
        Ok(CoreGatewayDescriptor {
            target: value.target,
            base_url: value.base_url,
            routes,
            authentication: value.authentication.into(),
            static_headers: value
                .static_headers
                .into_iter()
                .map(|header| (header.name, header.value))
                .collect(),
            timeouts: CoreGatewayTimeoutPolicy {
                connect: timeout,
                request: timeout,
                read: timeout,
            },
            protocol_options: json_object_or_empty(value.protocol_options, "protocolOptions")?,
        }
        .into())
    }
}

impl From<GatewayAuthentication> for CoreGatewayAuthentication {
    fn from(value: GatewayAuthentication) -> Self {
        match value {
            GatewayAuthentication::None => Self::None,
            GatewayAuthentication::Bearer(value) => Self::Bearer(CoreGatewaySecret::new(value)),
            GatewayAuthentication::Header { name, value } => Self::Header {
                name,
                value: CoreGatewaySecret::new(value),
            },
        }
    }
}

impl TryFrom<ProviderEndpointInput> for CoreEndpoint {
    type Error = ConvertError;

    fn try_from(value: ProviderEndpointInput) -> Result<Self> {
        let provider = match value {
            ProviderEndpointInput::OpenAi(input) => {
                CoreProviderDescriptor::OpenAi(CoreOpenAiProviderConfig {
                    model: input.model,
                    api_key: CoreProviderSecret::new(input.api_key),
                    base_url: input.base_url,
                    timeout: endpoint_timeout(input.timeout_ms)?,
                })
            }
            ProviderEndpointInput::Anthropic(input) => {
                CoreProviderDescriptor::Anthropic(CoreAnthropicProviderConfig {
                    model: input.model,
                    api_key: CoreProviderSecret::new(input.api_key),
                    base_url: input.base_url,
                    version: input.version,
                    timeout: endpoint_timeout(input.timeout_ms)?,
                })
            }
            ProviderEndpointInput::OpenAiCompatible(input) => {
                CoreProviderDescriptor::OpenAiCompatible(CoreOpenAiCompatibleProviderConfig {
                    model: input.model,
                    base_url: input.base_url,
                    auth: input.authentication.into(),
                    static_headers: input
                        .static_headers
                        .into_iter()
                        .map(|header| (header.name, CoreProviderSecret::new(header.value)))
                        .collect(),
                    correlation_header: input.correlation_header,
                    timeout: endpoint_timeout(input.timeout_ms)?,
                })
            }
        };
        Ok(provider.into())
    }
}

impl From<ProviderAuthentication> for CoreProviderAuthConfig {
    fn from(value: ProviderAuthentication) -> Self {
        match value {
            ProviderAuthentication::Bearer(value) => Self::Bearer(CoreProviderSecret::new(value)),
            ProviderAuthentication::Header { name, value } => Self::Header {
                name,
                value: CoreProviderSecret::new(value),
            },
        }
    }
}

/// Local runtime timing and cache statistics for a completed request.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct RequestStats {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cache_mode: String,
    pub cache_source: String,
    pub cache_hits: i32,
    pub prefill_tokens: i32,
    pub ttft_ms: Option<f64>,
    pub inter_token_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    pub e2e_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

impl From<CoreRequestStats> for RequestStats {
    fn from(stats: CoreRequestStats) -> Self {
        Self {
            input_tokens: stats.input_tokens,
            output_tokens: stats.output_tokens,
            cache_mode: kv_reuse_mode_to_string(stats.cache_mode),
            cache_source: cache_source_to_string(stats.cache_source),
            cache_hits: stats.cache_hits,
            prefill_tokens: stats.prefill_tokens,
            ttft_ms: stats.ttft_ms,
            inter_token_ms: stats.inter_token_ms,
            e2e_ms: stats.e2e_ms,
            e2e_tokens_per_second: stats.e2e_tokens_per_second,
            decode_tokens_per_second: stats.decode_tokens_per_second,
            prefill_tokens_per_second: stats.prefill_tokens_per_second,
            prefill_ms: stats.prefill_ms,
            decode_ms: stats.decode_ms,
        }
    }
}

/// Token counts reported by inference endpoints.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct TokenUsage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

impl From<CoreTokenUsage> for TokenUsage {
    fn from(usage: CoreTokenUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

fn json_object_or_empty(
    value: Option<serde_json::Value>,
    name: &'static str,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    match value {
        Some(serde_json::Value::Object(options)) => Ok(options),
        Some(_) => Err(invalid_arg(format!("{name} must be a JSON object"))),
        None => Ok(serde_json::Map::new()),
    }
}

fn optional_endpoint(endpoint: Option<&EndpointRef>) -> Result<Option<CoreEndpointRef>> {
    endpoint.map(CoreEndpointRef::try_from).transpose()
}

fn finite_f64_to_f32(value: f64, name: &'static str) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(invalid_arg(format!("{name} must be a finite f32")));
    }
    Ok(value as f32)
}

fn chat_messages_into_core(messages: Vec<ChatMessage>) -> Result<Vec<CoreChatMessage>> {
    if messages.is_empty() {
        return Err(invalid_arg("chat messages must not be empty"));
    }
    messages
        .into_iter()
        .map(|message| {
            Ok(CoreChatMessage {
                role: parse_chat_role(&message.role)?,
                content: message.content,
            })
        })
        .collect()
}

fn kv_reuse_mode_to_string(mode: KvReuseMode) -> String {
    match mode {
        KvReuseMode::Disabled => "disabled",
        KvReuseMode::LiveSlotPrefix => "live_slot_prefix",
        KvReuseMode::StateSnapshot => "state_snapshot",
        KvReuseMode::LiveSlotAndSnapshot => "live_slot_and_snapshot",
    }
    .to_string()
}

fn cache_source_to_string(source: CoreCacheSource) -> String {
    match source {
        CoreCacheSource::None => "none",
        CoreCacheSource::Live => "live",
        CoreCacheSource::Snapshot => "snapshot",
    }
    .to_string()
}

fn endpoint_timeout(timeout_ms: Option<u64>) -> Result<Option<Duration>> {
    match timeout_ms {
        Some(0) => Err(invalid_arg("timeoutMs must be a positive integer")),
        Some(timeout_ms) => Ok(Some(Duration::from_millis(timeout_ms))),
        None => Ok(None),
    }
}

fn parse_gpu_layers(value: &str) -> Result<GpuLayerConfig> {
    parse_choice(
        value,
        r#"gpu_layers must be "auto", "all", or { count: number }"#,
    )
}

fn parse_split_mode(value: &str) -> Result<SplitMode> {
    parse_choice(value, "split_mode must be one of: none, layer, row, tensor")
}

fn parse_flash_attention(value: &str) -> Result<FlashAttentionMode> {
    parse_choice(
        value,
        "flash_attention must be one of: auto, enabled, disabled",
    )
}

fn parse_kv_cache_type(value: &str) -> Result<KvCacheType> {
    parse_choice(
        value,
        "cache type must be one of: f16, f32, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1",
    )
}

fn parse_rope_scaling(value: &str) -> Result<RopeScaling> {
    parse_choice(value, "ropeScaling must be one of: none, linear, yarn")
}

fn parse_kv_reuse_mode(value: &str) -> Result<KvReuseMode> {
    parse_choice(
        value,
        "cache mode must be one of: disabled, live_slot_prefix, state_snapshot, live_slot_and_snapshot",
    )
}

fn parse_sampler_stage(value: &str) -> Result<SamplerStage> {
    parse_choice(
        value,
        "sampler stage must be one of: dry, top_k, typical_p, top_p, top_n_sigma, min_p, xtc, temperature, infill, penalties, adaptive_p",
    )
}

fn parse_scheduler_policy(value: &str) -> Result<SchedulerPolicyMode> {
    parse_choice(
        value,
        "scheduler.policy.mode must be one of: latency_first, balanced, throughput_first",
    )
}

fn parse_chat_role(value: &str) -> Result<CoreChatRole> {
    parse_choice(value, "chat role must be one of: system, user, assistant")
}

fn parse_choice<T>(value: &str, error_message: &'static str) -> Result<T>
where
    T: DeserializeOwned,
{
    serde_json::from_value(serde_json::Value::String(value.to_string()))
        .map_err(|_| invalid_arg(error_message))
}

fn optional_core_or_default<T, U>(value: Option<&T>, map: impl FnOnce(&T) -> Result<U>) -> Result<U>
where
    U: Default,
{
    value.map(map).transpose().map(Option::unwrap_or_default)
}

fn assign_if_some<T>(target: &mut T, value: Option<T>) {
    if let Some(value) = value {
        *target = value;
    }
}

fn assign_if_some_map<T, U>(target: &mut T, value: Option<U>, map: impl FnOnce(U) -> T) {
    if let Some(value) = value {
        *target = map(value);
    }
}

fn option_f32(value: Option<f64>, name: &'static str) -> Result<Option<f32>> {
    value
        .map(|value| finite_f64_to_f32(value, name))
        .transpose()
}
