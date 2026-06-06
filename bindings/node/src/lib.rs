//! N-API bindings for the public CogentLM Node package.
//!
//! This crate exposes the shared Rust client facade to JavaScript while
//! translating Node request objects, async tasks, and native errors.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use cogentlm_client::{
    AnthropicProviderConfig as CoreAnthropicProviderConfig,
    CogentChatRequest as CoreClientChatRequest, CogentClient as CoreClient,
    CogentEmbedRequest as CoreClientEmbedRequest,
    CogentEmbeddingResponse as CoreClientEmbeddingResponse,
    CogentEmbeddingResponseFuture as CoreClientEmbeddingResponseFuture,
    CogentEmbeddingRun as CoreClientEmbeddingRun, CogentError as CoreClientError,
    CogentQueryRequest as CoreClientQueryRequest, CogentTextOptions as CoreClientTextOptions,
    CogentTextResponse as CoreClientTextResponse,
    CogentTextResponseFuture as CoreClientTextResponseFuture, CogentTextRun as CoreClientTextRun,
    CogentTokenBatches as CoreClientTokenBatches, EndpointDescriptor as CoreEndpointDescriptor,
    EndpointRef as CoreEndpointRef, LocalEmbedOptions as CoreClientLocalEmbedOptions,
    LocalTextOptions as CoreClientLocalTextOptions,
    OpenAiCompatibleProviderConfig as CoreOpenAiCompatibleProviderConfig,
    OpenAiProviderConfig as CoreOpenAiProviderConfig, ProviderAuthConfig as CoreProviderAuthConfig,
    ProviderEndpointConfig as CoreProviderEndpointConfig,
    ProviderEndpointError as CoreProviderEndpointError,
    ProviderEndpointErrorKind as CoreProviderEndpointErrorKind,
    ProviderSecret as CoreProviderSecret, RemoteError as CoreRemoteError,
    RemoteErrorKind as CoreRemoteErrorKind, RemoteGatewayConfig as CoreRemoteGatewayConfig,
    RemoteSecret as CoreRemoteSecret,
};
use cogentlm_core::TokenUsage as CoreTokenUsage;
use cogentlm_engine::backend::{
    backend_observability_json as core_backend_observability_json,
    set_llama_log_quiet as core_set_llama_log_quiet,
};
use cogentlm_engine::engine::protocol::{
    CacheSource as CoreCacheSource, RequestStats as CoreRequestStats,
};
use cogentlm_engine::engine::{
    ChatMessage as CoreChatMessage, ChatRole as CoreChatRole, FlashAttentionMode, GpuLayerConfig,
    KvCacheType, KvReuseMode, LogitBias, ModelPlacementConfig as CoreModelPlacementConfig,
    MultimodalRuntimeConfig as CoreMultimodalRuntimeConfig,
    NativeRuntimeConfig as CoreNativeRuntimeConfig,
    ObservabilityRuntimeConfig as CoreObservabilityRuntimeConfig, PoolingType as CorePoolingType,
    ResidencyRuntimeConfig as CoreResidencyRuntimeConfig, RopeScaling, SamplerStage,
    SamplingRuntimeConfig as CoreSamplingRuntimeConfig,
    SchedulerRuntimeConfig as CoreSchedulerRuntimeConfig, SplitMode, TokenBatch as CoreTokenBatch,
};
use cogentlm_engine::runtime::config::{
    SchedulerPolicyConfig as CoreSchedulerPolicyConfig, SchedulerPolicyMode,
};
use futures::executor::block_on;
use futures::StreamExt;
use napi::bindgen_prelude::{AsyncTask, Buffer, Either, Env};
use napi::{Error, JsValue, Result, Status, Task};
use napi_derive::napi;
use serde::de::DeserializeOwned;

#[cfg(test)]
#[path = "tests/remote_tests.rs"]
mod remote_tests;
#[cfg(test)]
#[path = "tests/stats_tests.rs"]
mod stats_tests;

type SharedCogentClient = Arc<Mutex<CoreClient>>;
type SharedClientTextResponse = Arc<Mutex<Option<CoreClientTextResponseFuture>>>;
type SharedClientEmbeddingResponse = Arc<Mutex<Option<CoreClientEmbeddingResponseFuture>>>;
type SharedClientTokenBatches = Arc<Mutex<Option<CoreClientTokenBatches>>>;
type ClientTaskOutput<T> = std::result::Result<T, CoreClientError>;

const CLIENT_MUTEX_POISONED: &str = "client mutex is poisoned";
const CLIENT_TEXT_RESPONSE_CONSUMED: &str = "text response already consumed";
const CLIENT_EMBEDDING_RESPONSE_CONSUMED: &str = "embedding response already consumed";
const CLIENT_TOKEN_BATCHES_MUTEX_POISONED: &str = "token batches mutex is poisoned";
const CLIENT_TEXT_RESPONSE_MUTEX_POISONED: &str = "text response mutex is poisoned";
const CLIENT_EMBEDDING_RESPONSE_MUTEX_POISONED: &str = "embedding response mutex is poisoned";

/// Per-token logit bias applied during sampling.
#[napi(object)]
pub struct LogitBiasConfig {
    pub token: i32,
    pub bias: f64,
}

/// Sampling controls used by local text generation.
#[napi(object)]
pub struct SamplingRuntimeConfig {
    pub samplers: Option<Vec<String>>,
    pub seed: Option<i64>,
    #[napi(js_name = "top_k")]
    pub top_k: Option<i32>,
    #[napi(js_name = "top_p")]
    pub top_p: Option<f64>,
    #[napi(js_name = "min_p")]
    pub min_p: Option<f64>,
    #[napi(js_name = "typical_p")]
    pub typical_p: Option<f64>,
    #[napi(js_name = "xtc_probability")]
    pub xtc_probability: Option<f64>,
    #[napi(js_name = "xtc_threshold")]
    pub xtc_threshold: Option<f64>,
    #[napi(js_name = "top_n_sigma")]
    pub top_n_sigma: Option<f64>,
    pub temperature: Option<f64>,
    #[napi(js_name = "dynatemp_range")]
    pub dynatemp_range: Option<f64>,
    #[napi(js_name = "dynatemp_exponent")]
    pub dynatemp_exponent: Option<f64>,
    #[napi(js_name = "repeat_last_n")]
    pub repeat_last_n: Option<i32>,
    #[napi(js_name = "repeat_penalty")]
    pub repeat_penalty: Option<f64>,
    #[napi(js_name = "frequency_penalty")]
    pub frequency_penalty: Option<f64>,
    #[napi(js_name = "presence_penalty")]
    pub presence_penalty: Option<f64>,
    #[napi(js_name = "dry_multiplier")]
    pub dry_multiplier: Option<f64>,
    #[napi(js_name = "dry_base")]
    pub dry_base: Option<f64>,
    #[napi(js_name = "dry_allowed_length")]
    pub dry_allowed_length: Option<i32>,
    #[napi(js_name = "dry_penalty_last_n")]
    pub dry_penalty_last_n: Option<i32>,
    #[napi(js_name = "dry_sequence_breakers")]
    pub dry_sequence_breakers: Option<Vec<String>>,
    pub mirostat: Option<i32>,
    #[napi(js_name = "mirostat_tau")]
    pub mirostat_tau: Option<f64>,
    #[napi(js_name = "mirostat_eta")]
    pub mirostat_eta: Option<f64>,
    #[napi(js_name = "min_keep")]
    pub min_keep: Option<i32>,
    #[napi(js_name = "n_probs")]
    pub n_probs: Option<i32>,
    #[napi(js_name = "logit_bias")]
    pub logit_bias: Option<Vec<LogitBiasConfig>>,
    #[napi(js_name = "ignore_eos")]
    pub ignore_eos: Option<bool>,
    #[napi(js_name = "grammar_lazy")]
    pub grammar_lazy: Option<bool>,
    #[napi(js_name = "preserved_tokens")]
    pub preserved_tokens: Option<Vec<i32>>,
    #[napi(js_name = "backend_sampling")]
    pub backend_sampling: Option<bool>,
}

impl SamplingRuntimeConfig {
    fn to_core(&self) -> Result<CoreSamplingRuntimeConfig> {
        if self
            .seed
            .is_some_and(|value| value < 0 || value > u32::MAX as i64)
        {
            return Err(invalid_arg("seed must fit in an unsigned 32-bit integer"));
        }
        Ok(CoreSamplingRuntimeConfig {
            samplers: self
                .samplers
                .as_ref()
                .map(|samplers| {
                    samplers
                        .iter()
                        .map(|stage| parse_sampler_stage(stage))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?
                .unwrap_or_default(),
            seed: self.seed.map(|value| value as u32),
            top_k: self.top_k,
            top_p: option_f32(self.top_p),
            min_p: option_f32(self.min_p),
            typical_p: option_f32(self.typical_p),
            xtc_probability: option_f32(self.xtc_probability),
            xtc_threshold: option_f32(self.xtc_threshold),
            top_n_sigma: option_f32(self.top_n_sigma),
            temperature: option_f32(self.temperature),
            dynatemp_range: option_f32(self.dynatemp_range),
            dynatemp_exponent: option_f32(self.dynatemp_exponent),
            repeat_last_n: self.repeat_last_n,
            repeat_penalty: option_f32(self.repeat_penalty),
            frequency_penalty: option_f32(self.frequency_penalty),
            presence_penalty: option_f32(self.presence_penalty),
            dry_multiplier: option_f32(self.dry_multiplier),
            dry_base: option_f32(self.dry_base),
            dry_allowed_length: self.dry_allowed_length,
            dry_penalty_last_n: self.dry_penalty_last_n,
            dry_sequence_breakers: self.dry_sequence_breakers.clone().unwrap_or_default(),
            mirostat: self.mirostat,
            mirostat_tau: option_f32(self.mirostat_tau),
            mirostat_eta: option_f32(self.mirostat_eta),
            min_keep: self.min_keep,
            n_probs: self.n_probs,
            logit_bias: self
                .logit_bias
                .as_ref()
                .map(|biases| {
                    biases
                        .iter()
                        .map(|bias| LogitBias {
                            token: bias.token,
                            bias: bias.bias as f32,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            ignore_eos: self.ignore_eos.unwrap_or(false),
            grammar_lazy: self.grammar_lazy.unwrap_or(false),
            preserved_tokens: self.preserved_tokens.clone().unwrap_or_default(),
            backend_sampling: self.backend_sampling.unwrap_or(true),
        })
    }
}

/// Numeric GPU layer count for model placement configuration.
#[napi(object)]
pub struct GpuLayerCountConfig {
    pub count: i32,
}

/// Device placement and memory mapping settings for local model loading.
#[napi(object)]
pub struct ModelPlacementConfig {
    pub devices: Option<Vec<String>>,
    #[napi(js_name = "gpu_layers")]
    pub gpu_layers: Option<Either<String, GpuLayerCountConfig>>,
    #[napi(js_name = "split_mode")]
    pub split_mode: Option<String>,
    #[napi(js_name = "main_gpu")]
    pub main_gpu: Option<i32>,
    #[napi(js_name = "tensor_split")]
    pub tensor_split: Option<Vec<f64>>,
    #[napi(js_name = "use_mmap")]
    pub use_mmap: Option<bool>,
    #[napi(js_name = "use_mlock")]
    pub use_mlock: Option<bool>,
    #[napi(js_name = "fit_params")]
    pub fit_params: Option<bool>,
    #[napi(js_name = "fit_params_min_ctx")]
    pub fit_params_min_ctx: Option<i32>,
    #[napi(js_name = "fit_params_target_bytes")]
    pub fit_params_target_bytes: Option<Vec<f64>>,
    #[napi(js_name = "check_tensors")]
    pub check_tensors: Option<bool>,
    #[napi(js_name = "no_extra_bufts")]
    pub no_extra_bufts: Option<bool>,
    #[napi(js_name = "no_host")]
    pub no_host: Option<bool>,
}

impl ModelPlacementConfig {
    fn to_core(&self) -> Result<CoreModelPlacementConfig> {
        let mut core = CoreModelPlacementConfig::default();
        assign_if_some(&mut core.devices, self.devices.clone());
        if let Some(value) = &self.gpu_layers {
            core.gpu_layers = match value {
                Either::A(value) => parse_gpu_layers(value)?,
                Either::B(value) => GpuLayerConfig::from_layer_count(value.count),
            };
        }
        if let Some(value) = &self.split_mode {
            core.split_mode = parse_split_mode(value)?;
        }
        core.main_gpu = self.main_gpu;
        if let Some(value) = &self.tensor_split {
            core.tensor_split = value.iter().map(|value| *value as f32).collect();
        }
        assign_if_some(&mut core.use_mmap, self.use_mmap);
        assign_if_some(&mut core.use_mlock, self.use_mlock);
        assign_if_some(&mut core.fit_params, self.fit_params);
        core.fit_params_min_ctx = self.fit_params_min_ctx;
        if let Some(value) = &self.fit_params_target_bytes {
            core.fit_params_target_bytes = value.iter().map(|value| *value as u64).collect();
        }
        assign_if_some(&mut core.check_tensors, self.check_tensors);
        assign_if_some(&mut core.no_extra_bufts, self.no_extra_bufts);
        assign_if_some(&mut core.no_host, self.no_host);
        Ok(core)
    }
}

/// Context, threading, attention, and embedding settings for local runtime use.
#[napi(object)]
pub struct ContextRuntimeConfig {
    #[napi(js_name = "n_ctx")]
    pub n_ctx: Option<i32>,
    #[napi(js_name = "n_batch")]
    pub n_batch: Option<i32>,
    #[napi(js_name = "n_ubatch")]
    pub n_ubatch: Option<i32>,
    #[napi(js_name = "n_parallel")]
    pub n_parallel: Option<i32>,
    #[napi(js_name = "n_threads")]
    pub n_threads: Option<i32>,
    #[napi(js_name = "n_threads_batch")]
    pub n_threads_batch: Option<i32>,
    #[napi(js_name = "flash_attention")]
    pub flash_attention: Option<String>,
    #[napi(js_name = "kv_unified")]
    pub kv_unified: Option<bool>,
    #[napi(js_name = "cache_type_k")]
    pub cache_type_k: Option<String>,
    #[napi(js_name = "cache_type_v")]
    pub cache_type_v: Option<String>,
    #[napi(js_name = "offload_kqv")]
    pub offload_kqv: Option<bool>,
    #[napi(js_name = "op_offload")]
    pub op_offload: Option<bool>,
    #[napi(js_name = "swa_full")]
    pub swa_full: Option<bool>,
    pub warmup: Option<bool>,
    #[napi(js_name = "rope_scaling")]
    pub rope_scaling: Option<String>,
    #[napi(js_name = "rope_freq_base")]
    pub rope_freq_base: Option<f64>,
    #[napi(js_name = "rope_freq_scale")]
    pub rope_freq_scale: Option<f64>,
    #[napi(js_name = "yarn_orig_ctx")]
    pub yarn_orig_ctx: Option<i32>,
    #[napi(js_name = "yarn_ext_factor")]
    pub yarn_ext_factor: Option<f64>,
    #[napi(js_name = "yarn_attn_factor")]
    pub yarn_attn_factor: Option<f64>,
    #[napi(js_name = "yarn_beta_fast")]
    pub yarn_beta_fast: Option<f64>,
    #[napi(js_name = "yarn_beta_slow")]
    pub yarn_beta_slow: Option<f64>,
    pub embeddings: Option<bool>,
    pub pooling: Option<PoolingType>,
}

impl ContextRuntimeConfig {
    fn to_core(&self) -> Result<cogentlm_engine::engine::ContextRuntimeConfig> {
        let mut core = cogentlm_engine::engine::ContextRuntimeConfig {
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_parallel: self.n_parallel,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            flash_attention: self
                .flash_attention
                .as_deref()
                .map(parse_flash_attention)
                .transpose()?
                .unwrap_or_default(),
            kv_unified: self.kv_unified,
            ..Default::default()
        };
        if let Some(value) = &self.cache_type_k {
            core.cache_type_k = parse_kv_cache_type(value)?;
        }
        if let Some(value) = &self.cache_type_v {
            core.cache_type_v = parse_kv_cache_type(value)?;
        }
        assign_if_some(&mut core.offload_kqv, self.offload_kqv);
        assign_if_some(&mut core.op_offload, self.op_offload);
        assign_if_some(&mut core.swa_full, self.swa_full);
        assign_if_some(&mut core.warmup, self.warmup);
        if let Some(value) = &self.rope_scaling {
            core.rope_scaling = Some(parse_rope_scaling(value)?);
        }
        core.rope_freq_base = option_f32(self.rope_freq_base);
        core.rope_freq_scale = option_f32(self.rope_freq_scale);
        core.yarn_orig_ctx = self.yarn_orig_ctx;
        core.yarn_ext_factor = option_f32(self.yarn_ext_factor);
        core.yarn_attn_factor = option_f32(self.yarn_attn_factor);
        core.yarn_beta_fast = option_f32(self.yarn_beta_fast);
        core.yarn_beta_slow = option_f32(self.yarn_beta_slow);
        core.embeddings = self.embeddings;
        core.pooling = self.pooling.map(CorePoolingType::from);
        Ok(core)
    }
}

/// Scheduler policy knobs for latency, balance, or throughput behavior.
#[napi(object)]
pub struct SchedulerPolicyConfig {
    pub mode: Option<String>,
    #[napi(js_name = "decode_token_reserve")]
    pub decode_token_reserve: Option<i32>,
    #[napi(js_name = "enable_adaptive_prefill_chunking")]
    pub enable_adaptive_prefill_chunking: Option<bool>,
}

impl SchedulerPolicyConfig {
    fn apply_to_core(&self, core: &mut CoreSchedulerPolicyConfig) -> Result<()> {
        if let Some(value) = &self.mode {
            core.mode = parse_scheduler_policy(value)?;
        }
        assign_if_some(&mut core.decode_token_reserve, self.decode_token_reserve);
        assign_if_some(
            &mut core.enable_adaptive_prefill_chunking,
            self.enable_adaptive_prefill_chunking,
        );
        Ok(())
    }
}

/// Request scheduler and continuous batching settings.
#[napi(object)]
pub struct SchedulerRuntimeConfig {
    #[napi(js_name = "continuous_batching")]
    pub continuous_batching: Option<bool>,
    pub policy: Option<SchedulerPolicyConfig>,
    #[napi(js_name = "prefill_chunk_size")]
    pub prefill_chunk_size: Option<i32>,
    #[napi(js_name = "max_running_requests")]
    pub max_running_requests: Option<i32>,
    #[napi(js_name = "max_queued_requests")]
    pub max_queued_requests: Option<i32>,
}

impl SchedulerRuntimeConfig {
    fn to_core(&self) -> Result<CoreSchedulerRuntimeConfig> {
        let mut core = CoreSchedulerRuntimeConfig::default();
        assign_if_some(&mut core.continuous_batching, self.continuous_batching);
        if let Some(value) = &self.policy {
            value.apply_to_core(&mut core.policy)?;
        }
        assign_if_some(&mut core.prefill_chunk_size, self.prefill_chunk_size);
        core.max_running_requests = self.max_running_requests;
        core.max_queued_requests = self.max_queued_requests;
        Ok(core)
    }
}

/// Prefix KV-cache reuse and snapshot settings.
#[napi(object)]
pub struct CacheRuntimeConfig {
    pub mode: Option<String>,
    #[napi(js_name = "retained_prefix_tokens")]
    pub retained_prefix_tokens: Option<i32>,
    #[napi(js_name = "snapshot_interval_tokens")]
    pub snapshot_interval_tokens: Option<i32>,
    #[napi(js_name = "max_snapshot_entries")]
    pub max_snapshot_entries: Option<i32>,
    #[napi(js_name = "max_snapshot_bytes")]
    pub max_snapshot_bytes: Option<f64>,
}

impl CacheRuntimeConfig {
    fn to_core(&self) -> Result<cogentlm_engine::engine::CacheRuntimeConfig> {
        let mut core = cogentlm_engine::engine::CacheRuntimeConfig::default();
        if let Some(value) = &self.mode {
            core.mode = parse_kv_reuse_mode(value)?;
        }
        assign_if_some(
            &mut core.retained_prefix_tokens,
            self.retained_prefix_tokens,
        );
        assign_if_some(
            &mut core.snapshot_interval_tokens,
            self.snapshot_interval_tokens,
        );
        assign_if_some(&mut core.max_snapshot_entries, self.max_snapshot_entries);
        assign_if_some_map(
            &mut core.max_snapshot_bytes,
            self.max_snapshot_bytes,
            |value| value as usize,
        );
        Ok(core)
    }
}

/// Vision projector and image-token settings for multimodal models.
#[napi(object)]
pub struct MultimodalRuntimeConfig {
    #[napi(js_name = "projector_path")]
    pub projector_path: Option<String>,
    #[napi(js_name = "use_gpu")]
    pub use_gpu: Option<bool>,
    #[napi(js_name = "image_min_tokens")]
    pub image_min_tokens: Option<i32>,
    #[napi(js_name = "image_max_tokens")]
    pub image_max_tokens: Option<i32>,
}

impl MultimodalRuntimeConfig {
    fn to_core(&self) -> CoreMultimodalRuntimeConfig {
        CoreMultimodalRuntimeConfig {
            projector_path: self.projector_path.clone(),
            use_gpu: self.use_gpu,
            image_min_tokens: self.image_min_tokens,
            image_max_tokens: self.image_max_tokens,
        }
    }
}

/// GPU residency limits for concurrently loaded local models.
#[napi(object)]
pub struct ResidencyRuntimeConfig {
    #[napi(js_name = "max_gpu_models_per_device")]
    pub max_gpu_models_per_device: Option<f64>,
    #[napi(js_name = "allow_cpu_models_while_gpu_loaded")]
    pub allow_cpu_models_while_gpu_loaded: Option<bool>,
    #[napi(js_name = "require_gpu_lease")]
    pub require_gpu_lease: Option<bool>,
    #[napi(js_name = "gpu_memory_safety_margin_bytes")]
    pub gpu_memory_safety_margin_bytes: Option<f64>,
}

impl ResidencyRuntimeConfig {
    fn to_core(&self) -> CoreResidencyRuntimeConfig {
        let mut core = CoreResidencyRuntimeConfig::default();
        assign_if_some_map(
            &mut core.max_gpu_models_per_device,
            self.max_gpu_models_per_device,
            |value| value as usize,
        );
        assign_if_some(
            &mut core.allow_cpu_models_while_gpu_loaded,
            self.allow_cpu_models_while_gpu_loaded,
        );
        assign_if_some(&mut core.require_gpu_lease, self.require_gpu_lease);
        assign_if_some_map(
            &mut core.gpu_memory_safety_margin_bytes,
            self.gpu_memory_safety_margin_bytes,
            |value| value as u64,
        );
        core
    }
}

/// Runtime metrics and backend profiling options.
#[napi(object)]
pub struct ObservabilityRuntimeConfig {
    #[napi(js_name = "runtime_metrics")]
    pub runtime_metrics: Option<bool>,
    #[napi(js_name = "backend_profiling")]
    pub backend_profiling: Option<bool>,
}

impl ObservabilityRuntimeConfig {
    fn to_core(&self) -> CoreObservabilityRuntimeConfig {
        CoreObservabilityRuntimeConfig {
            runtime_metrics: self.runtime_metrics.unwrap_or(false),
            backend_profiling: self.backend_profiling.unwrap_or(false),
        }
    }
}

/// Complete native runtime configuration for local model loading.
#[napi(object)]
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

impl NativeRuntimeConfig {
    fn to_core(&self) -> Result<CoreNativeRuntimeConfig> {
        Ok(CoreNativeRuntimeConfig {
            placement: optional_core_or_default(
                self.placement.as_ref(),
                ModelPlacementConfig::to_core,
            )?,
            context: optional_core_or_default(
                self.context.as_ref(),
                ContextRuntimeConfig::to_core,
            )?,
            sampling: optional_core_or_default(
                self.sampling.as_ref(),
                SamplingRuntimeConfig::to_core,
            )?,
            scheduler: optional_core_or_default(
                self.scheduler.as_ref(),
                SchedulerRuntimeConfig::to_core,
            )?,
            cache: optional_core_or_default(self.cache.as_ref(), CacheRuntimeConfig::to_core)?,
            multimodal: optional_core_or_default(self.multimodal.as_ref(), |value| {
                Ok(value.to_core())
            })?,
            residency: optional_core_or_default(self.residency.as_ref(), |value| {
                Ok(value.to_core())
            })?,
            observability: optional_core_or_default(self.observability.as_ref(), |value| {
                Ok(value.to_core())
            })?,
        })
    }
}

/// Address of a registered local, remote gateway, or direct provider endpoint.
#[napi(object)]
pub struct EndpointRef {
    pub kind: String,
    pub id: String,
}

impl EndpointRef {
    fn to_core(&self) -> Result<CoreEndpointRef> {
        match self.kind.as_str() {
            "local" => Ok(CoreEndpointRef::Local {
                id: self.id.clone(),
            }),
            "remote" => Ok(CoreEndpointRef::Remote {
                id: self.id.clone(),
            }),
            "provider" => Ok(CoreEndpointRef::Provider {
                id: self.id.clone(),
            }),
            _ => Err(invalid_arg(
                "endpoint kind must be local, remote, or provider",
            )),
        }
    }
}

/// Shared generation options for text-producing requests.
#[napi(object)]
pub struct CogentTextOptions {
    #[napi(js_name = "maxTokens")]
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    #[napi(js_name = "topP")]
    pub top_p: Option<f64>,
    pub stop: Option<Vec<String>>,
}

impl CogentTextOptions {
    fn to_core(&self) -> Result<CoreClientTextOptions> {
        Ok(CoreClientTextOptions {
            max_tokens: self.max_tokens,
            temperature: self
                .temperature
                .map(|value| finite_f64_to_f32(value, "temperature"))
                .transpose()?,
            top_p: self
                .top_p
                .map(|value| finite_f64_to_f32(value, "topP"))
                .transpose()?,
            stop: self.stop.clone().unwrap_or_default(),
        })
    }
}

/// Local-only prompt options such as grammar constraints and image inputs.
#[napi(object)]
pub struct LocalTextOptions {
    #[napi(js_name = "contextKey")]
    pub context_key: Option<String>,
    pub grammar: Option<String>,
    #[napi(js_name = "jsonSchema")]
    pub json_schema: Option<String>,
    pub sampling: Option<SamplingRuntimeConfig>,
    pub media: Option<Vec<Buffer>>,
}

impl LocalTextOptions {
    fn to_core(&self) -> Result<CoreClientLocalTextOptions> {
        Ok(CoreClientLocalTextOptions {
            context_key: self.context_key.clone(),
            grammar: self.grammar.clone(),
            json_schema: self.json_schema.clone(),
            sampling: self
                .sampling
                .as_ref()
                .map(SamplingRuntimeConfig::to_core)
                .transpose()?,
            media: self
                .media
                .as_ref()
                .map(|buffers| {
                    buffers
                        .iter()
                        .map(|buffer| buffer.as_ref().to_vec())
                        .collect()
                })
                .unwrap_or_default(),
        })
    }
}

/// Local-only embedding options for context and vector normalization.
#[napi(object)]
pub struct LocalEmbedOptions {
    #[napi(js_name = "contextKey")]
    pub context_key: Option<String>,
    pub normalize: Option<bool>,
}

impl LocalEmbedOptions {
    fn to_core(&self) -> CoreClientLocalEmbedOptions {
        CoreClientLocalEmbedOptions {
            context_key: self.context_key.clone(),
            normalize: self.normalize,
        }
    }
}

/// Prompt completion request routed to a local endpoint or remote gateway.
#[napi(object)]
pub struct CogentQueryRequest {
    pub endpoint: Option<EndpointRef>,
    pub prompt: String,
    pub options: Option<CogentTextOptions>,
    pub local: Option<LocalTextOptions>,
    #[napi(js_name = "gatewayOptions")]
    pub gateway_options: Option<serde_json::Value>,
    #[napi(js_name = "providerOptions")]
    pub provider_options: Option<serde_json::Value>,
    #[napi(js_name = "emitTokens")]
    pub emit_tokens: Option<bool>,
}

impl CogentQueryRequest {
    fn to_core(&self) -> Result<CoreClientQueryRequest> {
        Ok(CoreClientQueryRequest {
            endpoint: optional_endpoint(self.endpoint.as_ref())?,
            prompt: self.prompt.clone(),
            options: optional_core_or_default(self.options.as_ref(), CogentTextOptions::to_core)?,
            local: optional_core_or_default(self.local.as_ref(), LocalTextOptions::to_core)?,
            gateway_options: gateway_options_or_empty(self.gateway_options.clone())?,
            provider_options: provider_options_or_empty(self.provider_options.clone())?,
            emit_tokens: self.emit_tokens.unwrap_or(false),
        })
    }
}

/// Chat completion request routed to a local endpoint or remote gateway.
#[napi(object)]
pub struct CogentChatRequest {
    pub endpoint: Option<EndpointRef>,
    pub messages: Vec<ChatMessage>,
    pub options: Option<CogentTextOptions>,
    pub local: Option<LocalTextOptions>,
    #[napi(js_name = "gatewayOptions")]
    pub gateway_options: Option<serde_json::Value>,
    #[napi(js_name = "providerOptions")]
    pub provider_options: Option<serde_json::Value>,
    #[napi(js_name = "emitTokens")]
    pub emit_tokens: Option<bool>,
}

impl CogentChatRequest {
    fn to_core(&self) -> Result<CoreClientChatRequest> {
        Ok(CoreClientChatRequest {
            endpoint: optional_endpoint(self.endpoint.as_ref())?,
            messages: chat_messages_to_core(self.messages.clone())?,
            options: optional_core_or_default(self.options.as_ref(), CogentTextOptions::to_core)?,
            local: optional_core_or_default(self.local.as_ref(), LocalTextOptions::to_core)?,
            gateway_options: gateway_options_or_empty(self.gateway_options.clone())?,
            provider_options: provider_options_or_empty(self.provider_options.clone())?,
            emit_tokens: self.emit_tokens.unwrap_or(false),
        })
    }
}

/// Embedding request routed to a local endpoint or remote gateway.
#[napi(object)]
pub struct CogentEmbedRequest {
    pub endpoint: Option<EndpointRef>,
    pub input: String,
    pub local: Option<LocalEmbedOptions>,
    #[napi(js_name = "gatewayOptions")]
    pub gateway_options: Option<serde_json::Value>,
    #[napi(js_name = "providerOptions")]
    pub provider_options: Option<serde_json::Value>,
}

impl CogentEmbedRequest {
    fn to_core(&self) -> Result<CoreClientEmbedRequest> {
        Ok(CoreClientEmbedRequest {
            endpoint: optional_endpoint(self.endpoint.as_ref())?,
            input: self.input.clone(),
            local: self
                .local
                .as_ref()
                .map(LocalEmbedOptions::to_core)
                .unwrap_or_default(),
            gateway_options: gateway_options_or_empty(self.gateway_options.clone())?,
            provider_options: provider_options_or_empty(self.provider_options.clone())?,
        })
    }
}

/// Role/content chat message accepted by local and remote chat requests.
#[napi(object)]
#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Remote CogentLM gateway alias, URL, token, and optional timeout.
#[napi(object)]
pub struct RemoteGatewayConfig {
    pub alias: String,
    #[napi(js_name = "baseUrl")]
    pub base_url: String,
    pub token: String,
    #[napi(js_name = "timeoutMs")]
    pub timeout_ms: Option<u32>,
}

impl RemoteGatewayConfig {
    fn to_core(&self) -> CoreRemoteGatewayConfig {
        let timeout = self
            .timeout_ms
            .map(|timeout_ms| Duration::from_millis(u64::from(timeout_ms)));
        CoreRemoteGatewayConfig {
            alias: self.alias.clone(),
            base_url: self.base_url.clone(),
            token: CoreRemoteSecret::new(self.token.clone()),
            timeout,
        }
    }
}

/// Static header entry for an OpenAI-compatible provider descriptor.
#[napi(object)]
pub struct ProviderStaticHeader {
    pub name: String,
    pub value: String,
}

/// Local, gateway, or direct provider endpoint descriptor accepted by add.
#[napi(object)]
pub struct EndpointDescriptor {
    pub kind: String,
    #[napi(js_name = "modelPath")]
    pub model_path: Option<String>,
    pub config: Option<NativeRuntimeConfig>,
    pub alias: Option<String>,
    #[napi(js_name = "baseUrl")]
    pub base_url: Option<String>,
    pub token: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    #[napi(js_name = "apiKey")]
    pub api_key: Option<String>,
    #[napi(js_name = "timeoutMs")]
    pub timeout_ms: Option<u32>,
    pub version: Option<String>,
    #[napi(js_name = "authHeaderName")]
    pub auth_header_name: Option<String>,
    #[napi(js_name = "authHeaderValue")]
    pub auth_header_value: Option<String>,
    #[napi(js_name = "staticHeaders")]
    pub static_headers: Option<Vec<ProviderStaticHeader>>,
}

impl EndpointDescriptor {
    fn to_core(&self) -> Result<CoreEndpointDescriptor> {
        match self.kind.as_str() {
            "local" => self.local_to_core(),
            "gateway" | "remote" => self.gateway_to_core(),
            "provider" => self.provider_to_core(),
            _ => Err(invalid_arg(
                "endpoint descriptor kind must be local, gateway, remote, or provider",
            )),
        }
    }

    fn local_to_core(&self) -> Result<CoreEndpointDescriptor> {
        reject_endpoint_descriptor_fields(
            &[
                ("alias", self.alias.is_some()),
                ("baseUrl", self.base_url.is_some()),
                ("token", self.token.is_some()),
                ("provider", self.provider.is_some()),
                ("model", self.model.is_some()),
                ("apiKey", self.api_key.is_some()),
                ("timeoutMs", self.timeout_ms.is_some()),
                ("version", self.version.is_some()),
                ("authHeaderName", self.auth_header_name.is_some()),
                ("authHeaderValue", self.auth_header_value.is_some()),
                ("staticHeaders", self.static_headers.is_some()),
            ],
            "local",
        )?;
        let model_path = self
            .model_path
            .clone()
            .ok_or_else(|| invalid_arg("local descriptor modelPath is required"))?;
        let config = self
            .config
            .as_ref()
            .map(NativeRuntimeConfig::to_core)
            .transpose()?
            .unwrap_or_default();
        Ok(CoreEndpointDescriptor::local(model_path, config))
    }

    fn gateway_to_core(&self) -> Result<CoreEndpointDescriptor> {
        reject_endpoint_descriptor_fields(
            &[
                ("modelPath", self.model_path.is_some()),
                ("config", self.config.is_some()),
                ("provider", self.provider.is_some()),
                ("model", self.model.is_some()),
                ("apiKey", self.api_key.is_some()),
                ("version", self.version.is_some()),
                ("authHeaderName", self.auth_header_name.is_some()),
                ("authHeaderValue", self.auth_header_value.is_some()),
                ("staticHeaders", self.static_headers.is_some()),
            ],
            "gateway",
        )?;
        Ok(CoreEndpointDescriptor::gateway(CoreRemoteGatewayConfig {
            alias: required_descriptor_string(self.alias.as_ref(), "gateway alias")?,
            base_url: required_descriptor_string(self.base_url.as_ref(), "gateway baseUrl")?,
            token: CoreRemoteSecret::new(required_descriptor_string(
                self.token.as_ref(),
                "gateway token",
            )?),
            timeout: endpoint_timeout(self.timeout_ms)?,
        }))
    }

    fn provider_to_core(&self) -> Result<CoreEndpointDescriptor> {
        reject_endpoint_descriptor_fields(
            &[
                ("modelPath", self.model_path.is_some()),
                ("config", self.config.is_some()),
                ("alias", self.alias.is_some()),
                ("token", self.token.is_some()),
            ],
            "provider",
        )?;
        let model = required_descriptor_string(self.model.as_ref(), "provider model")?;
        let provider = required_descriptor_string(self.provider.as_ref(), "provider")?;
        let timeout = endpoint_timeout(self.timeout_ms)?;
        let config = match provider.as_str() {
            "openai" => {
                reject_endpoint_descriptor_fields(
                    &[
                        ("version", self.version.is_some()),
                        ("authHeaderName", self.auth_header_name.is_some()),
                        ("authHeaderValue", self.auth_header_value.is_some()),
                        ("staticHeaders", self.static_headers.is_some()),
                    ],
                    "OpenAI provider",
                )?;
                CoreProviderEndpointConfig::OpenAi(CoreOpenAiProviderConfig {
                    model,
                    api_key: provider_secret(self.api_key.as_ref(), "provider apiKey")?,
                    base_url: self.base_url.clone(),
                    timeout,
                })
            }
            "anthropic" => {
                reject_endpoint_descriptor_fields(
                    &[
                        ("authHeaderName", self.auth_header_name.is_some()),
                        ("authHeaderValue", self.auth_header_value.is_some()),
                        ("staticHeaders", self.static_headers.is_some()),
                    ],
                    "Anthropic provider",
                )?;
                CoreProviderEndpointConfig::Anthropic(CoreAnthropicProviderConfig {
                    model,
                    api_key: provider_secret(self.api_key.as_ref(), "provider apiKey")?,
                    base_url: self.base_url.clone(),
                    version: self.version.clone(),
                    timeout,
                })
            }
            "openai_compatible" | "openai-compatible" => {
                reject_endpoint_descriptor_fields(
                    &[("version", self.version.is_some())],
                    "OpenAI-compatible provider",
                )?;
                CoreProviderEndpointConfig::OpenAiCompatible(CoreOpenAiCompatibleProviderConfig {
                    model,
                    base_url: required_descriptor_string(
                        self.base_url.as_ref(),
                        "provider baseUrl",
                    )?,
                    auth: provider_auth(self)?,
                    static_headers: self
                        .static_headers
                        .as_ref()
                        .map(|headers| {
                            headers
                                .iter()
                                .map(|header| {
                                    (
                                        header.name.clone(),
                                        CoreProviderSecret::new(header.value.clone()),
                                    )
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                    timeout,
                })
            }
            _ => {
                return Err(invalid_arg(
                    "provider must be one of: openai, anthropic, openai_compatible",
                ));
            }
        };
        Ok(CoreEndpointDescriptor::provider(config))
    }
}

fn reject_endpoint_descriptor_fields(
    fields: &[(&'static str, bool)],
    kind: &'static str,
) -> Result<()> {
    for (field, present) in fields {
        if *present {
            return Err(invalid_arg(format!(
                "{field} is not valid for {kind} endpoint descriptors"
            )));
        }
    }
    Ok(())
}

fn required_descriptor_string(value: Option<&String>, name: &'static str) -> Result<String> {
    value
        .cloned()
        .ok_or_else(|| invalid_arg(format!("{name} is required")))
}

fn endpoint_timeout(timeout_ms: Option<u32>) -> Result<Option<Duration>> {
    match timeout_ms {
        Some(0) => Err(invalid_arg("timeoutMs must be a positive integer")),
        Some(timeout_ms) => Ok(Some(Duration::from_millis(u64::from(timeout_ms)))),
        None => Ok(None),
    }
}

fn provider_secret(value: Option<&String>, name: &'static str) -> Result<CoreProviderSecret> {
    required_descriptor_string(value, name).map(CoreProviderSecret::new)
}

fn provider_auth(descriptor: &EndpointDescriptor) -> Result<CoreProviderAuthConfig> {
    match (
        descriptor.api_key.as_ref(),
        descriptor.auth_header_name.as_ref(),
        descriptor.auth_header_value.as_ref(),
    ) {
        (Some(key), None, None) => Ok(CoreProviderAuthConfig::Bearer(CoreProviderSecret::new(
            key.clone(),
        ))),
        (None, Some(name), Some(value)) => Ok(CoreProviderAuthConfig::Header {
            name: name.clone(),
            value: CoreProviderSecret::new(value.clone()),
        }),
        _ => Err(invalid_arg(
            "OpenAI-compatible provider requires either apiKey or authHeaderName/authHeaderValue",
        )),
    }
}

/// Token counts reported by local or remote inference backends.
#[napi(object)]
pub struct TokenUsage {
    #[napi(js_name = "inputTokens")]
    pub input_tokens: Option<u32>,
    #[napi(js_name = "outputTokens")]
    pub output_tokens: Option<u32>,
    #[napi(js_name = "totalTokens")]
    pub total_tokens: Option<u32>,
}

/// Local runtime timing and cache statistics for a completed request.
#[napi(object)]
pub struct RequestStats {
    #[napi(js_name = "inputTokens")]
    pub input_tokens: i32,
    #[napi(js_name = "outputTokens")]
    pub output_tokens: i32,
    #[napi(js_name = "cacheMode")]
    pub cache_mode: String,
    #[napi(js_name = "cacheSource")]
    pub cache_source: String,
    #[napi(js_name = "cacheHits")]
    pub cache_hits: i32,
    #[napi(js_name = "prefillTokens")]
    pub prefill_tokens: i32,
    #[napi(js_name = "ttftMs")]
    pub ttft_ms: Option<f64>,
    #[napi(js_name = "interTokenMs")]
    pub inter_token_ms: Option<f64>,
    #[napi(js_name = "e2eMs")]
    pub e2e_ms: Option<f64>,
    #[napi(js_name = "e2eTokensPerSecond")]
    pub e2e_tokens_per_second: Option<f64>,
    #[napi(js_name = "decodeTokensPerSecond")]
    pub decode_tokens_per_second: Option<f64>,
    #[napi(js_name = "prefillTokensPerSecond")]
    pub prefill_tokens_per_second: Option<f64>,
    #[napi(js_name = "prefillMs")]
    pub prefill_ms: f64,
    #[napi(js_name = "decodeMs")]
    pub decode_ms: f64,
}

/// Embedding pooling strategy used by the local runtime.
#[napi(string_enum = "snake_case")]
#[derive(Clone, Copy)]
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

impl From<CorePoolingType> for PoolingType {
    fn from(value: CorePoolingType) -> Self {
        match value {
            CorePoolingType::Unspecified => Self::Unspecified,
            CorePoolingType::None => Self::None,
            CorePoolingType::Mean => Self::Mean,
            CorePoolingType::Cls => Self::Cls,
            CorePoolingType::Last => Self::Last,
            CorePoolingType::Rank => Self::Rank,
        }
    }
}

/// Final text response from a query or chat request.
#[napi(object)]
pub struct CogentTextResponse {
    pub endpoint: EndpointRef,
    pub text: String,
    #[napi(js_name = "finishReason")]
    pub finish_reason: String,
    pub usage: Option<TokenUsage>,
    #[napi(js_name = "localStats")]
    pub local_stats: Option<RequestStats>,
}

/// Final vector response from an embedding request.
#[napi(object)]
pub struct CogentEmbeddingResponse {
    pub endpoint: EndpointRef,
    pub values: Vec<f64>,
    pub usage: Option<TokenUsage>,
    #[napi(js_name = "localStats")]
    pub local_stats: Option<RequestStats>,
    pub pooling: Option<PoolingType>,
    pub normalized: Option<bool>,
}

fn endpoint_ref_to_node(endpoint: CoreEndpointRef) -> EndpointRef {
    match endpoint {
        CoreEndpointRef::Local { id } => EndpointRef {
            kind: "local".to_string(),
            id,
        },
        CoreEndpointRef::Remote { id } => EndpointRef {
            kind: "remote".to_string(),
            id,
        },
        CoreEndpointRef::Provider { id } => EndpointRef {
            kind: "provider".to_string(),
            id,
        },
    }
}

fn token_usage_to_node(usage: CoreTokenUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
    }
}

fn cogent_text_response_to_node(response: CoreClientTextResponse) -> CogentTextResponse {
    CogentTextResponse {
        endpoint: endpoint_ref_to_node(response.endpoint),
        text: response.text,
        finish_reason: response.finish_reason.as_str().to_string(),
        usage: response.usage.map(token_usage_to_node),
        local_stats: response.local_stats.map(request_stats_to_node),
    }
}

fn cogent_embedding_response_to_node(
    response: CoreClientEmbeddingResponse,
) -> CogentEmbeddingResponse {
    CogentEmbeddingResponse {
        endpoint: endpoint_ref_to_node(response.endpoint),
        values: response.values.into_iter().map(f64::from).collect(),
        usage: response.usage.map(token_usage_to_node),
        local_stats: response.local_stats.map(request_stats_to_node),
        pooling: response.pooling.map(PoolingType::from),
        normalized: response.normalized,
    }
}

/// Aggregate transport statistics for emitted token batches.
#[napi(object)]
#[derive(Clone)]
pub struct TokenEmissionStats {
    pub frames_sent: f64,
    pub bytes_sent: f64,
    pub batches_sent: f64,
}

/// Streaming token payload emitted by an active text generation run.
#[napi(object)]
#[derive(Clone)]
pub struct TokenBatch {
    pub request_id: String,
    pub stream_id: u32,
    pub sequence_start: u32,
    pub text: String,
    pub frame_count: u32,
    pub byte_count: u32,
    pub stats: TokenEmissionStats,
}

/// Client facade for local CogentLM models and remote gateway aliases.
#[napi(js_name = "CogentClient")]
pub struct CogentClient {
    inner: SharedCogentClient,
}

#[napi]
impl CogentClient {
    #[napi(constructor)]
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(CoreClient::new())),
        })
    }

    /// Register or replace an endpoint and return its current reference.
    #[napi(ts_return_type = "Promise<EndpointRef>")]
    pub fn add(
        &self,
        id: String,
        descriptor: EndpointDescriptor,
    ) -> Result<AsyncTask<ClientAddTask>> {
        Ok(AsyncTask::new(ClientAddTask {
            client: self.inner.clone(),
            id,
            descriptor: descriptor.to_core()?,
        }))
    }

    #[napi(ts_return_type = "CogentTextRun")]
    pub fn query(&self, request: CogentQueryRequest) -> Result<CogentTextRun> {
        let request = request.to_core()?;
        let run = self
            .inner
            .lock()
            .map_err(|_| napi_error(CLIENT_MUTEX_POISONED))?
            .query(request);
        Ok(CogentTextRun::from_core(run))
    }

    #[napi(ts_return_type = "CogentTextRun")]
    pub fn chat(&self, request: CogentChatRequest) -> Result<CogentTextRun> {
        let request = request.to_core()?;
        let run = self
            .inner
            .lock()
            .map_err(|_| napi_error(CLIENT_MUTEX_POISONED))?
            .chat(request);
        Ok(CogentTextRun::from_core(run))
    }

    #[napi(ts_return_type = "CogentEmbeddingRun")]
    pub fn embed(&self, request: CogentEmbedRequest) -> Result<CogentEmbeddingRun> {
        let request = request.to_core()?;
        let run = self
            .inner
            .lock()
            .map_err(|_| napi_error(CLIENT_MUTEX_POISONED))?
            .embed(request);
        Ok(CogentEmbeddingRun::from_core(run))
    }
}

/// Text generation handle with a final response and optional token stream.
#[napi(js_name = "CogentTextRun")]
pub struct CogentTextRun {
    response: SharedClientTextResponse,
    tokens: SharedClientTokenBatches,
}

impl CogentTextRun {
    fn from_core(run: CoreClientTextRun) -> Self {
        let (tokens, response) = run.into_parts();
        Self {
            response: Arc::new(Mutex::new(Some(response))),
            tokens: Arc::new(Mutex::new(Some(tokens))),
        }
    }
}

#[napi]
impl CogentTextRun {
    #[napi(js_name = "__response", ts_return_type = "Promise<CogentTextResponse>")]
    pub fn response(&self) -> AsyncTask<ClientTextResultTask> {
        AsyncTask::new(ClientTextResultTask {
            response: self.response.clone(),
        })
    }

    #[napi(js_name = "__nextToken", ts_return_type = "Promise<TokenBatch | null>")]
    pub fn next_token(&self) -> AsyncTask<ClientNextTokenTask> {
        AsyncTask::new(ClientNextTokenTask {
            tokens: self.tokens.clone(),
        })
    }
}

/// Embedding request handle with a final embedding response.
#[napi(js_name = "CogentEmbeddingRun")]
pub struct CogentEmbeddingRun {
    response: SharedClientEmbeddingResponse,
}

impl CogentEmbeddingRun {
    fn from_core(run: CoreClientEmbeddingRun) -> Self {
        Self {
            response: Arc::new(Mutex::new(Some(run.into_response()))),
        }
    }
}

#[napi]
impl CogentEmbeddingRun {
    #[napi(
        js_name = "__response",
        ts_return_type = "Promise<CogentEmbeddingResponse>"
    )]
    pub fn response(&self) -> AsyncTask<ClientEmbeddingResultTask> {
        AsyncTask::new(ClientEmbeddingResultTask {
            response: self.response.clone(),
        })
    }
}

pub struct ClientAddTask {
    client: SharedCogentClient,
    id: String,
    descriptor: CoreEndpointDescriptor,
}

impl Task for ClientAddTask {
    type Output = ClientTaskOutput<CoreEndpointRef>;
    type JsValue = EndpointRef;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut client = self
            .client
            .lock()
            .map_err(|_| napi_error(CLIENT_MUTEX_POISONED))?;
        Ok(block_on(
            client.add(self.id.clone(), self.descriptor.clone()),
        ))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> Result<Self::JsValue> {
        output
            .map(endpoint_ref_to_node)
            .map_err(|error| client_error_to_node(env, error))
    }
}

pub struct ClientTextResultTask {
    response: SharedClientTextResponse,
}

impl Task for ClientTextResultTask {
    type Output = ClientTaskOutput<CoreClientTextResponse>;
    type JsValue = CogentTextResponse;

    fn compute(&mut self) -> Result<Self::Output> {
        let response = self
            .response
            .lock()
            .map_err(|_| napi_error(CLIENT_TEXT_RESPONSE_MUTEX_POISONED))?
            .take()
            .ok_or_else(|| napi_error(CLIENT_TEXT_RESPONSE_CONSUMED))?;
        Ok(block_on(response))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> Result<Self::JsValue> {
        output
            .map(cogent_text_response_to_node)
            .map_err(|error| client_error_to_node(env, error))
    }
}

pub struct ClientEmbeddingResultTask {
    response: SharedClientEmbeddingResponse,
}

impl Task for ClientEmbeddingResultTask {
    type Output = ClientTaskOutput<CoreClientEmbeddingResponse>;
    type JsValue = CogentEmbeddingResponse;

    fn compute(&mut self) -> Result<Self::Output> {
        let response = self
            .response
            .lock()
            .map_err(|_| napi_error(CLIENT_EMBEDDING_RESPONSE_MUTEX_POISONED))?
            .take()
            .ok_or_else(|| napi_error(CLIENT_EMBEDDING_RESPONSE_CONSUMED))?;
        Ok(block_on(response))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> Result<Self::JsValue> {
        output
            .map(cogent_embedding_response_to_node)
            .map_err(|error| client_error_to_node(env, error))
    }
}

pub struct ClientNextTokenTask {
    tokens: SharedClientTokenBatches,
}

impl Task for ClientNextTokenTask {
    type Output = Option<CoreTokenBatch>;
    type JsValue = Option<TokenBatch>;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut guard = self
            .tokens
            .lock()
            .map_err(|_| napi_error(CLIENT_TOKEN_BATCHES_MUTEX_POISONED))?;
        let Some(stream) = guard.as_mut() else {
            return Ok(None);
        };
        let next = block_on(stream.next());
        if next.is_none() {
            *guard = None;
        }
        Ok(next)
    }

    fn resolve(&mut self, _env: Env, batch: Self::Output) -> Result<Self::JsValue> {
        Ok(batch.map(token_batch_to_node))
    }
}

/// Return JSON backend and device observability from the native runtime.
#[napi]
pub fn backend_observability_json(include_details: Option<bool>) -> Result<String> {
    core_backend_observability_json(include_details.unwrap_or(true)).map_err(core_error)
}

/// Enable or suppress llama.cpp native logging.
#[napi]
pub fn set_llama_log_quiet(quiet: bool) {
    core_set_llama_log_quiet(quiet);
}

fn gateway_options_or_empty(
    value: Option<serde_json::Value>,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    json_options_or_empty(value, "gatewayOptions")
}

fn provider_options_or_empty(
    value: Option<serde_json::Value>,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    json_options_or_empty(value, "providerOptions")
}

fn json_options_or_empty(
    value: Option<serde_json::Value>,
    name: &'static str,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    match value {
        Some(serde_json::Value::Object(options)) => Ok(options),
        Some(_) => Err(napi_error(format!("{name} must be a JSON object"))),
        None => Ok(serde_json::Map::new()),
    }
}

fn optional_endpoint(endpoint: Option<&EndpointRef>) -> Result<Option<CoreEndpointRef>> {
    endpoint.map(EndpointRef::to_core).transpose()
}

fn finite_f64_to_f32(value: f64, name: &'static str) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(invalid_arg(format!("{name} must be a finite f32")));
    }
    Ok(value as f32)
}

fn chat_messages_to_core(messages: Vec<ChatMessage>) -> Result<Vec<CoreChatMessage>> {
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

fn token_batch_to_node(batch: CoreTokenBatch) -> TokenBatch {
    TokenBatch {
        request_id: batch.request_id,
        stream_id: batch.stream_id,
        sequence_start: batch.sequence_start,
        text: batch.text,
        frame_count: batch.frame_count,
        byte_count: batch.byte_count,
        stats: TokenEmissionStats {
            frames_sent: batch.stats.frames_sent as f64,
            bytes_sent: batch.stats.bytes_sent as f64,
            batches_sent: batch.stats.batches_sent as f64,
        },
    }
}

fn request_stats_to_node(stats: CoreRequestStats) -> RequestStats {
    RequestStats {
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

fn option_f32(value: Option<f64>) -> Option<f32> {
    value.map(|value| value as f32)
}

fn invalid_arg(message: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, message.into())
}

fn napi_error(message: impl ToString) -> Error {
    Error::new(Status::GenericFailure, message)
}

fn remote_error_message(error: &CoreRemoteError) -> String {
    format!(
        "remote gateway error ({}): {}",
        error.kind.as_str(),
        error.message
    )
}

fn remote_error_status(kind: CoreRemoteErrorKind) -> Status {
    match kind {
        CoreRemoteErrorKind::InvalidRequest | CoreRemoteErrorKind::UnsupportedFeature => {
            Status::InvalidArg
        }
        _ => Status::GenericFailure,
    }
}

fn provider_error_message(error: &CoreProviderEndpointError) -> String {
    format!(
        "provider error ({} {}): {}",
        error.kind.as_str(),
        error.provider,
        error.message
    )
}

fn provider_error_status(kind: CoreProviderEndpointErrorKind) -> Status {
    match kind {
        CoreProviderEndpointErrorKind::InvalidRequest
        | CoreProviderEndpointErrorKind::UnsupportedFeature => Status::InvalidArg,
        _ => Status::GenericFailure,
    }
}

fn remote_error_to_node(env: Env, error: CoreRemoteError) -> Error {
    match remote_error_to_node_result(env, error) {
        Ok(error) => error,
        Err(error) => error,
    }
}

fn remote_error_to_node_result(env: Env, error: CoreRemoteError) -> Result<Error> {
    let status = remote_error_status(error.kind);
    let message = remote_error_message(&error);
    let mut object = env.create_error(Error::new(status, message))?;
    let retry_after_ms = error
        .retry_after
        .map(|duration| duration.as_secs_f64() * 1000.0);
    let raw_body = error
        .raw
        .map(|value| *value)
        .unwrap_or(serde_json::Value::Null);

    object.set("name", "RemoteError")?;
    object.set("kind", error.kind.as_str())?;
    object.set("status", error.status)?;
    object.set("code", error.code)?;
    object.set("requestId", error.request_id)?;
    object.set("retryAfterMs", retry_after_ms)?;
    object.set("rawBody", raw_body)?;

    Ok(Error::from(object.to_unknown()))
}

fn provider_error_to_node(env: Env, error: CoreProviderEndpointError) -> Error {
    match provider_error_to_node_result(env, error) {
        Ok(error) => error,
        Err(error) => error,
    }
}

fn provider_error_to_node_result(env: Env, error: CoreProviderEndpointError) -> Result<Error> {
    let status = provider_error_status(error.kind);
    let message = provider_error_message(&error);
    let mut object = env.create_error(Error::new(status, message))?;
    let retry_after_ms = error
        .retry_after
        .map(|duration| duration.as_secs_f64() * 1000.0);
    let raw_body = error
        .raw
        .map(|value| *value)
        .unwrap_or(serde_json::Value::Null);

    object.set("name", "ProviderError")?;
    object.set("kind", error.kind.as_str())?;
    object.set("provider", error.provider)?;
    object.set("status", error.status)?;
    object.set("code", error.code)?;
    object.set("requestId", error.request_id)?;
    object.set("retryAfterMs", retry_after_ms)?;
    object.set("rawBody", raw_body)?;

    Ok(Error::from(object.to_unknown()))
}

fn client_error_without_env(error: CoreClientError) -> Error {
    match error {
        CoreClientError::Local(error) => core_error(error),
        CoreClientError::Remote(error) => Error::new(
            remote_error_status(error.kind),
            remote_error_message(&error),
        ),
        CoreClientError::Provider(error) => Error::new(
            provider_error_status(error.kind),
            provider_error_message(&error),
        ),
        CoreClientError::InvalidRequest(message) => invalid_arg(message),
        CoreClientError::UnsupportedOperation {
            endpoint,
            operation,
        } => invalid_arg(format!(
            "unsupported operation {operation} on endpoint {endpoint:?}"
        )),
        other => napi_error(other.to_string()),
    }
}

fn client_error_to_node(env: Env, error: CoreClientError) -> Error {
    match error {
        CoreClientError::Local(error) => core_error(error),
        CoreClientError::Remote(error) => remote_error_to_node(env, error),
        CoreClientError::Provider(error) => provider_error_to_node(env, error),
        CoreClientError::InvalidRequest(message) => invalid_arg(message),
        CoreClientError::UnsupportedOperation {
            endpoint,
            operation,
        } => invalid_arg(format!(
            "unsupported operation {operation} on endpoint {endpoint:?}"
        )),
        other => napi_error(other.to_string()),
    }
}

fn core_error(error: cogentlm_engine::Error) -> Error {
    match error {
        cogentlm_engine::Error::InvalidRequest(message)
        | cogentlm_engine::Error::InvalidConfig(message) => invalid_arg(message),
        cogentlm_engine::Error::UnsupportedOperation { operation, reason } => {
            invalid_arg(format!("unsupported operation {operation}: {reason}"))
        }
        other => napi_error(other.to_string()),
    }
}
