use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use cogentlm_client::{
    CogentChatRequest as CoreClientChatRequest, CogentClient as CoreClient,
    CogentEmbedRequest as CoreClientEmbedRequest,
    CogentEmbeddingResponse as CoreClientEmbeddingResponse,
    CogentEmbeddingResponseFuture as CoreClientEmbeddingResponseFuture,
    CogentEmbeddingRun as CoreClientEmbeddingRun, CogentError as CoreClientError,
    CogentQueryRequest as CoreClientQueryRequest, CogentTextOptions as CoreClientTextOptions,
    CogentTextResponse as CoreClientTextResponse,
    CogentTextResponseFuture as CoreClientTextResponseFuture, CogentTextRun as CoreClientTextRun,
    CogentTokenBatches as CoreClientTokenBatches, EndpointRef as CoreEndpointRef,
    LocalEmbedOptions as CoreClientLocalEmbedOptions,
    LocalTextOptions as CoreClientLocalTextOptions,
    RemoteAnthropicConfig as CoreRemoteAnthropicConfig, RemoteAuth as CoreRemoteAuth,
    RemoteConfig as CoreRemoteConfig, RemoteError as CoreRemoteError,
    RemoteErrorKind as CoreRemoteErrorKind, RemoteOpenAiConfig as CoreRemoteOpenAiConfig,
    RemoteProtocol as CoreRemoteProtocol, RemoteProxyConfig as CoreRemoteProxyConfig,
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

#[napi(object)]
pub struct LogitBiasConfig {
    pub token: i32,
    pub bias: f64,
}

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

#[napi(object)]
pub struct GpuLayerCountConfig {
    pub count: i32,
}

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
            _ => Err(invalid_arg("endpoint kind must be local or remote")),
        }
    }
}

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

#[napi(object)]
pub struct CogentQueryRequest {
    pub endpoint: Option<EndpointRef>,
    pub prompt: String,
    pub options: Option<CogentTextOptions>,
    pub local: Option<LocalTextOptions>,
    #[napi(js_name = "remoteOptions")]
    pub remote_options: Option<serde_json::Value>,
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
            remote_options: remote_options_or_empty(self.remote_options.clone())?,
            emit_tokens: self.emit_tokens.unwrap_or(false),
        })
    }
}

#[napi(object)]
pub struct CogentChatRequest {
    pub endpoint: Option<EndpointRef>,
    pub messages: Vec<ChatMessage>,
    pub options: Option<CogentTextOptions>,
    pub local: Option<LocalTextOptions>,
    #[napi(js_name = "remoteOptions")]
    pub remote_options: Option<serde_json::Value>,
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
            remote_options: remote_options_or_empty(self.remote_options.clone())?,
            emit_tokens: self.emit_tokens.unwrap_or(false),
        })
    }
}

#[napi(object)]
pub struct CogentEmbedRequest {
    pub endpoint: Option<EndpointRef>,
    pub input: String,
    pub local: Option<LocalEmbedOptions>,
    #[napi(js_name = "remoteOptions")]
    pub remote_options: Option<serde_json::Value>,
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
            remote_options: remote_options_or_empty(self.remote_options.clone())?,
        })
    }
}

#[napi(object)]
#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[napi(string_enum = "snake_case")]
#[derive(Clone, Copy)]
pub enum RemoteProxyProtocol {
    OpenAiCompatible,
}

impl From<RemoteProxyProtocol> for CoreRemoteProtocol {
    fn from(value: RemoteProxyProtocol) -> Self {
        match value {
            RemoteProxyProtocol::OpenAiCompatible => Self::OpenAiCompatible,
        }
    }
}

#[napi(object)]
pub struct RemoteAuthHeaderConfig {
    pub name: String,
    pub value: String,
}

#[napi(object)]
pub struct RemoteStaticHeaderConfig {
    pub name: String,
    pub value: String,
}

#[napi(object)]
pub struct RemoteAuthConfig {
    pub bearer: Option<String>,
    pub header: Option<RemoteAuthHeaderConfig>,
}

impl RemoteAuthConfig {
    fn to_core(&self) -> Result<CoreRemoteAuth> {
        match (&self.bearer, &self.header) {
            (Some(token), None) => Ok(CoreRemoteAuth::Bearer(CoreRemoteSecret::new(token.clone()))),
            (None, Some(header)) => Ok(CoreRemoteAuth::Header {
                name: header.name.clone(),
                value: CoreRemoteSecret::new(header.value.clone()),
            }),
            (Some(_), Some(_)) => Err(invalid_arg("remote auth must set bearer or header")),
            (None, None) => Err(invalid_arg("remote auth is required")),
        }
    }
}

#[napi(object)]
pub struct RemoteConfig {
    pub kind: String,
    pub model: String,
    #[napi(js_name = "apiKey")]
    pub api_key: Option<String>,
    #[napi(js_name = "baseUrl")]
    pub base_url: Option<String>,
    pub version: Option<String>,
    pub auth: Option<RemoteAuthConfig>,
    pub protocol: Option<RemoteProxyProtocol>,
    #[napi(js_name = "staticHeaders")]
    pub static_headers: Option<Vec<RemoteStaticHeaderConfig>>,
    #[napi(js_name = "timeoutMs")]
    pub timeout_ms: Option<u32>,
}

impl RemoteConfig {
    fn to_core(&self) -> Result<CoreRemoteConfig> {
        let timeout = self
            .timeout_ms
            .map(|timeout_ms| Duration::from_millis(u64::from(timeout_ms)));
        match self.kind.as_str() {
            "openai" => Ok(CoreRemoteConfig::OpenAi(CoreRemoteOpenAiConfig {
                model: self.model.clone(),
                api_key: CoreRemoteSecret::new(
                    self.api_key
                        .clone()
                        .ok_or_else(|| invalid_arg("openai remote requires apiKey"))?,
                ),
                base_url: self.base_url.clone(),
                timeout,
            })),
            "anthropic" => Ok(CoreRemoteConfig::Anthropic(CoreRemoteAnthropicConfig {
                model: self.model.clone(),
                api_key: CoreRemoteSecret::new(
                    self.api_key
                        .clone()
                        .ok_or_else(|| invalid_arg("anthropic remote requires apiKey"))?,
                ),
                base_url: self.base_url.clone(),
                version: self.version.clone(),
                timeout,
            })),
            "proxy" => Ok(CoreRemoteConfig::Proxy(CoreRemoteProxyConfig {
                model: self.model.clone(),
                base_url: self
                    .base_url
                    .clone()
                    .ok_or_else(|| invalid_arg("proxy remote requires baseUrl"))?,
                auth: self
                    .auth
                    .as_ref()
                    .ok_or_else(|| invalid_arg("proxy remote requires auth"))?
                    .to_core()?,
                protocol: self
                    .protocol
                    .unwrap_or(RemoteProxyProtocol::OpenAiCompatible)
                    .into(),
                static_headers: self
                    .static_headers
                    .as_ref()
                    .map(|headers| {
                        headers
                            .iter()
                            .map(|header| (header.name.clone(), header.value.clone()))
                            .collect()
                    })
                    .unwrap_or_default(),
                timeout,
            })),
            _ => Err(invalid_arg(
                "remote kind must be openai, anthropic, or proxy",
            )),
        }
    }
}

#[napi(object)]
pub struct TokenUsage {
    #[napi(js_name = "inputTokens")]
    pub input_tokens: Option<u32>,
    #[napi(js_name = "outputTokens")]
    pub output_tokens: Option<u32>,
    #[napi(js_name = "totalTokens")]
    pub total_tokens: Option<u32>,
}

#[napi(object)]
pub struct RequestStats {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cache_mode: String,
    pub cache_source: String,
    pub cache_hits: i32,
    pub prefill_tokens: i32,
    pub ttft_ms: Option<f64>,
    pub inter_token_ms: Option<f64>,
    #[napi(js_name = "e2eMs")]
    pub e2e_ms: Option<f64>,
    pub e2e_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

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

#[napi(object)]
#[derive(Clone)]
pub struct TokenEmissionStats {
    pub frames_sent: f64,
    pub bytes_sent: f64,
    pub batches_sent: f64,
}

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

    #[napi(ts_return_type = "Promise<EndpointRef>")]
    pub fn add_local(
        &self,
        id: String,
        model_path: String,
        config: Option<NativeRuntimeConfig>,
    ) -> Result<AsyncTask<ClientAddLocalTask>> {
        let config = config
            .as_ref()
            .map(NativeRuntimeConfig::to_core)
            .transpose()?
            .unwrap_or_default();
        Ok(AsyncTask::new(ClientAddLocalTask {
            client: self.inner.clone(),
            id,
            model_path,
            config,
        }))
    }

    #[napi(ts_return_type = "EndpointRef")]
    pub fn add_remote(&self, id: String, config: RemoteConfig) -> Result<EndpointRef> {
        let endpoint = self
            .inner
            .lock()
            .map_err(|_| napi_error(CLIENT_MUTEX_POISONED))?
            .add_remote(id, config.to_core()?)
            .map_err(client_error_without_env)?;
        Ok(endpoint_ref_to_node(endpoint))
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

pub struct ClientAddLocalTask {
    client: SharedCogentClient,
    id: String,
    model_path: String,
    config: CoreNativeRuntimeConfig,
}

impl Task for ClientAddLocalTask {
    type Output = ClientTaskOutput<CoreEndpointRef>;
    type JsValue = EndpointRef;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut client = self
            .client
            .lock()
            .map_err(|_| napi_error(CLIENT_MUTEX_POISONED))?;
        Ok(block_on(client.add_local(
            self.id.clone(),
            self.model_path.clone(),
            self.config.clone(),
        )))
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

#[napi]
pub fn backend_observability_json(include_details: Option<bool>) -> Result<String> {
    core_backend_observability_json(include_details.unwrap_or(true)).map_err(core_error)
}

#[napi]
pub fn set_llama_log_quiet(quiet: bool) {
    core_set_llama_log_quiet(quiet);
}

fn remote_options_or_empty(
    value: Option<serde_json::Value>,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    match value {
        Some(serde_json::Value::Object(options)) => Ok(options),
        Some(_) => Err(napi_error("remoteOptions must be a JSON object")),
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
        "{} remote error ({}): {}",
        error.remote_kind.as_str(),
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
    object.set("remoteKind", error.remote_kind.as_str())?;
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
        CoreClientError::Remote(error) => napi_error(remote_error_message(&error)),
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
