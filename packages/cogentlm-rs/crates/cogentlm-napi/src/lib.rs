use std::sync::{Arc, Mutex};

use cogentlm_engine::backend::{
    backend_observability_json as core_backend_observability_json,
    set_llama_log_quiet as core_set_llama_log_quiet,
};
use cogentlm_engine::engine::protocol::{
    BackendInfo as CoreBackendInfo, RequestState as CoreRequestState,
    RequestStats as CoreRequestStats,
};
use cogentlm_engine::engine::{
    CacheKeyPolicy, ChatMessage as CoreChatMessage, ChatRequest as CoreChatRequest,
    ChatRole as CoreChatRole, CogentEngine as CoreCogentEngine, EngineEvent as CoreEngineEvent,
    EngineEventReceiver as CoreEngineEventReceiver, EngineState as CoreEngineState,
    EngineStats as CoreEngineStats, FlashAttentionMode, GpuLayerConfig, KvCacheType, KvReuseMode,
    LogitBias, ModelPlacementConfig as CoreModelPlacementConfig,
    MultimodalRuntimeConfig as CoreMultimodalRuntimeConfig,
    NativeRuntimeConfig as CoreNativeRuntimeConfig,
    ObservabilityRuntimeConfig as CoreObservabilityRuntimeConfig, QueryOptions as CoreQueryOptions,
    QueryRequest as CoreQueryRequest, RequestResult as CoreRequestResult,
    ResidencyRuntimeConfig as CoreResidencyRuntimeConfig,
    ResolvedRuntimeLimits as CoreResolvedRuntimeLimits, RopeScaling, SamplerStage,
    SamplingRuntimeConfig as CoreSamplingRuntimeConfig,
    SchedulerRuntimeConfig as CoreSchedulerRuntimeConfig, SplitMode, TokenBatch as CoreTokenBatch,
    DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS,
};
use cogentlm_engine::lifecycle::{
    model_source_from_path as core_model_source_from_path,
    vision_model_source_from_paths as core_vision_model_source_from_paths,
    BackendPreference as CoreBackendPreference, LoadedModelInfo as CoreLoadedModelInfo,
    ModelInfo as CoreManagedModelInfo, ModelLoadOptions as CoreModelLoadOptions,
    ModelService as CoreModelService, ModelServiceState as CoreModelServiceState, StatsMode,
    DEFAULT_MODEL_BACKEND, DEFAULT_MODEL_STATS,
};
use cogentlm_engine::runtime::config::{
    SchedulerPolicyConfig as CoreSchedulerPolicyConfig, SchedulerPolicyMode,
};
use napi::bindgen_prelude::{AsyncTask, Buffer, Either, Env};
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{Error, Result, Status, Task};
use napi_derive::napi;
use serde::de::DeserializeOwned;

type SharedEngine = Arc<Mutex<Option<CoreCogentEngine>>>;
type SharedEvents = Arc<Mutex<CoreEngineEventReceiver>>;
type SharedModelService = Arc<Mutex<Option<CoreModelService>>>;
type SharedModelEvents = Arc<Mutex<Option<CoreEngineEventReceiver>>>;
type TokenBatchCallback = Arc<ThreadsafeFunction<TokenBatch, (), TokenBatch, Status, false>>;

const ENGINE_MUTEX_POISONED: &str = "engine mutex is poisoned";
const ENGINE_EVENTS_MUTEX_POISONED: &str = "engine events mutex is poisoned";
const ENGINE_CLOSED: &str = "engine is closed";
const MODEL_SERVICE_EVENTS_MUTEX_POISONED: &str = "model service events mutex is poisoned";
const MODEL_SERVICE_MUTEX_POISONED: &str = "model service mutex is poisoned";
const MODEL_SERVICE_CLOSED: &str = "model service is closed";
const EVENT_TYPE_STATE: &str = "state";
const EVENT_TYPE_LOAD_PROGRESS: &str = "load-progress";
const EVENT_TYPE_REQUEST_STARTED: &str = "request-started";
const EVENT_TYPE_REQUEST_COMPLETED: &str = "request-completed";
const EVENT_TYPE_REQUEST_FAILED: &str = "request-failed";
const EVENT_TYPE_CLOSED: &str = "closed";

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
    #[napi(js_name = "max_session_entries")]
    pub max_session_entries: Option<i32>,
    #[napi(js_name = "cache_key_policy")]
    pub cache_key_policy: Option<String>,
    #[napi(js_name = "enable_context_checkpoints")]
    pub enable_context_checkpoints: Option<bool>,
    #[napi(js_name = "checkpoint_every_tokens")]
    pub checkpoint_every_tokens: Option<i32>,
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
        assign_if_some(&mut core.max_session_entries, self.max_session_entries);
        if let Some(value) = &self.cache_key_policy {
            core.cache_key_policy = parse_cache_key_policy(value)?;
        }
        assign_if_some(
            &mut core.enable_context_checkpoints,
            self.enable_context_checkpoints,
        );
        assign_if_some(
            &mut core.checkpoint_every_tokens,
            self.checkpoint_every_tokens,
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
pub struct ModelLoadOptions {
    pub backend: Option<String>,
    pub stats: Option<String>,
    pub runtime: Option<NativeRuntimeConfig>,
}

impl ModelLoadOptions {
    fn to_core(&self) -> Result<CoreModelLoadOptions> {
        Ok(CoreModelLoadOptions {
            backend: parse_backend_preference(
                self.backend.as_deref().unwrap_or(DEFAULT_MODEL_BACKEND),
            )?,
            stats: parse_stats_mode(self.stats.as_deref().unwrap_or(DEFAULT_MODEL_STATS))?,
            runtime: optional_core_or_default(self.runtime.as_ref(), NativeRuntimeConfig::to_core)?,
        })
    }
}

#[napi(object)]
pub struct QueryOptions {
    pub context_key: Option<String>,
    pub max_tokens: Option<i32>,
    pub grammar: Option<String>,
    pub json_schema: Option<String>,
    pub stop: Option<Vec<String>>,
    pub sampling: Option<SamplingRuntimeConfig>,
    pub media: Option<Vec<Buffer>>,
}

impl QueryOptions {
    fn to_core(&self) -> Result<CoreQueryOptions> {
        let max_tokens = self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        if max_tokens <= 0 {
            return Err(invalid_arg("maxTokens must be positive"));
        }
        Ok(CoreQueryOptions {
            context_key: self
                .context_key
                .clone()
                .unwrap_or_else(|| DEFAULT_CONTEXT_KEY.to_string()),
            max_tokens,
            grammar: self.grammar.clone().unwrap_or_default(),
            json_schema: self.json_schema.clone().unwrap_or_default(),
            stop: self.stop.clone().unwrap_or_default(),
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
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[napi(object)]
pub struct BackendDevice {
    pub id: Option<String>,
    pub name: String,
    pub r#type: String,
    pub memory_total_bytes: Option<f64>,
    pub memory_free_bytes: Option<f64>,
}

#[napi(object)]
pub struct BackendInfo {
    pub selected: String,
    pub available: Vec<String>,
    pub devices: Vec<BackendDevice>,
}

#[napi(object)]
pub struct BackendSelection {
    pub requested: String,
    pub selected: String,
    pub available: Vec<String>,
    pub gpu_offload_expected: bool,
    pub reason: Option<String>,
}

#[napi(object)]
pub struct ManagedModelInfo {
    pub id: String,
    pub name: String,
    pub modality: String,
    pub status: String,
    pub source: String,
    pub bytes: f64,
    pub loaded: bool,
    pub chat_template: Option<String>,
    pub bos_text: String,
    pub eos_text: String,
    pub media_marker: Option<String>,
    pub created_at_unix_ms: f64,
    pub updated_at_unix_ms: f64,
}

#[napi(object)]
pub struct LoadedModelInfo {
    pub model: ManagedModelInfo,
    pub backend: BackendSelection,
    pub runtime_fingerprint: String,
}

#[napi(object)]
pub struct ModelState {
    pub id: String,
    pub name: String,
}

#[napi(object)]
pub struct RequestState {
    pub id: String,
    pub status: String,
    pub input_tokens: i32,
    pub output_tokens: i32,
}

#[napi(object)]
pub struct EngineStats {
    pub requests_running: i32,
    pub requests_queued: i32,
    pub requests_completed: f64,
    pub requests_failed: f64,
    pub input_tokens: f64,
    pub output_tokens: f64,
    pub cache_hits: f64,
    pub prefill_tokens: f64,
    pub ttft_ms: Option<f64>,
    pub inter_token_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub backend_ms: f64,
    pub sync_ms: f64,
    pub engine_overhead_ms: f64,
    pub debug_metrics_scheduler_ticks: f64,
    pub debug_metrics_decode_ticks: f64,
    pub debug_metrics_prefill_ticks: f64,
    pub debug_metrics_backend_sampler_attach_attempts: f64,
    pub debug_metrics_backend_sampler_attach_failures: f64,
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

#[napi(object)]
pub struct ResolvedRuntimeLimits {
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_parallel: i32,
    pub kv_unified: bool,
    pub flash_attention: String,
    pub cache_type_k: String,
    pub cache_type_v: String,
}

#[napi(object)]
pub struct EngineState {
    pub status: String,
    pub model: Option<ModelState>,
    pub backend: BackendInfo,
    pub runtime: Option<ResolvedRuntimeLimits>,
    pub requests: Vec<RequestState>,
    pub stats: EngineStats,
    pub updated_at_unix_ms: f64,
}

#[napi(object)]
pub struct ModelServiceState {
    pub status: String,
    pub model: Option<ManagedModelInfo>,
    pub backend: BackendInfo,
    pub runtime: Option<ResolvedRuntimeLimits>,
    pub requests: Vec<RequestState>,
    pub stats: EngineStats,
    pub updated_at_unix_ms: f64,
}

struct StateTail {
    backend: BackendInfo,
    runtime: Option<ResolvedRuntimeLimits>,
    requests: Vec<RequestState>,
    stats: EngineStats,
    updated_at_unix_ms: f64,
}

#[napi(object)]
pub struct RequestStats {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cache_hits: i32,
    pub ttft_ms: Option<f64>,
    pub inter_token_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
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

#[napi(object)]
pub struct RequestResult {
    pub id: String,
    pub text: String,
    pub finish_reason: String,
    pub stats: RequestStats,
}

#[napi(object)]
#[derive(Clone)]
pub struct StreamStats {
    pub frames_sent: f64,
    pub bytes_sent: f64,
    pub frames_dropped: f64,
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
    pub stats: StreamStats,
}

#[napi(object)]
pub struct EngineEvent {
    pub r#type: String,
    pub state: Option<EngineState>,
    pub loaded_bytes: Option<f64>,
    pub total_bytes: Option<f64>,
    pub asset_name: Option<String>,
    pub request_id: Option<String>,
    pub stream_id: Option<u32>,
    pub result: Option<RequestResult>,
    pub error: Option<String>,
}

#[napi(js_name = "CogentEngine")]
pub struct CogentEngine {
    inner: SharedEngine,
    events: SharedEvents,
}

#[napi]
impl CogentEngine {
    #[napi]
    pub fn load(
        model_path: String,
        config: Option<NativeRuntimeConfig>,
    ) -> Result<AsyncTask<LoadTask>> {
        let config = config
            .as_ref()
            .map(NativeRuntimeConfig::to_core)
            .transpose()?
            .unwrap_or_default();
        Ok(AsyncTask::new(LoadTask { model_path, config }))
    }

    #[napi]
    pub fn query(
        &self,
        prompt: String,
        options: Option<QueryOptions>,
        on_tokens: Option<TokenBatchCallback>,
    ) -> Result<AsyncTask<QueryTask>> {
        let options = optional_core_or_default(options.as_ref(), QueryOptions::to_core)?;
        Ok(AsyncTask::new(QueryTask {
            engine: self.inner.clone(),
            prompt,
            options,
            on_tokens,
        }))
    }

    #[napi]
    pub fn chat(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<QueryOptions>,
        on_tokens: Option<TokenBatchCallback>,
    ) -> Result<AsyncTask<ChatTextTask>> {
        let options = optional_core_or_default(options.as_ref(), QueryOptions::to_core)?;
        Ok(AsyncTask::new(ChatTextTask {
            engine: self.inner.clone(),
            messages: chat_messages_to_core(messages)?,
            options,
            on_tokens,
        }))
    }

    #[napi]
    pub fn state(&self) -> Result<AsyncTask<StateTask>> {
        Ok(AsyncTask::new(StateTask {
            engine: self.inner.clone(),
        }))
    }

    #[napi]
    pub fn drain_events(&self) -> Result<Vec<EngineEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|_| napi_error(ENGINE_EVENTS_MUTEX_POISONED))?;
        Ok(events.try_iter().map(engine_event_to_node).collect())
    }
}

#[napi(js_name = "ModelService")]
pub struct ModelService {
    inner: SharedModelService,
    events: SharedModelEvents,
}

#[napi]
impl ModelService {
    #[napi(constructor)]
    pub fn new(store_path: String) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(
                CoreModelService::local(store_path).map_err(model_error)?,
            ))),
            events: Arc::new(Mutex::new(None)),
        })
    }

    #[napi]
    pub fn load_path(
        &self,
        model_path: String,
        options: Option<ModelLoadOptions>,
    ) -> Result<AsyncTask<ModelLoadPathTask>> {
        Ok(AsyncTask::new(ModelLoadPathTask {
            service: self.inner.clone(),
            events: self.events.clone(),
            model_path,
            options: optional_core_or_default(options.as_ref(), ModelLoadOptions::to_core)?,
        }))
    }

    #[napi]
    pub fn load_vision(
        &self,
        model_path: String,
        projector_path: String,
        options: Option<ModelLoadOptions>,
    ) -> Result<AsyncTask<ModelLoadVisionTask>> {
        Ok(AsyncTask::new(ModelLoadVisionTask {
            service: self.inner.clone(),
            events: self.events.clone(),
            model_path,
            projector_path,
            options: optional_core_or_default(options.as_ref(), ModelLoadOptions::to_core)?,
        }))
    }

    #[napi]
    pub fn unload(&self) -> AsyncTask<ModelUnloadTask> {
        AsyncTask::new(ModelUnloadTask {
            service: self.inner.clone(),
            events: self.events.clone(),
        })
    }

    #[napi]
    pub fn remove(&self, model_id: String) -> AsyncTask<ModelRemoveTask> {
        AsyncTask::new(ModelRemoveTask {
            service: self.inner.clone(),
            model_id,
        })
    }

    #[napi]
    pub fn list(&self) -> Result<Vec<ManagedModelInfo>> {
        with_model_service(&self.inner, |service| Ok(service.list()))
            .map(|models| models.into_iter().map(model_info_to_node).collect())
    }

    #[napi]
    pub fn current(&self) -> Result<Option<ManagedModelInfo>> {
        with_model_service(&self.inner, |service| Ok(service.current()))
            .map(|model| model.map(model_info_to_node))
    }

    #[napi]
    pub fn query(
        &self,
        prompt: String,
        options: Option<QueryOptions>,
        on_tokens: Option<TokenBatchCallback>,
    ) -> Result<AsyncTask<ModelQueryTask>> {
        Ok(AsyncTask::new(ModelQueryTask {
            service: self.inner.clone(),
            prompt,
            options: optional_core_or_default(options.as_ref(), QueryOptions::to_core)?,
            on_tokens,
        }))
    }

    #[napi]
    pub fn chat(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<QueryOptions>,
        on_tokens: Option<TokenBatchCallback>,
    ) -> Result<AsyncTask<ModelChatTask>> {
        Ok(AsyncTask::new(ModelChatTask {
            service: self.inner.clone(),
            messages: chat_messages_to_core(messages)?,
            options: optional_core_or_default(options.as_ref(), QueryOptions::to_core)?,
            on_tokens,
        }))
    }

    #[napi]
    pub fn state(&self) -> Result<AsyncTask<ModelStateTask>> {
        Ok(AsyncTask::new(ModelStateTask {
            service: self.inner.clone(),
        }))
    }

    #[napi]
    pub fn drain_events(&self) -> Result<Vec<EngineEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|_| napi_error(MODEL_SERVICE_EVENTS_MUTEX_POISONED))?;
        Ok(events
            .as_ref()
            .map(|events| events.try_iter().map(engine_event_to_node).collect())
            .unwrap_or_default())
    }
}

pub struct LoadTask {
    model_path: String,
    config: CoreNativeRuntimeConfig,
}

impl Task for LoadTask {
    type Output = CoreCogentEngine;
    type JsValue = CogentEngine;

    fn compute(&mut self) -> Result<Self::Output> {
        CoreCogentEngine::load(&self.model_path, self.config.clone()).map_err(core_error)
    }

    fn resolve(&mut self, _env: Env, engine: Self::Output) -> Result<Self::JsValue> {
        let events = engine.subscribe_events();
        Ok(CogentEngine {
            inner: Arc::new(Mutex::new(Some(engine))),
            events: Arc::new(Mutex::new(events)),
        })
    }
}

pub struct QueryTask {
    engine: SharedEngine,
    prompt: String,
    options: CoreQueryOptions,
    on_tokens: Option<TokenBatchCallback>,
}

impl Task for QueryTask {
    type Output = CoreRequestResult;
    type JsValue = RequestResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let request = query_request_with_tokens(
            self.prompt.clone(),
            self.options.clone(),
            self.on_tokens.clone(),
        );
        with_engine(&self.engine, |engine| engine.query(request))
    }

    fn resolve(&mut self, _env: Env, result: Self::Output) -> Result<Self::JsValue> {
        Ok(request_result_to_node(result))
    }
}

pub struct ChatTextTask {
    engine: SharedEngine,
    messages: Vec<CoreChatMessage>,
    options: CoreQueryOptions,
    on_tokens: Option<TokenBatchCallback>,
}

impl Task for ChatTextTask {
    type Output = CoreRequestResult;
    type JsValue = RequestResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let request = chat_request_with_tokens(
            self.messages.clone(),
            self.options.clone(),
            self.on_tokens.clone(),
        );
        with_engine(&self.engine, |engine| engine.chat(request))
    }

    fn resolve(&mut self, _env: Env, result: Self::Output) -> Result<Self::JsValue> {
        Ok(request_result_to_node(result))
    }
}

pub struct StateTask {
    engine: SharedEngine,
}

impl Task for StateTask {
    type Output = CoreEngineState;
    type JsValue = EngineState;

    fn compute(&mut self) -> Result<Self::Output> {
        with_engine(&self.engine, |engine| engine.state())
    }

    fn resolve(&mut self, _env: Env, state: Self::Output) -> Result<Self::JsValue> {
        Ok(engine_state_to_node(state))
    }
}

pub struct ModelLoadPathTask {
    service: SharedModelService,
    events: SharedModelEvents,
    model_path: String,
    options: CoreModelLoadOptions,
}

impl Task for ModelLoadPathTask {
    type Output = CoreLoadedModelInfo;
    type JsValue = LoadedModelInfo;

    fn compute(&mut self) -> Result<Self::Output> {
        with_model_service_mut(&self.service, |service| {
            let loaded = service.load(
                core_model_source_from_path(&self.model_path),
                self.options.clone(),
            )?;
            refresh_model_events(&self.events, service)?;
            Ok(loaded)
        })
    }

    fn resolve(&mut self, _env: Env, loaded: Self::Output) -> Result<Self::JsValue> {
        Ok(loaded_model_info_to_node(loaded))
    }
}

pub struct ModelLoadVisionTask {
    service: SharedModelService,
    events: SharedModelEvents,
    model_path: String,
    projector_path: String,
    options: CoreModelLoadOptions,
}

impl Task for ModelLoadVisionTask {
    type Output = CoreLoadedModelInfo;
    type JsValue = LoadedModelInfo;

    fn compute(&mut self) -> Result<Self::Output> {
        with_model_service_mut(&self.service, |service| {
            let loaded = service.load(
                core_vision_model_source_from_paths(&self.model_path, &self.projector_path),
                self.options.clone(),
            )?;
            refresh_model_events(&self.events, service)?;
            Ok(loaded)
        })
    }

    fn resolve(&mut self, _env: Env, loaded: Self::Output) -> Result<Self::JsValue> {
        Ok(loaded_model_info_to_node(loaded))
    }
}

pub struct ModelUnloadTask {
    service: SharedModelService,
    events: SharedModelEvents,
}

impl Task for ModelUnloadTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<Self::Output> {
        with_model_service_mut(&self.service, |service| service.unload())?;
        clear_model_events(&self.events)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

pub struct ModelRemoveTask {
    service: SharedModelService,
    model_id: String,
}

impl Task for ModelRemoveTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<Self::Output> {
        with_model_service_mut(&self.service, |service| service.remove(&self.model_id))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

pub struct ModelQueryTask {
    service: SharedModelService,
    prompt: String,
    options: CoreQueryOptions,
    on_tokens: Option<TokenBatchCallback>,
}

impl Task for ModelQueryTask {
    type Output = CoreRequestResult;
    type JsValue = RequestResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let request = query_request_with_tokens(
            self.prompt.clone(),
            self.options.clone(),
            self.on_tokens.clone(),
        );
        with_model_service(&self.service, |service| service.query(request))
    }

    fn resolve(&mut self, _env: Env, result: Self::Output) -> Result<Self::JsValue> {
        Ok(request_result_to_node(result))
    }
}

pub struct ModelChatTask {
    service: SharedModelService,
    messages: Vec<CoreChatMessage>,
    options: CoreQueryOptions,
    on_tokens: Option<TokenBatchCallback>,
}

impl Task for ModelChatTask {
    type Output = CoreRequestResult;
    type JsValue = RequestResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let request = chat_request_with_tokens(
            self.messages.clone(),
            self.options.clone(),
            self.on_tokens.clone(),
        );
        with_model_service(&self.service, |service| service.chat(request))
    }

    fn resolve(&mut self, _env: Env, result: Self::Output) -> Result<Self::JsValue> {
        Ok(request_result_to_node(result))
    }
}

pub struct ModelStateTask {
    service: SharedModelService,
}

impl Task for ModelStateTask {
    type Output = CoreModelServiceState;
    type JsValue = ModelServiceState;

    fn compute(&mut self) -> Result<Self::Output> {
        with_model_service(&self.service, |service| service.state())
    }

    fn resolve(&mut self, _env: Env, state: Self::Output) -> Result<Self::JsValue> {
        Ok(model_service_state_to_node(state))
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

fn with_engine<T>(
    engine: &SharedEngine,
    f: impl FnOnce(&CoreCogentEngine) -> cogentlm_engine::Result<T>,
) -> Result<T> {
    let guard = engine
        .lock()
        .map_err(|_| napi_error(ENGINE_MUTEX_POISONED))?;
    let engine = guard.as_ref().ok_or_else(|| napi_error(ENGINE_CLOSED))?;
    f(engine).map_err(core_error)
}

fn with_model_service<T>(
    service: &SharedModelService,
    f: impl FnOnce(&CoreModelService) -> std::result::Result<T, cogentlm_engine::lifecycle::ModelError>,
) -> Result<T> {
    let guard = service
        .lock()
        .map_err(|_| napi_error(MODEL_SERVICE_MUTEX_POISONED))?;
    let service = guard
        .as_ref()
        .ok_or_else(|| napi_error(MODEL_SERVICE_CLOSED))?;
    f(service).map_err(model_error)
}

fn with_model_service_mut<T>(
    service: &SharedModelService,
    f: impl FnOnce(
        &mut CoreModelService,
    ) -> std::result::Result<T, cogentlm_engine::lifecycle::ModelError>,
) -> Result<T> {
    let mut guard = service
        .lock()
        .map_err(|_| napi_error(MODEL_SERVICE_MUTEX_POISONED))?;
    let service = guard
        .as_mut()
        .ok_or_else(|| napi_error(MODEL_SERVICE_CLOSED))?;
    f(service).map_err(model_error)
}

fn refresh_model_events(
    events: &SharedModelEvents,
    service: &CoreModelService,
) -> std::result::Result<(), cogentlm_engine::lifecycle::ModelError> {
    let receiver = service.subscribe_events()?;
    events
        .lock()
        .map_err(|_| {
            cogentlm_engine::lifecycle::ModelError::Runtime(
                MODEL_SERVICE_EVENTS_MUTEX_POISONED.to_string(),
            )
        })?
        .replace(receiver);
    Ok(())
}

fn clear_model_events(events: &SharedModelEvents) -> Result<()> {
    events
        .lock()
        .map_err(|_| napi_error(MODEL_SERVICE_EVENTS_MUTEX_POISONED))?
        .take();
    Ok(())
}

fn query_request_with_tokens(
    prompt: String,
    options: CoreQueryOptions,
    on_tokens: Option<TokenBatchCallback>,
) -> CoreQueryRequest {
    let request = CoreQueryRequest::new(prompt).options(options);
    if let Some(on_tokens) = on_tokens {
        request.on_tokens(move |batch| emit_token_batch(&on_tokens, batch))
    } else {
        request
    }
}

fn chat_request_with_tokens(
    messages: Vec<CoreChatMessage>,
    options: CoreQueryOptions,
    on_tokens: Option<TokenBatchCallback>,
) -> CoreChatRequest {
    let request = CoreChatRequest::new(messages).options(options);
    if let Some(on_tokens) = on_tokens {
        request.on_tokens(move |batch| emit_token_batch(&on_tokens, batch))
    } else {
        request
    }
}

fn emit_token_batch(
    callback: &TokenBatchCallback,
    batch: &CoreTokenBatch,
) -> cogentlm_engine::Result<()> {
    let status = callback.call(
        token_batch_to_node(batch.clone()),
        ThreadsafeFunctionCallMode::NonBlocking,
    );
    if status == Status::Ok {
        Ok(())
    } else {
        Err(cogentlm_engine::Error::RuntimeCommand(format!(
            "token batch callback failed with {status}"
        )))
    }
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

fn request_result_to_node(result: CoreRequestResult) -> RequestResult {
    RequestResult {
        id: result.id,
        text: result.text,
        finish_reason: result.finish_reason.as_str().to_string(),
        stats: request_stats_to_node(result.stats),
    }
}

fn token_batch_to_node(batch: CoreTokenBatch) -> TokenBatch {
    TokenBatch {
        request_id: batch.request_id,
        stream_id: batch.stream_id,
        sequence_start: batch.sequence_start,
        text: batch.text,
        frame_count: batch.frame_count,
        byte_count: batch.byte_count,
        stats: StreamStats {
            frames_sent: batch.stats.frames_sent as f64,
            bytes_sent: batch.stats.bytes_sent as f64,
            frames_dropped: batch.stats.frames_dropped as f64,
            batches_sent: batch.stats.batches_sent as f64,
        },
    }
}

fn loaded_model_info_to_node(loaded: CoreLoadedModelInfo) -> LoadedModelInfo {
    LoadedModelInfo {
        model: model_info_to_node(loaded.model),
        backend: BackendSelection {
            requested: loaded.backend.requested.as_str().to_string(),
            selected: loaded.backend.selected,
            available: loaded.backend.available,
            gpu_offload_expected: loaded.backend.gpu_offload_expected,
            reason: loaded.backend.reason,
        },
        runtime_fingerprint: loaded.runtime_fingerprint,
    }
}

fn model_info_to_node(model: CoreManagedModelInfo) -> ManagedModelInfo {
    ManagedModelInfo {
        id: model.id,
        name: model.name,
        modality: model.modality.as_str().to_string(),
        status: model.status.as_str().to_string(),
        source: model.source.as_str().to_string(),
        bytes: model.bytes as f64,
        loaded: model.loaded,
        chat_template: model.chat_template,
        bos_text: model.bos_text,
        eos_text: model.eos_text,
        media_marker: model.media_marker,
        created_at_unix_ms: model.created_at_unix_ms as f64,
        updated_at_unix_ms: model.updated_at_unix_ms as f64,
    }
}

fn engine_state_to_node(state: CoreEngineState) -> EngineState {
    let tail = state_tail_to_node(
        state.backend,
        state.runtime,
        state.requests,
        state.stats,
        state.updated_at_unix_ms,
    );
    EngineState {
        status: state.status.as_str().to_string(),
        model: state.model.map(|model| ModelState {
            id: model.id,
            name: model.name,
        }),
        backend: tail.backend,
        runtime: tail.runtime,
        requests: tail.requests,
        stats: tail.stats,
        updated_at_unix_ms: tail.updated_at_unix_ms,
    }
}

fn model_service_state_to_node(state: CoreModelServiceState) -> ModelServiceState {
    let tail = state_tail_to_node(
        state.backend,
        state.runtime,
        state.requests,
        state.stats,
        state.updated_at_unix_ms,
    );
    ModelServiceState {
        status: state.status.as_str().to_string(),
        model: state.model.map(model_info_to_node),
        backend: tail.backend,
        runtime: tail.runtime,
        requests: tail.requests,
        stats: tail.stats,
        updated_at_unix_ms: tail.updated_at_unix_ms,
    }
}

fn state_tail_to_node(
    backend: CoreBackendInfo,
    runtime: Option<CoreResolvedRuntimeLimits>,
    requests: Vec<CoreRequestState>,
    stats: CoreEngineStats,
    updated_at_unix_ms: u64,
) -> StateTail {
    StateTail {
        backend: backend_info_to_node(backend),
        runtime: runtime.map(resolved_runtime_limits_to_node),
        requests: requests
            .into_iter()
            .map(|request| RequestState {
                id: request.id,
                status: request.status.as_str().to_string(),
                input_tokens: request.input_tokens,
                output_tokens: request.output_tokens,
            })
            .collect(),
        stats: engine_stats_to_node(stats),
        updated_at_unix_ms: updated_at_unix_ms as f64,
    }
}

fn backend_info_to_node(backend: CoreBackendInfo) -> BackendInfo {
    BackendInfo {
        selected: backend.selected,
        available: backend.available,
        devices: backend
            .devices
            .into_iter()
            .map(|device| BackendDevice {
                id: device.id,
                name: device.name,
                r#type: device.device_type,
                memory_total_bytes: option_u64_f64(device.memory_total_bytes),
                memory_free_bytes: option_u64_f64(device.memory_free_bytes),
            })
            .collect(),
    }
}

fn resolved_runtime_limits_to_node(runtime: CoreResolvedRuntimeLimits) -> ResolvedRuntimeLimits {
    ResolvedRuntimeLimits {
        n_ctx: runtime.n_ctx,
        n_batch: runtime.n_batch,
        n_ubatch: runtime.n_ubatch,
        n_parallel: runtime.n_parallel,
        kv_unified: runtime.kv_unified,
        flash_attention: runtime.flash_attention,
        cache_type_k: runtime.cache_type_k,
        cache_type_v: runtime.cache_type_v,
    }
}

fn engine_stats_to_node(stats: CoreEngineStats) -> EngineStats {
    EngineStats {
        requests_running: stats.requests_running,
        requests_queued: stats.requests_queued,
        requests_completed: stats.requests_completed as f64,
        requests_failed: stats.requests_failed as f64,
        input_tokens: stats.input_tokens as f64,
        output_tokens: stats.output_tokens as f64,
        cache_hits: stats.cache_hits as f64,
        prefill_tokens: stats.prefill_tokens as f64,
        ttft_ms: stats.ttft_ms,
        inter_token_ms: stats.inter_token_ms,
        e2e_ms: stats.e2e_ms,
        tokens_per_second: stats.tokens_per_second,
        decode_tokens_per_second: stats.decode_tokens_per_second,
        prefill_tokens_per_second: stats.prefill_tokens_per_second,
        prefill_ms: stats.prefill_ms,
        decode_ms: stats.decode_ms,
        backend_ms: stats.backend_ms,
        sync_ms: stats.sync_ms,
        engine_overhead_ms: stats.engine_overhead_ms,
        debug_metrics_scheduler_ticks: stats.debug_metrics_scheduler_ticks as f64,
        debug_metrics_decode_ticks: stats.debug_metrics_decode_ticks as f64,
        debug_metrics_prefill_ticks: stats.debug_metrics_prefill_ticks as f64,
        debug_metrics_backend_sampler_attach_attempts: stats
            .debug_metrics_backend_sampler_attach_attempts
            as f64,
        debug_metrics_backend_sampler_attach_failures: stats
            .debug_metrics_backend_sampler_attach_failures
            as f64,
        debug_metrics_admit_ms: stats.debug_metrics_admit_ms,
        debug_metrics_normalize_ms: stats.debug_metrics_normalize_ms,
        debug_metrics_backend_sampler_attach_ms: stats.debug_metrics_backend_sampler_attach_ms,
        debug_metrics_select_slots_ms: stats.debug_metrics_select_slots_ms,
        debug_metrics_plan_ms: stats.debug_metrics_plan_ms,
        debug_metrics_batch_build_ms: stats.debug_metrics_batch_build_ms,
        debug_metrics_llama_decode_ms: stats.debug_metrics_llama_decode_ms,
        debug_metrics_llama_sync_ms: stats.debug_metrics_llama_sync_ms,
        debug_metrics_apply_bookkeeping_ms: stats.debug_metrics_apply_bookkeeping_ms,
        debug_metrics_apply_decode_results_ms: stats.debug_metrics_apply_decode_results_ms,
        debug_metrics_sample_ms: stats.debug_metrics_sample_ms,
        debug_metrics_token_piece_ms: stats.debug_metrics_token_piece_ms,
        debug_metrics_emit_ms: stats.debug_metrics_emit_ms,
        debug_metrics_prefix_queue_ms: stats.debug_metrics_prefix_queue_ms,
        debug_metrics_finalize_ms: stats.debug_metrics_finalize_ms,
        debug_metrics_commit_observability_ms: stats.debug_metrics_commit_observability_ms,
        debug_metrics_post_decode_ms: stats.debug_metrics_post_decode_ms,
    }
}

fn request_stats_to_node(stats: CoreRequestStats) -> RequestStats {
    RequestStats {
        input_tokens: stats.input_tokens,
        output_tokens: stats.output_tokens,
        cache_hits: stats.cache_hits,
        ttft_ms: stats.ttft_ms,
        inter_token_ms: stats.inter_token_ms,
        e2e_ms: stats.e2e_ms,
        tokens_per_second: stats.tokens_per_second,
        decode_tokens_per_second: stats.decode_tokens_per_second,
        prefill_ms: stats.prefill_ms,
        decode_ms: stats.decode_ms,
        debug_metrics_scheduler_ticks: stats.debug_metrics_scheduler_ticks,
        debug_metrics_decode_ticks: stats.debug_metrics_decode_ticks,
        debug_metrics_prefill_ticks: stats.debug_metrics_prefill_ticks,
        debug_metrics_backend_sampler_attach_attempts: stats
            .debug_metrics_backend_sampler_attach_attempts,
        debug_metrics_backend_sampler_attach_failures: stats
            .debug_metrics_backend_sampler_attach_failures,
        debug_metrics_admit_ms: stats.debug_metrics_admit_ms,
        debug_metrics_normalize_ms: stats.debug_metrics_normalize_ms,
        debug_metrics_backend_sampler_attach_ms: stats.debug_metrics_backend_sampler_attach_ms,
        debug_metrics_select_slots_ms: stats.debug_metrics_select_slots_ms,
        debug_metrics_plan_ms: stats.debug_metrics_plan_ms,
        debug_metrics_batch_build_ms: stats.debug_metrics_batch_build_ms,
        debug_metrics_llama_decode_ms: stats.debug_metrics_llama_decode_ms,
        debug_metrics_llama_sync_ms: stats.debug_metrics_llama_sync_ms,
        debug_metrics_apply_bookkeeping_ms: stats.debug_metrics_apply_bookkeeping_ms,
        debug_metrics_apply_decode_results_ms: stats.debug_metrics_apply_decode_results_ms,
        debug_metrics_sample_ms: stats.debug_metrics_sample_ms,
        debug_metrics_token_piece_ms: stats.debug_metrics_token_piece_ms,
        debug_metrics_emit_ms: stats.debug_metrics_emit_ms,
        debug_metrics_prefix_queue_ms: stats.debug_metrics_prefix_queue_ms,
        debug_metrics_finalize_ms: stats.debug_metrics_finalize_ms,
        debug_metrics_commit_observability_ms: stats.debug_metrics_commit_observability_ms,
        debug_metrics_post_decode_ms: stats.debug_metrics_post_decode_ms,
    }
}

fn engine_event_to_node(event: CoreEngineEvent) -> EngineEvent {
    match event {
        CoreEngineEvent::State(state) => EngineEvent {
            r#type: EVENT_TYPE_STATE.to_string(),
            state: Some(engine_state_to_node(*state)),
            loaded_bytes: None,
            total_bytes: None,
            asset_name: None,
            request_id: None,
            stream_id: None,
            result: None,
            error: None,
        },
        CoreEngineEvent::LoadProgress {
            loaded_bytes,
            total_bytes,
            asset_name,
        } => EngineEvent {
            r#type: EVENT_TYPE_LOAD_PROGRESS.to_string(),
            state: None,
            loaded_bytes: Some(loaded_bytes as f64),
            total_bytes: option_u64_f64(total_bytes),
            asset_name,
            request_id: None,
            stream_id: None,
            result: None,
            error: None,
        },
        CoreEngineEvent::RequestStarted {
            request_id,
            stream_id,
        } => EngineEvent {
            r#type: EVENT_TYPE_REQUEST_STARTED.to_string(),
            state: None,
            loaded_bytes: None,
            total_bytes: None,
            asset_name: None,
            request_id: Some(request_id),
            stream_id: Some(stream_id),
            result: None,
            error: None,
        },
        CoreEngineEvent::RequestCompleted { result } => EngineEvent {
            r#type: EVENT_TYPE_REQUEST_COMPLETED.to_string(),
            state: None,
            loaded_bytes: None,
            total_bytes: None,
            asset_name: None,
            request_id: None,
            stream_id: None,
            result: Some(request_result_to_node(*result)),
            error: None,
        },
        CoreEngineEvent::RequestFailed { request_id, error } => EngineEvent {
            r#type: EVENT_TYPE_REQUEST_FAILED.to_string(),
            state: None,
            loaded_bytes: None,
            total_bytes: None,
            asset_name: None,
            request_id: Some(request_id),
            stream_id: None,
            result: None,
            error: Some(error),
        },
        CoreEngineEvent::Closed => EngineEvent {
            r#type: EVENT_TYPE_CLOSED.to_string(),
            state: None,
            loaded_bytes: None,
            total_bytes: None,
            asset_name: None,
            request_id: None,
            stream_id: None,
            result: None,
            error: None,
        },
    }
}

fn parse_backend_preference(value: &str) -> Result<CoreBackendPreference> {
    parse_choice(
        value,
        "backend must be one of: auto, cpu, cuda, metal, vulkan, webgpu",
    )
}

fn parse_stats_mode(value: &str) -> Result<StatsMode> {
    parse_choice(value, "stats must be one of: off, basic, profile")
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

fn parse_cache_key_policy(value: &str) -> Result<CacheKeyPolicy> {
    parse_choice(
        value,
        "cache_key_policy must be one of: context_key, prompt_hash",
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

fn option_u64_f64(value: Option<u64>) -> Option<f64> {
    value.map(|value| value as f64)
}

fn invalid_arg(message: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, message.into())
}

fn napi_error(message: impl ToString) -> Error {
    Error::new(Status::GenericFailure, message)
}

fn core_error(error: cogentlm_engine::Error) -> Error {
    match error {
        cogentlm_engine::Error::InvalidRequest(message)
        | cogentlm_engine::Error::InvalidConfig(message) => invalid_arg(message),
        other => napi_error(other.to_string()),
    }
}

fn model_error(error: cogentlm_engine::lifecycle::ModelError) -> Error {
    match error {
        cogentlm_engine::lifecycle::ModelError::InvalidModelSource(message)
        | cogentlm_engine::lifecycle::ModelError::InvalidModelPairing(message) => {
            invalid_arg(message)
        }
        cogentlm_engine::lifecycle::ModelError::UnsupportedGgufVersion(version) => {
            invalid_arg(format!("unsupported GGUF version {version}"))
        }
        other => napi_error(other.to_string()),
    }
}
