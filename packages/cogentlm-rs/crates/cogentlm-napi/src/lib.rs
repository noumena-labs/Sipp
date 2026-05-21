use std::sync::{Arc, Mutex};

use cogentlm_engine::backend::{
    backend_observability_json as core_backend_observability_json,
    set_llama_log_quiet as core_set_llama_log_quiet,
};
use cogentlm_engine::engine::protocol::{
    BackendDevice as CoreBackendDevice, BackendInfo as CoreBackendInfo,
    EngineStatus as CoreEngineStatus, FinishReason as CoreFinishReason,
    ModelState as CoreModelState, RequestState as CoreRequestState,
    RequestStats as CoreRequestStats, RequestStatus as CoreRequestStatus,
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
};
use cogentlm_engine::lifecycle::{
    model_source_from_path as core_model_source_from_path,
    vision_model_source_from_paths as core_vision_model_source_from_paths,
    BackendPreference as CoreBackendPreference, BackendSelection as CoreBackendSelection,
    LoadedModelInfo as CoreLoadedModelInfo, ModelInfo as CoreManagedModelInfo,
    ModelLoadOptions as CoreModelLoadOptions, ModelModality as CoreModelModality,
    ModelService as CoreModelService, ModelServiceState as CoreModelServiceState,
    ModelSourceKind as CoreModelSourceKind, ModelStatus as CoreModelStatus, StatsMode,
};
use cogentlm_engine::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode};
use cogentlm_engine::runtime::metrics::RuntimeObservabilityMetrics;
use cogentlm_engine::runtime::request::{GenerateResponse, GenerateResponseStatus};
use napi::bindgen_prelude::{AsyncTask, Buffer, Env};
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{Error, Result, Status, Task};
use napi_derive::napi;

type SharedEngine = Arc<Mutex<Option<CoreCogentEngine>>>;
type SharedEvents = Arc<Mutex<CoreEngineEventReceiver>>;
type SharedModelService = Arc<Mutex<Option<CoreModelService>>>;
type SharedModelEvents = Arc<Mutex<Option<CoreEngineEventReceiver>>>;
type TokenBatchCallback = Arc<ThreadsafeFunction<TokenBatch, (), TokenBatch, Status, false>>;

#[napi(object)]
pub struct LogitBiasConfig {
    pub token: i32,
    pub bias: f64,
}

#[napi(object)]
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
            top_p: self.top_p.map(|value| value as f32),
            min_p: self.min_p.map(|value| value as f32),
            typical_p: self.typical_p.map(|value| value as f32),
            xtc_probability: self.xtc_probability.map(|value| value as f32),
            xtc_threshold: self.xtc_threshold.map(|value| value as f32),
            top_n_sigma: self.top_n_sigma.map(|value| value as f32),
            temperature: self.temperature.map(|value| value as f32),
            dynatemp_range: self.dynatemp_range.map(|value| value as f32),
            dynatemp_exponent: self.dynatemp_exponent.map(|value| value as f32),
            repeat_last_n: self.repeat_last_n,
            repeat_penalty: self.repeat_penalty.map(|value| value as f32),
            frequency_penalty: self.frequency_penalty.map(|value| value as f32),
            presence_penalty: self.presence_penalty.map(|value| value as f32),
            dry_multiplier: self.dry_multiplier.map(|value| value as f32),
            dry_base: self.dry_base.map(|value| value as f32),
            dry_allowed_length: self.dry_allowed_length,
            dry_penalty_last_n: self.dry_penalty_last_n,
            dry_sequence_breakers: self.dry_sequence_breakers.clone().unwrap_or_default(),
            mirostat: self.mirostat,
            mirostat_tau: self.mirostat_tau.map(|value| value as f32),
            mirostat_eta: self.mirostat_eta.map(|value| value as f32),
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
pub struct ModelPlacementConfig {
    pub devices: Option<Vec<String>>,
    pub gpu_layers: Option<String>,
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

impl ModelPlacementConfig {
    fn to_core(&self) -> Result<CoreModelPlacementConfig> {
        let mut core = CoreModelPlacementConfig::default();
        if let Some(devices) = &self.devices {
            core.devices = devices.clone();
        }
        if let Some(value) = &self.gpu_layers {
            core.gpu_layers = parse_gpu_layers(value)?;
        }
        if let Some(value) = &self.split_mode {
            core.split_mode = parse_split_mode(value)?;
        }
        core.main_gpu = self.main_gpu;
        if let Some(value) = &self.tensor_split {
            core.tensor_split = value.iter().map(|value| *value as f32).collect();
        }
        if let Some(value) = self.use_mmap {
            core.use_mmap = value;
        }
        if let Some(value) = self.use_mlock {
            core.use_mlock = value;
        }
        if let Some(value) = self.fit_params {
            core.fit_params = value;
        }
        core.fit_params_min_ctx = self.fit_params_min_ctx;
        if let Some(value) = &self.fit_params_target_bytes {
            core.fit_params_target_bytes = value.iter().map(|value| *value as u64).collect();
        }
        if let Some(value) = self.check_tensors {
            core.check_tensors = value;
        }
        if let Some(value) = self.no_extra_bufts {
            core.no_extra_bufts = value;
        }
        if let Some(value) = self.no_host {
            core.no_host = value;
        }
        Ok(core)
    }
}

#[napi(object)]
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
        if let Some(value) = self.offload_kqv {
            core.offload_kqv = value;
        }
        if let Some(value) = self.op_offload {
            core.op_offload = value;
        }
        if let Some(value) = self.swa_full {
            core.swa_full = value;
        }
        if let Some(value) = self.warmup {
            core.warmup = value;
        }
        if let Some(value) = &self.rope_scaling {
            core.rope_scaling = Some(parse_rope_scaling(value)?);
        }
        core.rope_freq_base = self.rope_freq_base.map(|value| value as f32);
        core.rope_freq_scale = self.rope_freq_scale.map(|value| value as f32);
        core.yarn_orig_ctx = self.yarn_orig_ctx;
        core.yarn_ext_factor = self.yarn_ext_factor.map(|value| value as f32);
        core.yarn_attn_factor = self.yarn_attn_factor.map(|value| value as f32);
        core.yarn_beta_fast = self.yarn_beta_fast.map(|value| value as f32);
        core.yarn_beta_slow = self.yarn_beta_slow.map(|value| value as f32);
        Ok(core)
    }
}

#[napi(object)]
pub struct SchedulerRuntimeConfig {
    pub continuous_batching: Option<bool>,
    pub policy: Option<String>,
    pub decode_token_reserve: Option<i32>,
    pub adaptive_prefill_chunking: Option<bool>,
    pub prefill_chunk_size: Option<i32>,
    pub max_running_requests: Option<i32>,
    pub max_queued_requests: Option<i32>,
}

impl SchedulerRuntimeConfig {
    fn to_core(&self) -> Result<CoreSchedulerRuntimeConfig> {
        let mut core = CoreSchedulerRuntimeConfig::default();
        if let Some(value) = self.continuous_batching {
            core.continuous_batching = value;
        }
        core.policy = SchedulerPolicyConfig {
            mode: if let Some(value) = &self.policy {
                parse_scheduler_policy(value)?
            } else {
                core.policy.mode
            },
            decode_token_reserve: self
                .decode_token_reserve
                .unwrap_or(core.policy.decode_token_reserve),
            enable_adaptive_prefill_chunking: self
                .adaptive_prefill_chunking
                .unwrap_or(core.policy.enable_adaptive_prefill_chunking),
        };
        if let Some(value) = self.prefill_chunk_size {
            core.prefill_chunk_size = value;
        }
        core.max_running_requests = self.max_running_requests;
        core.max_queued_requests = self.max_queued_requests;
        Ok(core)
    }
}

#[napi(object)]
pub struct CacheRuntimeConfig {
    pub mode: Option<String>,
    pub retained_prefix_tokens: Option<i32>,
    pub snapshot_interval_tokens: Option<i32>,
    pub max_snapshot_entries: Option<i32>,
    pub max_snapshot_bytes: Option<f64>,
    pub max_session_entries: Option<i32>,
    pub cache_key_policy: Option<String>,
    pub enable_context_checkpoints: Option<bool>,
    pub checkpoint_every_tokens: Option<i32>,
}

impl CacheRuntimeConfig {
    fn to_core(&self) -> Result<cogentlm_engine::engine::CacheRuntimeConfig> {
        let mut core = cogentlm_engine::engine::CacheRuntimeConfig::default();
        if let Some(value) = &self.mode {
            core.mode = parse_kv_reuse_mode(value)?;
        }
        if let Some(value) = self.retained_prefix_tokens {
            core.retained_prefix_tokens = value;
        }
        if let Some(value) = self.snapshot_interval_tokens {
            core.snapshot_interval_tokens = value;
        }
        if let Some(value) = self.max_snapshot_entries {
            core.max_snapshot_entries = value;
        }
        if let Some(value) = self.max_snapshot_bytes {
            core.max_snapshot_bytes = value as usize;
        }
        if let Some(value) = self.max_session_entries {
            core.max_session_entries = value;
        }
        if let Some(value) = &self.cache_key_policy {
            core.cache_key_policy = parse_cache_key_policy(value)?;
        }
        if let Some(value) = self.enable_context_checkpoints {
            core.enable_context_checkpoints = value;
        }
        if let Some(value) = self.checkpoint_every_tokens {
            core.checkpoint_every_tokens = value;
        }
        Ok(core)
    }
}

#[napi(object)]
pub struct MultimodalRuntimeConfig {
    pub projector_path: Option<String>,
    pub use_gpu: Option<bool>,
    pub image_min_tokens: Option<i32>,
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
    pub max_gpu_models_per_device: Option<f64>,
    pub allow_cpu_models_while_gpu_loaded: Option<bool>,
    pub require_gpu_lease: Option<bool>,
    pub gpu_memory_safety_margin_bytes: Option<f64>,
}

impl ResidencyRuntimeConfig {
    fn to_core(&self) -> CoreResidencyRuntimeConfig {
        let mut core = CoreResidencyRuntimeConfig::default();
        if let Some(value) = self.max_gpu_models_per_device {
            core.max_gpu_models_per_device = value as usize;
        }
        if let Some(value) = self.allow_cpu_models_while_gpu_loaded {
            core.allow_cpu_models_while_gpu_loaded = value;
        }
        if let Some(value) = self.require_gpu_lease {
            core.require_gpu_lease = value;
        }
        if let Some(value) = self.gpu_memory_safety_margin_bytes {
            core.gpu_memory_safety_margin_bytes = value as u64;
        }
        core
    }
}

#[napi(object)]
pub struct ObservabilityRuntimeConfig {
    pub runtime_metrics: Option<bool>,
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
        let mut core = CoreNativeRuntimeConfig::default();
        if let Some(value) = &self.placement {
            core.placement = value.to_core()?;
        }
        if let Some(value) = &self.context {
            core.context = value.to_core()?;
        }
        if let Some(value) = &self.sampling {
            core.sampling = value.to_core()?;
        }
        if let Some(value) = &self.scheduler {
            core.scheduler = value.to_core()?;
        }
        if let Some(value) = &self.cache {
            core.cache = value.to_core()?;
        }
        if let Some(value) = &self.multimodal {
            core.multimodal = value.to_core();
        }
        if let Some(value) = &self.residency {
            core.residency = value.to_core();
        }
        if let Some(value) = &self.observability {
            core.observability = value.to_core();
        }
        Ok(core)
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
            backend: parse_backend_preference(self.backend.as_deref().unwrap_or("auto"))?,
            stats: parse_stats_mode(self.stats.as_deref().unwrap_or("basic"))?,
            runtime: self
                .runtime
                .as_ref()
                .map(NativeRuntimeConfig::to_core)
                .transpose()?
                .unwrap_or_default(),
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
        let max_tokens = self.max_tokens.unwrap_or(64);
        if max_tokens <= 0 {
            return Err(invalid_arg("maxTokens must be positive"));
        }
        Ok(CoreQueryOptions {
            context_key: self
                .context_key
                .clone()
                .unwrap_or_else(|| "default".to_string()),
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
pub struct RequestObservabilityMetrics {
    pub ttft_ms: f64,
    pub itl_avg_ms: f64,
    pub itl_p99_ms: f64,
    pub e2e_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub native_gpu_ms: f64,
    pub native_sync_ms: f64,
    pub native_logic_ms: f64,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cache_hits: i32,
    pub prefill_tokens: i32,
}

#[napi(object)]
pub struct GenerationResponse {
    pub request_id: u32,
    pub status: String,
    pub completed: bool,
    pub failed: bool,
    pub cancelled: bool,
    pub output_text: String,
    pub error_message: Option<String>,
    pub observability: RequestObservabilityMetrics,
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

    #[napi(getter)]
    pub fn closed(&self) -> Result<bool> {
        Ok(self.engine_guard()?.is_none())
    }

    #[napi]
    pub fn query(
        &self,
        prompt: String,
        options: Option<QueryOptions>,
        on_tokens: Option<TokenBatchCallback>,
    ) -> Result<AsyncTask<QueryTask>> {
        let options = query_options_or_default(options)?;
        Ok(AsyncTask::new(QueryTask {
            engine: self.inner.clone(),
            prompt,
            options,
            on_tokens,
        }))
    }

    #[napi]
    pub fn query_response(
        &self,
        prompt: String,
        options: Option<QueryOptions>,
    ) -> Result<AsyncTask<QueryResponseTask>> {
        let options = query_options_or_default(options)?;
        Ok(AsyncTask::new(QueryResponseTask {
            engine: self.inner.clone(),
            prompt,
            options,
        }))
    }

    #[napi]
    pub fn query_result(
        &self,
        prompt: String,
        options: Option<QueryOptions>,
    ) -> Result<AsyncTask<QueryResultTask>> {
        let options = query_options_or_default(options)?;
        Ok(AsyncTask::new(QueryResultTask {
            engine: self.inner.clone(),
            prompt,
            options,
        }))
    }

    #[napi]
    pub fn chat(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<QueryOptions>,
        on_tokens: Option<TokenBatchCallback>,
    ) -> Result<AsyncTask<ChatTextTask>> {
        let options = query_options_or_default(options)?;
        Ok(AsyncTask::new(ChatTextTask {
            engine: self.inner.clone(),
            messages: chat_messages_to_core(messages)?,
            options,
            on_tokens,
        }))
    }

    #[napi]
    pub fn chat_response(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<QueryOptions>,
    ) -> Result<AsyncTask<ChatResponseTask>> {
        let options = query_options_or_default(options)?;
        Ok(AsyncTask::new(ChatResponseTask {
            engine: self.inner.clone(),
            messages: chat_messages_to_core(messages)?,
            options,
        }))
    }

    #[napi]
    pub fn chat_result(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<QueryOptions>,
    ) -> Result<AsyncTask<ChatResultTask>> {
        let options = query_options_or_default(options)?;
        Ok(AsyncTask::new(ChatResultTask {
            engine: self.inner.clone(),
            messages: chat_messages_to_core(messages)?,
            options,
        }))
    }

    #[napi]
    pub fn close(&self) -> AsyncTask<CloseTask> {
        AsyncTask::new(CloseTask {
            engine: self.inner.clone(),
        })
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
            .map_err(|_| napi_error("engine events mutex is poisoned"))?;
        Ok(events.try_iter().map(engine_event_to_node).collect())
    }
}

impl CogentEngine {
    fn engine_guard(&self) -> Result<std::sync::MutexGuard<'_, Option<CoreCogentEngine>>> {
        self.inner
            .lock()
            .map_err(|_| napi_error("engine mutex is poisoned"))
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

    #[napi(getter)]
    pub fn closed(&self) -> Result<bool> {
        Ok(self.service_guard()?.is_none())
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
            options: model_load_options_or_default(options)?,
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
            options: model_load_options_or_default(options)?,
        }))
    }

    #[napi]
    pub fn load_installed(
        &self,
        model_id: String,
        options: Option<ModelLoadOptions>,
    ) -> Result<AsyncTask<ModelLoadInstalledTask>> {
        Ok(AsyncTask::new(ModelLoadInstalledTask {
            service: self.inner.clone(),
            events: self.events.clone(),
            model_id,
            options: model_load_options_or_default(options)?,
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
            options: query_options_or_default(options)?,
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
            options: query_options_or_default(options)?,
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
    pub fn close(&self) -> AsyncTask<ModelCloseTask> {
        AsyncTask::new(ModelCloseTask {
            service: self.inner.clone(),
            events: self.events.clone(),
        })
    }

    #[napi]
    pub fn drain_events(&self) -> Result<Vec<EngineEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|_| napi_error("model service events mutex is poisoned"))?;
        Ok(events
            .as_ref()
            .map(|receiver| receiver.try_iter().map(engine_event_to_node).collect())
            .unwrap_or_default())
    }
}

impl ModelService {
    fn service_guard(&self) -> Result<std::sync::MutexGuard<'_, Option<CoreModelService>>> {
        self.inner
            .lock()
            .map_err(|_| napi_error("model service mutex is poisoned"))
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
        let mut request = CoreQueryRequest::new(self.prompt.clone()).options(self.options.clone());
        if let Some(on_tokens) = self.on_tokens.clone() {
            request = request.on_tokens(move |batch| emit_token_batch(&on_tokens, batch));
        }
        with_engine(&self.engine, |engine| engine.query(request))
    }

    fn resolve(&mut self, _env: Env, result: Self::Output) -> Result<Self::JsValue> {
        Ok(request_result_to_node(result))
    }
}

pub struct QueryResponseTask {
    engine: SharedEngine,
    prompt: String,
    options: CoreQueryOptions,
}

impl Task for QueryResponseTask {
    type Output = GenerateResponse;
    type JsValue = GenerationResponse;

    fn compute(&mut self) -> Result<Self::Output> {
        let request = CoreQueryRequest::new(self.prompt.clone()).options(self.options.clone());
        with_engine(&self.engine, |engine| engine.query_response(request))
    }

    fn resolve(&mut self, _env: Env, response: Self::Output) -> Result<Self::JsValue> {
        Ok(response_to_node(response))
    }
}

pub struct QueryResultTask {
    engine: SharedEngine,
    prompt: String,
    options: CoreQueryOptions,
}

impl Task for QueryResultTask {
    type Output = CoreRequestResult;
    type JsValue = RequestResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let request = CoreQueryRequest::new(self.prompt.clone()).options(self.options.clone());
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
        let mut request = CoreChatRequest::new(self.messages.clone()).options(self.options.clone());
        if let Some(on_tokens) = self.on_tokens.clone() {
            request = request.on_tokens(move |batch| emit_token_batch(&on_tokens, batch));
        }
        with_engine(&self.engine, |engine| engine.chat(request))
    }

    fn resolve(&mut self, _env: Env, result: Self::Output) -> Result<Self::JsValue> {
        Ok(request_result_to_node(result))
    }
}

pub struct ChatResponseTask {
    engine: SharedEngine,
    messages: Vec<CoreChatMessage>,
    options: CoreQueryOptions,
}

impl Task for ChatResponseTask {
    type Output = GenerateResponse;
    type JsValue = GenerationResponse;

    fn compute(&mut self) -> Result<Self::Output> {
        let request = CoreChatRequest::new(self.messages.clone()).options(self.options.clone());
        with_engine(&self.engine, |engine| engine.chat_response(request))
    }

    fn resolve(&mut self, _env: Env, response: Self::Output) -> Result<Self::JsValue> {
        Ok(response_to_node(response))
    }
}

pub struct ChatResultTask {
    engine: SharedEngine,
    messages: Vec<CoreChatMessage>,
    options: CoreQueryOptions,
}

impl Task for ChatResultTask {
    type Output = CoreRequestResult;
    type JsValue = RequestResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let request = CoreChatRequest::new(self.messages.clone()).options(self.options.clone());
        with_engine(&self.engine, |engine| engine.chat(request))
    }

    fn resolve(&mut self, _env: Env, result: Self::Output) -> Result<Self::JsValue> {
        Ok(request_result_to_node(result))
    }
}

pub struct CloseTask {
    engine: SharedEngine,
}

impl Task for CloseTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<Self::Output> {
        let engine = self
            .engine
            .lock()
            .map_err(|_| napi_error("engine mutex is poisoned"))?
            .take();
        if let Some(engine) = engine {
            engine.close().map_err(core_error)?;
        }
        Ok(())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
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

pub struct ModelLoadInstalledTask {
    service: SharedModelService,
    events: SharedModelEvents,
    model_id: String,
    options: CoreModelLoadOptions,
}

impl Task for ModelLoadInstalledTask {
    type Output = CoreLoadedModelInfo;
    type JsValue = LoadedModelInfo;

    fn compute(&mut self) -> Result<Self::Output> {
        with_model_service_mut(&self.service, |service| {
            let loaded = service.load_installed(&self.model_id, self.options.clone())?;
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
        self.events
            .lock()
            .map_err(|_| napi_error("model service events mutex is poisoned"))?
            .take();
        Ok(())
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
        let mut request = CoreQueryRequest::new(self.prompt.clone()).options(self.options.clone());
        if let Some(on_tokens) = self.on_tokens.clone() {
            request = request.on_tokens(move |batch| emit_token_batch(&on_tokens, batch));
        }
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
        let mut request = CoreChatRequest::new(self.messages.clone()).options(self.options.clone());
        if let Some(on_tokens) = self.on_tokens.clone() {
            request = request.on_tokens(move |batch| emit_token_batch(&on_tokens, batch));
        }
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

pub struct ModelCloseTask {
    service: SharedModelService,
    events: SharedModelEvents,
}

impl Task for ModelCloseTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<Self::Output> {
        let service = self
            .service
            .lock()
            .map_err(|_| napi_error("model service mutex is poisoned"))?
            .take();
        if let Some(mut service) = service {
            service.close().map_err(model_error)?;
        }
        self.events
            .lock()
            .map_err(|_| napi_error("model service events mutex is poisoned"))?
            .take();
        Ok(())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
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
        .map_err(|_| napi_error("engine mutex is poisoned"))?;
    let engine = guard
        .as_ref()
        .ok_or_else(|| napi_error("engine is closed"))?;
    f(engine).map_err(core_error)
}

fn with_model_service<T>(
    service: &SharedModelService,
    f: impl FnOnce(&CoreModelService) -> std::result::Result<T, cogentlm_engine::lifecycle::ModelError>,
) -> Result<T> {
    let guard = service
        .lock()
        .map_err(|_| napi_error("model service mutex is poisoned"))?;
    let service = guard
        .as_ref()
        .ok_or_else(|| napi_error("model service is closed"))?;
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
        .map_err(|_| napi_error("model service mutex is poisoned"))?;
    let service = guard
        .as_mut()
        .ok_or_else(|| napi_error("model service is closed"))?;
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
                "model service events mutex is poisoned".to_string(),
            )
        })?
        .replace(receiver);
    Ok(())
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

fn model_load_options_or_default(
    options: Option<ModelLoadOptions>,
) -> Result<CoreModelLoadOptions> {
    options
        .as_ref()
        .map(ModelLoadOptions::to_core)
        .transpose()
        .map(|options| options.unwrap_or_default())
}

fn query_options_or_default(options: Option<QueryOptions>) -> Result<CoreQueryOptions> {
    options
        .as_ref()
        .map(QueryOptions::to_core)
        .transpose()
        .map(|options| options.unwrap_or_default())
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

fn response_to_node(response: GenerateResponse) -> GenerationResponse {
    let status = response_status_name(response.status).to_string();
    GenerationResponse {
        request_id: response.request_id,
        status,
        completed: response.status == GenerateResponseStatus::Completed,
        failed: response.status == GenerateResponseStatus::Failed,
        cancelled: response.status == GenerateResponseStatus::Cancelled,
        output_text: response.output_text,
        error_message: (!response.error_message.is_empty()).then_some(response.error_message),
        observability: metrics_to_node(response.runtime_observability),
    }
}

fn request_result_to_node(result: CoreRequestResult) -> RequestResult {
    RequestResult {
        id: result.id,
        text: result.text,
        finish_reason: finish_reason_name(result.finish_reason).to_string(),
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
        backend: backend_selection_to_node(loaded.backend),
        runtime_fingerprint: loaded.runtime_fingerprint,
    }
}

fn model_info_to_node(model: CoreManagedModelInfo) -> ManagedModelInfo {
    ManagedModelInfo {
        id: model.id,
        name: model.name,
        modality: model_modality_name(model.modality).to_string(),
        status: model_status_name(model.status).to_string(),
        source: model_source_kind_name(model.source).to_string(),
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

fn backend_selection_to_node(selection: CoreBackendSelection) -> BackendSelection {
    BackendSelection {
        requested: backend_preference_name(selection.requested).to_string(),
        selected: selection.selected,
        available: selection.available,
        gpu_offload_expected: selection.gpu_offload_expected,
        reason: selection.reason,
    }
}

fn engine_state_to_node(state: CoreEngineState) -> EngineState {
    EngineState {
        status: engine_status_name(state.status).to_string(),
        model: state.model.map(model_state_to_node),
        backend: backend_info_to_node(state.backend),
        runtime: state.runtime.map(resolved_runtime_limits_to_node),
        requests: state
            .requests
            .into_iter()
            .map(request_state_to_node)
            .collect(),
        stats: engine_stats_to_node(state.stats),
        updated_at_unix_ms: state.updated_at_unix_ms as f64,
    }
}

fn model_service_state_to_node(state: CoreModelServiceState) -> ModelServiceState {
    ModelServiceState {
        status: engine_status_name(state.status).to_string(),
        model: state.model.map(model_info_to_node),
        backend: backend_info_to_node(state.backend),
        runtime: state.runtime.map(resolved_runtime_limits_to_node),
        requests: state
            .requests
            .into_iter()
            .map(request_state_to_node)
            .collect(),
        stats: engine_stats_to_node(state.stats),
        updated_at_unix_ms: state.updated_at_unix_ms as f64,
    }
}

fn model_state_to_node(model: CoreModelState) -> ModelState {
    ModelState {
        id: model.id,
        name: model.name,
    }
}

fn request_state_to_node(request: CoreRequestState) -> RequestState {
    RequestState {
        id: request.id,
        status: request_status_name(request.status).to_string(),
        input_tokens: request.input_tokens,
        output_tokens: request.output_tokens,
    }
}

fn backend_info_to_node(backend: CoreBackendInfo) -> BackendInfo {
    BackendInfo {
        selected: backend.selected,
        available: backend.available,
        devices: backend
            .devices
            .into_iter()
            .map(backend_device_to_node)
            .collect(),
    }
}

fn backend_device_to_node(device: CoreBackendDevice) -> BackendDevice {
    BackendDevice {
        id: device.id,
        name: device.name,
        r#type: device.device_type,
        memory_total_bytes: device.memory_total_bytes.map(|value| value as f64),
        memory_free_bytes: device.memory_free_bytes.map(|value| value as f64),
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
            r#type: "state".to_string(),
            state: Some(engine_state_to_node(state)),
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
            r#type: "load-progress".to_string(),
            state: None,
            loaded_bytes: Some(loaded_bytes as f64),
            total_bytes: total_bytes.map(|value| value as f64),
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
            r#type: "request-started".to_string(),
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
            r#type: "request-completed".to_string(),
            state: None,
            loaded_bytes: None,
            total_bytes: None,
            asset_name: None,
            request_id: None,
            stream_id: None,
            result: Some(request_result_to_node(result)),
            error: None,
        },
        CoreEngineEvent::RequestFailed { request_id, error } => EngineEvent {
            r#type: "request-failed".to_string(),
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
            r#type: "closed".to_string(),
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

fn metrics_to_node(metrics: RuntimeObservabilityMetrics) -> RequestObservabilityMetrics {
    RequestObservabilityMetrics {
        ttft_ms: metrics.ttft_ms,
        itl_avg_ms: metrics.itl_avg_ms,
        itl_p99_ms: metrics.itl_p99_ms,
        e2e_ms: metrics.e2e_ms,
        prefill_ms: metrics.prefill_ms,
        decode_ms: metrics.decode_ms,
        native_gpu_ms: metrics.native_gpu_ms,
        native_sync_ms: metrics.native_sync_ms,
        native_logic_ms: metrics.native_logic_ms,
        input_tokens: metrics.input_tokens,
        output_tokens: metrics.output_tokens,
        cache_hits: metrics.cache_hits,
        prefill_tokens: metrics.prefill_tokens,
    }
}

fn parse_backend_preference(value: &str) -> Result<CoreBackendPreference> {
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

fn parse_stats_mode(value: &str) -> Result<StatsMode> {
    match normalize_choice(value).as_str() {
        "off" => Ok(StatsMode::Off),
        "basic" => Ok(StatsMode::Basic),
        "profile" => Ok(StatsMode::Profile),
        _ => Err(invalid_arg("stats must be one of: off, basic, profile")),
    }
}

fn parse_gpu_layers(value: &str) -> Result<GpuLayerConfig> {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("auto") {
        Ok(GpuLayerConfig::Auto)
    } else if trimmed.eq_ignore_ascii_case("all") || trimmed.eq_ignore_ascii_case("full") {
        Ok(GpuLayerConfig::All)
    } else {
        trimmed
            .parse::<i32>()
            .map(GpuLayerConfig::Count)
            .map_err(|_| invalid_arg("gpuLayers must be one of: auto, all, or an integer count"))
    }
}

fn parse_split_mode(value: &str) -> Result<SplitMode> {
    match normalize_choice(value).as_str() {
        "none" => Ok(SplitMode::None),
        "layer" => Ok(SplitMode::Layer),
        "row" => Ok(SplitMode::Row),
        "tensor" => Ok(SplitMode::Tensor),
        _ => Err(invalid_arg(
            "splitMode must be one of: none, layer, row, tensor",
        )),
    }
}

fn parse_flash_attention(value: &str) -> Result<FlashAttentionMode> {
    match normalize_choice(value).as_str() {
        "auto" => Ok(FlashAttentionMode::Auto),
        "enabled" | "enable" | "on" | "true" => Ok(FlashAttentionMode::Enabled),
        "disabled" | "disable" | "off" | "false" => Ok(FlashAttentionMode::Disabled),
        _ => Err(invalid_arg(
            "flashAttention must be one of: auto, enabled, disabled",
        )),
    }
}

fn parse_kv_cache_type(value: &str) -> Result<KvCacheType> {
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

fn parse_rope_scaling(value: &str) -> Result<RopeScaling> {
    match normalize_choice(value).as_str() {
        "none" => Ok(RopeScaling::None),
        "linear" => Ok(RopeScaling::Linear),
        "yarn" => Ok(RopeScaling::Yarn),
        _ => Err(invalid_arg(
            "ropeScaling must be one of: none, linear, yarn",
        )),
    }
}

fn parse_kv_reuse_mode(value: &str) -> Result<KvReuseMode> {
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

fn parse_cache_key_policy(value: &str) -> Result<CacheKeyPolicy> {
    match normalize_choice(value).as_str() {
        "context_key" => Ok(CacheKeyPolicy::ContextKey),
        "prompt_hash" => Ok(CacheKeyPolicy::PromptHash),
        _ => Err(invalid_arg(
            "cacheKeyPolicy must be one of: context-key, prompt-hash",
        )),
    }
}

fn parse_sampler_stage(value: &str) -> Result<SamplerStage> {
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
            "sampler stage must be one of: dry, top-k, typical-p, top-p, top-n-sigma, min-p, xtc, temperature, infill, penalties, adaptive-p",
        )),
    }
}

fn parse_scheduler_policy(value: &str) -> Result<SchedulerPolicyMode> {
    match normalize_choice(value).as_str() {
        "latency_first" | "latency" => Ok(SchedulerPolicyMode::LatencyFirst),
        "balanced" | "balance" => Ok(SchedulerPolicyMode::Balanced),
        "throughput_first" | "throughput" => Ok(SchedulerPolicyMode::ThroughputFirst),
        _ => Err(invalid_arg(
            "schedulerPolicy must be one of: latency-first, balanced, throughput-first",
        )),
    }
}

fn parse_chat_role(value: &str) -> Result<CoreChatRole> {
    match normalize_choice(value).as_str() {
        "system" => Ok(CoreChatRole::System),
        "user" => Ok(CoreChatRole::User),
        "assistant" => Ok(CoreChatRole::Assistant),
        _ => Err(invalid_arg(
            "chat role must be one of: system, user, assistant",
        )),
    }
}

fn normalize_choice(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
}

fn response_status_name(status: GenerateResponseStatus) -> &'static str {
    match status {
        GenerateResponseStatus::Pending => "pending",
        GenerateResponseStatus::Completed => "completed",
        GenerateResponseStatus::Cancelled => "cancelled",
        GenerateResponseStatus::Failed => "failed",
    }
}

fn engine_status_name(status: CoreEngineStatus) -> &'static str {
    match status {
        CoreEngineStatus::Idle => "idle",
        CoreEngineStatus::Loading => "loading",
        CoreEngineStatus::Ready => "ready",
        CoreEngineStatus::Running => "running",
        CoreEngineStatus::Error => "error",
        CoreEngineStatus::Closed => "closed",
    }
}

fn request_status_name(status: CoreRequestStatus) -> &'static str {
    match status {
        CoreRequestStatus::Queued => "queued",
        CoreRequestStatus::Prefill => "prefill",
        CoreRequestStatus::Decode => "decode",
        CoreRequestStatus::Completed => "completed",
        CoreRequestStatus::Failed => "failed",
        CoreRequestStatus::Cancelled => "cancelled",
    }
}

fn finish_reason_name(reason: CoreFinishReason) -> &'static str {
    match reason {
        CoreFinishReason::Stop => "stop",
        CoreFinishReason::Length => "length",
        CoreFinishReason::Cancelled => "cancelled",
        CoreFinishReason::Error => "error",
    }
}

fn backend_preference_name(backend: CoreBackendPreference) -> &'static str {
    match backend {
        CoreBackendPreference::Auto => "auto",
        CoreBackendPreference::Cpu => "cpu",
        CoreBackendPreference::Cuda => "cuda",
        CoreBackendPreference::Metal => "metal",
        CoreBackendPreference::Vulkan => "vulkan",
        CoreBackendPreference::WebGpu => "webgpu",
    }
}

fn model_modality_name(modality: CoreModelModality) -> &'static str {
    match modality {
        CoreModelModality::Text => "text",
        CoreModelModality::Vision => "vision",
    }
}

fn model_status_name(status: CoreModelStatus) -> &'static str {
    match status {
        CoreModelStatus::Ready => "ready",
        CoreModelStatus::NeedsProjector => "needs_projector",
        CoreModelStatus::Broken => "broken",
    }
}

fn model_source_kind_name(source: CoreModelSourceKind) -> &'static str {
    match source {
        CoreModelSourceKind::Local => "local",
        CoreModelSourceKind::Remote => "remote",
    }
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
