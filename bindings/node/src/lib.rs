//! N-API bindings for the public Sipp Node package.
//!
//! This crate exposes the shared Rust client facade to JavaScript while
//! translating Node request objects, async tasks, and native errors.

use std::sync::{Arc, Mutex};

use futures::executor::block_on;
use futures::StreamExt;
use napi::bindgen_prelude::{AsyncTask, Buffer, Either, Env};
use napi::{Error, JsValue, Result, Status, Task};
use napi_derive::napi;
use sipp::backend::{
    backend_observability_json as core_backend_observability_json,
    set_llama_log_quiet as core_set_llama_log_quiet,
};
use sipp::core::TokenUsage as CoreTokenUsage;
use sipp::engine::protocol::RequestStats as CoreRequestStats;
use sipp::engine::{PoolingType as CorePoolingType, TokenBatch as CoreTokenBatch};
use sipp::{
    EndpointDescriptor as CoreEndpointDescriptor, EndpointRef as CoreEndpointRef,
    ProviderEndpointError as CoreProviderEndpointError,
    ProviderEndpointErrorKind as CoreProviderEndpointErrorKind,
    SippCancellationHandle as CoreCancellationHandle,
    SippCancellationReason as CoreCancellationReason, SippClient as CoreClient,
    SippEmbeddingResponse as CoreClientEmbeddingResponse,
    SippEmbeddingResponseFuture as CoreClientEmbeddingResponseFuture,
    SippEmbeddingRun as CoreClientEmbeddingRun, SippError as CoreClientError,
    SippRequestContext as CoreClientRequestContext, SippTextResponse as CoreClientTextResponse,
    SippTextResponseFuture as CoreClientTextResponseFuture, SippTextRun as CoreClientTextRun,
    SippTokenBatches as CoreClientTokenBatches,
};
use sipp_binding_dto as dto;

fn convert_error(error: dto::ConvertError) -> Error {
    match error {
        dto::ConvertError::InvalidArg(message) => Error::new(Status::InvalidArg, message),
    }
}

type SharedSippClient = Arc<Mutex<CoreClient>>;
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

impl From<&LogitBiasConfig> for dto::LogitBiasConfig {
    fn from(value: &LogitBiasConfig) -> Self {
        Self {
            token: value.token,
            bias: value.bias,
        }
    }
}

impl From<&SamplingRuntimeConfig> for dto::SamplingRuntimeConfig {
    fn from(value: &SamplingRuntimeConfig) -> Self {
        Self {
            samplers: value.samplers.clone(),
            seed: value.seed,
            top_k: value.top_k,
            top_p: value.top_p,
            min_p: value.min_p,
            typical_p: value.typical_p,
            xtc_probability: value.xtc_probability,
            xtc_threshold: value.xtc_threshold,
            top_n_sigma: value.top_n_sigma,
            temperature: value.temperature,
            dynatemp_range: value.dynatemp_range,
            dynatemp_exponent: value.dynatemp_exponent,
            repeat_last_n: value.repeat_last_n,
            repeat_penalty: value.repeat_penalty,
            frequency_penalty: value.frequency_penalty,
            presence_penalty: value.presence_penalty,
            dry_multiplier: value.dry_multiplier,
            dry_base: value.dry_base,
            dry_allowed_length: value.dry_allowed_length,
            dry_penalty_last_n: value.dry_penalty_last_n,
            dry_sequence_breakers: value.dry_sequence_breakers.clone(),
            mirostat: value.mirostat,
            mirostat_tau: value.mirostat_tau,
            mirostat_eta: value.mirostat_eta,
            min_keep: value.min_keep,
            n_probs: value.n_probs,
            logit_bias: value
                .logit_bias
                .as_ref()
                .map(|biases| biases.iter().map(dto::LogitBiasConfig::from).collect()),
            ignore_eos: value.ignore_eos,
            grammar_lazy: value.grammar_lazy,
            preserved_tokens: value.preserved_tokens.clone(),
            backend_sampling: value.backend_sampling,
        }
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

impl From<&GpuLayerCountConfig> for dto::GpuLayerCountConfig {
    fn from(value: &GpuLayerCountConfig) -> Self {
        Self { count: value.count }
    }
}

impl From<&ModelPlacementConfig> for dto::ModelPlacementConfig {
    fn from(value: &ModelPlacementConfig) -> Self {
        Self {
            devices: value.devices.clone(),
            gpu_layers: value.gpu_layers.as_ref().map(|value| match value {
                Either::A(preset) => dto::GpuLayers::Preset(preset.clone()),
                Either::B(count) => dto::GpuLayers::Count(count.into()),
            }),
            split_mode: value.split_mode.clone(),
            main_gpu: value.main_gpu,
            tensor_split: value.tensor_split.clone(),
            use_mmap: value.use_mmap,
            use_mlock: value.use_mlock,
            fit_params: value.fit_params,
            fit_params_min_ctx: value.fit_params_min_ctx,
            fit_params_target_bytes: value.fit_params_target_bytes.clone(),
            check_tensors: value.check_tensors,
            no_extra_bufts: value.no_extra_bufts,
            no_host: value.no_host,
        }
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

impl From<&ContextRuntimeConfig> for dto::ContextRuntimeConfig {
    fn from(value: &ContextRuntimeConfig) -> Self {
        Self {
            n_ctx: value.n_ctx,
            n_batch: value.n_batch,
            n_ubatch: value.n_ubatch,
            n_parallel: value.n_parallel,
            n_threads: value.n_threads,
            n_threads_batch: value.n_threads_batch,
            flash_attention: value.flash_attention.clone(),
            kv_unified: value.kv_unified,
            cache_type_k: value.cache_type_k.clone(),
            cache_type_v: value.cache_type_v.clone(),
            offload_kqv: value.offload_kqv,
            op_offload: value.op_offload,
            swa_full: value.swa_full,
            warmup: value.warmup,
            rope_scaling: value.rope_scaling.clone(),
            rope_freq_base: value.rope_freq_base,
            rope_freq_scale: value.rope_freq_scale,
            yarn_orig_ctx: value.yarn_orig_ctx,
            yarn_ext_factor: value.yarn_ext_factor,
            yarn_attn_factor: value.yarn_attn_factor,
            yarn_beta_fast: value.yarn_beta_fast,
            yarn_beta_slow: value.yarn_beta_slow,
            embeddings: value.embeddings,
            pooling: value.pooling.map(dto::PoolingType::from),
        }
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

impl From<&SchedulerPolicyConfig> for dto::SchedulerPolicyConfig {
    fn from(value: &SchedulerPolicyConfig) -> Self {
        Self {
            mode: value.mode.clone(),
            decode_token_reserve: value.decode_token_reserve,
            enable_adaptive_prefill_chunking: value.enable_adaptive_prefill_chunking,
        }
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

impl From<&SchedulerRuntimeConfig> for dto::SchedulerRuntimeConfig {
    fn from(value: &SchedulerRuntimeConfig) -> Self {
        Self {
            continuous_batching: value.continuous_batching,
            policy: value.policy.as_ref().map(dto::SchedulerPolicyConfig::from),
            prefill_chunk_size: value.prefill_chunk_size,
            max_running_requests: value.max_running_requests,
            max_queued_requests: value.max_queued_requests,
        }
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

impl From<&CacheRuntimeConfig> for dto::CacheRuntimeConfig {
    fn from(value: &CacheRuntimeConfig) -> Self {
        Self {
            mode: value.mode.clone(),
            retained_prefix_tokens: value.retained_prefix_tokens,
            snapshot_interval_tokens: value.snapshot_interval_tokens,
            max_snapshot_entries: value.max_snapshot_entries,
            max_snapshot_bytes: value.max_snapshot_bytes,
        }
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

impl From<&MultimodalRuntimeConfig> for dto::MultimodalRuntimeConfig {
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

impl From<&ResidencyRuntimeConfig> for dto::ResidencyRuntimeConfig {
    fn from(value: &ResidencyRuntimeConfig) -> Self {
        Self {
            max_gpu_models_per_device: value.max_gpu_models_per_device,
            allow_cpu_models_while_gpu_loaded: value.allow_cpu_models_while_gpu_loaded,
            require_gpu_lease: value.require_gpu_lease,
            gpu_memory_safety_margin_bytes: value.gpu_memory_safety_margin_bytes,
        }
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

impl From<&ObservabilityRuntimeConfig> for dto::ObservabilityRuntimeConfig {
    fn from(value: &ObservabilityRuntimeConfig) -> Self {
        Self {
            runtime_metrics: value.runtime_metrics,
            backend_profiling: value.backend_profiling,
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

impl From<&NativeRuntimeConfig> for dto::NativeRuntimeConfig {
    fn from(value: &NativeRuntimeConfig) -> Self {
        Self {
            placement: value
                .placement
                .as_ref()
                .map(dto::ModelPlacementConfig::from),
            context: value.context.as_ref().map(dto::ContextRuntimeConfig::from),
            sampling: value
                .sampling
                .as_ref()
                .map(dto::SamplingRuntimeConfig::from),
            scheduler: value
                .scheduler
                .as_ref()
                .map(dto::SchedulerRuntimeConfig::from),
            cache: value.cache.as_ref().map(dto::CacheRuntimeConfig::from),
            multimodal: value
                .multimodal
                .as_ref()
                .map(dto::MultimodalRuntimeConfig::from),
            residency: value
                .residency
                .as_ref()
                .map(dto::ResidencyRuntimeConfig::from),
            observability: value
                .observability
                .as_ref()
                .map(dto::ObservabilityRuntimeConfig::from),
        }
    }
}

/// Address of a registered inference endpoint.
#[napi(object)]
pub struct EndpointRef {
    pub kind: String,
    pub id: String,
}

impl From<EndpointRef> for dto::EndpointRef {
    fn from(value: EndpointRef) -> Self {
        Self {
            kind: value.kind,
            id: value.id,
        }
    }
}

/// Shared generation options for text-producing requests.
#[napi(object)]
pub struct SippTextOptions {
    #[napi(js_name = "maxTokens")]
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    #[napi(js_name = "topP")]
    pub top_p: Option<f64>,
    pub stop: Option<Vec<String>>,
}

impl From<SippTextOptions> for dto::SippTextOptions {
    fn from(value: SippTextOptions) -> Self {
        Self {
            max_tokens: value.max_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            stop: value.stop,
        }
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

impl From<LocalTextOptions> for dto::LocalTextOptions {
    fn from(value: LocalTextOptions) -> Self {
        Self {
            context_key: value.context_key,
            grammar: value.grammar,
            json_schema: value.json_schema,
            sampling: value
                .sampling
                .as_ref()
                .map(dto::SamplingRuntimeConfig::from),
            media: value
                .media
                .map(|buffers| buffers.into_iter().map(Vec::<u8>::from).collect())
                .unwrap_or_default(),
        }
    }
}

/// Local-only embedding options for context and vector normalization.
#[napi(object)]
pub struct LocalEmbedOptions {
    #[napi(js_name = "contextKey")]
    pub context_key: Option<String>,
    pub normalize: Option<bool>,
}

impl From<LocalEmbedOptions> for dto::LocalEmbedOptions {
    fn from(value: LocalEmbedOptions) -> Self {
        Self {
            context_key: value.context_key,
            normalize: value.normalize,
        }
    }
}

/// Prompt completion request routed to an inference endpoint.
#[napi(object)]
pub struct SippQueryRequest {
    #[napi(js_name = "requestId")]
    pub request_id: Option<String>,
    pub endpoint: Option<EndpointRef>,
    pub prompt: String,
    pub options: Option<SippTextOptions>,
    pub local: Option<LocalTextOptions>,
    #[napi(js_name = "endpointOptions")]
    pub endpoint_options: Option<serde_json::Value>,
    #[napi(js_name = "providerOptions")]
    pub provider_options: Option<serde_json::Value>,
    #[napi(js_name = "emitTokens")]
    pub emit_tokens: Option<bool>,
}

impl From<SippQueryRequest> for dto::SippQueryRequest {
    fn from(value: SippQueryRequest) -> Self {
        Self {
            request_id: value.request_id,
            endpoint: value.endpoint.map(dto::EndpointRef::from),
            prompt: value.prompt,
            options: value.options.map(dto::SippTextOptions::from),
            local: value.local.map(dto::LocalTextOptions::from),
            endpoint_options: value.endpoint_options,
            provider_options: value.provider_options,
            emit_tokens: value.emit_tokens,
        }
    }
}

/// Chat completion request routed to an inference endpoint.
#[napi(object)]
pub struct SippChatRequest {
    #[napi(js_name = "requestId")]
    pub request_id: Option<String>,
    pub endpoint: Option<EndpointRef>,
    pub messages: Vec<ChatMessage>,
    pub options: Option<SippTextOptions>,
    pub local: Option<LocalTextOptions>,
    #[napi(js_name = "endpointOptions")]
    pub endpoint_options: Option<serde_json::Value>,
    #[napi(js_name = "providerOptions")]
    pub provider_options: Option<serde_json::Value>,
    #[napi(js_name = "emitTokens")]
    pub emit_tokens: Option<bool>,
}

impl From<SippChatRequest> for dto::SippChatRequest {
    fn from(value: SippChatRequest) -> Self {
        Self {
            request_id: value.request_id,
            endpoint: value.endpoint.map(dto::EndpointRef::from),
            messages: value
                .messages
                .into_iter()
                .map(dto::ChatMessage::from)
                .collect(),
            options: value.options.map(dto::SippTextOptions::from),
            local: value.local.map(dto::LocalTextOptions::from),
            endpoint_options: value.endpoint_options,
            provider_options: value.provider_options,
            emit_tokens: value.emit_tokens,
        }
    }
}

/// Embedding request routed to an inference endpoint.
#[napi(object)]
pub struct SippEmbedRequest {
    #[napi(js_name = "requestId")]
    pub request_id: Option<String>,
    pub endpoint: Option<EndpointRef>,
    pub input: String,
    pub local: Option<LocalEmbedOptions>,
    #[napi(js_name = "endpointOptions")]
    pub endpoint_options: Option<serde_json::Value>,
    #[napi(js_name = "providerOptions")]
    pub provider_options: Option<serde_json::Value>,
}

impl From<SippEmbedRequest> for dto::SippEmbedRequest {
    fn from(value: SippEmbedRequest) -> Self {
        Self {
            request_id: value.request_id,
            endpoint: value.endpoint.map(dto::EndpointRef::from),
            input: value.input,
            local: value.local.map(dto::LocalEmbedOptions::from),
            endpoint_options: value.endpoint_options,
            provider_options: value.provider_options,
        }
    }
}

/// Role/content chat message accepted by chat requests.
#[napi(object)]
#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Authentication configuration for a gateway endpoint.
#[napi(object)]
pub struct GatewayAuthentication {
    pub kind: String,
    pub value: Option<String>,
    #[napi(js_name = "headerName")]
    pub header_name: Option<String>,
}

/// Static header entry for an OpenAI-compatible provider descriptor.
#[napi(object)]
pub struct ProviderStaticHeader {
    pub name: String,
    pub value: String,
}

impl From<ChatMessage> for dto::ChatMessage {
    fn from(value: ChatMessage) -> Self {
        Self {
            role: value.role,
            content: value.content,
        }
    }
}

impl From<&GatewayAuthentication> for dto::GatewayAuthentication {
    fn from(value: &GatewayAuthentication) -> Self {
        Self {
            kind: value.kind.clone(),
            value: value.value.clone(),
            header_name: value.header_name.clone(),
        }
    }
}

impl From<&ProviderStaticHeader> for dto::ProviderStaticHeader {
    fn from(value: &ProviderStaticHeader) -> Self {
        Self {
            name: value.name.clone(),
            value: value.value.clone(),
        }
    }
}

/// Local or direct provider endpoint descriptor accepted by add.
#[napi(object)]
pub struct EndpointDescriptor {
    pub kind: String,
    #[napi(js_name = "modelPath")]
    pub model_path: Option<String>,
    pub config: Option<NativeRuntimeConfig>,
    #[napi(js_name = "baseUrl")]
    pub base_url: Option<String>,
    pub target: Option<String>,
    pub authentication: Option<GatewayAuthentication>,
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
    #[napi(js_name = "correlationHeader")]
    pub correlation_header: Option<String>,
    #[napi(js_name = "queryRoute")]
    pub query_route: Option<String>,
    #[napi(js_name = "chatRoute")]
    pub chat_route: Option<String>,
    #[napi(js_name = "embedRoute")]
    pub embed_route: Option<String>,
    #[napi(js_name = "protocolOptions")]
    pub protocol_options: Option<serde_json::Value>,
}

impl From<&EndpointDescriptor> for dto::EndpointDescriptor {
    fn from(value: &EndpointDescriptor) -> Self {
        Self {
            kind: value.kind.clone(),
            model_path: value.model_path.clone(),
            config: value.config.as_ref().map(dto::NativeRuntimeConfig::from),
            base_url: value.base_url.clone(),
            target: value.target.clone(),
            authentication: value
                .authentication
                .as_ref()
                .map(dto::GatewayAuthentication::from),
            provider: value.provider.clone(),
            model: value.model.clone(),
            api_key: value.api_key.clone(),
            timeout_ms: value.timeout_ms.map(u64::from),
            version: value.version.clone(),
            auth_header_name: value.auth_header_name.clone(),
            auth_header_value: value.auth_header_value.clone(),
            static_headers: value.static_headers.as_ref().map(|headers| {
                headers
                    .iter()
                    .map(dto::ProviderStaticHeader::from)
                    .collect()
            }),
            correlation_header: value.correlation_header.clone(),
            query_route: value.query_route.clone(),
            chat_route: value.chat_route.clone(),
            embed_route: value.embed_route.clone(),
            protocol_options: value.protocol_options.clone(),
        }
    }
}

/// Token counts reported by inference endpoints.
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

impl From<PoolingType> for dto::PoolingType {
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
pub struct SippTextResponse {
    pub endpoint: EndpointRef,
    pub text: String,
    #[napi(js_name = "finishReason")]
    pub finish_reason: String,
    pub usage: Option<TokenUsage>,
    #[napi(js_name = "localStats")]
    pub local_stats: Option<RequestStats>,
    pub metadata: SippResponseMetadata,
}

/// Final vector response from an embedding request.
#[napi(object)]
pub struct SippEmbeddingResponse {
    pub endpoint: EndpointRef,
    pub values: Vec<f64>,
    pub usage: Option<TokenUsage>,
    #[napi(js_name = "localStats")]
    pub local_stats: Option<RequestStats>,
    pub pooling: Option<PoolingType>,
    pub normalized: Option<bool>,
    pub metadata: SippResponseMetadata,
}

/// Request and upstream correlation metadata.
#[napi(object)]
pub struct SippResponseMetadata {
    #[napi(js_name = "requestId")]
    pub request_id: Option<String>,
    #[napi(js_name = "upstreamRequestId")]
    pub upstream_request_id: Option<String>,
    #[napi(js_name = "upstreamResponseId")]
    pub upstream_response_id: Option<String>,
}

fn endpoint_ref_to_node(endpoint: CoreEndpointRef) -> EndpointRef {
    let endpoint = dto::EndpointRef::from(endpoint);
    EndpointRef {
        kind: endpoint.kind,
        id: endpoint.id,
    }
}

fn token_usage_to_node(usage: CoreTokenUsage) -> TokenUsage {
    let usage = dto::TokenUsage::from(usage);
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
    }
}

fn response_metadata_to_node(metadata: sipp::SippResponseMetadata) -> SippResponseMetadata {
    SippResponseMetadata {
        request_id: metadata.request_id,
        upstream_request_id: metadata.upstream_request_id,
        upstream_response_id: metadata.upstream_response_id,
    }
}

fn sipp_text_response_to_node(response: CoreClientTextResponse) -> SippTextResponse {
    SippTextResponse {
        endpoint: endpoint_ref_to_node(response.endpoint),
        text: response.text,
        finish_reason: response.finish_reason.as_str().to_string(),
        usage: response.usage.map(token_usage_to_node),
        local_stats: response.local_stats.map(request_stats_to_node),
        metadata: response_metadata_to_node(response.metadata),
    }
}

fn sipp_embedding_response_to_node(response: CoreClientEmbeddingResponse) -> SippEmbeddingResponse {
    SippEmbeddingResponse {
        endpoint: endpoint_ref_to_node(response.endpoint),
        values: response.values.into_iter().map(f64::from).collect(),
        usage: response.usage.map(token_usage_to_node),
        local_stats: response.local_stats.map(request_stats_to_node),
        pooling: response.pooling.map(PoolingType::from),
        normalized: response.normalized,
        metadata: response_metadata_to_node(response.metadata),
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

/// Client facade for registered inference endpoints.
#[napi(js_name = "SippClient")]
pub struct SippClient {
    inner: SharedSippClient,
}

#[napi]
impl SippClient {
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
            descriptor: {
                let descriptor = dto::EndpointDescriptor::from(&descriptor);
                CoreEndpointDescriptor::try_from(&descriptor).map_err(convert_error)?
            },
        }))
    }

    #[napi(ts_return_type = "SippTextRun")]
    pub fn query(&self, request: SippQueryRequest) -> Result<SippTextRun> {
        let context = CoreClientRequestContext {
            request_id: request.request_id.clone(),
        };
        let request = dto::SippQueryRequest::from(request);
        let request = sipp::SippQueryRequest::try_from(request).map_err(convert_error)?;
        let run = self
            .inner
            .lock()
            .map_err(|_| napi_error(CLIENT_MUTEX_POISONED))?
            .query_with_context(context, request);
        Ok(SippTextRun::from_core(run))
    }

    #[napi(ts_return_type = "SippTextRun")]
    pub fn chat(&self, request: SippChatRequest) -> Result<SippTextRun> {
        let context = CoreClientRequestContext {
            request_id: request.request_id.clone(),
        };
        let request = dto::SippChatRequest::from(request);
        let request = sipp::SippChatRequest::try_from(request).map_err(convert_error)?;
        let run = self
            .inner
            .lock()
            .map_err(|_| napi_error(CLIENT_MUTEX_POISONED))?
            .chat_with_context(context, request);
        Ok(SippTextRun::from_core(run))
    }

    #[napi(ts_return_type = "SippEmbeddingRun")]
    pub fn embed(&self, request: SippEmbedRequest) -> Result<SippEmbeddingRun> {
        let context = CoreClientRequestContext {
            request_id: request.request_id.clone(),
        };
        let request = dto::SippEmbedRequest::from(request);
        let request = sipp::SippEmbedRequest::try_from(request).map_err(convert_error)?;
        let run = self
            .inner
            .lock()
            .map_err(|_| napi_error(CLIENT_MUTEX_POISONED))?
            .embed_with_context(context, request);
        Ok(SippEmbeddingRun::from_core(run))
    }
}

/// Text generation handle with a final response and optional token stream.
#[napi(js_name = "SippTextRun")]
pub struct SippTextRun {
    response: SharedClientTextResponse,
    tokens: SharedClientTokenBatches,
    cancellation: CoreCancellationHandle,
}

impl SippTextRun {
    fn from_core(run: CoreClientTextRun) -> Self {
        let (tokens, response, cancellation) = run.into_parts_with_cancel();
        Self {
            response: Arc::new(Mutex::new(Some(response))),
            tokens: Arc::new(Mutex::new(Some(tokens))),
            cancellation,
        }
    }
}

#[napi]
impl SippTextRun {
    #[napi(js_name = "__response", ts_return_type = "Promise<SippTextResponse>")]
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

    /// Cancel the native run and abort local or upstream execution.
    #[napi]
    pub fn cancel(&self, reason: Option<String>) -> Result<()> {
        self.cancellation.cancel(cancellation_reason(reason)?);
        Ok(())
    }
}

/// Embedding request handle with a final embedding response.
#[napi(js_name = "SippEmbeddingRun")]
pub struct SippEmbeddingRun {
    response: SharedClientEmbeddingResponse,
    cancellation: CoreCancellationHandle,
}

impl SippEmbeddingRun {
    fn from_core(run: CoreClientEmbeddingRun) -> Self {
        let (response, cancellation) = run.into_parts();
        Self {
            response: Arc::new(Mutex::new(Some(response))),
            cancellation,
        }
    }
}

#[napi]
impl SippEmbeddingRun {
    #[napi(
        js_name = "__response",
        ts_return_type = "Promise<SippEmbeddingResponse>"
    )]
    pub fn response(&self) -> AsyncTask<ClientEmbeddingResultTask> {
        AsyncTask::new(ClientEmbeddingResultTask {
            response: self.response.clone(),
        })
    }

    /// Cancel the native run and abort local or upstream execution.
    #[napi]
    pub fn cancel(&self, reason: Option<String>) -> Result<()> {
        self.cancellation.cancel(cancellation_reason(reason)?);
        Ok(())
    }
}

fn cancellation_reason(reason: Option<String>) -> Result<CoreCancellationReason> {
    match reason.as_deref().unwrap_or("caller_cancelled") {
        "caller_cancelled" => Ok(CoreCancellationReason::CallerCancelled),
        "client_disconnected" => Ok(CoreCancellationReason::ClientDisconnected),
        "server_shutdown" => Ok(CoreCancellationReason::ServerShutdown),
        "deadline_exceeded" => Ok(CoreCancellationReason::DeadlineExceeded),
        _ => Err(invalid_arg(
            "cancellation reason must be caller_cancelled, client_disconnected, server_shutdown, or deadline_exceeded",
        )),
    }
}

pub struct ClientAddTask {
    client: SharedSippClient,
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
    type JsValue = SippTextResponse;

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
            .map(sipp_text_response_to_node)
            .map_err(|error| client_error_to_node(env, error))
    }
}

pub struct ClientEmbeddingResultTask {
    response: SharedClientEmbeddingResponse,
}

impl Task for ClientEmbeddingResultTask {
    type Output = ClientTaskOutput<CoreClientEmbeddingResponse>;
    type JsValue = SippEmbeddingResponse;

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
            .map(sipp_embedding_response_to_node)
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
    let stats = dto::RequestStats::from(stats);
    RequestStats {
        input_tokens: stats.input_tokens,
        output_tokens: stats.output_tokens,
        cache_mode: stats.cache_mode,
        cache_source: stats.cache_source,
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

fn invalid_arg(message: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, message.into())
}

fn napi_error(message: impl ToString) -> Error {
    Error::new(Status::GenericFailure, message)
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

fn provider_error_to_node(env: Env, error: CoreProviderEndpointError) -> Error {
    match provider_error_to_node_result(env, error) {
        Ok(error) => error,
        Err(error) => error,
    }
}

fn endpoint_error_to_node(env: Env, error: sipp::EndpointError) -> Error {
    match endpoint_error_to_node_result(env, error) {
        Ok(error) => error,
        Err(error) => error,
    }
}

fn endpoint_error_to_node_result(env: Env, error: sipp::EndpointError) -> Result<Error> {
    let mut object = env.create_error(Error::new(Status::GenericFailure, error.to_string()))?;
    object.set("name", "EndpointError")?;
    object.set("kind", error.kind)?;
    object.set("status", error.status)?;
    object.set("code", error.code)?;
    object.set("requestId", error.request_id)?;
    Ok(Error::from(object.to_unknown()))
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

fn client_error_to_node(env: Env, error: CoreClientError) -> Error {
    match error {
        CoreClientError::Local(error) => core_error(error),
        CoreClientError::Endpoint(error) => endpoint_error_to_node(env, error),
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

fn core_error(error: sipp::error::Error) -> Error {
    match error {
        sipp::error::Error::InvalidRequest(message)
        | sipp::error::Error::InvalidConfig(message) => invalid_arg(message),
        sipp::error::Error::UnsupportedOperation { operation, reason } => {
            invalid_arg(format!("unsupported operation {operation}: {reason}"))
        }
        other => napi_error(other.to_string()),
    }
}
