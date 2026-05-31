use std::{
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use cogentlm_client::{
    CogentChatRequest as ClientChatRequest, CogentClient as CoreCogentClient,
    CogentEmbedRequest as ClientEmbedRequest, CogentEmbeddingResponse as ClientEmbeddingResponse,
    CogentEmbeddingResponseFuture as ClientEmbeddingResponseFuture,
    CogentEmbeddingRun as CoreClientEmbeddingRun, CogentError as ClientError,
    CogentQueryRequest as ClientQueryRequest, CogentTextOptions as ClientTextOptions,
    CogentTextResponse as ClientTextResponse, CogentTextResponseFuture as ClientTextResponseFuture,
    CogentTextRun as CoreClientTextRun, CogentTokenStream as ClientTokenStream,
    EndpointRef as CoreEndpointRef, LocalEmbedOptions as ClientLocalEmbedOptions,
    LocalTextOptions as ClientLocalTextOptions, ProviderExecutor as CoreProviderExecutor,
};
use cogentlm_engine::backend::{
    backend_observability_json as core_backend_observability_json,
    set_llama_log_quiet as core_set_llama_log_quiet,
};
use cogentlm_engine::engine::protocol::RequestStats;
use cogentlm_engine::engine::{
    CacheKeyPolicy, ChatMessage, ChatRole, FlashAttentionMode, GpuLayerConfig, KvCacheType,
    KvReuseMode, LogitBias, ModelPlacementConfig, MultimodalRuntimeConfig, NativeRuntimeConfig,
    ObservabilityRuntimeConfig, ResidencyRuntimeConfig, RopeScaling, SamplerStage,
    SamplingRuntimeConfig, SchedulerRuntimeConfig, SplitMode, TokenBatch, DEFAULT_CONTEXT_KEY,
    DEFAULT_MAX_TOKENS,
};
use cogentlm_engine::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode};
use cogentlm_providers::{
    AnthropicConfig, CapabilitySupport as ProviderCapabilitySupport, OpenAiConfig, ProviderAuth,
    ProviderChatRequest, ProviderChatResponse, ProviderClient, ProviderEmbedRequest,
    ProviderEmbeddingOutput, ProviderEmbeddingResponse, ProviderError as CoreProviderError,
    ProviderGenerateRequest, ProviderGenerateResponse, ProviderGenerationOptions, ProviderModel,
    ProviderOptions, ProviderResponseMetadata, ProviderTextOutput, ProxyConfig, ProxyProtocol,
    SecretString, TokenUsage,
};
use futures::executor::block_on;
use futures::StreamExt;
use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::PyClass;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyList, PyLong, PyString, PyTuple};
use serde::de::DeserializeOwned;

#[cfg(test)]
#[path = "tests/provider_tests.rs"]
mod provider_tests;

pyo3::create_exception!(
    _native,
    UnsupportedOperationError,
    PyException,
    "The loaded model does not support the requested operation."
);

pyo3::create_exception!(_native, ProviderError, PyException, "Provider API error.");

const PY_CLIENT_MUTEX_POISONED: &str = "client mutex is poisoned";
const PY_CLIENT_TEXT_RESPONSE_MUTEX_POISONED: &str = "text response mutex is poisoned";
const PY_CLIENT_EMBEDDING_RESPONSE_MUTEX_POISONED: &str = "embedding response mutex is poisoned";
const PY_CLIENT_TOKEN_STREAM_MUTEX_POISONED: &str = "token stream mutex is poisoned";
const PY_CLIENT_TEXT_RESPONSE_CONSUMED: &str = "text response already consumed";
const PY_CLIENT_EMBEDDING_RESPONSE_CONSUMED: &str = "embedding response already consumed";

static PROVIDER_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

type PySharedClientTextResponse = Arc<Mutex<Option<ClientTextResponseFuture>>>;
type PySharedClientEmbeddingResponse = Arc<Mutex<Option<ClientEmbeddingResponseFuture>>>;
type PySharedClientTokenStream = Arc<Mutex<Option<ClientTokenStream>>>;

fn py_core_or_default<T, U>(py: Python<'_>, value: Option<Py<T>>, map: impl FnOnce(&T) -> U) -> U
where
    T: PyClass,
    U: Default,
{
    value
        .map(|value| map(&value.borrow(py)))
        .unwrap_or_default()
}

#[pyclass(name = "SamplingRuntimeConfig")]
#[derive(Debug, Clone)]
struct PySamplingRuntimeConfig {
    core: SamplingRuntimeConfig,
}

#[pymethods]
impl PySamplingRuntimeConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        *,
        samplers = None,
        seed = None,
        top_k = None,
        top_p = None,
        min_p = None,
        typical_p = None,
        xtc_probability = None,
        xtc_threshold = None,
        top_n_sigma = None,
        temperature = None,
        dynatemp_range = None,
        dynatemp_exponent = None,
        repeat_last_n = None,
        repeat_penalty = None,
        frequency_penalty = None,
        presence_penalty = None,
        dry_multiplier = None,
        dry_base = None,
        dry_allowed_length = None,
        dry_penalty_last_n = None,
        dry_sequence_breakers = None,
        mirostat = None,
        mirostat_tau = None,
        mirostat_eta = None,
        min_keep = None,
        n_probs = None,
        logit_bias = None,
        ignore_eos = false,
        grammar_lazy = false,
        preserved_tokens = None,
        backend_sampling = true
    ))]
    fn new(
        samplers: Option<Vec<String>>,
        seed: Option<i64>,
        top_k: Option<i32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        typical_p: Option<f32>,
        xtc_probability: Option<f32>,
        xtc_threshold: Option<f32>,
        top_n_sigma: Option<f32>,
        temperature: Option<f32>,
        dynatemp_range: Option<f32>,
        dynatemp_exponent: Option<f32>,
        repeat_last_n: Option<i32>,
        repeat_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        dry_multiplier: Option<f32>,
        dry_base: Option<f32>,
        dry_allowed_length: Option<i32>,
        dry_penalty_last_n: Option<i32>,
        dry_sequence_breakers: Option<Vec<String>>,
        mirostat: Option<i32>,
        mirostat_tau: Option<f32>,
        mirostat_eta: Option<f32>,
        min_keep: Option<i32>,
        n_probs: Option<i32>,
        logit_bias: Option<Vec<(i32, f32)>>,
        ignore_eos: bool,
        grammar_lazy: bool,
        preserved_tokens: Option<Vec<i32>>,
        backend_sampling: bool,
    ) -> PyResult<Self> {
        if seed.is_some_and(|value| value < 0 || value > u32::MAX as i64) {
            return Err(PyValueError::new_err(
                "seed must fit in an unsigned 32-bit integer",
            ));
        }
        let samplers = samplers
            .unwrap_or_default()
            .iter()
            .map(|stage| parse_sampler_stage(stage))
            .collect::<PyResult<Vec<_>>>()?;
        let logit_bias = logit_bias
            .unwrap_or_default()
            .into_iter()
            .map(|(token, bias)| LogitBias { token, bias })
            .collect();
        Ok(Self {
            core: SamplingRuntimeConfig {
                samplers,
                seed: seed.map(|value| value as u32),
                top_k,
                top_p,
                min_p,
                typical_p,
                xtc_probability,
                xtc_threshold,
                top_n_sigma,
                temperature,
                dynatemp_range,
                dynatemp_exponent,
                repeat_last_n,
                repeat_penalty,
                frequency_penalty,
                presence_penalty,
                dry_multiplier,
                dry_base,
                dry_allowed_length,
                dry_penalty_last_n,
                dry_sequence_breakers: dry_sequence_breakers.unwrap_or_default(),
                mirostat,
                mirostat_tau,
                mirostat_eta,
                min_keep,
                n_probs,
                logit_bias,
                ignore_eos,
                grammar_lazy,
                preserved_tokens: preserved_tokens.unwrap_or_default(),
                backend_sampling,
            },
        })
    }
}

impl PySamplingRuntimeConfig {
    fn to_core(&self) -> SamplingRuntimeConfig {
        self.core.clone()
    }
}

#[pyclass(name = "ModelPlacementConfig")]
#[derive(Debug, Clone)]
struct PyModelPlacementConfig {
    core: ModelPlacementConfig,
}

#[pymethods]
impl PyModelPlacementConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        *,
        devices = None,
        gpu_layers = None,
        split_mode = None,
        main_gpu = None,
        tensor_split = None,
        use_mmap = None,
        use_mlock = None,
        fit_params = None,
        fit_params_min_ctx = None,
        fit_params_target_bytes = None,
        check_tensors = None,
        no_extra_bufts = None,
        no_host = None
    ))]
    fn new(
        devices: Option<Vec<String>>,
        gpu_layers: Option<&Bound<'_, PyAny>>,
        split_mode: Option<String>,
        main_gpu: Option<i32>,
        tensor_split: Option<Vec<f32>>,
        use_mmap: Option<bool>,
        use_mlock: Option<bool>,
        fit_params: Option<bool>,
        fit_params_min_ctx: Option<i32>,
        fit_params_target_bytes: Option<Vec<u64>>,
        check_tensors: Option<bool>,
        no_extra_bufts: Option<bool>,
        no_host: Option<bool>,
    ) -> PyResult<Self> {
        let mut core = ModelPlacementConfig::default();
        assign_if_some(&mut core.devices, devices);
        if let Some(gpu_layers) = gpu_layers {
            core.gpu_layers = parse_gpu_layers_value(gpu_layers)?;
        }
        if let Some(split_mode) = split_mode {
            core.split_mode = parse_split_mode(&split_mode)?;
        }
        core.main_gpu = main_gpu;
        assign_if_some(&mut core.tensor_split, tensor_split);
        assign_if_some(&mut core.use_mmap, use_mmap);
        assign_if_some(&mut core.use_mlock, use_mlock);
        assign_if_some(&mut core.fit_params, fit_params);
        core.fit_params_min_ctx = fit_params_min_ctx;
        assign_if_some(&mut core.fit_params_target_bytes, fit_params_target_bytes);
        assign_if_some(&mut core.check_tensors, check_tensors);
        assign_if_some(&mut core.no_extra_bufts, no_extra_bufts);
        assign_if_some(&mut core.no_host, no_host);
        Ok(Self { core })
    }
}

#[pyclass(name = "ContextRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyContextRuntimeConfig {
    core: cogentlm_engine::engine::ContextRuntimeConfig,
}

#[pymethods]
impl PyContextRuntimeConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        *,
        n_ctx = None,
        n_batch = None,
        n_ubatch = None,
        n_parallel = None,
        n_threads = None,
        n_threads_batch = None,
        flash_attention = None,
        kv_unified = None,
        cache_type_k = None,
        cache_type_v = None,
        offload_kqv = None,
        op_offload = None,
        swa_full = None,
        warmup = None,
        rope_scaling = None,
        rope_freq_base = None,
        rope_freq_scale = None,
        yarn_orig_ctx = None,
        yarn_ext_factor = None,
        yarn_attn_factor = None,
        yarn_beta_fast = None,
        yarn_beta_slow = None,
        embeddings = None,
        pooling = None
    ))]
    fn new(
        n_ctx: Option<i32>,
        n_batch: Option<i32>,
        n_ubatch: Option<i32>,
        n_parallel: Option<i32>,
        n_threads: Option<i32>,
        n_threads_batch: Option<i32>,
        flash_attention: Option<String>,
        kv_unified: Option<bool>,
        cache_type_k: Option<String>,
        cache_type_v: Option<String>,
        offload_kqv: Option<bool>,
        op_offload: Option<bool>,
        swa_full: Option<bool>,
        warmup: Option<bool>,
        rope_scaling: Option<String>,
        rope_freq_base: Option<f32>,
        rope_freq_scale: Option<f32>,
        yarn_orig_ctx: Option<i32>,
        yarn_ext_factor: Option<f32>,
        yarn_attn_factor: Option<f32>,
        yarn_beta_fast: Option<f32>,
        yarn_beta_slow: Option<f32>,
        embeddings: Option<bool>,
        pooling: Option<String>,
    ) -> PyResult<Self> {
        let mut core = cogentlm_engine::engine::ContextRuntimeConfig {
            n_ctx,
            n_batch,
            n_ubatch,
            n_parallel,
            n_threads,
            n_threads_batch,
            kv_unified,
            rope_freq_base,
            rope_freq_scale,
            yarn_orig_ctx,
            yarn_ext_factor,
            yarn_attn_factor,
            yarn_beta_fast,
            yarn_beta_slow,
            embeddings,
            ..Default::default()
        };
        if let Some(value) = flash_attention {
            core.flash_attention = parse_flash_attention(&value)?;
        }
        if let Some(value) = cache_type_k {
            core.cache_type_k = parse_kv_cache_type(&value)?;
        }
        if let Some(value) = cache_type_v {
            core.cache_type_v = parse_kv_cache_type(&value)?;
        }
        assign_if_some(&mut core.offload_kqv, offload_kqv);
        assign_if_some(&mut core.op_offload, op_offload);
        assign_if_some(&mut core.swa_full, swa_full);
        assign_if_some(&mut core.warmup, warmup);
        if let Some(value) = rope_scaling {
            core.rope_scaling = Some(parse_rope_scaling(&value)?);
        }
        if let Some(value) = pooling {
            core.pooling = Some(parse_pooling_type(&value)?);
        }
        Ok(Self { core })
    }
}

fn parse_pooling_type(value: &str) -> PyResult<cogentlm_engine::engine::PoolingType> {
    cogentlm_engine::engine::PoolingType::from_name(value)
        .ok_or_else(|| PyValueError::new_err(format!("unknown pooling type: {value}")))
}

#[pyclass(name = "SchedulerPolicyConfig")]
#[derive(Debug, Clone)]
struct PySchedulerPolicyConfig {
    core: SchedulerPolicyConfig,
}

#[pymethods]
impl PySchedulerPolicyConfig {
    #[new]
    #[pyo3(signature = (
        *,
        mode = None,
        decode_token_reserve = None,
        enable_adaptive_prefill_chunking = None
    ))]
    fn new(
        mode: Option<String>,
        decode_token_reserve: Option<i32>,
        enable_adaptive_prefill_chunking: Option<bool>,
    ) -> PyResult<Self> {
        let mut core = SchedulerPolicyConfig::default();
        if let Some(value) = mode {
            core.mode = parse_scheduler_policy(&value)?;
        }
        assign_if_some(&mut core.decode_token_reserve, decode_token_reserve);
        assign_if_some(
            &mut core.enable_adaptive_prefill_chunking,
            enable_adaptive_prefill_chunking,
        );
        Ok(Self { core })
    }
}

#[pyclass(name = "SchedulerRuntimeConfig")]
#[derive(Debug, Clone)]
struct PySchedulerRuntimeConfig {
    core: SchedulerRuntimeConfig,
}

#[pymethods]
impl PySchedulerRuntimeConfig {
    #[new]
    #[pyo3(signature = (
        *,
        continuous_batching = None,
        policy = None,
        prefill_chunk_size = None,
        max_running_requests = None,
        max_queued_requests = None
    ))]
    fn new(
        py: Python<'_>,
        continuous_batching: Option<bool>,
        policy: Option<Py<PySchedulerPolicyConfig>>,
        prefill_chunk_size: Option<i32>,
        max_running_requests: Option<i32>,
        max_queued_requests: Option<i32>,
    ) -> PyResult<Self> {
        let mut core = SchedulerRuntimeConfig::default();
        assign_if_some(&mut core.continuous_batching, continuous_batching);
        if let Some(value) = policy {
            core.policy = value.borrow(py).core;
        }
        assign_if_some(&mut core.prefill_chunk_size, prefill_chunk_size);
        core.max_running_requests = max_running_requests;
        core.max_queued_requests = max_queued_requests;
        Ok(Self { core })
    }
}

#[pyclass(name = "CacheRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyCacheRuntimeConfig {
    core: cogentlm_engine::engine::CacheRuntimeConfig,
}

#[pymethods]
impl PyCacheRuntimeConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        *,
        mode = None,
        retained_prefix_tokens = None,
        snapshot_interval_tokens = None,
        max_snapshot_entries = None,
        max_snapshot_bytes = None,
        max_session_entries = None,
        cache_key_policy = None,
        enable_context_checkpoints = None,
        checkpoint_every_tokens = None
    ))]
    fn new(
        mode: Option<String>,
        retained_prefix_tokens: Option<i32>,
        snapshot_interval_tokens: Option<i32>,
        max_snapshot_entries: Option<i32>,
        max_snapshot_bytes: Option<usize>,
        max_session_entries: Option<i32>,
        cache_key_policy: Option<String>,
        enable_context_checkpoints: Option<bool>,
        checkpoint_every_tokens: Option<i32>,
    ) -> PyResult<Self> {
        let mut core = cogentlm_engine::engine::CacheRuntimeConfig::default();
        if let Some(value) = mode {
            core.mode = parse_kv_reuse_mode(&value)?;
        }
        assign_if_some(&mut core.retained_prefix_tokens, retained_prefix_tokens);
        assign_if_some(&mut core.snapshot_interval_tokens, snapshot_interval_tokens);
        assign_if_some(&mut core.max_snapshot_entries, max_snapshot_entries);
        assign_if_some(&mut core.max_snapshot_bytes, max_snapshot_bytes);
        assign_if_some(&mut core.max_session_entries, max_session_entries);
        if let Some(value) = cache_key_policy {
            core.cache_key_policy = parse_cache_key_policy(&value)?;
        }
        assign_if_some(
            &mut core.enable_context_checkpoints,
            enable_context_checkpoints,
        );
        assign_if_some(&mut core.checkpoint_every_tokens, checkpoint_every_tokens);
        Ok(Self { core })
    }
}

#[pyclass(name = "MultimodalRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyMultimodalRuntimeConfig {
    core: MultimodalRuntimeConfig,
}

#[pymethods]
impl PyMultimodalRuntimeConfig {
    #[new]
    #[pyo3(signature = (*, projector_path = None, use_gpu = None, image_min_tokens = None, image_max_tokens = None))]
    fn new(
        projector_path: Option<String>,
        use_gpu: Option<bool>,
        image_min_tokens: Option<i32>,
        image_max_tokens: Option<i32>,
    ) -> Self {
        Self {
            core: MultimodalRuntimeConfig {
                projector_path,
                use_gpu,
                image_min_tokens,
                image_max_tokens,
            },
        }
    }
}

#[pyclass(name = "ResidencyRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyResidencyRuntimeConfig {
    core: ResidencyRuntimeConfig,
}

#[pymethods]
impl PyResidencyRuntimeConfig {
    #[new]
    #[pyo3(signature = (
        *,
        max_gpu_models_per_device = None,
        allow_cpu_models_while_gpu_loaded = None,
        require_gpu_lease = None,
        gpu_memory_safety_margin_bytes = None
    ))]
    fn new(
        max_gpu_models_per_device: Option<usize>,
        allow_cpu_models_while_gpu_loaded: Option<bool>,
        require_gpu_lease: Option<bool>,
        gpu_memory_safety_margin_bytes: Option<u64>,
    ) -> Self {
        let mut core = ResidencyRuntimeConfig::default();
        assign_if_some(
            &mut core.max_gpu_models_per_device,
            max_gpu_models_per_device,
        );
        assign_if_some(
            &mut core.allow_cpu_models_while_gpu_loaded,
            allow_cpu_models_while_gpu_loaded,
        );
        assign_if_some(&mut core.require_gpu_lease, require_gpu_lease);
        assign_if_some(
            &mut core.gpu_memory_safety_margin_bytes,
            gpu_memory_safety_margin_bytes,
        );
        Self { core }
    }
}

#[pyclass(name = "ObservabilityRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyObservabilityRuntimeConfig {
    core: ObservabilityRuntimeConfig,
}

#[pymethods]
impl PyObservabilityRuntimeConfig {
    #[new]
    #[pyo3(signature = (*, runtime_metrics = false, backend_profiling = false))]
    fn new(runtime_metrics: bool, backend_profiling: bool) -> Self {
        Self {
            core: ObservabilityRuntimeConfig {
                runtime_metrics,
                backend_profiling,
            },
        }
    }
}

#[pyclass(name = "NativeRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyNativeRuntimeConfig {
    core: NativeRuntimeConfig,
}

#[pymethods]
impl PyNativeRuntimeConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        *,
        placement = None,
        context = None,
        sampling = None,
        scheduler = None,
        cache = None,
        multimodal = None,
        residency = None,
        observability = None
    ))]
    fn new(
        py: Python<'_>,
        placement: Option<Py<PyModelPlacementConfig>>,
        context: Option<Py<PyContextRuntimeConfig>>,
        sampling: Option<Py<PySamplingRuntimeConfig>>,
        scheduler: Option<Py<PySchedulerRuntimeConfig>>,
        cache: Option<Py<PyCacheRuntimeConfig>>,
        multimodal: Option<Py<PyMultimodalRuntimeConfig>>,
        residency: Option<Py<PyResidencyRuntimeConfig>>,
        observability: Option<Py<PyObservabilityRuntimeConfig>>,
    ) -> Self {
        Self {
            core: NativeRuntimeConfig {
                placement: py_core_or_default(py, placement, |value| value.core.clone()),
                context: py_core_or_default(py, context, |value| value.core.clone()),
                sampling: py_core_or_default(py, sampling, PySamplingRuntimeConfig::to_core),
                scheduler: py_core_or_default(py, scheduler, |value| value.core.clone()),
                cache: py_core_or_default(py, cache, |value| value.core.clone()),
                multimodal: py_core_or_default(py, multimodal, |value| value.core.clone()),
                residency: py_core_or_default(py, residency, |value| value.core.clone()),
                observability: py_core_or_default(py, observability, |value| value.core),
            },
        }
    }
}

impl PyNativeRuntimeConfig {
    fn to_core(&self) -> NativeRuntimeConfig {
        self.core.clone()
    }
}

#[pyclass(name = "ChatMessage")]
#[derive(Debug, Clone)]
struct PyChatMessage {
    #[pyo3(get)]
    role: String,
    #[pyo3(get)]
    content: String,
}

#[pymethods]
impl PyChatMessage {
    #[new]
    fn new(role: String, content: String) -> PyResult<Self> {
        parse_chat_role(&role)?;
        Ok(Self { role, content })
    }
}

impl PyChatMessage {
    fn to_core(&self) -> PyResult<ChatMessage> {
        Ok(ChatMessage {
            role: parse_chat_role(&self.role)?,
            content: self.content.clone(),
        })
    }
}

#[pyclass(name = "EndpointRef")]
#[derive(Clone)]
struct PyEndpointRef {
    core: CoreEndpointRef,
}

#[pymethods]
impl PyEndpointRef {
    #[staticmethod]
    fn local_model(model: String) -> Self {
        Self {
            core: CoreEndpointRef::LocalModel { model },
        }
    }

    #[staticmethod]
    fn provider_model(provider: String, model: String) -> Self {
        Self {
            core: CoreEndpointRef::ProviderModel { provider, model },
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.core {
            CoreEndpointRef::LocalModel { .. } => "local_model",
            CoreEndpointRef::ProviderModel { .. } => "provider_model",
        }
    }
}

impl PyEndpointRef {
    fn to_core(&self) -> CoreEndpointRef {
        self.core.clone()
    }
}

#[pyclass(name = "CogentTextOptions")]
#[derive(Clone)]
struct PyCogentTextOptions {
    core: ClientTextOptions,
}

#[pymethods]
impl PyCogentTextOptions {
    #[new]
    #[pyo3(signature = (*, max_tokens = None, temperature = None, top_p = None, stop = None))]
    fn new(
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        stop: Option<Vec<String>>,
    ) -> PyResult<Self> {
        Ok(Self {
            core: ClientTextOptions {
                max_tokens,
                temperature: py_optional_finite_f32(temperature, "temperature")?,
                top_p: py_optional_finite_f32(top_p, "top_p")?,
                stop: stop.unwrap_or_default(),
            },
        })
    }
}

impl PyCogentTextOptions {
    fn to_core(&self) -> ClientTextOptions {
        self.core.clone()
    }
}

#[pyclass(name = "LocalTextOptions")]
#[derive(Clone)]
struct PyLocalTextOptions {
    core: ClientLocalTextOptions,
}

#[pymethods]
impl PyLocalTextOptions {
    #[new]
    #[pyo3(signature = (*, context_key = None, grammar = None, json_schema = None, sampling = None, media = None))]
    fn new(
        py: Python<'_>,
        context_key: Option<String>,
        grammar: Option<String>,
        json_schema: Option<String>,
        sampling: Option<Py<PySamplingRuntimeConfig>>,
        media: Option<Vec<Vec<u8>>>,
    ) -> Self {
        Self {
            core: ClientLocalTextOptions {
                context_key,
                grammar,
                json_schema,
                sampling: sampling.as_ref().map(|config| config.borrow(py).to_core()),
                media: media.unwrap_or_default(),
            },
        }
    }
}

impl PyLocalTextOptions {
    fn to_core(&self) -> ClientLocalTextOptions {
        self.core.clone()
    }
}

#[pyclass(name = "LocalEmbedOptions")]
#[derive(Clone)]
struct PyLocalEmbedOptions {
    core: ClientLocalEmbedOptions,
}

#[pymethods]
impl PyLocalEmbedOptions {
    #[new]
    #[pyo3(signature = (*, context_key = None, normalize = None))]
    fn new(context_key: Option<String>, normalize: Option<bool>) -> Self {
        Self {
            core: ClientLocalEmbedOptions {
                context_key,
                normalize,
            },
        }
    }
}

impl PyLocalEmbedOptions {
    fn to_core(&self) -> ClientLocalEmbedOptions {
        self.core.clone()
    }
}

#[pyclass(name = "ProviderAuth")]
#[derive(Clone)]
struct PyProviderAuth {
    core: ProviderAuth,
}

#[pymethods]
impl PyProviderAuth {
    #[staticmethod]
    fn bearer(token: String) -> Self {
        Self {
            core: ProviderAuth::Bearer(SecretString::new(token)),
        }
    }

    #[staticmethod]
    fn header(name: String, value: String) -> Self {
        Self {
            core: ProviderAuth::Header {
                name,
                value: SecretString::new(value),
            },
        }
    }
}

impl PyProviderAuth {
    fn to_core(&self) -> ProviderAuth {
        self.core.clone()
    }
}

#[pyclass(name = "ProviderProxyConfig")]
#[derive(Clone)]
struct PyProviderProxyConfig {
    core: ProxyConfig,
}

#[pymethods]
impl PyProviderProxyConfig {
    #[new]
    #[pyo3(signature = (base_url, auth, protocol = "openai_compatible".to_string(), static_headers = None, timeout_ms = None))]
    fn new(
        py: Python<'_>,
        base_url: String,
        auth: Py<PyProviderAuth>,
        protocol: String,
        static_headers: Option<Vec<(String, String)>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Self> {
        Ok(Self {
            core: ProxyConfig {
                base_url,
                auth: auth.borrow(py).to_core(),
                protocol: parse_provider_proxy_protocol(&protocol)?,
                static_headers: static_headers.unwrap_or_default(),
                timeout: timeout_ms.map(Duration::from_millis),
            },
        })
    }
}

impl PyProviderProxyConfig {
    fn to_core(&self) -> ProxyConfig {
        self.core.clone()
    }
}

#[pyclass(name = "ProviderGenerationOptions")]
#[derive(Clone)]
struct PyProviderGenerationOptions {
    core: ProviderGenerationOptions,
}

#[pymethods]
impl PyProviderGenerationOptions {
    #[new]
    #[pyo3(signature = (*, max_tokens = None, temperature = None, top_p = None, stop = None))]
    fn new(
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        stop: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if max_tokens == Some(0) {
            return Err(PyValueError::new_err(
                "max_tokens must be greater than zero",
            ));
        }
        Ok(Self {
            core: ProviderGenerationOptions {
                max_tokens,
                temperature: py_optional_finite_f32(temperature, "temperature")?,
                top_p: py_optional_finite_f32(top_p, "top_p")?,
                stop: stop.unwrap_or_default(),
            },
        })
    }
}

impl PyProviderGenerationOptions {
    fn to_core(&self) -> ProviderGenerationOptions {
        self.core.clone()
    }
}

#[pyclass(name = "CogentClient")]
struct PyCogentClient {
    inner: Arc<Mutex<CoreCogentClient>>,
    executor: CoreProviderExecutor,
}

#[pymethods]
impl PyCogentClient {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(CoreCogentClient::new())),
            executor: CoreProviderExecutor::new().map_err(to_py_client_error)?,
        })
    }

    #[pyo3(signature = (id, model_path, config = None))]
    fn load_model(
        &self,
        py: Python<'_>,
        id: String,
        model_path: PathBuf,
        config: Option<Py<PyNativeRuntimeConfig>>,
    ) -> PyResult<()> {
        let config = py_core_or_default(py, config, PyNativeRuntimeConfig::to_core);
        let inner = self.inner.clone();
        py.allow_threads(move || {
            let mut client = inner
                .lock()
                .map_err(|_| ClientError::Internal(PY_CLIENT_MUTEX_POISONED.to_string()))?;
            block_on(client.load_model(id, model_path, config))
        })
        .map_err(to_py_client_error)
    }

    fn add_provider_model(
        &self,
        py: Python<'_>,
        provider: String,
        model: String,
        client: Py<PyProviderClient>,
    ) -> PyResult<()> {
        let provider_client = client.borrow(py).inner.clone();
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .add_provider_model(provider, model, provider_client, self.executor.clone())
            .map_err(to_py_client_error)
    }

    #[pyo3(signature = (prompt, *, endpoint = None, options = None, local = None, provider_options = None, stream_tokens = false))]
    fn query(
        &self,
        py: Python<'_>,
        prompt: String,
        endpoint: Option<Py<PyEndpointRef>>,
        options: Option<Py<PyCogentTextOptions>>,
        local: Option<Py<PyLocalTextOptions>>,
        provider_options: Option<PyObject>,
        stream_tokens: bool,
    ) -> PyResult<PyCogentTextRun> {
        let request = ClientQueryRequest {
            endpoint: endpoint
                .as_ref()
                .map(|endpoint| endpoint.borrow(py).to_core()),
            prompt,
            options: py_core_or_default(py, options, PyCogentTextOptions::to_core),
            local: py_core_or_default(py, local, PyLocalTextOptions::to_core),
            provider_options: py_provider_options_or_empty(py, provider_options)?,
            stream_tokens,
        };
        let run = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .query(request);
        Ok(PyCogentTextRun::from_core(run))
    }

    #[pyo3(signature = (messages, *, endpoint = None, options = None, local = None, provider_options = None, stream_tokens = false))]
    fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<Py<PyChatMessage>>,
        endpoint: Option<Py<PyEndpointRef>>,
        options: Option<Py<PyCogentTextOptions>>,
        local: Option<Py<PyLocalTextOptions>>,
        provider_options: Option<PyObject>,
        stream_tokens: bool,
    ) -> PyResult<PyCogentTextRun> {
        let request = ClientChatRequest {
            endpoint: endpoint
                .as_ref()
                .map(|endpoint| endpoint.borrow(py).to_core()),
            messages: chat_messages_to_core(py, messages)?,
            options: py_core_or_default(py, options, PyCogentTextOptions::to_core),
            local: py_core_or_default(py, local, PyLocalTextOptions::to_core),
            provider_options: py_provider_options_or_empty(py, provider_options)?,
            stream_tokens,
        };
        let run = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .chat(request);
        Ok(PyCogentTextRun::from_core(run))
    }

    #[pyo3(signature = (input, *, endpoint = None, local = None, provider_options = None))]
    fn embed(
        &self,
        py: Python<'_>,
        input: String,
        endpoint: Option<Py<PyEndpointRef>>,
        local: Option<Py<PyLocalEmbedOptions>>,
        provider_options: Option<PyObject>,
    ) -> PyResult<PyCogentEmbeddingRun> {
        let request = ClientEmbedRequest {
            endpoint: endpoint
                .as_ref()
                .map(|endpoint| endpoint.borrow(py).to_core()),
            input,
            local: py_core_or_default(py, local, PyLocalEmbedOptions::to_core),
            provider_options: py_provider_options_or_empty(py, provider_options)?,
        };
        let run = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .embed(request);
        Ok(PyCogentEmbeddingRun::from_core(run))
    }
}

#[pyclass(name = "CogentTextRun")]
struct PyCogentTextRun {
    response: PySharedClientTextResponse,
    tokens: PySharedClientTokenStream,
}

impl PyCogentTextRun {
    fn from_core(run: CoreClientTextRun) -> Self {
        let (tokens, response) = run.into_parts();
        Self {
            response: Arc::new(Mutex::new(Some(response))),
            tokens: Arc::new(Mutex::new(Some(tokens))),
        }
    }
}

#[pymethods]
impl PyCogentTextRun {
    fn result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let response = self
            .response
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_TEXT_RESPONSE_MUTEX_POISONED))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err(PY_CLIENT_TEXT_RESPONSE_CONSUMED))?;
        let response = py
            .allow_threads(|| block_on(response))
            .map_err(to_py_client_error)?;
        cogent_text_response_to_dict(py, response)
    }

    fn tokens(&self) -> PyCogentTokenIterator {
        PyCogentTokenIterator {
            tokens: self.tokens.clone(),
        }
    }
}

#[pyclass(name = "CogentTokenIterator")]
struct PyCogentTokenIterator {
    tokens: PySharedClientTokenStream,
}

#[pymethods]
impl PyCogentTokenIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut guard = self
            .tokens
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_TOKEN_STREAM_MUTEX_POISONED))?;
        let Some(stream) = guard.as_mut() else {
            return Err(pyo3::exceptions::PyStopIteration::new_err(()));
        };
        let batch = py.allow_threads(|| block_on(stream.next()));
        match batch {
            Some(batch) => token_batch_to_dict(py, batch),
            None => {
                *guard = None;
                Err(pyo3::exceptions::PyStopIteration::new_err(()))
            }
        }
    }
}

#[pyclass(name = "CogentEmbeddingRun")]
struct PyCogentEmbeddingRun {
    response: PySharedClientEmbeddingResponse,
}

impl PyCogentEmbeddingRun {
    fn from_core(run: CoreClientEmbeddingRun) -> Self {
        Self {
            response: Arc::new(Mutex::new(Some(run.into_response()))),
        }
    }
}

#[pymethods]
impl PyCogentEmbeddingRun {
    fn result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let response = self
            .response
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_EMBEDDING_RESPONSE_MUTEX_POISONED))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err(PY_CLIENT_EMBEDDING_RESPONSE_CONSUMED))?;
        let response = py
            .allow_threads(|| block_on(response))
            .map_err(to_py_client_error)?;
        cogent_embedding_response_to_dict(py, response)
    }
}

#[pyclass(name = "ProviderClient")]
struct PyProviderClient {
    inner: ProviderClient,
}

#[pymethods]
impl PyProviderClient {
    #[staticmethod]
    fn proxy(py: Python<'_>, config: Py<PyProviderProxyConfig>) -> PyResult<Self> {
        Ok(Self {
            inner: ProviderClient::proxy(config.borrow(py).to_core())
                .map_err(to_py_provider_error)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (api_key, base_url = None, timeout_ms = None))]
    fn openai(
        api_key: String,
        base_url: Option<String>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ProviderClient::openai(OpenAiConfig {
                api_key: SecretString::new(api_key),
                base_url,
                timeout: timeout_ms.map(Duration::from_millis),
            })
            .map_err(to_py_provider_error)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (api_key, base_url = None, version = None, timeout_ms = None))]
    fn anthropic(
        api_key: String,
        base_url: Option<String>,
        version: Option<String>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ProviderClient::anthropic(AnthropicConfig {
                api_key: SecretString::new(api_key),
                base_url,
                version,
                timeout: timeout_ms.map(Duration::from_millis),
            })
            .map_err(to_py_provider_error)?,
        })
    }

    fn kind(&self) -> String {
        self.inner.kind().as_str().to_string()
    }

    fn list_models(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let client = self.inner.clone();
        let runtime = provider_runtime()?;
        let models = py
            .allow_threads(|| runtime.block_on(client.list_models()))
            .map_err(to_py_provider_error)?;
        let output = PyList::empty_bound(py);
        for model in models {
            output.append(provider_model_to_dict(py, model)?)?;
        }
        Ok(output.into_py(py))
    }

    fn get_model(&self, py: Python<'_>, model: String) -> PyResult<Py<PyAny>> {
        let client = self.inner.clone();
        let runtime = provider_runtime()?;
        let model = py
            .allow_threads(|| runtime.block_on(client.get_model(&model)))
            .map_err(to_py_provider_error)?;
        provider_model_to_dict(py, model)
    }

    #[pyo3(signature = (model, messages, options = None, provider_options = None))]
    fn chat(
        &self,
        py: Python<'_>,
        model: String,
        messages: Vec<Py<PyChatMessage>>,
        options: Option<Py<PyProviderGenerationOptions>>,
        provider_options: Option<PyObject>,
    ) -> PyResult<Py<PyAny>> {
        let request = ProviderChatRequest {
            model,
            messages: chat_messages_to_core(py, messages)?,
            options: py_core_or_default(py, options, PyProviderGenerationOptions::to_core),
            provider_options: py_provider_options_or_empty(py, provider_options)?,
        };
        let client = self.inner.clone();
        let runtime = provider_runtime()?;
        let response = py
            .allow_threads(|| runtime.block_on(client.chat(request)))
            .map_err(to_py_provider_error)?;
        provider_chat_response_to_dict(py, response)
    }

    #[pyo3(signature = (model, prompt, options = None, provider_options = None))]
    fn generate(
        &self,
        py: Python<'_>,
        model: String,
        prompt: String,
        options: Option<Py<PyProviderGenerationOptions>>,
        provider_options: Option<PyObject>,
    ) -> PyResult<Py<PyAny>> {
        let request = ProviderGenerateRequest {
            model,
            prompt,
            options: py_core_or_default(py, options, PyProviderGenerationOptions::to_core),
            provider_options: py_provider_options_or_empty(py, provider_options)?,
        };
        let client = self.inner.clone();
        let runtime = provider_runtime()?;
        let response = py
            .allow_threads(|| runtime.block_on(client.generate(request)))
            .map_err(to_py_provider_error)?;
        provider_generate_response_to_dict(py, response)
    }

    #[pyo3(signature = (model, input, provider_options = None))]
    fn embed(
        &self,
        py: Python<'_>,
        model: String,
        input: String,
        provider_options: Option<PyObject>,
    ) -> PyResult<Py<PyAny>> {
        let request = ProviderEmbedRequest {
            model,
            input,
            provider_options: py_provider_options_or_empty(py, provider_options)?,
        };
        let client = self.inner.clone();
        let runtime = provider_runtime()?;
        let response = py
            .allow_threads(|| runtime.block_on(client.embed(request)))
            .map_err(to_py_provider_error)?;
        provider_embedding_response_to_dict(py, response)
    }
}

#[pyfunction]
#[pyo3(signature = (include_details = true))]
fn backend_observability_json(include_details: bool) -> PyResult<String> {
    core_backend_observability_json(include_details).map_err(to_py_error)
}

#[pyfunction]
fn set_llama_log_quiet(quiet: bool) {
    core_set_llama_log_quiet(quiet);
}

#[pymodule]
fn _native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "UnsupportedOperationError",
        module.py().get_type_bound::<UnsupportedOperationError>(),
    )?;
    module.add(
        "ProviderError",
        module.py().get_type_bound::<ProviderError>(),
    )?;
    module.add_class::<PyModelPlacementConfig>()?;
    module.add_class::<PyContextRuntimeConfig>()?;
    module.add_class::<PySamplingRuntimeConfig>()?;
    module.add_class::<PySchedulerPolicyConfig>()?;
    module.add_class::<PySchedulerRuntimeConfig>()?;
    module.add_class::<PyCacheRuntimeConfig>()?;
    module.add_class::<PyMultimodalRuntimeConfig>()?;
    module.add_class::<PyResidencyRuntimeConfig>()?;
    module.add_class::<PyObservabilityRuntimeConfig>()?;
    module.add_class::<PyNativeRuntimeConfig>()?;
    module.add_class::<PyChatMessage>()?;
    module.add_class::<PyEndpointRef>()?;
    module.add_class::<PyCogentTextOptions>()?;
    module.add_class::<PyLocalTextOptions>()?;
    module.add_class::<PyLocalEmbedOptions>()?;
    module.add_class::<PyCogentClient>()?;
    module.add_class::<PyCogentTextRun>()?;
    module.add_class::<PyCogentTokenIterator>()?;
    module.add_class::<PyCogentEmbeddingRun>()?;
    module.add_class::<PyProviderAuth>()?;
    module.add_class::<PyProviderProxyConfig>()?;
    module.add_class::<PyProviderGenerationOptions>()?;
    module.add_class::<PyProviderClient>()?;
    module.add("DEFAULT_CONTEXT_KEY", DEFAULT_CONTEXT_KEY)?;
    module.add("DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS)?;
    module.add_function(wrap_pyfunction!(backend_observability_json, module)?)?;
    module.add_function(wrap_pyfunction!(set_llama_log_quiet, module)?)?;
    Ok(())
}

fn provider_runtime() -> PyResult<&'static tokio::runtime::Runtime> {
    if let Some(runtime) = PROVIDER_RUNTIME.get() {
        return Ok(runtime);
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    match PROVIDER_RUNTIME.set(runtime) {
        Ok(()) => Ok(PROVIDER_RUNTIME
            .get()
            .expect("provider runtime set before get")),
        Err(_) => Ok(PROVIDER_RUNTIME
            .get()
            .expect("provider runtime initialized concurrently")),
    }
}

fn py_provider_options_or_empty(
    py: Python<'_>,
    value: Option<PyObject>,
) -> PyResult<ProviderOptions> {
    match value {
        Some(value) => match py_to_json(value.bind(py))? {
            serde_json::Value::Object(options) => Ok(options),
            _ => Err(PyTypeError::new_err(
                "provider_options must be a JSON object",
            )),
        },
        None => Ok(ProviderOptions::new()),
    }
}

fn chat_messages_to_core(
    py: Python<'_>,
    messages: Vec<Py<PyChatMessage>>,
) -> PyResult<Vec<ChatMessage>> {
    if messages.is_empty() {
        return Err(PyValueError::new_err("chat messages must not be empty"));
    }
    messages
        .iter()
        .map(|message| message.borrow(py).to_core())
        .collect()
}

fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if value.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if value.downcast::<PyBool>().is_ok() {
        return Ok(serde_json::Value::Bool(value.extract()?));
    }
    if value.downcast::<PyString>().is_ok() {
        return Ok(serde_json::Value::String(value.extract()?));
    }
    if value.downcast::<PyLong>().is_ok() {
        if let Ok(number) = value.extract::<i64>() {
            return Ok(serde_json::Value::Number(number.into()));
        }
        if let Ok(number) = value.extract::<u64>() {
            return Ok(serde_json::Value::Number(number.into()));
        }
    }
    if value.downcast::<PyFloat>().is_ok() {
        let number = value.extract::<f64>()?;
        return serde_json::Number::from_f64(number)
            .map(serde_json::Value::Number)
            .ok_or_else(|| {
                PyValueError::new_err("provider_options cannot contain non-finite floats")
            });
    }
    if let Ok(items) = value.downcast::<PyList>() {
        let mut output = Vec::with_capacity(items.len());
        for item in items.iter() {
            output.push(py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(output));
    }
    if let Ok(items) = value.downcast::<PyTuple>() {
        let mut output = Vec::with_capacity(items.len());
        for item in items.iter() {
            output.push(py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(output));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut output = serde_json::Map::new();
        for (key, item) in dict.iter() {
            output.insert(key.extract()?, py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Object(output));
    }
    Err(PyTypeError::new_err(
        "provider_options must contain JSON-compatible values",
    ))
}

fn json_to_py(py: Python<'_>, value: serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(value) => Ok(value.into_py(py)),
        serde_json::Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                Ok(value.into_py(py))
            } else if let Some(value) = number.as_u64() {
                Ok(value.into_py(py))
            } else if let Some(value) = number.as_f64() {
                Ok(value.into_py(py))
            } else {
                Err(PyValueError::new_err(
                    "provider JSON number is not representable in Python",
                ))
            }
        }
        serde_json::Value::String(value) => Ok(value.into_py(py)),
        serde_json::Value::Array(values) => {
            let output = PyList::empty_bound(py);
            for value in values {
                output.append(json_to_py(py, value)?)?;
            }
            Ok(output.into_py(py))
        }
        serde_json::Value::Object(values) => {
            let output = PyDict::new_bound(py);
            for (key, value) in values {
                output.set_item(key, json_to_py(py, value)?)?;
            }
            Ok(output.into_py(py))
        }
    }
}

fn endpoint_ref_to_dict(py: Python<'_>, endpoint: CoreEndpointRef) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    match endpoint {
        CoreEndpointRef::LocalModel { model } => {
            dict.set_item("kind", "local_model")?;
            dict.set_item("provider", py.None())?;
            dict.set_item("model", model)?;
        }
        CoreEndpointRef::ProviderModel { provider, model } => {
            dict.set_item("kind", "provider_model")?;
            dict.set_item("provider", provider)?;
            dict.set_item("model", model)?;
        }
    }
    Ok(dict.into_py(py))
}

fn cogent_text_response_to_dict(
    py: Python<'_>,
    response: ClientTextResponse,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("endpoint", endpoint_ref_to_dict(py, response.endpoint)?)?;
    dict.set_item("text", response.text)?;
    dict.set_item("finish_reason", response.finish_reason.as_str())?;
    match response.usage {
        Some(usage) => dict.set_item("usage", provider_usage_to_dict(py, usage)?)?,
        None => dict.set_item("usage", py.None())?,
    }
    match response.local_stats {
        Some(stats) => dict.set_item("local_stats", request_stats_to_dict(py, stats)?)?,
        None => dict.set_item("local_stats", py.None())?,
    }
    Ok(dict.into_py(py))
}

fn cogent_embedding_response_to_dict(
    py: Python<'_>,
    response: ClientEmbeddingResponse,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("endpoint", endpoint_ref_to_dict(py, response.endpoint)?)?;
    dict.set_item("values", response.values)?;
    match response.usage {
        Some(usage) => dict.set_item("usage", provider_usage_to_dict(py, usage)?)?,
        None => dict.set_item("usage", py.None())?,
    }
    match response.local_stats {
        Some(stats) => dict.set_item("local_stats", request_stats_to_dict(py, stats)?)?,
        None => dict.set_item("local_stats", py.None())?,
    }
    match response.pooling {
        Some(pooling) => dict.set_item("pooling", pooling.as_str())?,
        None => dict.set_item("pooling", py.None())?,
    }
    dict.set_item("normalized", response.normalized)?;
    Ok(dict.into_py(py))
}

fn provider_model_to_dict(py: Python<'_>, model: ProviderModel) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", model.id)?;
    dict.set_item("provider", model.provider.as_str())?;
    dict.set_item("display_name", model.display_name)?;
    dict.set_item(
        "capabilities",
        provider_capabilities_to_dict(py, model.capabilities)?,
    )?;
    dict.set_item("context_window", model.context_window)?;
    dict.set_item("max_output_tokens", model.max_output_tokens)?;
    dict.set_item("raw", json_to_py(py, model.raw)?)?;
    Ok(dict.into_py(py))
}

fn provider_capabilities_to_dict(
    py: Python<'_>,
    capabilities: cogentlm_providers::ProviderCapabilities,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("chat", provider_capability_support_str(capabilities.chat))?;
    dict.set_item(
        "generate",
        provider_capability_support_str(capabilities.generate),
    )?;
    dict.set_item(
        "embeddings",
        provider_capability_support_str(capabilities.embeddings),
    )?;
    dict.set_item(
        "streaming",
        provider_capability_support_str(capabilities.streaming),
    )?;
    Ok(dict.into_py(py))
}

fn provider_capability_support_str(value: ProviderCapabilitySupport) -> &'static str {
    match value {
        ProviderCapabilitySupport::Supported => "supported",
        ProviderCapabilitySupport::Unsupported => "unsupported",
        ProviderCapabilitySupport::Unknown => "unknown",
    }
}

fn provider_text_output_to_dict(py: Python<'_>, output: ProviderTextOutput) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("text", output.text)?;
    dict.set_item("finish_reason", output.finish_reason.as_str())?;
    Ok(dict.into_py(py))
}

fn provider_embedding_output_to_dict(
    py: Python<'_>,
    output: ProviderEmbeddingOutput,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("values", output.values)?;
    Ok(dict.into_py(py))
}

fn provider_usage_to_dict(py: Python<'_>, usage: TokenUsage) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("input_tokens", usage.input_tokens)?;
    dict.set_item("output_tokens", usage.output_tokens)?;
    dict.set_item("total_tokens", usage.total_tokens)?;
    Ok(dict.into_py(py))
}

fn provider_metadata_to_dict(
    py: Python<'_>,
    metadata: ProviderResponseMetadata,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("provider", metadata.provider.as_str())?;
    dict.set_item("model", metadata.model)?;
    dict.set_item("request_id", metadata.request_id)?;
    dict.set_item("response_id", metadata.response_id)?;
    dict.set_item("finish_reason_raw", metadata.finish_reason_raw)?;
    dict.set_item("raw", json_to_py(py, metadata.raw)?)?;
    Ok(dict.into_py(py))
}

fn provider_chat_response_to_dict(
    py: Python<'_>,
    response: ProviderChatResponse,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("result", provider_text_output_to_dict(py, response.result)?)?;
    match response.usage {
        Some(usage) => dict.set_item("usage", provider_usage_to_dict(py, usage)?)?,
        None => dict.set_item("usage", py.None())?,
    }
    dict.set_item(
        "metadata",
        provider_metadata_to_dict(py, response.metadata)?,
    )?;
    Ok(dict.into_py(py))
}

fn provider_generate_response_to_dict(
    py: Python<'_>,
    response: ProviderGenerateResponse,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("result", provider_text_output_to_dict(py, response.result)?)?;
    match response.usage {
        Some(usage) => dict.set_item("usage", provider_usage_to_dict(py, usage)?)?,
        None => dict.set_item("usage", py.None())?,
    }
    dict.set_item(
        "metadata",
        provider_metadata_to_dict(py, response.metadata)?,
    )?;
    Ok(dict.into_py(py))
}

fn provider_embedding_response_to_dict(
    py: Python<'_>,
    response: ProviderEmbeddingResponse,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "result",
        provider_embedding_output_to_dict(py, response.result)?,
    )?;
    match response.usage {
        Some(usage) => dict.set_item("usage", provider_usage_to_dict(py, usage)?)?,
        None => dict.set_item("usage", py.None())?,
    }
    dict.set_item(
        "metadata",
        provider_metadata_to_dict(py, response.metadata)?,
    )?;
    Ok(dict.into_py(py))
}

fn token_batch_to_dict(py: Python<'_>, batch: TokenBatch) -> PyResult<Py<PyAny>> {
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

fn request_stats_to_dict(py: Python<'_>, stats: RequestStats) -> PyResult<Py<PyAny>> {
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

fn parse_gpu_layers(value: &str) -> PyResult<GpuLayerConfig> {
    parse_choice(
        value,
        r#"gpu_layers must be "auto", "all", or {"count": int}"#,
    )
}

fn parse_gpu_layers_value(value: &Bound<'_, PyAny>) -> PyResult<GpuLayerConfig> {
    if let Ok(value) = value.extract::<String>() {
        return parse_gpu_layers(&value);
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let count = dict
            .get_item("count")?
            .ok_or_else(|| PyValueError::new_err("gpu_layers.count is required"))?
            .extract::<i32>()?;
        return Ok(GpuLayerConfig::from_layer_count(count));
    }
    Err(PyTypeError::new_err(
        r#"gpu_layers must be "auto", "all", or {"count": int}"#,
    ))
}

fn parse_split_mode(value: &str) -> PyResult<SplitMode> {
    parse_choice(value, "split_mode must be one of: none, layer, row, tensor")
}

fn parse_flash_attention(value: &str) -> PyResult<FlashAttentionMode> {
    parse_choice(
        value,
        "flash_attention must be one of: auto, enabled, disabled",
    )
}

fn parse_kv_cache_type(value: &str) -> PyResult<KvCacheType> {
    parse_choice(
        value,
        "cache type must be one of: f16, f32, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1",
    )
}

fn parse_rope_scaling(value: &str) -> PyResult<RopeScaling> {
    parse_choice(value, "rope_scaling must be one of: none, linear, yarn")
}

fn parse_kv_reuse_mode(value: &str) -> PyResult<KvReuseMode> {
    parse_choice(
        value,
        "cache mode must be one of: disabled, live_slot_prefix, state_snapshot, live_slot_and_snapshot",
    )
}

fn parse_cache_key_policy(value: &str) -> PyResult<CacheKeyPolicy> {
    parse_choice(
        value,
        "cache_key_policy must be one of: context_key, prompt_hash",
    )
}

fn parse_sampler_stage(value: &str) -> PyResult<SamplerStage> {
    parse_choice(
        value,
        "sampler stage must be one of: dry, top_k, typical_p, top_p, top_n_sigma, min_p, xtc, temperature, infill, penalties, adaptive_p",
    )
}

fn parse_scheduler_policy(value: &str) -> PyResult<SchedulerPolicyMode> {
    parse_choice(
        value,
        "scheduler.policy.mode must be one of: latency_first, balanced, throughput_first",
    )
}

fn parse_chat_role(value: &str) -> PyResult<ChatRole> {
    parse_choice(value, "chat role must be one of: system, user, assistant")
}

fn parse_provider_proxy_protocol(value: &str) -> PyResult<ProxyProtocol> {
    match value {
        "openai_compatible" => Ok(ProxyProtocol::OpenAiCompatible),
        _ => Err(PyValueError::new_err(
            "provider proxy protocol must be: openai_compatible",
        )),
    }
}

fn parse_choice<T>(value: &str, error_message: &'static str) -> PyResult<T>
where
    T: DeserializeOwned,
{
    serde_json::from_value(serde_json::Value::String(value.to_string()))
        .map_err(|_| PyValueError::new_err(error_message))
}

fn py_finite_f32(value: f32, name: &'static str) -> PyResult<f32> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!("{name} must be finite")));
    }
    Ok(value)
}

fn py_optional_finite_f32(value: Option<f32>, name: &'static str) -> PyResult<Option<f32>> {
    value.map(|value| py_finite_f32(value, name)).transpose()
}

fn assign_if_some<T>(target: &mut T, value: Option<T>) {
    if let Some(value) = value {
        *target = value;
    }
}

fn provider_error_message(error: &CoreProviderError) -> String {
    format!(
        "{} provider error ({}): {}",
        error.provider.as_str(),
        error.kind.as_str(),
        error.message
    )
}

fn to_py_provider_error(error: CoreProviderError) -> PyErr {
    Python::with_gil(|py| {
        let message = provider_error_message(&error);
        let instance = py
            .get_type_bound::<ProviderError>()
            .call1((message,))
            .expect("ProviderError constructor should accept a message");
        let retry_after_ms = error
            .retry_after
            .map(|duration| duration.as_secs_f64() * 1000.0);
        let raw_body = error
            .raw
            .map(|value| json_to_py(py, *value).expect("ProviderError.raw_body should be JSON"))
            .unwrap_or_else(|| py.None());

        instance
            .setattr("kind", error.kind.as_str())
            .expect("setting ProviderError.kind should not fail");
        instance
            .setattr("provider", error.provider.as_str())
            .expect("setting ProviderError.provider should not fail");
        instance
            .setattr("status", error.status)
            .expect("setting ProviderError.status should not fail");
        instance
            .setattr("code", error.code)
            .expect("setting ProviderError.code should not fail");
        instance
            .setattr("request_id", error.request_id)
            .expect("setting ProviderError.request_id should not fail");
        instance
            .setattr("retry_after_ms", retry_after_ms)
            .expect("setting ProviderError.retry_after_ms should not fail");
        instance
            .setattr("raw_body", raw_body)
            .expect("setting ProviderError.raw_body should not fail");

        PyErr::from_value_bound(instance)
    })
}

fn to_py_client_error(error: ClientError) -> PyErr {
    match error {
        ClientError::Local(error) => to_py_error(error),
        ClientError::Provider(error) => to_py_provider_error(error),
        ClientError::InvalidRequest(message) => PyValueError::new_err(message),
        ClientError::UnsupportedOperation {
            endpoint,
            operation,
        } => UnsupportedOperationError::new_err(format!(
            "unsupported operation {operation} on endpoint {endpoint:?}"
        )),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn to_py_error(error: cogentlm_engine::Error) -> PyErr {
    match error {
        cogentlm_engine::Error::InvalidRequest(message)
        | cogentlm_engine::Error::InvalidConfig(message) => PyValueError::new_err(message),
        cogentlm_engine::Error::UnsupportedOperation { operation, reason } => {
            UnsupportedOperationError::new_err(format!(
                "unsupported operation {operation}: {reason}"
            ))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}
