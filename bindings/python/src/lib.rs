//! PyO3 bindings for the public CogentLM Python package.
//!
//! This crate translates Python configuration objects and requests into the
//! shared Rust client facade while preserving Python-specific validation and
//! exception surfaces.

use std::{
    collections::HashSet,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use cogentlm_client::{
    CogentChatRequest as ClientChatRequest, CogentClient as CoreCogentClient,
    CogentEmbedRequest as ClientEmbedRequest, CogentEmbeddingResponse as ClientEmbeddingResponse,
    CogentEmbeddingResponseFuture as ClientEmbeddingResponseFuture,
    CogentEmbeddingRun as CoreClientEmbeddingRun, CogentError as ClientError,
    CogentQueryRequest as ClientQueryRequest, CogentTextOptions as ClientTextOptions,
    CogentTextResponse as ClientTextResponse, CogentTextResponseFuture as ClientTextResponseFuture,
    CogentTextRun as CoreClientTextRun, CogentTokenBatches as ClientTokenBatches,
    EndpointRef as CoreEndpointRef, LocalEmbedOptions as ClientLocalEmbedOptions,
    LocalTextOptions as ClientLocalTextOptions, RemoteError as CoreRemoteError,
    RemoteGatewayConfig as CoreRemoteGatewayConfig, RemoteSecret as CoreRemoteSecret,
};
use cogentlm_core::TokenUsage;
use cogentlm_engine::backend::{
    backend_observability_json as core_backend_observability_json,
    set_llama_log_quiet as core_set_llama_log_quiet,
};
use cogentlm_engine::engine::protocol::{CacheSource, RequestStats};
use cogentlm_engine::engine::{
    ChatMessage, ChatRole, FlashAttentionMode, GpuLayerConfig, KvCacheType, KvReuseMode, LogitBias,
    ModelPlacementConfig, MultimodalRuntimeConfig, NativeRuntimeConfig, ObservabilityRuntimeConfig,
    ResidencyRuntimeConfig, RopeScaling, SamplerStage, SamplingRuntimeConfig,
    SchedulerRuntimeConfig, SplitMode, TokenBatch, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS,
};
use cogentlm_engine::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode};
use futures::executor::block_on;
use futures::StreamExt;
use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::PyClass;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyList, PyLong, PyString, PyTuple};
use serde::de::DeserializeOwned;

#[cfg(test)]
#[path = "tests/remote_tests.rs"]
mod remote_tests;

pyo3::create_exception!(
    _native,
    UnsupportedOperationError,
    PyException,
    "The loaded model does not support the requested operation."
);

pyo3::create_exception!(_native, RemoteError, PyException, "Remote API error.");

const PY_CLIENT_MUTEX_POISONED: &str = "client mutex is poisoned";
const PY_CLIENT_TEXT_RESPONSE_MUTEX_POISONED: &str = "text response mutex is poisoned";
const PY_CLIENT_EMBEDDING_RESPONSE_MUTEX_POISONED: &str = "embedding response mutex is poisoned";
const PY_CLIENT_TOKEN_BATCHES_MUTEX_POISONED: &str = "token batches mutex is poisoned";
const PY_CLIENT_TEXT_RESPONSE_CONSUMED: &str = "text response already consumed";
const PY_CLIENT_EMBEDDING_RESPONSE_CONSUMED: &str = "embedding response already consumed";

type PySharedClientTextResponse = Arc<Mutex<Option<ClientTextResponseFuture>>>;
type PySharedClientEmbeddingResponse = Arc<Mutex<Option<ClientEmbeddingResponseFuture>>>;
type PySharedClientTokenBatches = Arc<Mutex<Option<ClientTokenBatches>>>;

fn py_core_or_default<T, U>(py: Python<'_>, value: Option<Py<T>>, map: impl FnOnce(&T) -> U) -> U
where
    T: PyClass,
    U: Default,
{
    value
        .map(|value| map(&value.borrow(py)))
        .unwrap_or_default()
}

/// Sampling controls used by local text generation.
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

/// Device placement and memory mapping settings for local model loading.
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

/// Context, threading, attention, and embedding settings for local runtime use.
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

/// Scheduler policy knobs for latency, balance, or throughput behavior.
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

/// Request scheduler and continuous batching settings.
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

/// Prefix KV-cache reuse and snapshot settings.
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
        max_snapshot_bytes = None
    ))]
    fn new(
        mode: Option<String>,
        retained_prefix_tokens: Option<i32>,
        snapshot_interval_tokens: Option<i32>,
        max_snapshot_entries: Option<i32>,
        max_snapshot_bytes: Option<usize>,
    ) -> PyResult<Self> {
        let mut core = cogentlm_engine::engine::CacheRuntimeConfig::default();
        if let Some(value) = mode {
            core.mode = parse_kv_reuse_mode(&value)?;
        }
        assign_if_some(&mut core.retained_prefix_tokens, retained_prefix_tokens);
        assign_if_some(&mut core.snapshot_interval_tokens, snapshot_interval_tokens);
        assign_if_some(&mut core.max_snapshot_entries, max_snapshot_entries);
        assign_if_some(&mut core.max_snapshot_bytes, max_snapshot_bytes);
        Ok(Self { core })
    }
}

/// Vision projector and image-token settings for multimodal models.
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

/// GPU residency limits for concurrently loaded local models.
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

/// Runtime metrics and backend profiling options.
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

/// Complete native runtime configuration for local model loading.
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

/// Role/content chat message accepted by local and remote chat requests.
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

/// Address of a local model or remote gateway endpoint registered in a client.
#[pyclass(name = "EndpointRef")]
#[derive(Clone)]
struct PyEndpointRef {
    core: CoreEndpointRef,
}

#[pymethods]
impl PyEndpointRef {
    #[staticmethod]
    fn local(id: String) -> Self {
        Self {
            core: CoreEndpointRef::Local { id },
        }
    }

    #[staticmethod]
    fn remote(id: String) -> Self {
        Self {
            core: CoreEndpointRef::Remote { id },
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.core {
            CoreEndpointRef::Local { .. } => "local",
            CoreEndpointRef::Remote { .. } => "remote",
        }
    }
}

impl PyEndpointRef {
    fn to_core(&self) -> CoreEndpointRef {
        self.core.clone()
    }
}

/// Shared generation options for text-producing requests.
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

/// Local-only prompt options such as grammar constraints and image inputs.
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

/// Local-only embedding options for context and vector normalization.
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

/// Remote CogentLM gateway alias, URL, token, and optional timeout.
#[pyclass(name = "RemoteGatewayConfig")]
#[derive(Clone)]
struct PyRemoteGatewayConfig {
    core: CoreRemoteGatewayConfig,
}

#[pymethods]
impl PyRemoteGatewayConfig {
    #[new]
    #[pyo3(signature = (alias, base_url, token, *, timeout_ms = None))]
    fn new(
        alias: String,
        base_url: String,
        token: String,
        timeout_ms: Option<u64>,
    ) -> PyResult<Self> {
        if timeout_ms == Some(0) {
            return Err(PyValueError::new_err(
                "RemoteGatewayConfig.timeout_ms must be a positive integer",
            ));
        }
        Ok(Self {
            core: CoreRemoteGatewayConfig {
                alias,
                base_url,
                token: CoreRemoteSecret::new(token),
                timeout: timeout_ms.map(Duration::from_millis),
            },
        })
    }
}

impl PyRemoteGatewayConfig {
    fn to_core(&self) -> CoreRemoteGatewayConfig {
        self.core.clone()
    }
}

/// Client facade for local CogentLM models and remote gateway aliases.
#[pyclass(name = "CogentClient")]
struct PyCogentClient {
    inner: Arc<Mutex<CoreCogentClient>>,
}

#[pymethods]
impl PyCogentClient {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(CoreCogentClient::new())),
        })
    }

    #[pyo3(signature = (id, model_path, config = None))]
    fn add_local(
        &self,
        py: Python<'_>,
        id: String,
        model_path: PathBuf,
        config: Option<Py<PyNativeRuntimeConfig>>,
    ) -> PyResult<PyEndpointRef> {
        let config = py_core_or_default(py, config, PyNativeRuntimeConfig::to_core);
        let inner = self.inner.clone();
        let endpoint = py
            .allow_threads(move || {
                let mut client = inner
                    .lock()
                    .map_err(|_| ClientError::Internal(PY_CLIENT_MUTEX_POISONED.to_string()))?;
                block_on(client.add_local(id, model_path, config))
            })
            .map_err(to_py_client_error)?;
        Ok(PyEndpointRef { core: endpoint })
    }

    fn add_remote(
        &self,
        py: Python<'_>,
        id: String,
        config: Py<PyRemoteGatewayConfig>,
    ) -> PyResult<PyEndpointRef> {
        let endpoint = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .add_remote(id, config.borrow(py).to_core())
            .map_err(to_py_client_error)?;
        Ok(PyEndpointRef { core: endpoint })
    }

    fn update_remote(
        &self,
        py: Python<'_>,
        id: String,
        config: Py<PyRemoteGatewayConfig>,
    ) -> PyResult<PyEndpointRef> {
        let endpoint = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .update_remote(id, config.borrow(py).to_core())
            .map_err(to_py_client_error)?;
        Ok(PyEndpointRef { core: endpoint })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (prompt, *, endpoint = None, options = None, local = None, gateway_options = None, emit_tokens = false))]
    fn query(
        &self,
        py: Python<'_>,
        prompt: String,
        endpoint: Option<Py<PyEndpointRef>>,
        options: Option<Py<PyCogentTextOptions>>,
        local: Option<Py<PyLocalTextOptions>>,
        gateway_options: Option<PyObject>,
        emit_tokens: bool,
    ) -> PyResult<PyCogentTextRun> {
        let request = ClientQueryRequest {
            endpoint: endpoint
                .as_ref()
                .map(|endpoint| endpoint.borrow(py).to_core()),
            prompt,
            options: py_core_or_default(py, options, PyCogentTextOptions::to_core),
            local: py_core_or_default(py, local, PyLocalTextOptions::to_core),
            gateway_options: py_gateway_options_or_empty(py, gateway_options)?,
            emit_tokens,
        };
        let run = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .query(request);
        Ok(PyCogentTextRun::from_core(run))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (messages, *, endpoint = None, options = None, local = None, gateway_options = None, emit_tokens = false))]
    fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<Py<PyChatMessage>>,
        endpoint: Option<Py<PyEndpointRef>>,
        options: Option<Py<PyCogentTextOptions>>,
        local: Option<Py<PyLocalTextOptions>>,
        gateway_options: Option<PyObject>,
        emit_tokens: bool,
    ) -> PyResult<PyCogentTextRun> {
        let request = ClientChatRequest {
            endpoint: endpoint
                .as_ref()
                .map(|endpoint| endpoint.borrow(py).to_core()),
            messages: chat_messages_to_core(py, messages)?,
            options: py_core_or_default(py, options, PyCogentTextOptions::to_core),
            local: py_core_or_default(py, local, PyLocalTextOptions::to_core),
            gateway_options: py_gateway_options_or_empty(py, gateway_options)?,
            emit_tokens,
        };
        let run = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .chat(request);
        Ok(PyCogentTextRun::from_core(run))
    }

    #[pyo3(signature = (input, *, endpoint = None, local = None, gateway_options = None))]
    fn embed(
        &self,
        py: Python<'_>,
        input: String,
        endpoint: Option<Py<PyEndpointRef>>,
        local: Option<Py<PyLocalEmbedOptions>>,
        gateway_options: Option<PyObject>,
    ) -> PyResult<PyCogentEmbeddingRun> {
        let request = ClientEmbedRequest {
            endpoint: endpoint
                .as_ref()
                .map(|endpoint| endpoint.borrow(py).to_core()),
            input,
            local: py_core_or_default(py, local, PyLocalEmbedOptions::to_core),
            gateway_options: py_gateway_options_or_empty(py, gateway_options)?,
        };
        let run = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .embed(request);
        Ok(PyCogentEmbeddingRun::from_core(run))
    }
}

/// Text generation handle with a final response and optional token stream.
#[pyclass(name = "CogentTextRun")]
struct PyCogentTextRun {
    response: PySharedClientTextResponse,
    tokens: PySharedClientTokenBatches,
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

/// Iterator over token batches emitted by a text generation run.
#[pyclass(name = "CogentTokenIterator")]
struct PyCogentTokenIterator {
    tokens: PySharedClientTokenBatches,
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
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_TOKEN_BATCHES_MUTEX_POISONED))?;
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

/// Embedding request handle with a final embedding response.
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

/// Return JSON backend and device observability from the native runtime.
#[pyfunction]
#[pyo3(signature = (include_details = true))]
fn backend_observability_json(include_details: bool) -> PyResult<String> {
    core_backend_observability_json(include_details).map_err(to_py_error)
}

/// Enable or suppress llama.cpp native logging.
#[pyfunction]
fn set_llama_log_quiet(quiet: bool) {
    core_set_llama_log_quiet(quiet);
}

/// Initialize the native Python extension module.
#[pymodule]
fn _native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "UnsupportedOperationError",
        module.py().get_type_bound::<UnsupportedOperationError>(),
    )?;
    module.add("RemoteError", module.py().get_type_bound::<RemoteError>())?;
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
    module.add_class::<PyRemoteGatewayConfig>()?;
    module.add("DEFAULT_CONTEXT_KEY", DEFAULT_CONTEXT_KEY)?;
    module.add("DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS)?;
    module.add_function(wrap_pyfunction!(backend_observability_json, module)?)?;
    module.add_function(wrap_pyfunction!(set_llama_log_quiet, module)?)?;
    Ok(())
}

fn py_gateway_options_or_empty(
    py: Python<'_>,
    value: Option<PyObject>,
) -> PyResult<serde_json::Map<String, serde_json::Value>> {
    match value {
        Some(value) => match py_to_json(value.bind(py))? {
            serde_json::Value::Object(options) => Ok(options),
            _ => Err(PyTypeError::new_err(
                "gateway_options must be a JSON object",
            )),
        },
        None => Ok(serde_json::Map::new()),
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
    let mut ancestors = HashSet::new();
    py_to_json_inner(value, &mut ancestors)
}

fn py_to_json_inner(
    value: &Bound<'_, PyAny>,
    ancestors: &mut HashSet<usize>,
) -> PyResult<serde_json::Value> {
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
                PyValueError::new_err("gateway_options cannot contain non-finite floats")
            });
    }
    if let Ok(items) = value.downcast::<PyList>() {
        let container = enter_json_container(value, ancestors)?;
        let result = (|| {
            let mut output = Vec::with_capacity(items.len());
            for item in items.iter() {
                output.push(py_to_json_inner(&item, ancestors)?);
            }
            Ok(serde_json::Value::Array(output))
        })();
        ancestors.remove(&container);
        return result;
    }
    if let Ok(items) = value.downcast::<PyTuple>() {
        let container = enter_json_container(value, ancestors)?;
        let result = (|| {
            let mut output = Vec::with_capacity(items.len());
            for item in items.iter() {
                output.push(py_to_json_inner(&item, ancestors)?);
            }
            Ok(serde_json::Value::Array(output))
        })();
        ancestors.remove(&container);
        return result;
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let container = enter_json_container(value, ancestors)?;
        let result = (|| {
            let mut output = serde_json::Map::new();
            for (key, item) in dict.iter() {
                let key = key.extract::<String>().map_err(|_| {
                    PyTypeError::new_err("gateway_options object keys must be strings")
                })?;
                output.insert(key, py_to_json_inner(&item, ancestors)?);
            }
            Ok(serde_json::Value::Object(output))
        })();
        ancestors.remove(&container);
        return result;
    }
    Err(PyTypeError::new_err(
        "gateway_options must contain JSON-compatible values",
    ))
}

fn enter_json_container(
    value: &Bound<'_, PyAny>,
    ancestors: &mut HashSet<usize>,
) -> PyResult<usize> {
    let container = value.as_ptr() as usize;
    if ancestors.insert(container) {
        Ok(container)
    } else {
        Err(PyTypeError::new_err(
            "gateway_options must contain JSON-compatible values",
        ))
    }
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
                    "gateway JSON number is not representable in Python",
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
        CoreEndpointRef::Local { id } => {
            dict.set_item("kind", "local")?;
            dict.set_item("id", id)?;
        }
        CoreEndpointRef::Remote { id } => {
            dict.set_item("kind", "remote")?;
            dict.set_item("id", id)?;
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
        Some(usage) => dict.set_item("usage", token_usage_to_dict(py, usage)?)?,
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
        Some(usage) => dict.set_item("usage", token_usage_to_dict(py, usage)?)?,
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

fn token_usage_to_dict(py: Python<'_>, usage: TokenUsage) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("input_tokens", usage.input_tokens)?;
    dict.set_item("output_tokens", usage.output_tokens)?;
    dict.set_item("total_tokens", usage.total_tokens)?;
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
    stats.set_item("batches_sent", batch.stats.batches_sent)?;
    dict.set_item("stats", stats)?;
    Ok(dict.into_py(py))
}

fn request_stats_to_dict(py: Python<'_>, stats: RequestStats) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("input_tokens", stats.input_tokens)?;
    dict.set_item("output_tokens", stats.output_tokens)?;
    dict.set_item("cache_mode", kv_reuse_mode_to_str(stats.cache_mode))?;
    dict.set_item("cache_source", cache_source_to_str(stats.cache_source))?;
    dict.set_item("cache_hits", stats.cache_hits)?;
    dict.set_item("prefill_tokens", stats.prefill_tokens)?;
    dict.set_item("ttft_ms", stats.ttft_ms)?;
    dict.set_item("inter_token_ms", stats.inter_token_ms)?;
    dict.set_item("e2e_ms", stats.e2e_ms)?;
    dict.set_item("e2e_tokens_per_second", stats.e2e_tokens_per_second)?;
    dict.set_item("decode_tokens_per_second", stats.decode_tokens_per_second)?;
    dict.set_item("prefill_tokens_per_second", stats.prefill_tokens_per_second)?;
    dict.set_item("prefill_ms", stats.prefill_ms)?;
    dict.set_item("decode_ms", stats.decode_ms)?;
    Ok(dict.into_py(py))
}

fn kv_reuse_mode_to_str(mode: KvReuseMode) -> &'static str {
    match mode {
        KvReuseMode::Disabled => "disabled",
        KvReuseMode::LiveSlotPrefix => "live_slot_prefix",
        KvReuseMode::StateSnapshot => "state_snapshot",
        KvReuseMode::LiveSlotAndSnapshot => "live_slot_and_snapshot",
    }
}

fn cache_source_to_str(source: CacheSource) -> &'static str {
    match source {
        CacheSource::None => "none",
        CacheSource::Live => "live",
        CacheSource::Snapshot => "snapshot",
    }
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

fn remote_error_message(error: &CoreRemoteError) -> String {
    format!(
        "remote gateway error ({}): {}",
        error.kind.as_str(),
        error.message
    )
}

fn to_py_remote_error(error: CoreRemoteError) -> PyErr {
    Python::with_gil(|py| match to_py_remote_error_result(py, error) {
        Ok(error) => error,
        Err(error) => error,
    })
}

fn to_py_remote_error_result(py: Python<'_>, error: CoreRemoteError) -> PyResult<PyErr> {
    let message = remote_error_message(&error);
    let instance = py.get_type_bound::<RemoteError>().call1((message,))?;
    let retry_after_ms = error
        .retry_after
        .map(|duration| duration.as_secs_f64() * 1000.0);
    let raw_body = match error.raw {
        Some(value) => json_to_py(py, *value)?,
        None => py.None(),
    };

    instance.setattr("kind", error.kind.as_str())?;
    instance.setattr("status", error.status)?;
    instance.setattr("code", error.code)?;
    instance.setattr("request_id", error.request_id)?;
    instance.setattr("retry_after_ms", retry_after_ms)?;
    instance.setattr("raw_body", raw_body)?;

    Ok(PyErr::from_value_bound(instance))
}

fn to_py_client_error(error: ClientError) -> PyErr {
    match error {
        ClientError::Local(error) => to_py_error(error),
        ClientError::Remote(error) => to_py_remote_error(error),
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
