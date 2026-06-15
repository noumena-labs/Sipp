//! PyO3 bindings for the public Sipp Python package.
//!
//! This crate translates Python configuration objects and requests into the
//! shared Rust client facade while preserving Python-specific validation and
//! exception surfaces.

use std::{
    collections::{BTreeMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use futures::executor::block_on;
use futures::StreamExt;
use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyList, PyLong, PyString, PyTuple};
use sipp::backend::{
    backend_observability_json as core_backend_observability_json,
    set_llama_log_quiet as core_set_llama_log_quiet,
};
use sipp::core::TokenUsage;
use sipp::engine::protocol::RequestStats;
use sipp::engine::{
    ChatMessage, ModelPlacementConfig, NativeRuntimeConfig, SamplingRuntimeConfig,
    SchedulerRuntimeConfig, TokenBatch, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS,
};
use sipp::{
    EndpointDescriptor as CoreEndpointDescriptor, EndpointRef as CoreEndpointRef,
    ProviderEndpointError as CoreProviderEndpointError, SippChatRequest as ClientChatRequest,
    SippClient as CoreSippClient, SippEmbedRequest as ClientEmbedRequest,
    SippEmbeddingResponse as ClientEmbeddingResponse,
    SippEmbeddingResponseFuture as ClientEmbeddingResponseFuture,
    SippEmbeddingRun as CoreClientEmbeddingRun, SippError as ClientError,
    SippQueryRequest as ClientQueryRequest, SippTextOptions as ClientTextOptions,
    SippTextResponse as ClientTextResponse, SippTextResponseFuture as ClientTextResponseFuture,
    SippTextRun as CoreClientTextRun, SippTokenBatches as ClientTokenBatches,
};
use sipp_binding_dto as dto;

pyo3::create_exception!(
    _native,
    UnsupportedOperationError,
    PyException,
    "The loaded model does not support the requested operation."
);

pyo3::create_exception!(_native, EndpointError, PyException, "Endpoint error.");
pyo3::create_exception!(_native, ProviderError, PyException, "Provider API error.");

const PY_CLIENT_MUTEX_POISONED: &str = "client mutex is poisoned";
const PY_CLIENT_TEXT_RESPONSE_MUTEX_POISONED: &str = "text response mutex is poisoned";
const PY_CLIENT_EMBEDDING_RESPONSE_MUTEX_POISONED: &str = "embedding response mutex is poisoned";
const PY_CLIENT_TOKEN_BATCHES_MUTEX_POISONED: &str = "token batches mutex is poisoned";
const PY_CLIENT_TEXT_RESPONSE_CONSUMED: &str = "text response already consumed";
const PY_CLIENT_EMBEDDING_RESPONSE_CONSUMED: &str = "embedding response already consumed";

type PySharedClientTextResponse = Arc<Mutex<Option<ClientTextResponseFuture>>>;
type PySharedClientEmbeddingResponse = Arc<Mutex<Option<ClientEmbeddingResponseFuture>>>;
type PySharedClientTokenBatches = Arc<Mutex<Option<ClientTokenBatches>>>;

/// Sampling controls used by local text generation.
#[pyclass(name = "SamplingRuntimeConfig")]
#[derive(Debug, Clone)]
struct PySamplingRuntimeConfig {
    dto: dto::SamplingRuntimeConfig,
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
        let dto = dto::SamplingRuntimeConfig {
            samplers,
            seed,
            top_k,
            top_p: top_p.map(f64::from),
            min_p: min_p.map(f64::from),
            typical_p: typical_p.map(f64::from),
            xtc_probability: xtc_probability.map(f64::from),
            xtc_threshold: xtc_threshold.map(f64::from),
            top_n_sigma: top_n_sigma.map(f64::from),
            temperature: temperature.map(f64::from),
            dynatemp_range: dynatemp_range.map(f64::from),
            dynatemp_exponent: dynatemp_exponent.map(f64::from),
            repeat_last_n,
            repeat_penalty: repeat_penalty.map(f64::from),
            frequency_penalty: frequency_penalty.map(f64::from),
            presence_penalty: presence_penalty.map(f64::from),
            dry_multiplier: dry_multiplier.map(f64::from),
            dry_base: dry_base.map(f64::from),
            dry_allowed_length,
            dry_penalty_last_n,
            dry_sequence_breakers,
            mirostat,
            mirostat_tau: mirostat_tau.map(f64::from),
            mirostat_eta: mirostat_eta.map(f64::from),
            min_keep,
            n_probs,
            logit_bias: logit_bias.map(|biases| {
                biases
                    .into_iter()
                    .map(|(token, bias)| dto::LogitBiasConfig {
                        token,
                        bias: f64::from(bias),
                    })
                    .collect()
            }),
            ignore_eos: Some(ignore_eos),
            grammar_lazy: Some(grammar_lazy),
            preserved_tokens,
            backend_sampling: Some(backend_sampling),
        };
        SamplingRuntimeConfig::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

impl PySamplingRuntimeConfig {
    fn to_dto(&self) -> dto::SamplingRuntimeConfig {
        self.dto.clone()
    }
}

/// Device placement and memory mapping settings for local model loading.
#[pyclass(name = "ModelPlacementConfig")]
#[derive(Debug, Clone)]
struct PyModelPlacementConfig {
    dto: dto::ModelPlacementConfig,
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
        let dto = dto::ModelPlacementConfig {
            devices,
            gpu_layers: gpu_layers.map(py_gpu_layers).transpose()?,
            split_mode,
            main_gpu,
            tensor_split: tensor_split.map(|values| values.into_iter().map(f64::from).collect()),
            use_mmap,
            use_mlock,
            fit_params,
            fit_params_min_ctx,
            fit_params_target_bytes: fit_params_target_bytes
                .map(|values| values.into_iter().map(|value| value as f64).collect()),
            check_tensors,
            no_extra_bufts,
            no_host,
        };
        ModelPlacementConfig::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

/// Context, threading, attention, and embedding settings for local runtime use.
#[pyclass(name = "ContextRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyContextRuntimeConfig {
    dto: dto::ContextRuntimeConfig,
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
        let dto = dto::ContextRuntimeConfig {
            n_ctx,
            n_batch,
            n_ubatch,
            n_parallel,
            n_threads,
            n_threads_batch,
            flash_attention,
            kv_unified,
            cache_type_k,
            cache_type_v,
            offload_kqv,
            op_offload,
            swa_full,
            warmup,
            rope_scaling,
            rope_freq_base: rope_freq_base.map(f64::from),
            rope_freq_scale: rope_freq_scale.map(f64::from),
            yarn_orig_ctx,
            yarn_ext_factor: yarn_ext_factor.map(f64::from),
            yarn_attn_factor: yarn_attn_factor.map(f64::from),
            yarn_beta_fast: yarn_beta_fast.map(f64::from),
            yarn_beta_slow: yarn_beta_slow.map(f64::from),
            embeddings,
            pooling: pooling
                .map(|value| dto::PoolingType::try_from(value.as_str()).map_err(convert_error))
                .transpose()?,
        };
        sipp::engine::ContextRuntimeConfig::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

/// Scheduler policy knobs for latency, balance, or throughput behavior.
#[pyclass(name = "SchedulerPolicyConfig")]
#[derive(Debug, Clone)]
struct PySchedulerPolicyConfig {
    dto: dto::SchedulerPolicyConfig,
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
        let dto = dto::SchedulerPolicyConfig {
            mode,
            decode_token_reserve,
            enable_adaptive_prefill_chunking,
        };
        sipp::runtime::config::SchedulerPolicyConfig::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

/// Request scheduler and continuous batching settings.
#[pyclass(name = "SchedulerRuntimeConfig")]
#[derive(Debug, Clone)]
struct PySchedulerRuntimeConfig {
    dto: dto::SchedulerRuntimeConfig,
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
        let dto = dto::SchedulerRuntimeConfig {
            continuous_batching,
            policy: policy.as_ref().map(|value| value.borrow(py).dto.clone()),
            prefill_chunk_size,
            max_running_requests,
            max_queued_requests,
            ..Default::default()
        };
        SchedulerRuntimeConfig::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

/// Prefix KV-cache reuse and snapshot settings.
#[pyclass(name = "CacheRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyCacheRuntimeConfig {
    dto: dto::CacheRuntimeConfig,
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
        let dto = dto::CacheRuntimeConfig {
            mode,
            retained_prefix_tokens,
            snapshot_interval_tokens,
            max_snapshot_entries,
            max_snapshot_bytes: max_snapshot_bytes.map(|value| value as f64),
        };
        sipp::engine::CacheRuntimeConfig::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

/// Vision projector and image-token settings for multimodal models.
#[pyclass(name = "MultimodalRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyMultimodalRuntimeConfig {
    dto: dto::MultimodalRuntimeConfig,
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
            dto: dto::MultimodalRuntimeConfig {
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
    dto: dto::ResidencyRuntimeConfig,
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
        Self {
            dto: dto::ResidencyRuntimeConfig {
                max_gpu_models_per_device: max_gpu_models_per_device.map(|value| value as f64),
                allow_cpu_models_while_gpu_loaded,
                require_gpu_lease,
                gpu_memory_safety_margin_bytes: gpu_memory_safety_margin_bytes
                    .map(|value| value as f64),
            },
        }
    }
}

/// Runtime metrics and backend profiling options.
#[pyclass(name = "ObservabilityRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyObservabilityRuntimeConfig {
    dto: dto::ObservabilityRuntimeConfig,
}

#[pymethods]
impl PyObservabilityRuntimeConfig {
    #[new]
    #[pyo3(signature = (*, runtime_metrics = false, backend_profiling = false))]
    fn new(runtime_metrics: bool, backend_profiling: bool) -> Self {
        Self {
            dto: dto::ObservabilityRuntimeConfig {
                runtime_metrics: Some(runtime_metrics),
                backend_profiling: Some(backend_profiling),
            },
        }
    }
}

/// Complete native runtime configuration for local model loading.
#[pyclass(name = "NativeRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyNativeRuntimeConfig {
    dto: dto::NativeRuntimeConfig,
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
    ) -> PyResult<Self> {
        let dto = dto::NativeRuntimeConfig {
            placement: placement.as_ref().map(|value| value.borrow(py).dto.clone()),
            context: context.as_ref().map(|value| value.borrow(py).dto.clone()),
            sampling: sampling.as_ref().map(|value| value.borrow(py).to_dto()),
            scheduler: scheduler.as_ref().map(|value| value.borrow(py).dto.clone()),
            cache: cache.as_ref().map(|value| value.borrow(py).dto.clone()),
            multimodal: multimodal
                .as_ref()
                .map(|value| value.borrow(py).dto.clone()),
            residency: residency.as_ref().map(|value| value.borrow(py).dto.clone()),
            observability: observability
                .as_ref()
                .map(|value| value.borrow(py).dto.clone()),
        };
        NativeRuntimeConfig::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

impl PyNativeRuntimeConfig {
    fn to_dto(&self) -> dto::NativeRuntimeConfig {
        self.dto.clone()
    }
}

/// Role/content chat message accepted by chat requests.
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
        let dto = dto::ChatMessage {
            role: role.clone(),
            content: content.clone(),
        };
        ChatMessage::try_from(&dto).map_err(convert_error)?;
        Ok(Self { role, content })
    }
}

impl PyChatMessage {
    fn to_dto(&self) -> dto::ChatMessage {
        dto::ChatMessage {
            role: self.role.clone(),
            content: self.content.clone(),
        }
    }
}

/// Address of a registered inference endpoint.
#[pyclass(name = "EndpointRef")]
#[derive(Clone)]
struct PyEndpointRef {
    dto: dto::EndpointRef,
}

#[pymethods]
impl PyEndpointRef {
    #[staticmethod]
    fn local(id: String) -> Self {
        Self {
            dto: dto::EndpointRef {
                kind: "local".to_string(),
                id,
            },
        }
    }

    #[staticmethod]
    fn gateway(id: String) -> Self {
        Self {
            dto: dto::EndpointRef {
                kind: "gateway".to_string(),
                id,
            },
        }
    }

    #[staticmethod]
    fn provider(id: String) -> Self {
        Self {
            dto: dto::EndpointRef {
                kind: "provider".to_string(),
                id,
            },
        }
    }

    #[getter]
    fn kind(&self) -> &str {
        &self.dto.kind
    }
}

impl PyEndpointRef {
    fn to_dto(&self) -> dto::EndpointRef {
        self.dto.clone()
    }
}

/// Shared generation options for text-producing requests.
#[pyclass(name = "SippTextOptions")]
#[derive(Clone)]
struct PySippTextOptions {
    dto: dto::SippTextOptions,
}

#[pymethods]
impl PySippTextOptions {
    #[new]
    #[pyo3(signature = (*, max_tokens = None, temperature = None, top_p = None, stop = None))]
    fn new(
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        stop: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let dto = dto::SippTextOptions {
            max_tokens,
            temperature: temperature.map(f64::from),
            top_p: top_p.map(f64::from),
            stop,
        };
        ClientTextOptions::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

impl PySippTextOptions {
    fn to_dto(&self) -> dto::SippTextOptions {
        self.dto.clone()
    }
}

/// Local-only prompt options such as grammar constraints and image inputs.
#[pyclass(name = "LocalTextOptions")]
#[derive(Clone)]
struct PyLocalTextOptions {
    dto: dto::LocalTextOptions,
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
    ) -> PyResult<Self> {
        let dto = dto::LocalTextOptions {
            context_key,
            grammar,
            json_schema,
            sampling: sampling.as_ref().map(|config| config.borrow(py).to_dto()),
            media: media.unwrap_or_default(),
        };
        Ok(Self { dto })
    }
}

impl PyLocalTextOptions {
    fn to_dto(&self) -> dto::LocalTextOptions {
        self.dto.clone()
    }
}

/// Local-only embedding options for context and vector normalization.
#[pyclass(name = "LocalEmbedOptions")]
#[derive(Clone)]
struct PyLocalEmbedOptions {
    dto: dto::LocalEmbedOptions,
}

#[pymethods]
impl PyLocalEmbedOptions {
    #[new]
    #[pyo3(signature = (*, context_key = None, normalize = None))]
    fn new(context_key: Option<String>, normalize: Option<bool>) -> Self {
        Self {
            dto: dto::LocalEmbedOptions {
                context_key,
                normalize,
            },
        }
    }
}

impl PyLocalEmbedOptions {
    fn to_dto(&self) -> dto::LocalEmbedOptions {
        self.dto.clone()
    }
}

/// Gateway endpoint descriptor accepted by SippClient.add.
#[pyclass(name = "GatewayDescriptor")]
#[derive(Clone)]
struct PyGatewayDescriptor {
    dto: dto::EndpointDescriptor,
}

#[pymethods]
impl PyGatewayDescriptor {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (target, base_url, *, authentication_kind = "none", authentication_value = None, authentication_header = None, static_headers = None, timeout_ms = None, query_route = None, chat_route = None, embed_route = None, protocol_options = None))]
    fn new(
        py: Python<'_>,
        target: String,
        base_url: String,
        authentication_kind: &str,
        authentication_value: Option<String>,
        authentication_header: Option<String>,
        static_headers: Option<BTreeMap<String, String>>,
        timeout_ms: Option<u64>,
        query_route: Option<String>,
        chat_route: Option<String>,
        embed_route: Option<String>,
        protocol_options: Option<PyObject>,
    ) -> PyResult<Self> {
        let dto = dto::EndpointDescriptor {
            kind: "gateway".to_string(),
            target: Some(target),
            base_url: Some(base_url),
            authentication: Some(dto::GatewayAuthentication {
                kind: authentication_kind.to_string(),
                value: authentication_value,
                header_name: authentication_header,
            }),
            static_headers: static_headers.map(py_static_headers),
            timeout_ms,
            query_route,
            chat_route,
            embed_route,
            protocol_options: protocol_options
                .map(|value| py_to_json(value.bind(py)))
                .transpose()?,
            ..dto::EndpointDescriptor::default()
        };
        CoreEndpointDescriptor::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

/// Local model descriptor accepted by SippClient.add.
#[pyclass(name = "LocalModelDescriptor")]
#[derive(Clone)]
struct PyLocalModelDescriptor {
    dto: dto::EndpointDescriptor,
}

#[pymethods]
impl PyLocalModelDescriptor {
    #[new]
    #[pyo3(signature = (model_path, config = None))]
    fn new(
        py: Python<'_>,
        model_path: PathBuf,
        config: Option<Py<PyNativeRuntimeConfig>>,
    ) -> PyResult<Self> {
        let dto = dto::EndpointDescriptor {
            kind: "local".to_string(),
            model_path: Some(model_path.to_string_lossy().into_owned()),
            config: config.map(|value| value.borrow(py).to_dto()),
            ..dto::EndpointDescriptor::default()
        };
        CoreEndpointDescriptor::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

/// Direct provider descriptor accepted by SippClient.add.
#[pyclass(name = "ProviderDescriptor")]
#[derive(Clone)]
struct PyProviderDescriptor {
    dto: dto::EndpointDescriptor,
}

#[pymethods]
impl PyProviderDescriptor {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (provider, model, *, api_key = None, base_url = None, timeout_ms = None, version = None, auth_header_name = None, auth_header_value = None, static_headers = None))]
    fn new(
        provider: String,
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
        timeout_ms: Option<u64>,
        version: Option<String>,
        auth_header_name: Option<String>,
        auth_header_value: Option<String>,
        static_headers: Option<Vec<(String, String)>>,
    ) -> PyResult<Self> {
        let dto = dto::EndpointDescriptor {
            kind: "provider".to_string(),
            provider: Some(provider),
            model: Some(model),
            api_key,
            base_url,
            timeout_ms,
            version,
            auth_header_name,
            auth_header_value,
            static_headers: static_headers.map(py_static_headers),
            ..dto::EndpointDescriptor::default()
        };
        CoreEndpointDescriptor::try_from(&dto).map_err(convert_error)?;
        Ok(Self { dto })
    }
}

/// Client facade for registered inference endpoints.
#[pyclass(name = "SippClient")]
struct PySippClient {
    inner: Arc<Mutex<CoreSippClient>>,
}

#[pymethods]
impl PySippClient {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(CoreSippClient::new())),
        })
    }

    /// Register or replace an endpoint and return its current reference.
    fn add(&self, py: Python<'_>, id: String, descriptor: PyObject) -> PyResult<PyEndpointRef> {
        let descriptor = py_endpoint_descriptor_to_core(py, descriptor)?;
        let inner = self.inner.clone();
        let endpoint = py
            .allow_threads(move || {
                let mut client = inner
                    .lock()
                    .map_err(|_| ClientError::Internal(PY_CLIENT_MUTEX_POISONED.to_string()))?;
                block_on(client.add(id, descriptor))
            })
            .map_err(to_py_client_error)?;
        Ok(PyEndpointRef {
            dto: endpoint_ref_to_dto(endpoint),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (prompt, *, endpoint = None, options = None, local = None, endpoint_options = None, provider_options = None, emit_tokens = false))]
    fn query(
        &self,
        py: Python<'_>,
        prompt: String,
        endpoint: Option<Py<PyEndpointRef>>,
        options: Option<Py<PySippTextOptions>>,
        local: Option<Py<PyLocalTextOptions>>,
        endpoint_options: Option<PyObject>,
        provider_options: Option<PyObject>,
        emit_tokens: bool,
    ) -> PyResult<PySippTextRun> {
        let request = dto::SippQueryRequest {
            request_id: None,
            endpoint: endpoint
                .as_ref()
                .map(|endpoint| endpoint.borrow(py).to_dto()),
            prompt,
            options: options.as_ref().map(|value| value.borrow(py).to_dto()),
            local: local.as_ref().map(|value| value.borrow(py).to_dto()),
            endpoint_options: endpoint_options
                .map(|value| py_to_json(value.bind(py)))
                .transpose()?,
            provider_options: provider_options
                .map(|value| py_to_json(value.bind(py)))
                .transpose()?,
            emit_tokens: Some(emit_tokens),
        };
        let request = ClientQueryRequest::try_from(request).map_err(convert_error)?;
        let run = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .query(request);
        Ok(PySippTextRun::from_core(run))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (messages, *, endpoint = None, options = None, local = None, endpoint_options = None, provider_options = None, emit_tokens = false))]
    fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<Py<PyChatMessage>>,
        endpoint: Option<Py<PyEndpointRef>>,
        options: Option<Py<PySippTextOptions>>,
        local: Option<Py<PyLocalTextOptions>>,
        endpoint_options: Option<PyObject>,
        provider_options: Option<PyObject>,
        emit_tokens: bool,
    ) -> PyResult<PySippTextRun> {
        let request = dto::SippChatRequest {
            request_id: None,
            endpoint: endpoint
                .as_ref()
                .map(|endpoint| endpoint.borrow(py).to_dto()),
            messages: messages
                .iter()
                .map(|message| message.borrow(py).to_dto())
                .collect(),
            options: options.as_ref().map(|value| value.borrow(py).to_dto()),
            local: local.as_ref().map(|value| value.borrow(py).to_dto()),
            endpoint_options: endpoint_options
                .map(|value| py_to_json(value.bind(py)))
                .transpose()?,
            provider_options: provider_options
                .map(|value| py_to_json(value.bind(py)))
                .transpose()?,
            emit_tokens: Some(emit_tokens),
        };
        let request = ClientChatRequest::try_from(request).map_err(convert_error)?;
        let run = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .chat(request);
        Ok(PySippTextRun::from_core(run))
    }

    #[pyo3(signature = (input, *, endpoint = None, local = None, endpoint_options = None, provider_options = None))]
    fn embed(
        &self,
        py: Python<'_>,
        input: String,
        endpoint: Option<Py<PyEndpointRef>>,
        local: Option<Py<PyLocalEmbedOptions>>,
        endpoint_options: Option<PyObject>,
        provider_options: Option<PyObject>,
    ) -> PyResult<PySippEmbeddingRun> {
        let request = dto::SippEmbedRequest {
            request_id: None,
            endpoint: endpoint
                .as_ref()
                .map(|endpoint| endpoint.borrow(py).to_dto()),
            input,
            local: local.as_ref().map(|value| value.borrow(py).to_dto()),
            endpoint_options: endpoint_options
                .map(|value| py_to_json(value.bind(py)))
                .transpose()?,
            provider_options: provider_options
                .map(|value| py_to_json(value.bind(py)))
                .transpose()?,
        };
        let request = ClientEmbedRequest::try_from(request).map_err(convert_error)?;
        let run = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_CLIENT_MUTEX_POISONED))?
            .embed(request);
        Ok(PySippEmbeddingRun::from_core(run))
    }
}

/// Text generation handle with a final response and optional token stream.
#[pyclass(name = "SippTextRun")]
struct PySippTextRun {
    response: PySharedClientTextResponse,
    tokens: PySharedClientTokenBatches,
}

impl PySippTextRun {
    fn from_core(run: CoreClientTextRun) -> Self {
        let (tokens, response) = run.into_parts();
        Self {
            response: Arc::new(Mutex::new(Some(response))),
            tokens: Arc::new(Mutex::new(Some(tokens))),
        }
    }
}

#[pymethods]
impl PySippTextRun {
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
        sipp_text_response_to_dict(py, response)
    }

    fn tokens(&self) -> PySippTokenIterator {
        PySippTokenIterator {
            tokens: self.tokens.clone(),
        }
    }
}

/// Iterator over token batches emitted by a text generation run.
#[pyclass(name = "SippTokenIterator")]
struct PySippTokenIterator {
    tokens: PySharedClientTokenBatches,
}

#[pymethods]
impl PySippTokenIterator {
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
#[pyclass(name = "SippEmbeddingRun")]
struct PySippEmbeddingRun {
    response: PySharedClientEmbeddingResponse,
}

impl PySippEmbeddingRun {
    fn from_core(run: CoreClientEmbeddingRun) -> Self {
        Self {
            response: Arc::new(Mutex::new(Some(run.into_response()))),
        }
    }
}

#[pymethods]
impl PySippEmbeddingRun {
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
        sipp_embedding_response_to_dict(py, response)
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
    module.add(
        "EndpointError",
        module.py().get_type_bound::<EndpointError>(),
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
    module.add_class::<PySippTextOptions>()?;
    module.add_class::<PyLocalTextOptions>()?;
    module.add_class::<PyLocalEmbedOptions>()?;
    module.add_class::<PySippClient>()?;
    module.add_class::<PySippTextRun>()?;
    module.add_class::<PySippTokenIterator>()?;
    module.add_class::<PySippEmbeddingRun>()?;
    module.add_class::<PyGatewayDescriptor>()?;
    module.add_class::<PyLocalModelDescriptor>()?;
    module.add_class::<PyProviderDescriptor>()?;
    module.add("DEFAULT_CONTEXT_KEY", DEFAULT_CONTEXT_KEY)?;
    module.add("DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS)?;
    module.add_function(wrap_pyfunction!(backend_observability_json, module)?)?;
    module.add_function(wrap_pyfunction!(set_llama_log_quiet, module)?)?;
    Ok(())
}

fn py_endpoint_descriptor_to_core(
    py: Python<'_>,
    descriptor: PyObject,
) -> PyResult<CoreEndpointDescriptor> {
    let descriptor = descriptor.bind(py);
    if let Ok(descriptor) = descriptor.extract::<PyRef<'_, PyLocalModelDescriptor>>() {
        return CoreEndpointDescriptor::try_from(&descriptor.dto).map_err(convert_error);
    }
    if let Ok(descriptor) = descriptor.extract::<PyRef<'_, PyProviderDescriptor>>() {
        return CoreEndpointDescriptor::try_from(&descriptor.dto).map_err(convert_error);
    }
    if let Ok(descriptor) = descriptor.extract::<PyRef<'_, PyGatewayDescriptor>>() {
        return CoreEndpointDescriptor::try_from(&descriptor.dto).map_err(convert_error);
    }
    Err(PyTypeError::new_err(
        "descriptor must be LocalModelDescriptor, ProviderDescriptor, or GatewayDescriptor",
    ))
}

fn py_static_headers(
    headers: impl IntoIterator<Item = (String, String)>,
) -> Vec<dto::ProviderStaticHeader> {
    headers
        .into_iter()
        .map(|(name, value)| dto::ProviderStaticHeader { name, value })
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
            .ok_or_else(|| PyValueError::new_err("JSON options cannot contain non-finite floats"));
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
                let key = key
                    .extract::<String>()
                    .map_err(|_| PyTypeError::new_err("JSON option object keys must be strings"))?;
                output.insert(key, py_to_json_inner(&item, ancestors)?);
            }
            Ok(serde_json::Value::Object(output))
        })();
        ancestors.remove(&container);
        return result;
    }
    Err(PyTypeError::new_err(
        "JSON options must contain JSON-compatible values",
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
            "JSON options must contain JSON-compatible values",
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
    let endpoint = endpoint_ref_to_dto(endpoint);
    dict.set_item("kind", endpoint.kind)?;
    dict.set_item("id", endpoint.id)?;
    Ok(dict.into_py(py))
}

fn endpoint_ref_to_dto(endpoint: CoreEndpointRef) -> dto::EndpointRef {
    dto::EndpointRef::from(endpoint)
}

fn sipp_text_response_to_dict(py: Python<'_>, response: ClientTextResponse) -> PyResult<Py<PyAny>> {
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

fn sipp_embedding_response_to_dict(
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
    let usage = dto::TokenUsage::from(usage);
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
    let stats = dto::RequestStats::from(stats);
    let dict = PyDict::new_bound(py);
    dict.set_item("input_tokens", stats.input_tokens)?;
    dict.set_item("output_tokens", stats.output_tokens)?;
    dict.set_item("cache_mode", stats.cache_mode)?;
    dict.set_item("cache_source", stats.cache_source)?;
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

fn py_gpu_layers(value: &Bound<'_, PyAny>) -> PyResult<dto::GpuLayers> {
    serde_json::from_value(py_to_json(value)?)
        .map_err(|_| PyTypeError::new_err(r#"gpu_layers must be "auto", "all", or {"count": int}"#))
}

fn convert_error(error: dto::ConvertError) -> PyErr {
    match error {
        dto::ConvertError::InvalidArg(message) => PyValueError::new_err(message),
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

fn to_py_provider_error(error: CoreProviderEndpointError) -> PyErr {
    Python::with_gil(|py| match to_py_provider_error_result(py, error) {
        Ok(error) => error,
        Err(error) => error,
    })
}

fn to_py_provider_error_result(
    py: Python<'_>,
    error: CoreProviderEndpointError,
) -> PyResult<PyErr> {
    let message = provider_error_message(&error);
    let instance = py.get_type_bound::<ProviderError>().call1((message,))?;
    let retry_after_ms = error
        .retry_after
        .map(|duration| duration.as_secs_f64() * 1000.0);
    let raw_body = match error.raw {
        Some(value) => json_to_py(py, *value)?,
        None => py.None(),
    };

    instance.setattr("kind", error.kind.as_str())?;
    instance.setattr("provider", error.provider)?;
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
        ClientError::Endpoint(error) => Python::with_gil(|py| {
            let instance = py
                .get_type_bound::<EndpointError>()
                .call1((error.to_string(),))
                .and_then(|instance| {
                    instance.setattr("kind", error.kind)?;
                    instance.setattr("status", error.status)?;
                    instance.setattr("code", error.code)?;
                    instance.setattr("request_id", error.request_id)?;
                    Ok(instance)
                });
            match instance {
                Ok(instance) => PyErr::from_value_bound(instance),
                Err(error) => error,
            }
        }),
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

fn to_py_error(error: sipp::error::Error) -> PyErr {
    match error {
        sipp::error::Error::InvalidRequest(message)
        | sipp::error::Error::InvalidConfig(message) => PyValueError::new_err(message),
        sipp::error::Error::UnsupportedOperation { operation, reason } => {
            UnsupportedOperationError::new_err(format!(
                "unsupported operation {operation}: {reason}"
            ))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}
