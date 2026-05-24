use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use cogentlm_engine::backend::{
    backend_observability_json as core_backend_observability_json,
    set_llama_log_quiet as core_set_llama_log_quiet,
};
use cogentlm_engine::engine::protocol::{BackendInfo, RequestState, RequestStats};
use cogentlm_engine::engine::{
    CacheKeyPolicy, ChatMessage, ChatRequest, ChatRole, CogentEngine, EngineEvent,
    EngineEventReceiver, EngineState, EngineStats, FlashAttentionMode, GenerationResult,
    GpuLayerConfig, KvCacheType, KvReuseMode, LogitBias, ModelPlacementConfig,
    MultimodalRuntimeConfig, NativeRuntimeConfig, ObservabilityRuntimeConfig, QueryOptions,
    QueryRequest, ResidencyRuntimeConfig, ResolvedRuntimeLimits, RopeScaling, SamplerStage,
    SamplingRuntimeConfig, SchedulerRuntimeConfig, SplitMode, TokenBatch, DEFAULT_CONTEXT_KEY,
    DEFAULT_MAX_TOKENS,
};
use cogentlm_engine::lifecycle::{
    model_source_from_path as core_model_source_from_path,
    vision_model_source_from_paths as core_vision_model_source_from_paths, BackendPreference,
    LoadedModelInfo, ModelInfo, ModelLoadOptions, ModelService, ModelServiceState, StatsMode,
    DEFAULT_MODEL_BACKEND, DEFAULT_MODEL_STATS,
};
use cogentlm_engine::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode};
use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::PyClass;
use pyo3::types::{PyAny, PyDict, PyList};
use serde::de::DeserializeOwned;

pyo3::create_exception!(
    _native,
    UnsupportedOperationError,
    PyException,
    "The loaded model does not support the requested operation."
);

const PY_CALLBACK_FAILED_MESSAGE: &str = "Python token callback failed";
const PY_ENGINE_EVENTS_MUTEX_POISONED: &str = "engine events mutex is poisoned";
const PY_ENGINE_MUTEX_POISONED: &str = "engine mutex is poisoned";
const PY_ENGINE_CLOSED: &str = "engine is closed";
const PY_MODEL_SERVICE_EVENTS_MUTEX_POISONED: &str = "model service events mutex is poisoned";
const PY_MODEL_SERVICE_MUTEX_POISONED: &str = "model service mutex is poisoned";
const PY_MODEL_SERVICE_CLOSED: &str = "model service is closed";
const EVENT_TYPE_STATE: &str = "state";
const EVENT_TYPE_LOAD_PROGRESS: &str = "load-progress";
const EVENT_TYPE_REQUEST_STARTED: &str = "request-started";
const EVENT_TYPE_REQUEST_COMPLETED: &str = "request-completed";
const EVENT_TYPE_REQUEST_FAILED: &str = "request-failed";
const EVENT_TYPE_CLOSED: &str = "closed";

fn clear_events(events: &Mutex<Option<EngineEventReceiver>>) {
    if let Ok(mut events) = events.lock() {
        events.take();
    }
}

fn drain_events_to_list(
    py: Python<'_>,
    events: &Mutex<Option<EngineEventReceiver>>,
    poison_message: &'static str,
) -> PyResult<Py<PyAny>> {
    let output = PyList::empty_bound(py);
    let guard = events
        .lock()
        .map_err(|_| PyRuntimeError::new_err(poison_message))?;
    if let Some(receiver) = guard.as_ref() {
        for event in receiver.try_iter() {
            output.append(engine_event_to_dict(py, event)?)?;
        }
    }
    Ok(output.into_py(py))
}

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
        yarn_beta_slow = None
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
        Ok(Self { core })
    }
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

#[pyclass(name = "ModelLoadOptions")]
#[derive(Debug, Clone)]
struct PyModelLoadOptions {
    #[pyo3(get)]
    backend: String,
    #[pyo3(get)]
    stats: String,
    runtime: NativeRuntimeConfig,
}

#[pymethods]
impl PyModelLoadOptions {
    #[new]
    #[pyo3(signature = (*, backend = DEFAULT_MODEL_BACKEND.to_string(), stats = DEFAULT_MODEL_STATS.to_string(), runtime = None))]
    fn new(
        py: Python<'_>,
        backend: String,
        stats: String,
        runtime: Option<Py<PyNativeRuntimeConfig>>,
    ) -> PyResult<Self> {
        let runtime = py_core_or_default(py, runtime, PyNativeRuntimeConfig::to_core);
        parse_backend_preference(&backend)?;
        parse_stats_mode(&stats)?;
        Ok(Self {
            backend,
            stats,
            runtime,
        })
    }
}

impl PyModelLoadOptions {
    fn to_core(&self) -> PyResult<ModelLoadOptions> {
        Ok(ModelLoadOptions {
            backend: parse_backend_preference(&self.backend)?,
            stats: parse_stats_mode(&self.stats)?,
            runtime: self.runtime.clone(),
        })
    }
}

#[pyclass(name = "QueryOptions")]
#[derive(Debug, Clone)]
struct PyQueryOptions {
    #[pyo3(get)]
    context_key: String,
    #[pyo3(get)]
    max_tokens: i32,
    #[pyo3(get)]
    grammar: String,
    #[pyo3(get)]
    json_schema: String,
    #[pyo3(get)]
    stop: Vec<String>,
    #[pyo3(get)]
    media: Vec<Vec<u8>>,
    sampling: Option<SamplingRuntimeConfig>,
}

#[pymethods]
impl PyQueryOptions {
    #[new]
    #[pyo3(signature = (
        context_key = DEFAULT_CONTEXT_KEY.to_string(),
        max_tokens = DEFAULT_MAX_TOKENS,
        grammar = "".to_string(),
        *,
        json_schema = "".to_string(),
        stop = None,
        sampling = None,
        media = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        context_key: String,
        max_tokens: i32,
        grammar: String,
        json_schema: String,
        stop: Option<Vec<String>>,
        sampling: Option<Py<PySamplingRuntimeConfig>>,
        media: Option<Vec<Vec<u8>>>,
    ) -> PyResult<Self> {
        if max_tokens <= 0 {
            return Err(PyValueError::new_err("max_tokens must be positive"));
        }
        Ok(Self {
            context_key,
            max_tokens,
            grammar,
            json_schema,
            stop: stop.unwrap_or_default(),
            sampling: sampling.as_ref().map(|config| config.borrow(py).to_core()),
            media: media.unwrap_or_default(),
        })
    }
}

impl PyQueryOptions {
    fn to_core(&self) -> QueryOptions {
        QueryOptions {
            context_key: self.context_key.clone(),
            max_tokens: self.max_tokens,
            grammar: self.grammar.clone(),
            json_schema: self.json_schema.clone(),
            stop: self.stop.clone(),
            sampling: self.sampling.clone(),
            media: self.media.clone(),
        }
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

#[pyclass(name = "CogentEngine")]
struct PyCogentEngine {
    inner: Mutex<Option<CogentEngine>>,
    events: Mutex<Option<EngineEventReceiver>>,
}

#[pymethods]
impl PyCogentEngine {
    #[new]
    #[pyo3(signature = (model_path, config = None))]
    fn new(
        py: Python<'_>,
        model_path: PathBuf,
        config: Option<Py<PyNativeRuntimeConfig>>,
    ) -> PyResult<Self> {
        let config = py_core_or_default(py, config, PyNativeRuntimeConfig::to_core);
        let engine = py
            .allow_threads(|| CogentEngine::load(model_path, config))
            .map_err(to_py_error)?;
        let events = engine.subscribe_events();
        Ok(Self {
            inner: Mutex::new(Some(engine)),
            events: Mutex::new(Some(events)),
        })
    }

    #[pyo3(signature = (prompt, options = None, on_tokens = None))]
    fn query(
        &self,
        py: Python<'_>,
        prompt: String,
        options: Option<Py<PyQueryOptions>>,
        on_tokens: Option<PyObject>,
    ) -> PyResult<Py<PyAny>> {
        let options = py_core_or_default(py, options, PyQueryOptions::to_core);
        let callback_error = Arc::new(Mutex::new(None));
        let request =
            query_request_with_tokens(py, prompt, options, on_tokens, callback_error.clone())?;
        let guard = self.engine_guard()?;
        let engine = engine_ref(&guard)?;
        let result = generation_result_or_callback_error(
            py.allow_threads(move || engine.query(request)),
            callback_error,
        )?;
        generation_result_to_dict(py, result)
    }

    #[pyo3(signature = (messages, options = None, on_tokens = None))]
    fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<Py<PyChatMessage>>,
        options: Option<Py<PyQueryOptions>>,
        on_tokens: Option<PyObject>,
    ) -> PyResult<Py<PyAny>> {
        let options = py_core_or_default(py, options, PyQueryOptions::to_core);
        let messages = chat_messages_to_core(py, messages)?;
        let callback_error = Arc::new(Mutex::new(None));
        let request =
            chat_request_with_tokens(py, messages, options, on_tokens, callback_error.clone())?;
        let guard = self.engine_guard()?;
        let engine = engine_ref(&guard)?;
        let result = generation_result_or_callback_error(
            py.allow_threads(move || engine.chat(request)),
            callback_error,
        )?;
        generation_result_to_dict(py, result)
    }

    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = self.engine_guard()?;
        let engine = engine_ref(&guard)?;
        let state = py.allow_threads(|| engine.state()).map_err(to_py_error)?;
        engine_state_to_dict(py, state)
    }

    fn drain_events(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        drain_events_to_list(py, &self.events, PY_ENGINE_EVENTS_MUTEX_POISONED)
    }
}

impl PyCogentEngine {
    fn engine_guard(&self) -> PyResult<MutexGuard<'_, Option<CogentEngine>>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_ENGINE_MUTEX_POISONED))
    }
}

#[pyclass(name = "ModelService")]
struct PyModelService {
    inner: Mutex<Option<ModelService>>,
    events: Mutex<Option<EngineEventReceiver>>,
}

#[pymethods]
impl PyModelService {
    #[new]
    fn new(store_path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: Mutex::new(Some(
                ModelService::local(store_path).map_err(to_py_model_error)?,
            )),
            events: Mutex::new(None),
        })
    }

    #[pyo3(signature = (model_path, options = None))]
    fn load_path(
        &self,
        py: Python<'_>,
        model_path: PathBuf,
        options: Option<Py<PyModelLoadOptions>>,
    ) -> PyResult<Py<PyAny>> {
        let options = options
            .as_ref()
            .map(|options| options.borrow(py).to_core())
            .transpose()?
            .unwrap_or_default();
        let loaded = self.load_and_refresh(py, |service| {
            service.load(core_model_source_from_path(model_path), options)
        })?;
        loaded_model_info_to_dict(py, loaded)
    }

    #[pyo3(signature = (model_path, projector_path, options = None))]
    fn load_vision(
        &self,
        py: Python<'_>,
        model_path: PathBuf,
        projector_path: PathBuf,
        options: Option<Py<PyModelLoadOptions>>,
    ) -> PyResult<Py<PyAny>> {
        let options = options
            .as_ref()
            .map(|options| options.borrow(py).to_core())
            .transpose()?
            .unwrap_or_default();
        let source = core_vision_model_source_from_paths(model_path, projector_path);
        let loaded = self.load_and_refresh(py, |service| service.load(source, options))?;
        loaded_model_info_to_dict(py, loaded)
    }

    fn unload(&self, py: Python<'_>) -> PyResult<()> {
        let mut guard = self.service_guard()?;
        let service = service_mut(&mut guard)?;
        py.allow_threads(|| service.unload())
            .map_err(to_py_model_error)?;
        clear_events(&self.events);
        Ok(())
    }

    fn remove(&self, py: Python<'_>, model_id: String) -> PyResult<()> {
        let mut guard = self.service_guard()?;
        let service = service_mut(&mut guard)?;
        py.allow_threads(|| service.remove(model_id))
            .map_err(to_py_model_error)
    }

    fn list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = self.service_guard()?;
        let service = service_ref(&guard)?;
        let output = PyList::empty_bound(py);
        for model in service.list() {
            output.append(model_info_to_dict(py, model)?)?;
        }
        Ok(output.into_py(py))
    }

    fn current(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = self.service_guard()?;
        let service = service_ref(&guard)?;
        if let Some(model) = service.current() {
            model_info_to_dict(py, model)
        } else {
            Ok(py.None())
        }
    }

    #[pyo3(signature = (prompt, options = None, on_tokens = None))]
    fn query(
        &self,
        py: Python<'_>,
        prompt: String,
        options: Option<Py<PyQueryOptions>>,
        on_tokens: Option<PyObject>,
    ) -> PyResult<Py<PyAny>> {
        let options = py_core_or_default(py, options, PyQueryOptions::to_core);
        let callback_error = Arc::new(Mutex::new(None));
        let request =
            query_request_with_tokens(py, prompt, options, on_tokens, callback_error.clone())?;
        let guard = self.service_guard()?;
        let service = service_ref(&guard)?;
        let result = model_generation_result_or_callback_error(
            py.allow_threads(|| service.query(request)),
            callback_error,
        )?;
        generation_result_to_dict(py, result)
    }

    #[pyo3(signature = (messages, options = None, on_tokens = None))]
    fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<Py<PyChatMessage>>,
        options: Option<Py<PyQueryOptions>>,
        on_tokens: Option<PyObject>,
    ) -> PyResult<Py<PyAny>> {
        let options = py_core_or_default(py, options, PyQueryOptions::to_core);
        let messages = chat_messages_to_core(py, messages)?;
        let callback_error = Arc::new(Mutex::new(None));
        let request =
            chat_request_with_tokens(py, messages, options, on_tokens, callback_error.clone())?;
        let guard = self.service_guard()?;
        let service = service_ref(&guard)?;
        let result = model_generation_result_or_callback_error(
            py.allow_threads(|| service.chat(request)),
            callback_error,
        )?;
        generation_result_to_dict(py, result)
    }

    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = self.service_guard()?;
        let service = service_ref(&guard)?;
        let state = py
            .allow_threads(|| service.state())
            .map_err(to_py_model_error)?;
        model_service_state_to_dict(py, state)
    }

    fn drain_events(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        drain_events_to_list(py, &self.events, PY_MODEL_SERVICE_EVENTS_MUTEX_POISONED)
    }
}

impl PyModelService {
    fn service_guard(&self) -> PyResult<MutexGuard<'_, Option<ModelService>>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_MODEL_SERVICE_MUTEX_POISONED))
    }

    fn refresh_events(&self, service: &ModelService) -> PyResult<()> {
        let events = service.subscribe_events().map_err(to_py_model_error)?;
        self.events
            .lock()
            .map_err(|_| PyRuntimeError::new_err(PY_MODEL_SERVICE_EVENTS_MUTEX_POISONED))?
            .replace(events);
        Ok(())
    }

    fn load_and_refresh<T: Send>(
        &self,
        py: Python<'_>,
        load: impl FnOnce(&mut ModelService) -> Result<T, cogentlm_engine::lifecycle::ModelError> + Send,
    ) -> PyResult<T> {
        let mut guard = self.service_guard()?;
        let service = service_mut(&mut guard)?;
        let loaded = py
            .allow_threads(|| load(service))
            .map_err(to_py_model_error)?;
        self.refresh_events(service)?;
        Ok(loaded)
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
    module.add_class::<PyModelLoadOptions>()?;
    module.add_class::<PyQueryOptions>()?;
    module.add_class::<PyChatMessage>()?;
    module.add_class::<PyCogentEngine>()?;
    module.add_class::<PyModelService>()?;
    module.add("DEFAULT_CONTEXT_KEY", DEFAULT_CONTEXT_KEY)?;
    module.add("DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS)?;
    module.add("DEFAULT_MODEL_BACKEND", DEFAULT_MODEL_BACKEND)?;
    module.add("DEFAULT_MODEL_STATS", DEFAULT_MODEL_STATS)?;
    module.add_function(wrap_pyfunction!(backend_observability_json, module)?)?;
    module.add_function(wrap_pyfunction!(set_llama_log_quiet, module)?)?;
    Ok(())
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

fn query_request_with_tokens(
    py: Python<'_>,
    prompt: String,
    options: QueryOptions,
    on_tokens: Option<PyObject>,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyResult<QueryRequest> {
    let mut request = QueryRequest::new(prompt).options(options);
    if let Some(callback) = on_tokens {
        require_callable(py, &callback)?;
        request = request.on_tokens(make_python_tokens_callback(callback, callback_error));
    }
    Ok(request)
}

fn chat_request_with_tokens(
    py: Python<'_>,
    messages: Vec<ChatMessage>,
    options: QueryOptions,
    on_tokens: Option<PyObject>,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyResult<ChatRequest> {
    let mut request = ChatRequest::new(messages).options(options);
    if let Some(callback) = on_tokens {
        require_callable(py, &callback)?;
        request = request.on_tokens(make_python_tokens_callback(callback, callback_error));
    }
    Ok(request)
}

fn engine_ref<'a>(guard: &'a MutexGuard<'_, Option<CogentEngine>>) -> PyResult<&'a CogentEngine> {
    guard
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err(PY_ENGINE_CLOSED))
}

fn service_ref<'a>(guard: &'a MutexGuard<'_, Option<ModelService>>) -> PyResult<&'a ModelService> {
    guard
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err(PY_MODEL_SERVICE_CLOSED))
}

fn service_mut<'a>(
    guard: &'a mut MutexGuard<'_, Option<ModelService>>,
) -> PyResult<&'a mut ModelService> {
    guard
        .as_mut()
        .ok_or_else(|| PyRuntimeError::new_err(PY_MODEL_SERVICE_CLOSED))
}

fn require_callable(py: Python<'_>, callback: &PyObject) -> PyResult<()> {
    if callback.bind(py).is_callable() {
        Ok(())
    } else {
        Err(PyTypeError::new_err("on_tokens must be callable"))
    }
}

fn make_python_tokens_callback(
    callback: PyObject,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> impl FnMut(&TokenBatch) -> cogentlm_engine::Result<()> + Send + 'static {
    move |batch| {
        if has_callback_error(&callback_error) {
            return Err(cogentlm_engine::Error::RuntimeCommand(
                PY_CALLBACK_FAILED_MESSAGE.to_string(),
            ));
        }

        Python::with_gil(|py| {
            let batch = token_batch_to_dict(py, batch.clone()).map_err(|error| {
                store_callback_error(&callback_error, error);
                cogentlm_engine::Error::RuntimeCommand(PY_CALLBACK_FAILED_MESSAGE.to_string())
            })?;
            callback.call1(py, (batch,)).map(|_| ()).map_err(|error| {
                store_callback_error(&callback_error, error);
                cogentlm_engine::Error::RuntimeCommand(PY_CALLBACK_FAILED_MESSAGE.to_string())
            })
        })
    }
}

fn generation_result_or_callback_error(
    result: cogentlm_engine::Result<GenerationResult>,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyResult<GenerationResult> {
    match result {
        Ok(result) => Ok(result),
        Err(error) => Err(callback_error_or_core_error(error, callback_error)),
    }
}

fn model_generation_result_or_callback_error(
    result: Result<GenerationResult, cogentlm_engine::lifecycle::ModelError>,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyResult<GenerationResult> {
    match result {
        Ok(result) => Ok(result),
        Err(error) => Err(callback_error_or_model_error(error, callback_error)),
    }
}

fn callback_error_or_model_error(
    error: cogentlm_engine::lifecycle::ModelError,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyErr {
    if matches!(
        &error,
        cogentlm_engine::lifecycle::ModelError::Runtime(message) if message.contains(PY_CALLBACK_FAILED_MESSAGE)
    ) {
        if let Some(error) = take_callback_error(&callback_error) {
            return error;
        }
    }
    to_py_model_error(error)
}

fn callback_error_or_core_error(
    error: cogentlm_engine::Error,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyErr {
    if matches!(
        &error,
        cogentlm_engine::Error::RuntimeCommand(message) if message == PY_CALLBACK_FAILED_MESSAGE
    ) {
        if let Some(error) = take_callback_error(&callback_error) {
            return error;
        }
    }
    to_py_error(error)
}

fn has_callback_error(callback_error: &Arc<Mutex<Option<PyErr>>>) -> bool {
    callback_error
        .lock()
        .map(|error| error.is_some())
        .unwrap_or(true)
}

fn store_callback_error(callback_error: &Arc<Mutex<Option<PyErr>>>, error: PyErr) {
    if let Ok(mut stored) = callback_error.lock() {
        if stored.is_none() {
            *stored = Some(error);
        }
    }
}

fn take_callback_error(callback_error: &Arc<Mutex<Option<PyErr>>>) -> Option<PyErr> {
    callback_error
        .lock()
        .ok()
        .and_then(|mut error| error.take())
}

fn generation_result_to_dict(py: Python<'_>, result: GenerationResult) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", result.id)?;
    dict.set_item("text", result.text)?;
    dict.set_item("finish_reason", result.finish_reason.as_str())?;
    dict.set_item("stats", request_stats_to_dict(py, result.stats)?)?;
    Ok(dict.into_py(py))
}

fn loaded_model_info_to_dict(py: Python<'_>, loaded: LoadedModelInfo) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("model", model_info_to_dict(py, loaded.model)?)?;
    let backend = PyDict::new_bound(py);
    backend.set_item("requested", loaded.backend.requested.as_str())?;
    backend.set_item("selected", loaded.backend.selected)?;
    backend.set_item("available", loaded.backend.available)?;
    backend.set_item("gpu_offload_expected", loaded.backend.gpu_offload_expected)?;
    backend.set_item("reason", loaded.backend.reason)?;
    dict.set_item("backend", backend)?;
    dict.set_item("runtime_fingerprint", loaded.runtime_fingerprint)?;
    Ok(dict.into_py(py))
}

fn model_info_to_dict(py: Python<'_>, model: ModelInfo) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", model.id)?;
    dict.set_item("name", model.name)?;
    dict.set_item("modality", model.modality.as_str())?;
    dict.set_item("status", model.status.as_str())?;
    dict.set_item("source", model.source.as_str())?;
    dict.set_item("bytes", model.bytes)?;
    dict.set_item("loaded", model.loaded)?;
    dict.set_item("chat_template", model.chat_template)?;
    dict.set_item("bos_text", model.bos_text)?;
    dict.set_item("eos_text", model.eos_text)?;
    dict.set_item("media_marker", model.media_marker)?;
    dict.set_item("created_at_unix_ms", model.created_at_unix_ms)?;
    dict.set_item("updated_at_unix_ms", model.updated_at_unix_ms)?;
    Ok(dict.into_py(py))
}

fn model_service_state_to_dict(py: Python<'_>, state: ModelServiceState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("status", state.status.as_str())?;
    if let Some(model) = state.model {
        dict.set_item("model", model_info_to_dict(py, model)?)?;
    } else {
        dict.set_item("model", py.None())?;
    }
    set_state_tail_fields(
        py,
        &dict,
        state.backend,
        state.runtime,
        state.requests,
        state.stats,
        state.updated_at_unix_ms,
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

fn engine_state_to_dict(py: Python<'_>, state: EngineState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("status", state.status.as_str())?;
    if let Some(model) = state.model {
        let model_dict = PyDict::new_bound(py);
        model_dict.set_item("id", model.id)?;
        model_dict.set_item("name", model.name)?;
        dict.set_item("model", model_dict)?;
    } else {
        dict.set_item("model", py.None())?;
    }
    set_state_tail_fields(
        py,
        &dict,
        state.backend,
        state.runtime,
        state.requests,
        state.stats,
        state.updated_at_unix_ms,
    )?;
    Ok(dict.into_py(py))
}

fn set_state_tail_fields(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    backend: BackendInfo,
    runtime: Option<ResolvedRuntimeLimits>,
    requests: Vec<RequestState>,
    stats: EngineStats,
    updated_at_unix_ms: u64,
) -> PyResult<()> {
    dict.set_item("backend", backend_info_to_dict(py, backend)?)?;
    dict.set_item("runtime", resolved_runtime_limits_to_dict(py, runtime)?)?;
    let request_items = PyList::empty_bound(py);
    for request in requests {
        let item = PyDict::new_bound(py);
        item.set_item("id", request.id)?;
        item.set_item("status", request.status.as_str())?;
        item.set_item("input_tokens", request.input_tokens)?;
        item.set_item("output_tokens", request.output_tokens)?;
        request_items.append(item)?;
    }
    dict.set_item("requests", request_items)?;
    dict.set_item("stats", engine_stats_to_dict(py, stats)?)?;
    dict.set_item("updated_at_unix_ms", updated_at_unix_ms)?;
    Ok(())
}

fn backend_info_to_dict(py: Python<'_>, backend: BackendInfo) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("selected", backend.selected)?;
    dict.set_item("available", backend.available)?;
    let devices = PyList::empty_bound(py);
    for device in backend.devices {
        let item = PyDict::new_bound(py);
        item.set_item("id", device.id)?;
        item.set_item("name", device.name)?;
        item.set_item("type", device.device_type)?;
        item.set_item("memory_total_bytes", device.memory_total_bytes)?;
        item.set_item("memory_free_bytes", device.memory_free_bytes)?;
        devices.append(item)?;
    }
    dict.set_item("devices", devices)?;
    Ok(dict.into_py(py))
}

fn resolved_runtime_limits_to_dict(
    py: Python<'_>,
    runtime: Option<ResolvedRuntimeLimits>,
) -> PyResult<Py<PyAny>> {
    let Some(runtime) = runtime else {
        return Ok(py.None());
    };
    let dict = PyDict::new_bound(py);
    dict.set_item("n_ctx", runtime.n_ctx)?;
    dict.set_item("n_batch", runtime.n_batch)?;
    dict.set_item("n_ubatch", runtime.n_ubatch)?;
    dict.set_item("n_parallel", runtime.n_parallel)?;
    dict.set_item("kv_unified", runtime.kv_unified)?;
    dict.set_item("flash_attention", runtime.flash_attention)?;
    dict.set_item("cache_type_k", runtime.cache_type_k)?;
    dict.set_item("cache_type_v", runtime.cache_type_v)?;
    Ok(dict.into_py(py))
}

fn engine_stats_to_dict(py: Python<'_>, stats: EngineStats) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("requests_running", stats.requests_running)?;
    dict.set_item("requests_queued", stats.requests_queued)?;
    dict.set_item("requests_completed", stats.requests_completed)?;
    dict.set_item("requests_failed", stats.requests_failed)?;
    dict.set_item("input_tokens", stats.input_tokens)?;
    dict.set_item("output_tokens", stats.output_tokens)?;
    dict.set_item("cache_hits", stats.cache_hits)?;
    dict.set_item("prefill_tokens", stats.prefill_tokens)?;
    dict.set_item("ttft_ms", stats.ttft_ms)?;
    dict.set_item("inter_token_ms", stats.inter_token_ms)?;
    dict.set_item("e2e_ms", stats.e2e_ms)?;
    dict.set_item("tokens_per_second", stats.tokens_per_second)?;
    dict.set_item("decode_tokens_per_second", stats.decode_tokens_per_second)?;
    dict.set_item("prefill_tokens_per_second", stats.prefill_tokens_per_second)?;
    dict.set_item("prefill_ms", stats.prefill_ms)?;
    dict.set_item("decode_ms", stats.decode_ms)?;
    dict.set_item("backend_ms", stats.backend_ms)?;
    dict.set_item("sync_ms", stats.sync_ms)?;
    dict.set_item("engine_overhead_ms", stats.engine_overhead_ms)?;
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

fn engine_event_to_dict(py: Python<'_>, event: EngineEvent) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    match event {
        EngineEvent::State(state) => {
            dict.set_item("type", EVENT_TYPE_STATE)?;
            dict.set_item("state", engine_state_to_dict(py, *state)?)?;
        }
        EngineEvent::LoadProgress {
            loaded_bytes,
            total_bytes,
            asset_name,
        } => {
            dict.set_item("type", EVENT_TYPE_LOAD_PROGRESS)?;
            dict.set_item("loaded_bytes", loaded_bytes)?;
            dict.set_item("total_bytes", total_bytes)?;
            dict.set_item("asset_name", asset_name)?;
        }
        EngineEvent::RequestStarted {
            request_id,
            stream_id,
        } => {
            dict.set_item("type", EVENT_TYPE_REQUEST_STARTED)?;
            dict.set_item("request_id", request_id)?;
            dict.set_item("stream_id", stream_id)?;
        }
        EngineEvent::RequestCompleted { result } => {
            dict.set_item("type", EVENT_TYPE_REQUEST_COMPLETED)?;
            dict.set_item("result", generation_result_to_dict(py, *result)?)?;
        }
        EngineEvent::RequestFailed { request_id, error } => {
            dict.set_item("type", EVENT_TYPE_REQUEST_FAILED)?;
            dict.set_item("request_id", request_id)?;
            dict.set_item("error", error)?;
        }
        EngineEvent::Closed => {
            dict.set_item("type", EVENT_TYPE_CLOSED)?;
        }
    }
    Ok(dict.into_py(py))
}

fn parse_backend_preference(value: &str) -> PyResult<BackendPreference> {
    parse_choice(
        value,
        "backend must be one of: auto, cpu, cuda, metal, vulkan, webgpu",
    )
}

fn parse_stats_mode(value: &str) -> PyResult<StatsMode> {
    parse_choice(value, "stats must be one of: off, basic, profile")
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

fn parse_choice<T>(value: &str, error_message: &'static str) -> PyResult<T>
where
    T: DeserializeOwned,
{
    serde_json::from_value(serde_json::Value::String(value.to_string()))
        .map_err(|_| PyValueError::new_err(error_message))
}

fn assign_if_some<T>(target: &mut T, value: Option<T>) {
    if let Some(value) = value {
        *target = value;
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

fn to_py_model_error(error: cogentlm_engine::lifecycle::ModelError) -> PyErr {
    match error {
        cogentlm_engine::lifecycle::ModelError::InvalidModelSource(message)
        | cogentlm_engine::lifecycle::ModelError::InvalidModelPairing(message) => {
            PyValueError::new_err(message)
        }
        cogentlm_engine::lifecycle::ModelError::UnsupportedGgufVersion(version) => {
            PyValueError::new_err(format!("unsupported GGUF version {version}"))
        }
        cogentlm_engine::lifecycle::ModelError::UnsupportedOperation { operation, reason } => {
            UnsupportedOperationError::new_err(format!(
                "unsupported operation {operation}: {reason}"
            ))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}
