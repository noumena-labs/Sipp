use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use cogentlm_core::engine::protocol::{
    BackendDevice, BackendInfo, EngineStatus, FinishReason, ModelState, RequestState, RequestStats,
    RequestStatus,
};
use cogentlm_core::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode};
use cogentlm_core::runtime::metrics::RuntimeObservabilityMetrics;
use cogentlm_core::runtime::request::{GenerateResponse, GenerateResponseStatus};
use cogentlm_core::{
    backend_observability_json as core_backend_observability_json,
    model_source_from_path as core_model_source_from_path,
    set_llama_log_quiet as core_set_llama_log_quiet, CacheKeyPolicy, ChatMessage, ChatRequest,
    ChatRole, CogentEngine, EngineEvent, EngineEventReceiver, EngineState, EngineStats,
    FlashAttentionMode, GpuLayerConfig, KvCacheType, KvReuseMode, LoadedModelInfo, LogitBias,
    ModelInfo, ModelLoadOptions, ModelModality, ModelPlacementConfig, ModelService,
    ModelServiceState, ModelSourceKind, ModelStatus, MultimodalRuntimeConfig, NativeRuntimeConfig,
    ObservabilityRuntimeConfig, QueryOptions, QueryRequest, RequestResult, ResidencyRuntimeConfig,
    ResolvedRuntimeLimits, RopeScaling, SamplerStage, SamplingRuntimeConfig,
    SchedulerRuntimeConfig, SplitMode, StatsMode, TokenBatch,
};
use cogentlm_core::{
    vision_model_source_from_paths as core_vision_model_source_from_paths, BackendPreference,
    BackendSelection,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const PY_CALLBACK_FAILED_MESSAGE: &str = "Python token callback failed";

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

    fn __repr__(&self) -> String {
        format!("{:?}", self.core)
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
        gpu_layers: Option<String>,
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
        if let Some(devices) = devices {
            core.devices = devices;
        }
        if let Some(gpu_layers) = gpu_layers {
            core.gpu_layers = parse_gpu_layers(&gpu_layers)?;
        }
        if let Some(split_mode) = split_mode {
            core.split_mode = parse_split_mode(&split_mode)?;
        }
        core.main_gpu = main_gpu;
        if let Some(tensor_split) = tensor_split {
            core.tensor_split = tensor_split;
        }
        if let Some(use_mmap) = use_mmap {
            core.use_mmap = use_mmap;
        }
        if let Some(use_mlock) = use_mlock {
            core.use_mlock = use_mlock;
        }
        if let Some(fit_params) = fit_params {
            core.fit_params = fit_params;
        }
        core.fit_params_min_ctx = fit_params_min_ctx;
        if let Some(target_bytes) = fit_params_target_bytes {
            core.fit_params_target_bytes = target_bytes;
        }
        if let Some(check_tensors) = check_tensors {
            core.check_tensors = check_tensors;
        }
        if let Some(no_extra_bufts) = no_extra_bufts {
            core.no_extra_bufts = no_extra_bufts;
        }
        if let Some(no_host) = no_host {
            core.no_host = no_host;
        }
        Ok(Self { core })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.core)
    }
}

#[pyclass(name = "ContextRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyContextRuntimeConfig {
    core: cogentlm_core::ContextRuntimeConfig,
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
        let mut core = cogentlm_core::ContextRuntimeConfig::default();
        core.n_ctx = n_ctx;
        core.n_batch = n_batch;
        core.n_ubatch = n_ubatch;
        core.n_parallel = n_parallel;
        core.n_threads = n_threads;
        core.n_threads_batch = n_threads_batch;
        if let Some(value) = flash_attention {
            core.flash_attention = parse_flash_attention(&value)?;
        }
        core.kv_unified = kv_unified;
        if let Some(value) = cache_type_k {
            core.cache_type_k = parse_kv_cache_type(&value)?;
        }
        if let Some(value) = cache_type_v {
            core.cache_type_v = parse_kv_cache_type(&value)?;
        }
        if let Some(value) = offload_kqv {
            core.offload_kqv = value;
        }
        if let Some(value) = op_offload {
            core.op_offload = value;
        }
        if let Some(value) = swa_full {
            core.swa_full = value;
        }
        if let Some(value) = warmup {
            core.warmup = value;
        }
        if let Some(value) = rope_scaling {
            core.rope_scaling = Some(parse_rope_scaling(&value)?);
        }
        core.rope_freq_base = rope_freq_base;
        core.rope_freq_scale = rope_freq_scale;
        core.yarn_orig_ctx = yarn_orig_ctx;
        core.yarn_ext_factor = yarn_ext_factor;
        core.yarn_attn_factor = yarn_attn_factor;
        core.yarn_beta_fast = yarn_beta_fast;
        core.yarn_beta_slow = yarn_beta_slow;
        Ok(Self { core })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.core)
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
        decode_token_reserve = None,
        adaptive_prefill_chunking = None,
        prefill_chunk_size = None,
        max_running_requests = None,
        max_queued_requests = None
    ))]
    fn new(
        continuous_batching: Option<bool>,
        policy: Option<String>,
        decode_token_reserve: Option<i32>,
        adaptive_prefill_chunking: Option<bool>,
        prefill_chunk_size: Option<i32>,
        max_running_requests: Option<i32>,
        max_queued_requests: Option<i32>,
    ) -> PyResult<Self> {
        let mut core = SchedulerRuntimeConfig::default();
        if let Some(value) = continuous_batching {
            core.continuous_batching = value;
        }
        core.policy = SchedulerPolicyConfig {
            mode: if let Some(policy) = policy {
                parse_scheduler_policy(&policy)?
            } else {
                core.policy.mode
            },
            decode_token_reserve: decode_token_reserve.unwrap_or(core.policy.decode_token_reserve),
            enable_adaptive_prefill_chunking: adaptive_prefill_chunking
                .unwrap_or(core.policy.enable_adaptive_prefill_chunking),
        };
        if let Some(value) = prefill_chunk_size {
            core.prefill_chunk_size = value;
        }
        core.max_running_requests = max_running_requests;
        core.max_queued_requests = max_queued_requests;
        Ok(Self { core })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.core)
    }
}

#[pyclass(name = "CacheRuntimeConfig")]
#[derive(Debug, Clone)]
struct PyCacheRuntimeConfig {
    core: cogentlm_core::CacheRuntimeConfig,
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
        let mut core = cogentlm_core::CacheRuntimeConfig::default();
        if let Some(value) = mode {
            core.mode = parse_kv_reuse_mode(&value)?;
        }
        if let Some(value) = retained_prefix_tokens {
            core.retained_prefix_tokens = value;
        }
        if let Some(value) = snapshot_interval_tokens {
            core.snapshot_interval_tokens = value;
        }
        if let Some(value) = max_snapshot_entries {
            core.max_snapshot_entries = value;
        }
        if let Some(value) = max_snapshot_bytes {
            core.max_snapshot_bytes = value;
        }
        if let Some(value) = max_session_entries {
            core.max_session_entries = value;
        }
        if let Some(value) = cache_key_policy {
            core.cache_key_policy = parse_cache_key_policy(&value)?;
        }
        if let Some(value) = enable_context_checkpoints {
            core.enable_context_checkpoints = value;
        }
        if let Some(value) = checkpoint_every_tokens {
            core.checkpoint_every_tokens = value;
        }
        Ok(Self { core })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.core)
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

    fn __repr__(&self) -> String {
        format!("{:?}", self.core)
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
        if let Some(value) = max_gpu_models_per_device {
            core.max_gpu_models_per_device = value;
        }
        if let Some(value) = allow_cpu_models_while_gpu_loaded {
            core.allow_cpu_models_while_gpu_loaded = value;
        }
        if let Some(value) = require_gpu_lease {
            core.require_gpu_lease = value;
        }
        if let Some(value) = gpu_memory_safety_margin_bytes {
            core.gpu_memory_safety_margin_bytes = value;
        }
        Self { core }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.core)
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

    fn __repr__(&self) -> String {
        format!("{:?}", self.core)
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
        let mut core = NativeRuntimeConfig::default();
        if let Some(value) = placement {
            core.placement = value.borrow(py).core.clone();
        }
        if let Some(value) = context {
            core.context = value.borrow(py).core.clone();
        }
        if let Some(value) = sampling {
            core.sampling = value.borrow(py).to_core();
        }
        if let Some(value) = scheduler {
            core.scheduler = value.borrow(py).core.clone();
        }
        if let Some(value) = cache {
            core.cache = value.borrow(py).core.clone();
        }
        if let Some(value) = multimodal {
            core.multimodal = value.borrow(py).core.clone();
        }
        if let Some(value) = residency {
            core.residency = value.borrow(py).core.clone();
        }
        if let Some(value) = observability {
            core.observability = value.borrow(py).core;
        }
        Self { core }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.core)
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
    #[pyo3(signature = (*, backend = "auto".to_string(), stats = "basic".to_string(), runtime = None))]
    fn new(
        py: Python<'_>,
        backend: String,
        stats: String,
        runtime: Option<Py<PyNativeRuntimeConfig>>,
    ) -> PyResult<Self> {
        let runtime = runtime
            .as_ref()
            .map(|config| config.borrow(py).to_core())
            .unwrap_or_default();
        parse_backend_preference(&backend)?;
        parse_stats_mode(&stats)?;
        Ok(Self {
            backend,
            stats,
            runtime,
        })
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
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
        context_key = "default".to_string(),
        max_tokens = 64,
        grammar = "".to_string(),
        *,
        json_schema = "".to_string(),
        stop = None,
        sampling = None,
        media = None
    ))]
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

    fn __repr__(&self) -> String {
        format!("{self:?}")
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

    #[staticmethod]
    fn system(content: String) -> Self {
        Self {
            role: "system".to_string(),
            content,
        }
    }

    #[staticmethod]
    fn user(content: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }

    #[staticmethod]
    fn assistant(content: String) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
        }
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
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
        let config = config
            .as_ref()
            .map(|config| config.borrow(py).to_core())
            .unwrap_or_default();
        let engine = py
            .allow_threads(|| CogentEngine::load(model_path, config))
            .map_err(to_py_error)?;
        let events = engine.subscribe_events();
        Ok(Self {
            inner: Mutex::new(Some(engine)),
            events: Mutex::new(Some(events)),
        })
    }

    #[getter]
    fn closed(&self) -> PyResult<bool> {
        Ok(self.engine_guard()?.is_none())
    }

    #[pyo3(signature = (prompt, options = None, on_tokens = None))]
    fn query(
        &self,
        py: Python<'_>,
        prompt: String,
        options: Option<Py<PyQueryOptions>>,
        on_tokens: Option<PyObject>,
    ) -> PyResult<Py<PyAny>> {
        let options = query_options_or_default(py, options);
        let callback_error = Arc::new(Mutex::new(None));
        let request =
            query_request_with_tokens(py, prompt, options, on_tokens, callback_error.clone())?;
        let guard = self.engine_guard()?;
        let engine = engine_ref(&guard)?;
        let result = py_token_result_to_request_result(
            py.allow_threads(move || engine.query(request)),
            callback_error,
        )?;
        request_result_to_dict(py, result)
    }

    #[pyo3(signature = (prompt, options = None))]
    fn query_response(
        &self,
        py: Python<'_>,
        prompt: String,
        options: Option<Py<PyQueryOptions>>,
    ) -> PyResult<Py<PyAny>> {
        let options = query_options_or_default(py, options);
        let guard = self.engine_guard()?;
        let engine = engine_ref(&guard)?;
        let request = QueryRequest::new(prompt).options(options);
        let response = py
            .allow_threads(move || engine.query_response(request))
            .map_err(to_py_error)?;
        response_to_dict(py, response)
    }

    #[pyo3(signature = (prompt, options = None))]
    fn query_result(
        &self,
        py: Python<'_>,
        prompt: String,
        options: Option<Py<PyQueryOptions>>,
    ) -> PyResult<Py<PyAny>> {
        let options = query_options_or_default(py, options);
        let guard = self.engine_guard()?;
        let engine = engine_ref(&guard)?;
        let request = QueryRequest::new(prompt).options(options);
        let result = py
            .allow_threads(move || engine.query(request))
            .map_err(to_py_error)?;
        request_result_to_dict(py, result)
    }

    #[pyo3(signature = (messages, options = None, on_tokens = None))]
    fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<Py<PyChatMessage>>,
        options: Option<Py<PyQueryOptions>>,
        on_tokens: Option<PyObject>,
    ) -> PyResult<Py<PyAny>> {
        let options = query_options_or_default(py, options);
        let messages = chat_messages_to_core(py, messages)?;
        let callback_error = Arc::new(Mutex::new(None));
        let request =
            chat_request_with_tokens(py, messages, options, on_tokens, callback_error.clone())?;
        let guard = self.engine_guard()?;
        let engine = engine_ref(&guard)?;
        let result = py_token_result_to_request_result(
            py.allow_threads(move || engine.chat(request)),
            callback_error,
        )?;
        request_result_to_dict(py, result)
    }

    #[pyo3(signature = (messages, options = None))]
    fn chat_response(
        &self,
        py: Python<'_>,
        messages: Vec<Py<PyChatMessage>>,
        options: Option<Py<PyQueryOptions>>,
    ) -> PyResult<Py<PyAny>> {
        let options = query_options_or_default(py, options);
        let messages = chat_messages_to_core(py, messages)?;
        let guard = self.engine_guard()?;
        let engine = engine_ref(&guard)?;
        let request = ChatRequest::new(messages).options(options);
        let response = py
            .allow_threads(move || engine.chat_response(request))
            .map_err(to_py_error)?;
        response_to_dict(py, response)
    }

    #[pyo3(signature = (messages, options = None))]
    fn chat_result(
        &self,
        py: Python<'_>,
        messages: Vec<Py<PyChatMessage>>,
        options: Option<Py<PyQueryOptions>>,
    ) -> PyResult<Py<PyAny>> {
        let options = query_options_or_default(py, options);
        let messages = chat_messages_to_core(py, messages)?;
        let guard = self.engine_guard()?;
        let engine = engine_ref(&guard)?;
        let request = ChatRequest::new(messages).options(options);
        let result = py
            .allow_threads(move || engine.chat(request))
            .map_err(to_py_error)?;
        request_result_to_dict(py, result)
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let engine = self.engine_guard()?.take();
        if let Some(engine) = engine {
            py.allow_threads(|| engine.close()).map_err(to_py_error)?;
        }
        if let Ok(mut events) = self.events.lock() {
            events.take();
        }
        Ok(())
    }

    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = self.engine_guard()?;
        let engine = engine_ref(&guard)?;
        let state = py.allow_threads(|| engine.state()).map_err(to_py_error)?;
        engine_state_to_dict(py, state)
    }

    fn drain_events(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let events = PyList::empty_bound(py);
        let guard = self
            .events
            .lock()
            .map_err(|_| PyRuntimeError::new_err("engine events mutex is poisoned"))?;
        if let Some(receiver) = guard.as_ref() {
            for event in receiver.try_iter() {
                events.append(engine_event_to_dict(py, event)?)?;
            }
        }
        Ok(events.into_py(py))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("CogentEngine(closed={})", self.closed()?))
    }
}

impl PyCogentEngine {
    fn engine_guard(&self) -> PyResult<MutexGuard<'_, Option<CogentEngine>>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("engine mutex is poisoned"))
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

    #[getter]
    fn closed(&self) -> PyResult<bool> {
        Ok(self.service_guard()?.is_none())
    }

    #[pyo3(signature = (model_path, options = None))]
    fn load_path(
        &self,
        py: Python<'_>,
        model_path: PathBuf,
        options: Option<Py<PyModelLoadOptions>>,
    ) -> PyResult<Py<PyAny>> {
        let options = model_load_options_or_default(py, options)?;
        let mut guard = self.service_guard()?;
        let service = service_mut(&mut guard)?;
        let loaded = py
            .allow_threads(|| service.load(core_model_source_from_path(model_path), options))
            .map_err(to_py_model_error)?;
        self.refresh_events(service)?;
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
        let options = model_load_options_or_default(py, options)?;
        let mut guard = self.service_guard()?;
        let service = service_mut(&mut guard)?;
        let source = core_vision_model_source_from_paths(model_path, projector_path);
        let loaded = py
            .allow_threads(|| service.load(source, options))
            .map_err(to_py_model_error)?;
        self.refresh_events(service)?;
        loaded_model_info_to_dict(py, loaded)
    }

    #[pyo3(signature = (model_id, options = None))]
    fn load_installed(
        &self,
        py: Python<'_>,
        model_id: String,
        options: Option<Py<PyModelLoadOptions>>,
    ) -> PyResult<Py<PyAny>> {
        let options = model_load_options_or_default(py, options)?;
        let mut guard = self.service_guard()?;
        let service = service_mut(&mut guard)?;
        let loaded = py
            .allow_threads(|| service.load_installed(model_id, options))
            .map_err(to_py_model_error)?;
        self.refresh_events(service)?;
        loaded_model_info_to_dict(py, loaded)
    }

    fn unload(&self, py: Python<'_>) -> PyResult<()> {
        let mut guard = self.service_guard()?;
        let service = service_mut(&mut guard)?;
        py.allow_threads(|| service.unload())
            .map_err(to_py_model_error)?;
        if let Ok(mut events) = self.events.lock() {
            events.take();
        }
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
        let models = PyList::empty_bound(py);
        for model in service.list() {
            models.append(model_info_to_dict(py, model)?)?;
        }
        Ok(models.into_py(py))
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
        let options = query_options_or_default(py, options);
        let callback_error = Arc::new(Mutex::new(None));
        let request =
            query_request_with_tokens(py, prompt, options, on_tokens, callback_error.clone())?;
        let guard = self.service_guard()?;
        let service = service_ref(&guard)?;
        let result = py_model_token_result_to_request_result(
            py.allow_threads(|| service.query(request)),
            callback_error,
        )?;
        request_result_to_dict(py, result)
    }

    #[pyo3(signature = (messages, options = None, on_tokens = None))]
    fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<Py<PyChatMessage>>,
        options: Option<Py<PyQueryOptions>>,
        on_tokens: Option<PyObject>,
    ) -> PyResult<Py<PyAny>> {
        let options = query_options_or_default(py, options);
        let messages = chat_messages_to_core(py, messages)?;
        let callback_error = Arc::new(Mutex::new(None));
        let request =
            chat_request_with_tokens(py, messages, options, on_tokens, callback_error.clone())?;
        let guard = self.service_guard()?;
        let service = service_ref(&guard)?;
        let result = py_model_token_result_to_request_result(
            py.allow_threads(|| service.chat(request)),
            callback_error,
        )?;
        request_result_to_dict(py, result)
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
        let events = PyList::empty_bound(py);
        let guard = self
            .events
            .lock()
            .map_err(|_| PyRuntimeError::new_err("model service events mutex is poisoned"))?;
        if let Some(receiver) = guard.as_ref() {
            for event in receiver.try_iter() {
                events.append(engine_event_to_dict(py, event)?)?;
            }
        }
        Ok(events.into_py(py))
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let service = self.service_guard()?.take();
        if let Some(mut service) = service {
            py.allow_threads(|| service.close())
                .map_err(to_py_model_error)?;
        }
        if let Ok(mut events) = self.events.lock() {
            events.take();
        }
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("ModelService(closed={})", self.closed()?))
    }
}

impl PyModelService {
    fn service_guard(&self) -> PyResult<MutexGuard<'_, Option<ModelService>>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("model service mutex is poisoned"))
    }

    fn refresh_events(&self, service: &ModelService) -> PyResult<()> {
        let events = service.subscribe_events().map_err(to_py_model_error)?;
        self.events
            .lock()
            .map_err(|_| PyRuntimeError::new_err("model service events mutex is poisoned"))?
            .replace(events);
        Ok(())
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
    module.add_class::<PyModelPlacementConfig>()?;
    module.add_class::<PyContextRuntimeConfig>()?;
    module.add_class::<PySamplingRuntimeConfig>()?;
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
    module.add_function(wrap_pyfunction!(backend_observability_json, module)?)?;
    module.add_function(wrap_pyfunction!(set_llama_log_quiet, module)?)?;
    Ok(())
}

fn model_load_options_or_default(
    py: Python<'_>,
    options: Option<Py<PyModelLoadOptions>>,
) -> PyResult<ModelLoadOptions> {
    options
        .as_ref()
        .map(|options| options.borrow(py).to_core())
        .transpose()
        .map(|options| options.unwrap_or_default())
}

fn query_options_or_default(py: Python<'_>, options: Option<Py<PyQueryOptions>>) -> QueryOptions {
    options
        .as_ref()
        .map(|options| options.borrow(py).to_core())
        .unwrap_or_default()
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
        .ok_or_else(|| PyRuntimeError::new_err("engine is closed"))
}

fn service_ref<'a>(guard: &'a MutexGuard<'_, Option<ModelService>>) -> PyResult<&'a ModelService> {
    guard
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("model service is closed"))
}

fn service_mut<'a>(
    guard: &'a mut MutexGuard<'_, Option<ModelService>>,
) -> PyResult<&'a mut ModelService> {
    guard
        .as_mut()
        .ok_or_else(|| PyRuntimeError::new_err("model service is closed"))
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
) -> impl FnMut(&TokenBatch) -> cogentlm_core::Result<()> + Send + 'static {
    move |batch| {
        if has_callback_error(&callback_error) {
            return Err(cogentlm_core::Error::RuntimeCommand(
                PY_CALLBACK_FAILED_MESSAGE.to_string(),
            ));
        }

        Python::with_gil(|py| {
            let batch = token_batch_to_dict(py, batch.clone()).map_err(|error| {
                store_callback_error(&callback_error, error);
                cogentlm_core::Error::RuntimeCommand(PY_CALLBACK_FAILED_MESSAGE.to_string())
            })?;
            callback.call1(py, (batch,)).map(|_| ()).map_err(|error| {
                store_callback_error(&callback_error, error);
                cogentlm_core::Error::RuntimeCommand(PY_CALLBACK_FAILED_MESSAGE.to_string())
            })
        })
    }
}

fn py_token_result_to_request_result(
    result: cogentlm_core::Result<RequestResult>,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyResult<RequestResult> {
    match result {
        Ok(result) => Ok(result),
        Err(error) => Err(callback_error_or_core_error(error, callback_error)),
    }
}

fn py_model_token_result_to_request_result(
    result: Result<RequestResult, cogentlm_core::ModelError>,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyResult<RequestResult> {
    match result {
        Ok(result) => Ok(result),
        Err(error) => Err(callback_error_or_model_error(error, callback_error)),
    }
}

fn callback_error_or_model_error(
    error: cogentlm_core::ModelError,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyErr {
    if is_python_model_callback_error(&error) {
        if let Some(error) = take_callback_error(&callback_error) {
            return error;
        }
    }
    to_py_model_error(error)
}

fn is_python_model_callback_error(error: &cogentlm_core::ModelError) -> bool {
    matches!(
        error,
        cogentlm_core::ModelError::Runtime(message) if message.contains(PY_CALLBACK_FAILED_MESSAGE)
    )
}

fn callback_error_or_core_error(
    error: cogentlm_core::Error,
    callback_error: Arc<Mutex<Option<PyErr>>>,
) -> PyErr {
    if is_python_callback_error(&error) {
        if let Some(error) = take_callback_error(&callback_error) {
            return error;
        }
    }
    to_py_error(error)
}

fn is_python_callback_error(error: &cogentlm_core::Error) -> bool {
    matches!(
        error,
        cogentlm_core::Error::RuntimeCommand(message) if message == PY_CALLBACK_FAILED_MESSAGE
    )
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

fn response_to_dict(py: Python<'_>, response: GenerateResponse) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("request_id", response.request_id)?;
    dict.set_item("status", response_status_name(response.status))?;
    dict.set_item("output_text", response.output_text)?;
    dict.set_item("error_message", response.error_message)?;
    dict.set_item(
        "runtime_observability",
        metrics_to_dict(py, response.runtime_observability)?,
    )?;
    Ok(dict.into_py(py))
}

fn request_result_to_dict(py: Python<'_>, result: RequestResult) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", result.id)?;
    dict.set_item("text", result.text)?;
    dict.set_item("finish_reason", finish_reason_name(result.finish_reason))?;
    dict.set_item("stats", request_stats_to_dict(py, result.stats)?)?;
    Ok(dict.into_py(py))
}

fn loaded_model_info_to_dict(py: Python<'_>, loaded: LoadedModelInfo) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("model", model_info_to_dict(py, loaded.model)?)?;
    dict.set_item("backend", backend_selection_to_dict(py, loaded.backend)?)?;
    dict.set_item("runtime_fingerprint", loaded.runtime_fingerprint)?;
    Ok(dict.into_py(py))
}

fn model_info_to_dict(py: Python<'_>, model: ModelInfo) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", model.id)?;
    dict.set_item("name", model.name)?;
    dict.set_item("modality", model_modality_name(model.modality))?;
    dict.set_item("status", model_status_name(model.status))?;
    dict.set_item("source", model_source_kind_name(model.source))?;
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

fn backend_selection_to_dict(py: Python<'_>, backend: BackendSelection) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("requested", backend_preference_name(backend.requested))?;
    dict.set_item("selected", backend.selected)?;
    dict.set_item("available", backend.available)?;
    dict.set_item("gpu_offload_expected", backend.gpu_offload_expected)?;
    dict.set_item("reason", backend.reason)?;
    Ok(dict.into_py(py))
}

fn model_service_state_to_dict(py: Python<'_>, state: ModelServiceState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("status", engine_status_name(state.status))?;
    if let Some(model) = state.model {
        dict.set_item("model", model_info_to_dict(py, model)?)?;
    } else {
        dict.set_item("model", py.None())?;
    }
    dict.set_item("backend", backend_info_to_dict(py, state.backend)?)?;
    dict.set_item(
        "runtime",
        resolved_runtime_limits_to_dict(py, state.runtime)?,
    )?;
    let requests = PyList::empty_bound(py);
    for request in state.requests {
        requests.append(request_state_to_dict(py, request)?)?;
    }
    dict.set_item("requests", requests)?;
    dict.set_item("stats", engine_stats_to_dict(py, state.stats)?)?;
    dict.set_item("updated_at_unix_ms", state.updated_at_unix_ms)?;
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
    dict.set_item("status", engine_status_name(state.status))?;
    if let Some(model) = state.model {
        dict.set_item("model", model_state_to_dict(py, model)?)?;
    } else {
        dict.set_item("model", py.None())?;
    }
    dict.set_item("backend", backend_info_to_dict(py, state.backend)?)?;
    dict.set_item(
        "runtime",
        resolved_runtime_limits_to_dict(py, state.runtime)?,
    )?;
    let requests = PyList::empty_bound(py);
    for request in state.requests {
        requests.append(request_state_to_dict(py, request)?)?;
    }
    dict.set_item("requests", requests)?;
    dict.set_item("stats", engine_stats_to_dict(py, state.stats)?)?;
    dict.set_item("updated_at_unix_ms", state.updated_at_unix_ms)?;
    Ok(dict.into_py(py))
}

fn model_state_to_dict(py: Python<'_>, model: ModelState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", model.id)?;
    dict.set_item("name", model.name)?;
    Ok(dict.into_py(py))
}

fn request_state_to_dict(py: Python<'_>, request: RequestState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", request.id)?;
    dict.set_item("status", request_status_name(request.status))?;
    dict.set_item("input_tokens", request.input_tokens)?;
    dict.set_item("output_tokens", request.output_tokens)?;
    Ok(dict.into_py(py))
}

fn backend_info_to_dict(py: Python<'_>, backend: BackendInfo) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("selected", backend.selected)?;
    dict.set_item("available", backend.available)?;
    let devices = PyList::empty_bound(py);
    for device in backend.devices {
        devices.append(backend_device_to_dict(py, device)?)?;
    }
    dict.set_item("devices", devices)?;
    Ok(dict.into_py(py))
}

fn backend_device_to_dict(py: Python<'_>, device: BackendDevice) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", device.id)?;
    dict.set_item("name", device.name)?;
    dict.set_item("type", device.device_type)?;
    dict.set_item("memory_total_bytes", device.memory_total_bytes)?;
    dict.set_item("memory_free_bytes", device.memory_free_bytes)?;
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
            dict.set_item("type", "state")?;
            dict.set_item("state", engine_state_to_dict(py, state)?)?;
        }
        EngineEvent::LoadProgress {
            loaded_bytes,
            total_bytes,
            asset_name,
        } => {
            dict.set_item("type", "load-progress")?;
            dict.set_item("loaded_bytes", loaded_bytes)?;
            dict.set_item("total_bytes", total_bytes)?;
            dict.set_item("asset_name", asset_name)?;
        }
        EngineEvent::RequestStarted {
            request_id,
            stream_id,
        } => {
            dict.set_item("type", "request-started")?;
            dict.set_item("request_id", request_id)?;
            dict.set_item("stream_id", stream_id)?;
        }
        EngineEvent::RequestCompleted { result } => {
            dict.set_item("type", "request-completed")?;
            dict.set_item("result", request_result_to_dict(py, result)?)?;
        }
        EngineEvent::RequestFailed { request_id, error } => {
            dict.set_item("type", "request-failed")?;
            dict.set_item("request_id", request_id)?;
            dict.set_item("error", error)?;
        }
        EngineEvent::Closed => {
            dict.set_item("type", "closed")?;
        }
    }
    Ok(dict.into_py(py))
}

fn metrics_to_dict(
    py: Python<'_>,
    metrics: RuntimeObservabilityMetrics,
) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("ttft_ms", metrics.ttft_ms)?;
    dict.set_item("itl_avg_ms", metrics.itl_avg_ms)?;
    dict.set_item("itl_p99_ms", metrics.itl_p99_ms)?;
    dict.set_item("e2e_ms", metrics.e2e_ms)?;
    dict.set_item("prefill_ms", metrics.prefill_ms)?;
    dict.set_item("decode_ms", metrics.decode_ms)?;
    dict.set_item("native_gpu_ms", metrics.native_gpu_ms)?;
    dict.set_item("native_sync_ms", metrics.native_sync_ms)?;
    dict.set_item("native_logic_ms", metrics.native_logic_ms)?;
    dict.set_item("input_tokens", metrics.input_tokens)?;
    dict.set_item("output_tokens", metrics.output_tokens)?;
    dict.set_item("cache_hits", metrics.cache_hits)?;
    dict.set_item("prefill_tokens", metrics.prefill_tokens)?;
    Ok(dict)
}

fn parse_backend_preference(value: &str) -> PyResult<BackendPreference> {
    match normalize_choice(value).as_str() {
        "auto" => Ok(BackendPreference::Auto),
        "cpu" => Ok(BackendPreference::Cpu),
        "cuda" => Ok(BackendPreference::Cuda),
        "metal" => Ok(BackendPreference::Metal),
        "vulkan" => Ok(BackendPreference::Vulkan),
        "webgpu" | "web_gpu" => Ok(BackendPreference::WebGpu),
        _ => Err(PyValueError::new_err(
            "backend must be one of: auto, cpu, cuda, metal, vulkan, webgpu",
        )),
    }
}

fn parse_stats_mode(value: &str) -> PyResult<StatsMode> {
    match normalize_choice(value).as_str() {
        "off" => Ok(StatsMode::Off),
        "basic" => Ok(StatsMode::Basic),
        "profile" => Ok(StatsMode::Profile),
        _ => Err(PyValueError::new_err(
            "stats must be one of: off, basic, profile",
        )),
    }
}

fn parse_gpu_layers(value: &str) -> PyResult<GpuLayerConfig> {
    let normalized = normalize_choice(value);
    match normalized.as_str() {
        "auto" => Ok(GpuLayerConfig::Auto),
        "all" | "full" => Ok(GpuLayerConfig::All),
        _ => normalized
            .parse::<i32>()
            .map(GpuLayerConfig::Count)
            .map_err(|_| {
                PyValueError::new_err("gpu_layers must be one of: auto, all, or an integer count")
            }),
    }
}

fn parse_split_mode(value: &str) -> PyResult<SplitMode> {
    match normalize_choice(value).as_str() {
        "none" => Ok(SplitMode::None),
        "layer" => Ok(SplitMode::Layer),
        "row" => Ok(SplitMode::Row),
        "tensor" => Ok(SplitMode::Tensor),
        _ => Err(PyValueError::new_err(
            "split_mode must be one of: none, layer, row, tensor",
        )),
    }
}

fn parse_flash_attention(value: &str) -> PyResult<FlashAttentionMode> {
    match normalize_choice(value).as_str() {
        "auto" => Ok(FlashAttentionMode::Auto),
        "enabled" | "enable" | "on" | "true" => Ok(FlashAttentionMode::Enabled),
        "disabled" | "disable" | "off" | "false" => Ok(FlashAttentionMode::Disabled),
        _ => Err(PyValueError::new_err(
            "flash_attention must be one of: auto, enabled, disabled",
        )),
    }
}

fn parse_kv_cache_type(value: &str) -> PyResult<KvCacheType> {
    match normalize_choice(value).as_str() {
        "f16" => Ok(KvCacheType::F16),
        "f32" => Ok(KvCacheType::F32),
        "q8_0" => Ok(KvCacheType::Q8_0),
        "q4_0" => Ok(KvCacheType::Q4_0),
        "q4_1" => Ok(KvCacheType::Q4_1),
        "iq4_nl" => Ok(KvCacheType::Iq4Nl),
        "q5_0" => Ok(KvCacheType::Q5_0),
        "q5_1" => Ok(KvCacheType::Q5_1),
        _ => Err(PyValueError::new_err(
            "cache type must be one of: f16, f32, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1",
        )),
    }
}

fn parse_rope_scaling(value: &str) -> PyResult<RopeScaling> {
    match normalize_choice(value).as_str() {
        "none" => Ok(RopeScaling::None),
        "linear" => Ok(RopeScaling::Linear),
        "yarn" => Ok(RopeScaling::Yarn),
        _ => Err(PyValueError::new_err(
            "rope_scaling must be one of: none, linear, yarn",
        )),
    }
}

fn parse_kv_reuse_mode(value: &str) -> PyResult<KvReuseMode> {
    match normalize_choice(value).as_str() {
        "disabled" | "none" => Ok(KvReuseMode::Disabled),
        "live_slot_prefix" | "live_slot" => Ok(KvReuseMode::LiveSlotPrefix),
        "state_snapshot" | "snapshot" => Ok(KvReuseMode::StateSnapshot),
        "live_slot_and_snapshot" | "both" => Ok(KvReuseMode::LiveSlotAndSnapshot),
        _ => Err(PyValueError::new_err(
            "cache mode must be one of: disabled, live_slot_prefix, state_snapshot, live_slot_and_snapshot",
        )),
    }
}

fn parse_cache_key_policy(value: &str) -> PyResult<CacheKeyPolicy> {
    match normalize_choice(value).as_str() {
        "context_key" => Ok(CacheKeyPolicy::ContextKey),
        "prompt_hash" => Ok(CacheKeyPolicy::PromptHash),
        _ => Err(PyValueError::new_err(
            "cache_key_policy must be one of: context_key, prompt_hash",
        )),
    }
}

fn parse_sampler_stage(value: &str) -> PyResult<SamplerStage> {
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
        _ => Err(PyValueError::new_err(
            "sampler stage must be one of: dry, top_k, typical_p, top_p, top_n_sigma, min_p, xtc, temperature, infill, penalties, adaptive_p",
        )),
    }
}

fn parse_scheduler_policy(value: &str) -> PyResult<SchedulerPolicyMode> {
    match normalize_choice(value).as_str() {
        "latency_first" | "latency" => Ok(SchedulerPolicyMode::LatencyFirst),
        "balanced" | "balance" => Ok(SchedulerPolicyMode::Balanced),
        "throughput_first" | "throughput" => Ok(SchedulerPolicyMode::ThroughputFirst),
        _ => Err(PyValueError::new_err(
            "scheduler_policy must be one of: latency_first, balanced, throughput_first",
        )),
    }
}

fn parse_chat_role(value: &str) -> PyResult<ChatRole> {
    match normalize_choice(value).as_str() {
        "system" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        _ => Err(PyValueError::new_err(
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

fn engine_status_name(status: EngineStatus) -> &'static str {
    match status {
        EngineStatus::Idle => "idle",
        EngineStatus::Loading => "loading",
        EngineStatus::Ready => "ready",
        EngineStatus::Running => "running",
        EngineStatus::Error => "error",
        EngineStatus::Closed => "closed",
    }
}

fn request_status_name(status: RequestStatus) -> &'static str {
    match status {
        RequestStatus::Queued => "queued",
        RequestStatus::Prefill => "prefill",
        RequestStatus::Decode => "decode",
        RequestStatus::Completed => "completed",
        RequestStatus::Failed => "failed",
        RequestStatus::Cancelled => "cancelled",
    }
}

fn finish_reason_name(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Cancelled => "cancelled",
        FinishReason::Error => "error",
    }
}

fn backend_preference_name(backend: BackendPreference) -> &'static str {
    match backend {
        BackendPreference::Auto => "auto",
        BackendPreference::Cpu => "cpu",
        BackendPreference::Cuda => "cuda",
        BackendPreference::Metal => "metal",
        BackendPreference::Vulkan => "vulkan",
        BackendPreference::WebGpu => "webgpu",
    }
}

fn model_modality_name(modality: ModelModality) -> &'static str {
    match modality {
        ModelModality::Text => "text",
        ModelModality::Vision => "vision",
    }
}

fn model_status_name(status: ModelStatus) -> &'static str {
    match status {
        ModelStatus::Ready => "ready",
        ModelStatus::NeedsProjector => "needs_projector",
        ModelStatus::Broken => "broken",
    }
}

fn model_source_kind_name(source: ModelSourceKind) -> &'static str {
    match source {
        ModelSourceKind::Local => "local",
        ModelSourceKind::Remote => "remote",
    }
}

fn to_py_error(error: cogentlm_core::Error) -> PyErr {
    match error {
        cogentlm_core::Error::InvalidRequest(message)
        | cogentlm_core::Error::InvalidConfig(message) => PyValueError::new_err(message),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn to_py_model_error(error: cogentlm_core::ModelError) -> PyErr {
    match error {
        cogentlm_core::ModelError::InvalidModelSource(message)
        | cogentlm_core::ModelError::InvalidModelPairing(message) => PyValueError::new_err(message),
        cogentlm_core::ModelError::UnsupportedGgufVersion(version) => {
            PyValueError::new_err(format!("unsupported GGUF version {version}"))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}
