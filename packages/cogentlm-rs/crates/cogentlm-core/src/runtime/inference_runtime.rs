use std::cmp;
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use cogentlm_sys as ffi;

use crate::backend::{backend_observability_json, ensure_backend_initialized};
use crate::error::{Error, Result};
use crate::runtime::config::{
    KvReuseMode, NativeRuntimeConfig, ResolvedRuntimeLimits, SamplingRuntimeConfig,
    SchedulerTickBudget,
};
use crate::runtime::llama::LlamaBatchBuilder;
use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::request::{
    GenerateRequest, GenerateRequestId, GenerateRequestLifecycle, GenerateResponse,
    GenerateTokenEmissionMode, RequestQueue, TokenByteRingProducer,
};
use crate::runtime::residency::{acquire_residency_lease, ResidencyLease};
use crate::runtime::scheduler::{
    BatchContributionKind, BatchPlanner, SamplerCacheKey, SharedBatchPlan, SlotPhase,
    SlotScheduler, SlotState,
};
use crate::runtime::session::{
    PendingPrefixSnapshot, PrefixCachePolicy, PrefixStateCache, SequenceState, SessionStore,
};
use crate::runtime::{llama_seq_id, llama_token};
use crate::token::{token_to_piece, tokenize};

const DEFAULT_PROMPT_CONTEXT_KEY: &str = "__primary_prompt__";
const PREFIX_SNAPSHOT_COMMIT_BUDGET: usize = 2;

// In practice `n_parallel` for browser inference sits in 1..=8 and never
// exceeds 32, so a single u64 covers the "unique slot indices in this plan"
// sets we used to allocate fresh HashSets for every tick. Falling back to a
// no-op (treat as "already seen") for slots ≥64 preserves correctness — the
// scheduler simply won't double-credit a request, which is fine.
#[inline(always)]
fn unique_slot_first_use(seen: &mut u64, slot_index: usize) -> bool {
    if slot_index >= 64 {
        return false;
    }
    let bit = 1u64 << slot_index;
    let already = (*seen & bit) != 0;
    *seen |= bit;
    !already
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestStepResult {
    Invalid = -1,
    FatalNoProgress = -2,
    #[default]
    Waiting = 0,
    Progressed = 1,
    Terminal = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SchedulerBurstResult {
    pub status: RequestStepResult,
    pub ticks_executed: i32,
    pub progressed_ticks: i32,
    pub completed_response_count: i32,
    pub emitted_token_count: i32,
}

#[derive(Debug)]
struct PendingLogitsContribution {
    slot_index: usize,
    batch_token_index: i32,
    sampled_token: llama_token,
}

#[allow(dead_code)]
struct PendingTickBookkeeping {
    contributions: Vec<PendingLogitsContribution>,
    timing: (f64, f64, f64),
}

#[derive(Debug, Clone, Copy, Default)]
struct DebugMetricsTick {
    normalize_ms: f64,
    select_slots_ms: f64,
    plan_ms: f64,
    batch_build_ms: f64,
    llama_decode_ms: f64,
    llama_sync_ms: f64,
    apply_bookkeeping_ms: f64,
    apply_decode_results_ms: f64,
    sample_ms: f64,
    token_piece_ms: f64,
    emit_ms: f64,
    prefix_queue_ms: f64,
    post_decode_ms: f64,
}

pub struct InferenceRuntime {
    config: NativeRuntimeConfig,
    resolved_limits: ResolvedRuntimeLimits,
    residency_lease: Option<ResidencyLease>,
    common_init: *mut ffi::cogent_common_init,
    primary_model: *mut ffi::llama_model,
    shared_context: *mut ffi::llama_context,
    chat_templates: *mut ffi::cogent_chat_templates,
    mtmd_context: *mut ffi::cogent_mtmd_context,
    last_runtime_observability: RuntimeObservabilityMetrics,
    has_last_runtime_observability: bool,
    session_store: SessionStore,
    request_queue: RequestQueue,
    slot_scheduler: SlotScheduler,
    batch_planner: BatchPlanner,
    shared_batch_builder: LlamaBatchBuilder,
    prefix_state_cache: PrefixStateCache,
    prefix_cache_policy: PrefixCachePolicy,
    next_request_id: GenerateRequestId,
    model_fingerprint: u64,
    committed_observability_request_ids: HashSet<GenerateRequestId>,
    scratch_decode_ready_slots: Vec<usize>,
    scratch_prefill_ready_slots: Vec<usize>,
    scratch_logits_contributions: Vec<PendingLogitsContribution>,
    /// Reused across every tick to avoid allocating a fresh ~16 KiB Vec for
    /// the batch contributions each scheduler iteration.
    scratch_plan: SharedBatchPlan,
    /// Reused by `token_to_piece` to avoid a 128-byte Vec allocation per
    /// emitted token. Sized once and cleared per call.
    scratch_token_piece: Vec<i8>,
    /// Cached result of `config.observability.effective_runtime_metrics()`
    /// so the tick loop can skip ~10 wasm→JS Instant::now() round-trips per
    /// tick when nobody asked for debug metrics.
    debug_metrics_enabled: bool,
    total_decode_ms: f64,
    total_prefill_ms: f64,
    total_input_tokens: usize,
    total_output_tokens: usize,
    total_cache_hits: usize,
    total_prefill_tokens: usize,
    sampler_pool:
        std::collections::HashMap<SamplerCacheKey, Vec<NonNull<ffi::cogent_common_sampler>>>,
    /// Pending bookkeeping deferred to next-tick start (C++ pattern).
    /// Metrics + prefix-cache work are applied at the START of the next tick
    /// so the decode→sync→sample path is as short as possible.
    #[allow(dead_code)]
    pending_bookkeeping: Option<PendingTickBookkeeping>,
}

// InferenceRuntime owns llama.cpp pointers and is intended to be moved into one
// engine thread, not shared concurrently. Public mutation remains `&mut self`.
unsafe impl Send for InferenceRuntime {}

impl InferenceRuntime {
    pub fn load(
        model_path: impl AsRef<std::path::Path>,
        config: NativeRuntimeConfig,
    ) -> Result<Self> {
        ensure_backend_initialized();

        let model_path = model_path.as_ref();
        let model_path_string = model_path.display().to_string();
        if model_path_string.is_empty() {
            return Err(Error::InvalidRequest("model path is required"));
        }

        let config = config.normalize();
        let c_model_path = CString::new(model_path_string.clone())?;
        let common_args = config.llama_common_args();
        let common_arg_cstrings = c_strings_from_args(&common_args)?;
        let common_arg_ptrs = c_ptrs_from_strings(&common_arg_cstrings);
        let mut parse_error = std::ptr::null_mut();
        let common_params = unsafe {
            ffi::cogent_common_params_parse_server(
                c_model_path.as_ptr(),
                common_arg_ptrs.len() as i32,
                common_arg_ptrs.as_ptr(),
                &mut parse_error,
            )
        };
        if common_params.is_null() {
            return Err(runtime_command_from_shim_error(
                parse_error,
                "failed to parse llama.cpp common params",
            ));
        }

        let residency_lease = match admit_runtime_residency(&config) {
            Ok(lease) => lease,
            Err(error) => {
                unsafe {
                    ffi::cogent_common_params_free(common_params);
                }
                return Err(error);
            }
        };

        let mut init_error = std::ptr::null_mut();
        let common_init =
            unsafe { ffi::cogent_common_init_from_params(common_params, &mut init_error) };
        unsafe {
            ffi::cogent_common_params_free(common_params);
        }
        if common_init.is_null() {
            drop(residency_lease);
            return Err(runtime_command_from_shim_error(
                init_error,
                "failed to initialize llama.cpp common runtime",
            ));
        }

        let resolved_limits = resolved_runtime_limits(common_init);

        let primary_model = unsafe { ffi::cogent_common_init_model(common_init) };
        let shared_context = unsafe { ffi::cogent_common_init_context(common_init) };
        if primary_model.is_null() || shared_context.is_null() {
            unsafe {
                ffi::cogent_common_init_free(common_init);
            }
            drop(residency_lease);
            return Err(Error::ModelLoad {
                path: model_path_string,
            });
        }

        let vocab = unsafe { ffi::cogent_common_init_vocab(common_init) };
        if vocab.is_null() {
            unsafe {
                ffi::cogent_common_init_free(common_init);
            }
            drop(residency_lease);
            return Err(Error::NullPointer("llama_model_get_vocab"));
        }

        let chat_templates =
            unsafe { ffi::cogent_chat_templates_init(primary_model, std::ptr::null()) };
        let mtmd_context = if config.multimodal.projector_path.is_none() {
            std::ptr::null_mut()
        } else {
            let c_mmproj_path =
                CString::new(config.multimodal.projector_path.clone().unwrap_or_default())?;
            let use_gpu = config.multimodal.use_gpu.unwrap_or(true);
            let mtmd = unsafe {
                ffi::cogent_mtmd_init_from_file(
                    c_mmproj_path.as_ptr(),
                    primary_model,
                    use_gpu,
                    config.context.n_threads.unwrap_or(0),
                )
            };
            if mtmd.is_null() || !unsafe { ffi::cogent_mtmd_support_vision(mtmd) } {
                if !mtmd.is_null() {
                    unsafe {
                        ffi::cogent_mtmd_free(mtmd);
                    }
                }
                unsafe {
                    if !chat_templates.is_null() {
                        ffi::cogent_chat_templates_free(chat_templates);
                    }
                    ffi::cogent_common_init_free(common_init);
                }
                drop(residency_lease);
                return Err(Error::NullPointer("cogent_mtmd_init_from_file"));
            }
            mtmd
        };

        let max_cached_sessions = config.cache.max_session_entries.max(1) as usize;
        let resolved_parallel = resolved_limits.n_parallel.max(1);
        let max_sequences = resolved_parallel as usize;
        let mut session_store = SessionStore::new(max_cached_sessions, max_sequences);
        session_store.bind_shared_context(shared_context);

        let mut slot_scheduler = SlotScheduler::default();
        slot_scheduler.resize(max_sequences);

        let mut shared_batch_builder = LlamaBatchBuilder::default();
        shared_batch_builder.ensure_capacity(
            resolve_batch_token_budget(shared_context, &config),
            resolved_parallel,
        )?;

        let max_prefix_cache_entries = config.cache.max_snapshot_entries.max(1) as usize;
        let prefix_cache_interval_tokens = if snapshot_prefix_cache_enabled(config.cache.mode) {
            config.cache.snapshot_interval_tokens.max(0) as usize
        } else {
            0
        };
        let max_prefix_cache_bytes = config.cache.max_snapshot_bytes;
        let debug_metrics_enabled = config.observability.effective_runtime_metrics();

        Ok(Self {
            config,
            resolved_limits,
            residency_lease,
            common_init,
            primary_model,
            shared_context,
            chat_templates,
            mtmd_context,
            last_runtime_observability: RuntimeObservabilityMetrics::default(),
            has_last_runtime_observability: false,
            session_store,
            request_queue: RequestQueue::new(),
            slot_scheduler,
            batch_planner: BatchPlanner,
            shared_batch_builder,
            prefix_state_cache: PrefixStateCache::new(
                max_prefix_cache_entries,
                max_prefix_cache_bytes,
            ),
            prefix_cache_policy: PrefixCachePolicy::new(prefix_cache_interval_tokens),
            next_request_id: 1,
            model_fingerprint: fingerprint_path(model_path),
            committed_observability_request_ids: HashSet::new(),
            scratch_decode_ready_slots: Vec::new(),
            scratch_prefill_ready_slots: Vec::new(),
            scratch_logits_contributions: Vec::new(),
            scratch_plan: SharedBatchPlan::default(),
            scratch_token_piece: Vec::with_capacity(128),
            debug_metrics_enabled,
            total_decode_ms: 0.0,
            total_prefill_ms: 0.0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_hits: 0,
            total_prefill_tokens: 0,
            sampler_pool: std::collections::HashMap::new(),
            pending_bookkeeping: None,
        })
    }

    pub fn is_ready(&self) -> bool {
        !self.common_init.is_null()
            && !self.primary_model.is_null()
            && !self.shared_context.is_null()
            && (self.config.multimodal.projector_path.is_none() || !self.mtmd_context.is_null())
    }

    pub fn resolved_runtime_limits(&self) -> ResolvedRuntimeLimits {
        self.resolved_limits.clone()
    }

    pub fn runtime_observability_enabled(&self) -> bool {
        self.config.observability.effective_runtime_metrics()
    }

    pub fn backend_profiling_enabled(&self) -> bool {
        self.config.observability.backend_profiling
    }

    pub fn try_get_runtime_observability(&self) -> Option<RuntimeObservabilityMetrics> {
        if !self.config.observability.effective_runtime_metrics() {
            return None;
        }

        for slot in self.slot_scheduler.slots() {
            let Some(request) = slot.request() else {
                continue;
            };

            let mut metrics = RuntimeObservabilityMetrics {
                input_tokens: request.input_tokens,
                output_tokens: request.output_tokens,
                cache_hits: request.cache_hits,
                prefill_tokens: request.prefill_tokens,
                prefill_ms: request.prefill_ms,
                decode_ms: request.decode_ms,
                native_gpu_ms: request.native_gpu_ms,
                native_sync_ms: request.native_sync_ms,
                native_logic_ms: request.native_logic_ms,
                debug_metrics_scheduler_ticks: request.debug_metrics_scheduler_ticks,
                debug_metrics_decode_ticks: request.debug_metrics_decode_ticks,
                debug_metrics_prefill_ticks: request.debug_metrics_prefill_ticks,
                debug_metrics_backend_sampler_attach_attempts: request
                    .debug_metrics_backend_sampler_attach_attempts,
                debug_metrics_backend_sampler_attach_failures: request
                    .debug_metrics_backend_sampler_attach_failures,
                debug_metrics_admit_ms: request.debug_metrics_admit_ms,
                debug_metrics_normalize_ms: request.debug_metrics_normalize_ms,
                debug_metrics_backend_sampler_attach_ms: request
                    .debug_metrics_backend_sampler_attach_ms,
                debug_metrics_select_slots_ms: request.debug_metrics_select_slots_ms,
                debug_metrics_plan_ms: request.debug_metrics_plan_ms,
                debug_metrics_batch_build_ms: request.debug_metrics_batch_build_ms,
                debug_metrics_llama_decode_ms: request.debug_metrics_llama_decode_ms,
                debug_metrics_llama_sync_ms: request.debug_metrics_llama_sync_ms,
                debug_metrics_apply_bookkeeping_ms: request.debug_metrics_apply_bookkeeping_ms,
                debug_metrics_apply_decode_results_ms: request
                    .debug_metrics_apply_decode_results_ms,
                debug_metrics_sample_ms: request.debug_metrics_sample_ms,
                debug_metrics_token_piece_ms: request.debug_metrics_token_piece_ms,
                debug_metrics_emit_ms: request.debug_metrics_emit_ms,
                debug_metrics_prefix_queue_ms: request.debug_metrics_prefix_queue_ms,
                debug_metrics_finalize_ms: request.debug_metrics_finalize_ms,
                debug_metrics_commit_observability_ms: request
                    .debug_metrics_commit_observability_ms,
                debug_metrics_post_decode_ms: request.debug_metrics_post_decode_ms,
                ..RuntimeObservabilityMetrics::default()
            };
            if request.output_tokens > 1 {
                metrics.itl_avg_ms = request.decode_ms / f64::from(request.output_tokens - 1);
            }
            if let (Some(enqueued), Some(first_token)) =
                (request.enqueued_at, request.first_token_at)
            {
                metrics.ttft_ms = duration_ms(enqueued, first_token);
            }
            return Some(metrics);
        }

        if self.has_last_runtime_observability {
            return Some(self.last_runtime_observability.clone());
        }

        Some(RuntimeObservabilityMetrics {
            input_tokens: self.total_input_tokens as i32,
            output_tokens: self.total_output_tokens as i32,
            cache_hits: self.total_cache_hits as i32,
            prefill_tokens: self.total_prefill_tokens as i32,
            prefill_ms: self.total_prefill_ms,
            decode_ms: self.total_decode_ms,
            ..RuntimeObservabilityMetrics::default()
        })
    }

    pub fn enqueue_request(
        &mut self,
        context_key: impl Into<String>,
        prompt: impl Into<String>,
        n_tokens_predict: i32,
        grammar: impl Into<String>,
        json_schema: impl Into<String>,
        stop: Vec<String>,
        sampling: Option<SamplingRuntimeConfig>,
        token_emission_mode: GenerateTokenEmissionMode,
    ) -> Result<GenerateRequestId> {
        if !self.is_ready() {
            return Err(Error::RuntimeNotReady);
        }
        if n_tokens_predict <= 0 {
            return Err(Error::InvalidRequest("n_tokens_predict must be positive"));
        }

        let mut context_key = context_key.into();
        if context_key.is_empty() {
            context_key = DEFAULT_PROMPT_CONTEXT_KEY.to_string();
        }
        let prompt = prompt.into();
        let grammar = grammar.into();
        let json_schema = json_schema.into();

        let vocab = self.vocab()?;
        let prompt_tokens = tokenize(vocab, &prompt, true, true)?;
        if prompt_tokens.is_empty() {
            return Err(Error::Tokenize);
        }

        let request_id = self.next_request_id;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(Error::InvalidRequest("request id overflow"))?;

        let mut request = GenerateRequest::new(request_id, context_key);
        request.original_prompt = prompt;
        request.max_output_tokens = n_tokens_predict;
        request.token_emission_mode = token_emission_mode;
        request.prompt_tokens = prompt_tokens;
        request.grammar = grammar;
        request.json_schema = json_schema;
        request.stop = normalize_stop_sequences(stop);
        request.sampling = sampling;
        request.input_tokens = request.prompt_tokens.len() as i32;
        self.total_input_tokens += request.prompt_tokens.len();

        if !self.request_queue.push(request) {
            return Err(Error::InvalidRequest("failed to enqueue request"));
        }

        Ok(request_id)
    }

    pub fn enqueue_multimodal_request(
        &mut self,
        context_key: impl Into<String>,
        prompt: impl Into<String>,
        n_tokens_predict: i32,
        image_buffers: Vec<Vec<u8>>,
        grammar: impl Into<String>,
        json_schema: impl Into<String>,
        stop: Vec<String>,
        sampling: Option<SamplingRuntimeConfig>,
        token_emission_mode: GenerateTokenEmissionMode,
    ) -> Result<GenerateRequestId> {
        if !self.is_ready() || self.mtmd_context.is_null() {
            return Err(Error::RuntimeNotReady);
        }
        if n_tokens_predict <= 0 {
            return Err(Error::InvalidRequest("n_tokens_predict must be positive"));
        }
        if image_buffers.is_empty() {
            return Err(Error::InvalidRequest("image_buffers must not be empty"));
        }

        let mut context_key = context_key.into();
        if context_key.is_empty() {
            context_key = DEFAULT_PROMPT_CONTEXT_KEY.to_string();
        }
        let prompt = prompt.into();
        let grammar = grammar.into();
        let json_schema = json_schema.into();

        let vocab = self.vocab()?;
        let prompt_tokens = tokenize(vocab, &prompt, false, true)?;

        let request_id = self.next_request_id;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(Error::InvalidRequest("request id overflow"))?;

        let mut request = GenerateRequest::new(request_id, context_key);
        request.original_prompt = prompt;
        request.prompt_tokens = prompt_tokens;
        request.multimodal = Some(crate::runtime::request::MultimodalPayload { image_buffers });
        request.max_output_tokens = n_tokens_predict;
        request.token_emission_mode = token_emission_mode;
        request.is_multimodal_turn = true;
        request.grammar = grammar;
        request.json_schema = json_schema;
        request.stop = normalize_stop_sequences(stop);
        request.sampling = sampling;
        request.input_tokens = request.prompt_tokens.len() as i32;
        self.total_input_tokens += request.prompt_tokens.len();

        if !self.request_queue.push(request) {
            return Err(Error::InvalidRequest("failed to enqueue request"));
        }

        Ok(request_id)
    }

    pub fn cancel_request(&mut self, request_id: GenerateRequestId) -> bool {
        if request_id == 0 {
            return false;
        }
        let mut cancelled = self
            .request_queue
            .cancel(request_id, "Request cancelled.".to_string());

        for slot in self.slot_scheduler.mutable_slots() {
            if slot.request_id != request_id {
                continue;
            }
            if let Some(request) = slot.request_mut() {
                request.cancel_requested = true;
                cancelled = true;
            }
        }

        cancelled
    }

    pub fn run_scheduler_tick(&mut self) -> RequestStepResult {
        self.run_scheduler_tick_locked()
    }

    pub fn run_scheduler_burst(
        &mut self,
        max_ticks: i32,
        max_completed_responses: i32,
        max_emitted_tokens: i32,
        max_duration: Duration,
    ) -> SchedulerBurstResult {
        let mut burst_result = SchedulerBurstResult::default();
        if max_ticks <= 0 || !self.is_ready() {
            burst_result.status = RequestStepResult::Invalid;
            return burst_result;
        }

        let max_completed = max_completed_responses.max(0);
        let max_emitted = max_emitted_tokens.max(0);
        let deadline = (!max_duration.is_zero()).then(|| Instant::now() + max_duration);

        for _ in 0..max_ticks {
            let completed_before = self.request_queue.completed_response_count();
            let emitted_before = self.request_queue.total_emitted_token_count();
            let step_result = self.run_scheduler_tick_locked();
            let completed_after = self.request_queue.completed_response_count();
            let emitted_after = self.request_queue.total_emitted_token_count();

            burst_result.ticks_executed += 1;
            if completed_after > completed_before {
                burst_result.completed_response_count +=
                    (completed_after - completed_before) as i32;
            }
            if emitted_after > emitted_before {
                burst_result.emitted_token_count += emitted_after - emitted_before;
            }
            if matches!(
                step_result,
                RequestStepResult::Progressed | RequestStepResult::Terminal
            ) {
                burst_result.progressed_ticks += 1;
            }

            if matches!(
                step_result,
                RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
            ) {
                burst_result.status = step_result;
                return burst_result;
            }

            if step_result == RequestStepResult::Waiting {
                self.commit_pending_prefix_snapshots();
                burst_result.status = if burst_result.progressed_ticks > 0
                    || burst_result.completed_response_count > 0
                {
                    RequestStepResult::Progressed
                } else {
                    RequestStepResult::Waiting
                };
                return burst_result;
            }

            let completed_limit_reached =
                max_completed > 0 && burst_result.completed_response_count >= max_completed;
            let emitted_limit_reached =
                max_emitted > 0 && burst_result.emitted_token_count >= max_emitted;
            let duration_limit_reached =
                deadline.is_some_and(|deadline| Instant::now() >= deadline);

            if completed_limit_reached || emitted_limit_reached || duration_limit_reached {
                if burst_result.completed_response_count > 0 {
                    self.commit_pending_prefix_snapshots();
                }
                burst_result.status = if burst_result.progressed_ticks > 0
                    || burst_result.completed_response_count > 0
                {
                    RequestStepResult::Progressed
                } else {
                    RequestStepResult::Waiting
                };
                return burst_result;
            }
        }

        if burst_result.completed_response_count > 0 {
            self.commit_pending_prefix_snapshots();
        }
        burst_result.status =
            if burst_result.progressed_ticks > 0 || burst_result.completed_response_count > 0 {
                RequestStepResult::Progressed
            } else {
                RequestStepResult::Waiting
            };
        burst_result
    }

    pub fn run_scheduler_loop(
        &mut self,
        max_ticks: i32,
        max_completed_responses: i32,
        max_emitted_tokens: i32,
        max_duration: Duration,
    ) -> SchedulerBurstResult {
        let mut loop_result = SchedulerBurstResult::default();
        if !self.is_ready() {
            loop_result.status = RequestStepResult::Invalid;
            return loop_result;
        }

        let loop_start = Instant::now();
        loop {
            if self.request_queue.live_request_count() == 0 {
                loop_result.status = RequestStepResult::Waiting;
                break;
            }

            let completed_before = self.request_queue.completed_response_count();
            let emitted_before = self.request_queue.total_emitted_token_count();
            let step_result = self.run_scheduler_tick_locked();
            let completed_after = self.request_queue.completed_response_count();
            let emitted_after = self.request_queue.total_emitted_token_count();

            loop_result.ticks_executed += 1;
            if completed_after > completed_before {
                loop_result.completed_response_count += (completed_after - completed_before) as i32;
            }
            if emitted_after > emitted_before {
                loop_result.emitted_token_count += emitted_after - emitted_before;
            }
            if matches!(
                step_result,
                RequestStepResult::Progressed | RequestStepResult::Terminal
            ) {
                loop_result.progressed_ticks += 1;
            }

            if matches!(
                step_result,
                RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
            ) {
                loop_result.status = step_result;
                break;
            }

            if max_ticks > 0 && loop_result.ticks_executed >= max_ticks {
                loop_result.status = RequestStepResult::Progressed;
                break;
            }
            if max_completed_responses > 0
                && loop_result.completed_response_count >= max_completed_responses
            {
                loop_result.status = RequestStepResult::Progressed;
                break;
            }
            if max_emitted_tokens > 0 && loop_result.emitted_token_count >= max_emitted_tokens {
                loop_result.status = RequestStepResult::Progressed;
                break;
            }
            if !max_duration.is_zero() && loop_start.elapsed() >= max_duration {
                loop_result.status = if loop_result.progressed_ticks > 0
                    || loop_result.completed_response_count > 0
                {
                    RequestStepResult::Progressed
                } else {
                    RequestStepResult::Waiting
                };
                break;
            }
            if step_result == RequestStepResult::Waiting {
                loop_result.status = if loop_result.progressed_ticks > 0
                    || loop_result.completed_response_count > 0
                {
                    RequestStepResult::Progressed
                } else {
                    RequestStepResult::Waiting
                };
                break;
            }
        }

        if loop_result.completed_response_count > 0
            || loop_result.status == RequestStepResult::Waiting
        {
            self.commit_pending_prefix_snapshots();
        }
        loop_result
    }

    pub fn try_peek_completed_response(
        &self,
        request_id: GenerateRequestId,
    ) -> Option<GenerateResponse> {
        self.request_queue
            .peek_completed_response(request_id)
            .cloned()
    }

    pub fn has_request(&self, request_id: GenerateRequestId) -> bool {
        self.request_queue.contains(request_id)
    }

    pub fn consume_completed_response(&mut self, request_id: GenerateRequestId) -> bool {
        self.committed_observability_request_ids.remove(&request_id);
        self.request_queue.consume_completed_response(request_id)
    }

    pub fn set_token_ring_producer(&mut self, producer: Option<TokenByteRingProducer>) {
        self.request_queue.set_token_ring_producer(producer);
    }

    fn commit_pending_prefix_snapshots(&mut self) {
        if !self.slot_scheduler.is_idle() {
            return;
        }
        self.prefix_state_cache
            .drain_pending_snapshots(self.shared_context, PREFIX_SNAPSHOT_COMMIT_BUDGET);
    }

    pub fn get_bos_text(&self) -> Result<String> {
        let vocab = self.vocab()?;
        let bos = unsafe { ffi::llama_vocab_bos(vocab.as_ptr()) };
        if bos == ffi::LLAMA_TOKEN_NULL {
            return Ok(String::new());
        }
        token_to_piece(vocab, bos, true)
    }

    pub fn get_eos_text(&self) -> Result<String> {
        let vocab = self.vocab()?;
        let eos = unsafe { ffi::llama_vocab_eos(vocab.as_ptr()) };
        if eos == ffi::LLAMA_TOKEN_NULL {
            return Ok(String::new());
        }
        token_to_piece(vocab, eos, true)
    }

    pub fn chat_template_source(&self) -> Result<Option<String>> {
        if self.chat_templates.is_null() {
            return Ok(None);
        }
        owned_shim_string(
            unsafe { ffi::cogent_chat_templates_source(self.chat_templates) },
            "cogent_chat_templates_source",
        )
        .map(Some)
    }

    pub fn apply_chat_template_json(
        &self,
        messages_json: &str,
        add_assistant: bool,
    ) -> Result<String> {
        if self.chat_templates.is_null() {
            return Err(Error::NullPointer("cogent_chat_templates_init"));
        }
        let messages_json = CString::new(messages_json)?;
        owned_shim_string(
            unsafe {
                ffi::cogent_apply_chat_template(
                    self.chat_templates,
                    messages_json.as_ptr(),
                    add_assistant,
                )
            },
            "cogent_apply_chat_template",
        )
    }

    pub fn media_marker(&self) -> Result<String> {
        let marker = unsafe { ffi::cogent_mtmd_default_marker() };
        if marker.is_null() {
            return Err(Error::NullPointer("cogent_mtmd_default_marker"));
        }
        Ok(unsafe { CStr::from_ptr(marker) }
            .to_string_lossy()
            .into_owned())
    }

    fn run_scheduler_tick_locked(&mut self) -> RequestStepResult {
        if !self.is_ready() {
            return RequestStepResult::Invalid;
        }

        let completed_before = self.request_queue.completed_response_count();
        let mut admitted_any = false;
        while self
            .slot_scheduler
            .admit_pending_requests(&mut self.request_queue, &mut self.session_store)
        {
            admitted_any = true;
        }

        let tick_executed = self.run_policy_batch_tick_locked();
        self.resolve_terminal_prefix_snapshots_locked();
        self.detach_terminal_backend_samplers_locked();
        self.reclaim_and_pool_samplers_locked();
        self.slot_scheduler
            .finalize_completed_slots(&mut self.request_queue, &mut self.session_store);
        self.commit_new_completed_responses_observability_locked();

        let completed_after = self.request_queue.completed_response_count();
        if completed_after > completed_before {
            return RequestStepResult::Progressed;
        }

        if !tick_executed {
            let Some(active_slot_index) = self.slot_scheduler.find_first_active_slot() else {
                return if admitted_any {
                    RequestStepResult::Progressed
                } else {
                    RequestStepResult::Waiting
                };
            };

            let diagnostic = self.build_no_progress_diagnostic_locked();
            if let Some(active_slot) = self
                .slot_scheduler
                .mutable_slots()
                .get_mut(active_slot_index)
            {
                if active_slot.phase != SlotPhase::Failed
                    && active_slot.phase != SlotPhase::Completed
                {
                    active_slot.terminal_error_message = diagnostic;
                    active_slot.phase = SlotPhase::Failed;
                    if let Some(request) = active_slot.request_mut() {
                        request.lifecycle = GenerateRequestLifecycle::Failed;
                    }
                }
            }

            self.resolve_terminal_prefix_snapshots_locked();
            self.detach_terminal_backend_samplers_locked();
            self.reclaim_and_pool_samplers_locked();
            self.slot_scheduler
                .finalize_completed_slots(&mut self.request_queue, &mut self.session_store);
            self.commit_new_completed_responses_observability_locked();
            if self.request_queue.completed_response_count() > completed_before {
                return RequestStepResult::Progressed;
            }
            return RequestStepResult::FatalNoProgress;
        }

        if tick_executed || admitted_any {
            RequestStepResult::Progressed
        } else {
            RequestStepResult::Waiting
        }
    }

    fn run_policy_batch_tick_locked(&mut self) -> bool {
        let vocab = match self.vocab() {
            Ok(vocab) => vocab.as_ptr(),
            Err(_) => return false,
        };

        let debug_metrics_enabled = self.debug_metrics_enabled;
        let normalize_start = debug_metrics_enabled.then(Instant::now);
        self.normalize_slots_for_tick(vocab);
        let mut debug_metrics = DebugMetricsTick::default();
        if let Some(start) = normalize_start {
            debug_metrics.normalize_ms = duration_ms(start, Instant::now());
        }

        let select_slots_start = debug_metrics_enabled.then(Instant::now);
        self.slot_scheduler
            .select_decode_ready_slots_into(&mut self.scratch_decode_ready_slots);
        self.slot_scheduler
            .select_prefill_ready_slots_into(&mut self.scratch_prefill_ready_slots);
        if let Some(start) = select_slots_start {
            debug_metrics.select_slots_ms = duration_ms(start, Instant::now());
        }
        if self.scratch_decode_ready_slots.is_empty() && self.scratch_prefill_ready_slots.is_empty()
        {
            return false;
        }

        let batch_token_budget = resolve_batch_token_budget(self.shared_context, &self.config);
        let tick_budget = SlotScheduler::build_tick_budget(
            self.config.scheduler.policy,
            self.scratch_decode_ready_slots.len() as i32,
            self.scratch_prefill_ready_slots.len() as i32,
            batch_token_budget,
            self.config.scheduler.prefill_chunk_size,
        );
        let effective_prefill_chunk_size = self.resolve_prefill_chunk_size_locked(
            tick_budget,
            self.scratch_decode_ready_slots.len() as i32,
            self.scratch_prefill_ready_slots.len() as i32,
        );

        // Borrow `scratch_plan` out so the rest of this function can pass `&plan`
        // to helpers that also need `&mut self`. The Vec allocations live on
        // through the move, so we still get the alloc-reuse benefit.
        let mut plan = std::mem::take(&mut self.scratch_plan);
        let plan_start = debug_metrics_enabled.then(Instant::now);
        self.batch_planner.build_policy_batch_into(
            &mut plan,
            self.slot_scheduler.slots(),
            &self.scratch_decode_ready_slots,
            &self.scratch_prefill_ready_slots,
            tick_budget,
            effective_prefill_chunk_size,
        );
        if let Some(start) = plan_start {
            debug_metrics.plan_ms = duration_ms(start, Instant::now());
        }
        if plan.is_empty() {
            self.scratch_plan = plan;
            return false;
        }

        let batch_build_start = debug_metrics_enabled.then(Instant::now);
        if self
            .shared_batch_builder
            .ensure_capacity(batch_token_budget, self.resolved_limits.n_parallel.max(1))
            .is_err()
        {
            self.fail_plan_slots(&plan, "Shared batch allocation failed.");
            return false;
        }
        self.shared_batch_builder.reset();
        self.scratch_logits_contributions.clear();

        let mut batch_token_index = 0_i32;
        for contribution in plan.contributions.iter() {
            let Some(slot) = self.slot_scheduler.slots().get(contribution.slot_index) else {
                continue;
            };
            if slot.seq_id < 0 {
                continue;
            }
            if !self.shared_batch_builder.add_token(
                contribution.token,
                contribution.position,
                slot.seq_id,
                contribution.request_logits,
            ) {
                self.fail_plan_slots(&plan, "Shared batch builder capacity was exceeded.");
                self.scratch_plan = plan;
                return false;
            }
            if contribution.request_logits {
                self.scratch_logits_contributions
                    .push(PendingLogitsContribution {
                        slot_index: contribution.slot_index,
                        batch_token_index,
                        sampled_token: -1,
                    });
            }
            batch_token_index += 1;
        }
        if let Some(start) = batch_build_start {
            debug_metrics.batch_build_ms = duration_ms(start, Instant::now());
        }

        // The decode/sync/post-decode timings are *production* metrics
        // (native_gpu_ms/native_sync_ms/native_logic_ms) consumed by the
        // observability surface — keep them unconditional.
        let decode_start = Instant::now();
        let decode_status = unsafe {
            ffi::cogent_llama_decode(self.shared_context, self.shared_batch_builder.raw())
        };
        let decode_submitted = Instant::now();
        let sync_ok = unsafe { ffi::cogent_llama_synchronize(self.shared_context) };
        let decode_end = Instant::now();
        debug_metrics.llama_decode_ms = duration_ms(decode_start, decode_submitted);
        debug_metrics.llama_sync_ms = duration_ms(decode_submitted, decode_end);
        if decode_status != 0 {
            let diagnostic = format!(
                "llama_decode() failed in shared tick (status={decode_status}, n_tokens={})",
                self.shared_batch_builder.raw().n_tokens
            );
            self.fail_plan_slots(&plan, &diagnostic);
            self.scratch_plan = plan;
            return false;
        }
        if !sync_ok {
            self.fail_plan_slots(&plan, "llama_synchronize() failed in shared tick.");
            self.scratch_plan = plan;
            return false;
        }

        let native_decode_ms = debug_metrics.llama_decode_ms;
        let native_sync_ms = debug_metrics.llama_sync_ms;
        let native_logic_ms = debug_metrics.plan_ms + debug_metrics.batch_build_ms;
        let apply_start = debug_metrics_enabled.then(Instant::now);
        self.apply_bookkeeping_and_emit(&plan, native_decode_ms, native_sync_ms, native_logic_ms);
        if let Some(start) = apply_start {
            debug_metrics.apply_bookkeeping_ms = duration_ms(start, Instant::now());
        }
        let (sample_ms, token_piece_ms) = self.sample_logits_and_buffer_output(vocab);
        debug_metrics.sample_ms = sample_ms;
        debug_metrics.token_piece_ms = token_piece_ms;
        let emit_start = debug_metrics_enabled.then(Instant::now);
        for slot in self.slot_scheduler.mutable_slots() {
            if slot.phase == SlotPhase::Streaming && !slot.buffered_output_text.is_empty() {
                SlotScheduler::emit_buffered_token_piece(&mut self.request_queue, slot);
            }
        }
        if let Some(start) = emit_start {
            debug_metrics.emit_ms = duration_ms(start, Instant::now());
        }
        let prefix_queue_start = debug_metrics_enabled.then(Instant::now);
        if snapshot_prefix_cache_enabled(self.config.cache.mode) {
            self.queue_prefix_snapshots(&plan);
        }
        if let Some(start) = prefix_queue_start {
            debug_metrics.prefix_queue_ms = duration_ms(start, Instant::now());
            debug_metrics.post_decode_ms = duration_ms(decode_end, Instant::now());
            self.apply_debug_metrics_to_plan(&plan, debug_metrics);
        }
        // Return the plan's allocations to the scratch slot so the next tick
        // can refill it without paying for fresh Vec capacity.
        self.scratch_plan = plan;
        true
    }

    fn normalize_slots_for_tick(&mut self, vocab: *const ffi::llama_vocab) {
        let slot_count = self.slot_scheduler.slots().len();
        for slot_index in 0..slot_count {
            let slot = &mut self.slot_scheduler.mutable_slots()[slot_index];
            if slot.request().is_none() || slot.session.is_none() || slot.seq_id < 0 {
                continue;
            }

            // `InferenceRuntime::cancel_request` mirrors the flag onto the
            // slot's own copy of the request, so we can read it directly
            // here without a per-tick `RequestQueue::find` HashMap lookup.
            if slot
                .request()
                .is_some_and(|request| request.cancel_requested)
            {
                slot.terminal_error_message = "Request cancelled.".to_string();
                slot.phase = SlotPhase::Failed;
                if let Some(request) = slot.request_mut() {
                    request.lifecycle = GenerateRequestLifecycle::Cancelled;
                }
                continue;
            }

            normalize_runnable_slot_state(
                slot,
                self.shared_context,
                self.primary_model,
                live_retained_prefix_tokens(&self.config),
            );

            if slot.sampler().is_none() {
                let (grammar, json_schema, sampling) = slot
                    .request()
                    .map(|request| {
                        (
                            request.grammar.clone(),
                            request.json_schema.clone(),
                            request.sampling.clone(),
                        )
                    })
                    .unwrap_or_default();

                let sampling_json = self.config.sampling_json_with_override(sampling.as_ref());
                let key = SamplerCacheKey {
                    sampling_json,
                    grammar: grammar.clone(),
                    json_schema: json_schema.clone(),
                };

                let cached_sampler = self.sampler_pool.get_mut(&key).and_then(|vec| vec.pop());
                if let Some(sampler) = cached_sampler {
                    slot.set_sampler(sampler.as_ptr());
                    slot.sampler_key = Some(key);
                    let attach_start = Instant::now();
                    let attached = attach_backend_sampler(self.shared_context, slot);
                    let attach_ms = duration_ms(attach_start, Instant::now());
                    if let Some(request) = slot.request_mut() {
                        request.debug_metrics_backend_sampler_attach_attempts += 1;
                        request.debug_metrics_backend_sampler_attach_ms += attach_ms;
                        if !attached {
                            request.debug_metrics_backend_sampler_attach_failures += 1;
                        }
                    }
                } else {
                    match create_sampler(
                        self.common_init,
                        &self.config,
                        sampling.as_ref(),
                        Some(&grammar),
                        Some(&json_schema),
                    ) {
                        Ok(sampler) => {
                            slot.set_sampler(sampler);
                            slot.sampler_key = Some(key);
                            let attach_start = Instant::now();
                            let attached = attach_backend_sampler(self.shared_context, slot);
                            let attach_ms = duration_ms(attach_start, Instant::now());
                            if let Some(request) = slot.request_mut() {
                                request.debug_metrics_backend_sampler_attach_attempts += 1;
                                request.debug_metrics_backend_sampler_attach_ms += attach_ms;
                                if !attached {
                                    request.debug_metrics_backend_sampler_attach_failures += 1;
                                }
                            }
                        }
                        Err(_) => {
                            slot.terminal_error_message = if grammar.is_empty() {
                                "Failed to create per-slot sampler.".to_string()
                            } else {
                                "Failed to create per-slot grammar sampler.".to_string()
                            };
                            slot.phase = SlotPhase::Failed;
                            if let Some(request) = slot.request_mut() {
                                request.lifecycle = GenerateRequestLifecycle::Failed;
                            }
                            continue;
                        }
                    }
                }
            }

            if slot.phase == SlotPhase::Prefill && slot.prefill_cursor == 0 {
                if slot
                    .request()
                    .is_some_and(|request| request.is_multimodal_turn)
                {
                    let ok = run_multimodal_prefill(
                        self.mtmd_context,
                        self.shared_context,
                        vocab,
                        resolve_batch_token_budget(self.shared_context, &self.config),
                        &mut self.request_queue,
                        slot,
                        &mut self.scratch_token_piece,
                    );
                    if !ok {
                        if slot.terminal_error_message.is_empty() {
                            slot.terminal_error_message =
                                "Failed to evaluate multimodal prompt.".to_string();
                        }
                        slot.phase = SlotPhase::Failed;
                        if let Some(request) = slot.request_mut() {
                            request.lifecycle = GenerateRequestLifecycle::Failed;
                            request.multimodal = None;
                        }
                    }
                    continue;
                }

                let request_snapshot = slot.request().cloned();
                if let Some(mut request) = request_snapshot {
                    let context_key = request.context_key.clone();
                    let prompt_tokens = request.prompt_tokens.clone();
                    let max_output_tokens = request.max_output_tokens;
                    let mut prefill_cursor = 0;
                    let prepared = prepare_sequence_for_prompt(
                        self.shared_context,
                        self.primary_model,
                        live_retained_prefix_tokens(&self.config),
                        snapshot_prefix_cache_enabled(self.config.cache.mode),
                        self.config.scheduler.policy.decode_token_reserve,
                        self.model_fingerprint,
                        &self.session_store,
                        &mut self.prefix_state_cache,
                        &mut self.prefix_cache_policy,
                        &context_key,
                        &prompt_tokens,
                        max_output_tokens,
                        &mut slot.mirror,
                        slot.seq_id,
                        &mut request,
                        &mut prefill_cursor,
                    );

                    if !prepared {
                        slot.terminal_error_message =
                            "Failed to prepare sequence for prompt reuse.".to_string();
                        slot.phase = SlotPhase::Failed;
                        if let Some(slot_request) = slot.request_mut() {
                            slot_request.lifecycle = GenerateRequestLifecycle::Failed;
                        }
                        continue;
                    }

                    if let Some(slot_request) = slot.request_mut() {
                        slot_request.cache_hits = request.cache_hits;
                    }
                    seed_sampler_with_prompt(slot, &prompt_tokens);
                    slot.prefill_cursor = prefill_cursor;
                    slot.phase = if slot
                        .request()
                        .is_some_and(|request| slot.prefill_cursor >= request.prompt_tokens.len())
                    {
                        SlotPhase::Decode
                    } else {
                        SlotPhase::Prefill
                    };
                }
            }

            if slot.phase == SlotPhase::Decode
                && !ensure_decode_step_context_space(
                    self.shared_context,
                    live_retained_prefix_tokens(&self.config),
                    slot,
                )
            {
                slot.terminal_error_message =
                    "Failed to extend decode context headroom.".to_string();
                slot.phase = SlotPhase::Failed;
                if let Some(request) = slot.request_mut() {
                    request.lifecycle = GenerateRequestLifecycle::Failed;
                }
                continue;
            }

            if let Some(request) = slot.request_mut() {
                request.lifecycle = GenerateRequestLifecycle::Running;
            }
        }
    }

    fn apply_bookkeeping_and_emit(
        &mut self,
        plan: &crate::runtime::scheduler::SharedBatchPlan,
        native_decode_ms: f64,
        native_sync_ms: f64,
        native_logic_ms: f64,
    ) {
        let mut tick_had_prefill = false;
        let mut tick_had_decode = false;
        let tick_ms = native_decode_ms + native_sync_ms + native_logic_ms;
        let mut timed_slots: u64 = 0;
        let mut prefill_timed_slots: u64 = 0;
        let mut decode_timed_slots: u64 = 0;
        let mut emitted_slots: u64 = 0;

        for contribution in &plan.contributions {
            let Some(slot) = self
                .slot_scheduler
                .mutable_slots()
                .get_mut(contribution.slot_index)
            else {
                continue;
            };

            // Mirror update (no request borrow needed)
            slot.mirror.current_kv_tokens.push(contribution.token);
            slot.mirror.n_past += 1;
            slot.batch_participation_count += 1;

            // Phase transition — read prompt_tokens.len() via short immutable borrow
            let is_prefill = contribution.kind == BatchContributionKind::Prefill;
            if is_prefill {
                let prompt_len = slot.request().map(|r| r.prompt_tokens.len()).unwrap_or(0);
                slot.prefill_cursor += 1;
                slot.phase = if slot.prefill_cursor >= prompt_len {
                    SlotPhase::Decode
                } else {
                    SlotPhase::Prefill
                };
            } else {
                slot.decode_step_count += 1;
            }

            // Metrics accumulation (separate short request borrow)
            if unique_slot_first_use(&mut timed_slots, contribution.slot_index) {
                if let Some(request) = slot.request_mut() {
                    request.native_gpu_ms += native_decode_ms;
                    request.native_sync_ms += native_sync_ms;
                    request.native_logic_ms += native_logic_ms;
                }
            }
            if is_prefill {
                if let Some(request) = slot.request_mut() {
                    request.prefill_tokens += 1;
                }
                if unique_slot_first_use(&mut prefill_timed_slots, contribution.slot_index) {
                    if let Some(request) = slot.request_mut() {
                        request.prefill_ms += tick_ms;
                    }
                }
                self.total_prefill_tokens += 1;
                tick_had_prefill = true;
            } else {
                if unique_slot_first_use(&mut decode_timed_slots, contribution.slot_index) {
                    if let Some(request) = slot.request_mut() {
                        request.decode_ms += tick_ms;
                    }
                }
                tick_had_decode = true;
            }

            // Emission
            if unique_slot_first_use(&mut emitted_slots, contribution.slot_index)
                && !slot.buffered_output_text.is_empty()
            {
                SlotScheduler::emit_buffered_token_piece(&mut self.request_queue, slot);
            }
        }

        if tick_had_decode {
            self.total_decode_ms += tick_ms;
        }
        if tick_had_prefill {
            self.total_prefill_ms += tick_ms;
        }
    }

    fn sample_logits_and_buffer_output(&mut self, vocab: *const ffi::llama_vocab) -> (f64, f64) {
        let mut sample_ms = 0.0;
        let mut token_piece_ms = 0.0;
        let now = Instant::now();
        let enable_metrics = self.debug_metrics_enabled;
        for pending_logits in &mut self.scratch_logits_contributions {
            let Some(slot) = self
                .slot_scheduler
                .mutable_slots()
                .get_mut(pending_logits.slot_index)
            else {
                continue;
            };
            let Some(sampler) = slot.sampler() else {
                continue;
            };

            let sample_start = enable_metrics.then(Instant::now);
            let next_token = unsafe {
                ffi::cogent_common_sampler_sample(
                    sampler.as_ptr(),
                    self.shared_context,
                    pending_logits.batch_token_index,
                )
            };
            if let Some(start) = sample_start {
                sample_ms += duration_ms(start, Instant::now());
            }
            pending_logits.sampled_token = next_token;

            if next_token == ffi::LLAMA_TOKEN_NULL {
                slot.terminal_error_message = "llama_sampler_sample() failed.".to_string();
                slot.phase = SlotPhase::Failed;
                if let Some(request) = slot.request_mut() {
                    request.lifecycle = GenerateRequestLifecycle::Failed;
                }
                continue;
            }
            unsafe {
                ffi::cogent_common_sampler_accept(sampler.as_ptr(), next_token, true);
            }

            // EOG check (immutable slot borrow, dropped before mutable ops).
            let is_eog = unsafe { ffi::llama_vocab_is_eog(vocab, next_token) };
            if is_eog {
                if let Some(request) = slot.request_mut() {
                    request.first_token_at.get_or_insert(now);
                    request.first_sampled_token_id = next_token;
                    request.lifecycle = GenerateRequestLifecycle::Completed;
                }
                flush_pending_utf8(slot);
                slot.phase = SlotPhase::Completed;
                continue;
            }

            slot.generated_tokens.push(next_token);
            self.total_output_tokens += 1;

            let piece_start = enable_metrics.then(Instant::now);
            append_token_piece_to_slot(vocab, next_token, slot, &mut self.scratch_token_piece);
            if let Some(start) = piece_start {
                token_piece_ms += duration_ms(start, Instant::now());
            }

            // Stop sequence check (mutates slot phase on match).
            let stop_matched = slot.request().is_some_and(|r| !r.stop.is_empty())
                && apply_stop_sequences_to_slot(slot);

            // Read fields before entering request borrow.
            let gen_len = slot.generated_tokens.len();
            let cancel = slot.request().map(|r| r.cancel_requested).unwrap_or(false);
            let max_output_tokens = slot.request().map(|r| r.max_output_tokens).unwrap_or(0);
            let should_complete = stop_matched
                || cancel
                || (max_output_tokens > 0 && gen_len >= max_output_tokens as usize);

            // Request-only block (no slot field access inside).
            {
                let request = match slot.request_mut() {
                    Some(r) => r,
                    None => continue,
                };
                request.first_token_at.get_or_insert(now);
                request.first_sampled_token_id = next_token;
                request.output_tokens += 1;
                request.emitted_token_count += 1;
                request.last_token_at = Some(now);
                request.lifecycle = if should_complete {
                    GenerateRequestLifecycle::Completed
                } else {
                    GenerateRequestLifecycle::Streaming
                };
            }
            // request borrow ended — slot is available again.

            if should_complete {
                flush_pending_utf8(slot);
                slot.phase = SlotPhase::Completed;
            } else {
                slot.phase = SlotPhase::Streaming;
            }
            if stop_matched || cancel {
                continue;
            }
        }
        (sample_ms, token_piece_ms)
    }

    fn queue_prefix_snapshots(&mut self, plan: &crate::runtime::scheduler::SharedBatchPlan) {
        if !self.scratch_decode_ready_slots.is_empty() {
            return;
        }

        let mut seen_slots: u64 = 0;
        for contribution in &plan.contributions {
            if contribution.kind != BatchContributionKind::Prefill
                || !unique_slot_first_use(&mut seen_slots, contribution.slot_index)
            {
                continue;
            }
            let Some(slot) = self.slot_scheduler.slots().get(contribution.slot_index) else {
                continue;
            };
            let Some(request) = slot.request() else {
                continue;
            };
            let token_count = slot.mirror.current_kv_tokens.len();
            if !self
                .prefix_cache_policy
                .should_store_boundary(token_count, request.prompt_tokens.len())
            {
                continue;
            }
            self.prefix_state_cache
                .enqueue_pending_snapshot(PendingPrefixSnapshot {
                    model_fingerprint: self.model_fingerprint,
                    context_key: request.context_key.clone(),
                    seq_id: slot.seq_id,
                    token_count,
                    prefix_hash: self
                        .prefix_cache_policy
                        .hash_prefix(&slot.mirror.current_kv_tokens, token_count),
                    retention_priority: token_count as u64,
                    prefix_tokens: slot.mirror.current_kv_tokens[..token_count].to_vec(),
                });
            self.prefix_cache_policy.record_store(token_count);
        }
    }

    fn apply_debug_metrics_to_plan(
        &mut self,
        plan: &crate::runtime::scheduler::SharedBatchPlan,
        debug_metrics: DebugMetricsTick,
    ) {
        let mut timed_slots: u64 = 0;
        let mut decode_slots: u64 = 0;
        let mut prefill_slots: u64 = 0;

        for contribution in &plan.contributions {
            let Some(slot) = self
                .slot_scheduler
                .mutable_slots()
                .get_mut(contribution.slot_index)
            else {
                continue;
            };
            let Some(request) = slot.request_mut() else {
                continue;
            };

            if unique_slot_first_use(&mut timed_slots, contribution.slot_index) {
                request.debug_metrics_scheduler_ticks += 1;
                request.debug_metrics_normalize_ms += debug_metrics.normalize_ms;
                request.debug_metrics_select_slots_ms += debug_metrics.select_slots_ms;
                request.debug_metrics_plan_ms += debug_metrics.plan_ms;
                request.debug_metrics_batch_build_ms += debug_metrics.batch_build_ms;
                request.debug_metrics_llama_decode_ms += debug_metrics.llama_decode_ms;
                request.debug_metrics_llama_sync_ms += debug_metrics.llama_sync_ms;
                request.debug_metrics_apply_bookkeeping_ms += debug_metrics.apply_bookkeeping_ms;
                request.debug_metrics_apply_decode_results_ms +=
                    debug_metrics.apply_decode_results_ms;
                request.debug_metrics_sample_ms += debug_metrics.sample_ms;
                request.debug_metrics_token_piece_ms += debug_metrics.token_piece_ms;
                request.debug_metrics_emit_ms += debug_metrics.emit_ms;
                request.debug_metrics_prefix_queue_ms += debug_metrics.prefix_queue_ms;
                request.debug_metrics_post_decode_ms += debug_metrics.post_decode_ms;
            }

            match contribution.kind {
                BatchContributionKind::Prefill => {
                    if unique_slot_first_use(&mut prefill_slots, contribution.slot_index) {
                        request.debug_metrics_prefill_ticks += 1;
                    }
                }
                BatchContributionKind::Decode => {
                    if unique_slot_first_use(&mut decode_slots, contribution.slot_index) {
                        request.debug_metrics_decode_ticks += 1;
                    }
                }
            }
        }
    }

    fn resolve_terminal_prefix_snapshots_locked(&mut self) {
        let terminal_sequences: Vec<_> = self
            .slot_scheduler
            .slots()
            .iter()
            .filter_map(|slot| {
                if slot.seq_id < 0 {
                    return None;
                }
                match slot.phase {
                    SlotPhase::Completed => Some((slot.seq_id, true)),
                    SlotPhase::Failed => Some((slot.seq_id, false)),
                    _ => None,
                }
            })
            .collect();

        for (seq_id, completed) in terminal_sequences {
            if completed {
                self.prefix_state_cache
                    .drain_best_pending_snapshot_for_seq(self.shared_context, seq_id);
            } else {
                self.prefix_state_cache
                    .drop_pending_snapshots_for_seq(seq_id);
            }
        }
    }

    fn detach_terminal_backend_samplers_locked(&mut self) {
        for slot in self.slot_scheduler.mutable_slots() {
            if matches!(slot.phase, SlotPhase::Completed | SlotPhase::Failed) {
                detach_backend_sampler(self.shared_context, slot);
            }
        }
    }

    fn reclaim_and_pool_samplers_locked(&mut self) {
        for slot in self.slot_scheduler.mutable_slots() {
            if matches!(slot.phase, SlotPhase::Completed | SlotPhase::Failed) {
                if let Some(sampler) = slot.take_sampler() {
                    if let Some(key) = slot.sampler_key.take() {
                        unsafe {
                            let raw = ffi::cogent_common_sampler_raw(sampler.as_ptr());
                            if !raw.is_null() {
                                ffi::llama_sampler_reset(raw);
                            }
                        }
                        self.sampler_pool.entry(key).or_default().push(sampler);
                    } else {
                        unsafe {
                            ffi::cogent_common_sampler_free(sampler.as_ptr());
                        }
                    }
                }
            }
        }
    }

    fn detach_all_backend_samplers_locked(&mut self) {
        for slot in self.slot_scheduler.mutable_slots() {
            detach_backend_sampler(self.shared_context, slot);
        }
    }

    fn fail_plan_slots(
        &mut self,
        plan: &crate::runtime::scheduler::SharedBatchPlan,
        message: &str,
    ) {
        let mut failed_slots: u64 = 0;
        for contribution in &plan.contributions {
            if !unique_slot_first_use(&mut failed_slots, contribution.slot_index) {
                continue;
            }
            let Some(slot) = self
                .slot_scheduler
                .mutable_slots()
                .get_mut(contribution.slot_index)
            else {
                continue;
            };
            slot.terminal_error_message = message.to_string();
            slot.phase = SlotPhase::Failed;
            if let Some(request) = slot.request_mut() {
                request.lifecycle = GenerateRequestLifecycle::Failed;
            }
        }
    }

    fn resolve_prefill_chunk_size_locked(
        &self,
        tick_budget: SchedulerTickBudget,
        decode_ready_count: i32,
        prefill_ready_count: i32,
    ) -> i32 {
        let configured_chunk_size = self.config.scheduler.prefill_chunk_size.max(0);
        if !self
            .config
            .scheduler
            .policy
            .enable_adaptive_prefill_chunking
            || prefill_ready_count <= 0
        {
            return configured_chunk_size;
        }

        if decode_ready_count <= 0 && configured_chunk_size <= 0 {
            return 0;
        }

        let prefill_budget = tick_budget.effective_prefill_budget();
        if prefill_budget <= 0 {
            return configured_chunk_size;
        }

        let fair_share = (prefill_budget / prefill_ready_count.max(1)).max(1);
        if configured_chunk_size > 0 {
            configured_chunk_size.min(fair_share)
        } else {
            fair_share
        }
    }

    fn commit_new_completed_responses_observability_locked(&mut self) {
        // Hot path optimisation: this function runs *every* scheduler tick.
        // When nothing new has completed since the last commit (the common
        // case during long decodes) we skip the Vec allocation, the sort, and
        // the per-id HashMap lookups. `committed_observability_request_ids`
        // only ever shrinks on `consume_completed_response`, which also
        // removes the entry from `completed_responses`, so the counts move in
        // lockstep when nothing new lands.
        if self.request_queue.completed_response_count()
            == self.committed_observability_request_ids.len()
        {
            return;
        }
        let mut completed_request_ids = self.request_queue.completed_response_ids();
        if completed_request_ids.is_empty() {
            return;
        }
        completed_request_ids.sort_unstable();

        for request_id in completed_request_ids {
            if request_id == 0
                || self
                    .committed_observability_request_ids
                    .contains(&request_id)
            {
                continue;
            }

            if self
                .request_queue
                .peek_completed_response(request_id)
                .is_none()
            {
                continue;
            };
            let debug_metrics_commit_start = Instant::now();
            self.committed_observability_request_ids.insert(request_id);

            let committed_metrics = {
                let Some(response) = self.request_queue.find_mut_completed_response(request_id)
                else {
                    continue;
                };
                response
                    .runtime_observability
                    .debug_metrics_commit_observability_ms +=
                    duration_ms(debug_metrics_commit_start, Instant::now());
                if self.config.observability.effective_runtime_metrics() {
                    Some(response.runtime_observability)
                } else {
                    None
                }
            };

            if let Some(metrics) = committed_metrics {
                self.last_runtime_observability = metrics;
                self.has_last_runtime_observability = true;
            }
        }
    }

    fn build_no_progress_diagnostic_locked(&self) -> String {
        let mut active_count = 0;
        let mut decode_ready_count = 0;
        let mut prefill_ready_count = 0;
        let mut decode_without_seed_count = 0;
        let mut streaming_without_buffer_count = 0;

        for slot in self.slot_scheduler.slots() {
            let Some(request) = slot.request() else {
                continue;
            };
            if !matches!(
                slot.phase,
                SlotPhase::Idle | SlotPhase::Completed | SlotPhase::Failed
            ) {
                active_count += 1;
            }
            if slot.phase == SlotPhase::Decode
                && slot.buffered_output_text.is_empty()
                && !slot.generated_tokens.is_empty()
            {
                decode_ready_count += 1;
            }
            if slot.phase == SlotPhase::Prefill
                && (request.is_multimodal_turn || slot.prefill_cursor < request.prompt_tokens.len())
            {
                prefill_ready_count += 1;
            }
            if slot.phase == SlotPhase::Decode && slot.generated_tokens.is_empty() {
                decode_without_seed_count += 1;
            }
            if slot.phase == SlotPhase::Streaming && slot.buffered_output_text.is_empty() {
                streaming_without_buffer_count += 1;
            }
        }

        format!(
            "Shared batch tick could not make progress (active={active_count}, decode_ready={decode_ready_count}, prefill_ready={prefill_ready_count}, decode_without_seed={decode_without_seed_count}, streaming_without_buffer={streaming_without_buffer_count})."
        )
    }

    fn vocab(&self) -> Result<NonNull<ffi::llama_vocab>> {
        if self.primary_model.is_null() {
            return Err(Error::RuntimeNotReady);
        }
        let vocab =
            unsafe { ffi::llama_model_get_vocab(self.primary_model) as *mut ffi::llama_vocab };
        NonNull::new(vocab).ok_or(Error::NullPointer("llama_model_get_vocab"))
    }
}

impl Drop for InferenceRuntime {
    fn drop(&mut self) {
        self.detach_all_backend_samplers_locked();
        self.slot_scheduler.resize(0);
        self.session_store.clear();

        for samplers in self.sampler_pool.values_mut() {
            while let Some(sampler) = samplers.pop() {
                unsafe {
                    ffi::cogent_common_sampler_free(sampler.as_ptr());
                }
            }
        }

        if !self.chat_templates.is_null() {
            unsafe {
                ffi::cogent_chat_templates_free(self.chat_templates);
            }
            self.chat_templates = std::ptr::null_mut();
        }
        if !self.mtmd_context.is_null() {
            unsafe {
                ffi::cogent_mtmd_free(self.mtmd_context);
            }
            self.mtmd_context = std::ptr::null_mut();
        }
        if !self.common_init.is_null() {
            unsafe {
                ffi::cogent_common_init_free(self.common_init);
            }
            self.common_init = std::ptr::null_mut();
        }
        if !self.shared_context.is_null() {
            self.shared_context = std::ptr::null_mut();
        }
        if !self.primary_model.is_null() {
            self.primary_model = std::ptr::null_mut();
        }
        drop(self.residency_lease.take());
    }
}

fn attach_backend_sampler(shared_context: *mut ffi::llama_context, slot: &mut SlotState) -> bool {
    if shared_context.is_null() || slot.seq_id < 0 || slot.backend_sampler_attached() {
        return false;
    }

    let Some(sampler) = slot.sampler() else {
        return false;
    };
    if !unsafe { ffi::cogent_common_sampler_backend_sampling(sampler.as_ptr()) } {
        return false;
    }
    let raw_sampler = unsafe { ffi::cogent_common_sampler_raw(sampler.as_ptr()) };
    if raw_sampler.is_null() {
        return false;
    }

    let attached =
        unsafe { ffi::cogent_llama_set_sampler(shared_context, slot.seq_id, raw_sampler) };
    if attached {
        slot.mark_backend_sampler_attached(true);
    }
    attached
}

fn detach_backend_sampler(shared_context: *mut ffi::llama_context, slot: &mut SlotState) {
    if !slot.backend_sampler_attached() {
        return;
    }

    if !shared_context.is_null() && slot.seq_id >= 0 {
        unsafe {
            ffi::cogent_llama_set_sampler(shared_context, slot.seq_id, std::ptr::null_mut());
        }
    }
    slot.mark_backend_sampler_attached(false);
}

fn create_sampler(
    common_init: *mut ffi::cogent_common_init,
    config: &NativeRuntimeConfig,
    sampling_override: Option<&SamplingRuntimeConfig>,
    grammar: Option<&str>,
    json_schema: Option<&str>,
) -> Result<*mut ffi::cogent_common_sampler> {
    if common_init.is_null() {
        return Err(Error::RuntimeNotReady);
    }
    let sampling_json = CString::new(config.sampling_json_with_override(sampling_override))?;
    let grammar = CString::new(grammar.unwrap_or(""))?;
    let json_schema = CString::new(json_schema.unwrap_or(""))?;
    let mut error = std::ptr::null_mut();
    let sampler = unsafe {
        ffi::cogent_common_sampler_init_from_json(
            common_init,
            sampling_json.as_ptr(),
            grammar.as_ptr(),
            json_schema.as_ptr(),
            &mut error,
        )
    };
    if sampler.is_null() {
        return Err(runtime_command_from_shim_error(
            error,
            "common sampler initialization failed",
        ));
    }
    Ok(sampler)
}

fn resolved_runtime_limits(common_init: *mut ffi::cogent_common_init) -> ResolvedRuntimeLimits {
    ResolvedRuntimeLimits {
        n_ctx: unsafe { ffi::cogent_common_init_n_ctx(common_init) }.max(0),
        n_batch: unsafe { ffi::cogent_common_init_n_batch(common_init) }.max(0),
        n_ubatch: unsafe { ffi::cogent_common_init_n_ubatch(common_init) }.max(0),
        n_parallel: unsafe { ffi::cogent_common_init_n_parallel(common_init) }.max(0),
        kv_unified: unsafe { ffi::cogent_common_init_kv_unified(common_init) },
        flash_attention: owned_shim_string(
            unsafe { ffi::cogent_common_init_flash_attention(common_init) },
            "cogent_common_init_flash_attention",
        )
        .unwrap_or_else(|_| "unknown".to_string()),
        cache_type_k: owned_shim_string(
            unsafe { ffi::cogent_common_init_cache_type_k(common_init) },
            "cogent_common_init_cache_type_k",
        )
        .unwrap_or_else(|_| "unknown".to_string()),
        cache_type_v: owned_shim_string(
            unsafe { ffi::cogent_common_init_cache_type_v(common_init) },
            "cogent_common_init_cache_type_v",
        )
        .unwrap_or_else(|_| "unknown".to_string()),
    }
}

fn seed_sampler_with_prompt(slot: &mut SlotState, prompt_tokens: &[llama_token]) {
    if slot.sampler_prompt_seeded {
        return;
    }
    if slot
        .request()
        .is_some_and(|request| !request.grammar.is_empty() || !request.json_schema.is_empty())
    {
        return;
    }
    let Some(sampler) = slot.sampler() else {
        return;
    };

    for &token in prompt_tokens {
        if !unsafe { ffi::cogent_common_sampler_accept(sampler.as_ptr(), token, false) } {
            return;
        }
    }
    slot.sampler_prompt_seeded = true;
}

fn resolve_batch_token_budget(
    shared_context: *mut ffi::llama_context,
    config: &NativeRuntimeConfig,
) -> i32 {
    if !shared_context.is_null() {
        return unsafe { ffi::llama_n_batch(shared_context) }.max(1) as i32;
    }
    config.context.n_batch.unwrap_or(1).max(1)
}

fn live_prefix_reuse_enabled(mode: KvReuseMode) -> bool {
    matches!(
        mode,
        KvReuseMode::LiveSlotPrefix | KvReuseMode::LiveSlotAndSnapshot
    )
}

fn snapshot_prefix_cache_enabled(mode: KvReuseMode) -> bool {
    matches!(
        mode,
        KvReuseMode::StateSnapshot | KvReuseMode::LiveSlotAndSnapshot
    )
}

fn live_retained_prefix_tokens(config: &NativeRuntimeConfig) -> i32 {
    if live_prefix_reuse_enabled(config.cache.mode) {
        config.cache.retained_prefix_tokens
    } else {
        0
    }
}

fn normalize_stop_sequences(stop: Vec<String>) -> Vec<String> {
    let mut stop = stop
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    stop.sort();
    stop.dedup();
    stop
}

fn admit_runtime_residency(config: &NativeRuntimeConfig) -> Result<Option<ResidencyLease>> {
    let raw = backend_observability_json(true)?;
    acquire_residency_lease(config, &raw)
}

fn ensure_context_space(
    shared_context: *mut ffi::llama_context,
    retained_prefix_tokens: i32,
    state: &mut SequenceState,
    seq_id: llama_seq_id,
    new_tokens_needed: i32,
) -> bool {
    if shared_context.is_null() || seq_id < 0 {
        return false;
    }
    if new_tokens_needed <= 0 {
        return true;
    }

    let n_ctx = unsafe { ffi::llama_n_ctx(shared_context) } as i32;
    if n_ctx <= 0 || new_tokens_needed > n_ctx {
        return false;
    }
    if state.n_past + new_tokens_needed <= n_ctx {
        return true;
    }

    let mem = unsafe { ffi::llama_get_memory(shared_context) };
    let n_keep = retained_prefix_tokens.min(state.n_past).max(0);
    let required_discard = state.n_past + new_tokens_needed - n_ctx;
    let max_discard = (state.n_past - n_keep).max(0);
    let n_discard = required_discard.clamp(0, max_discard);

    if n_discard <= 0 {
        if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, 0, -1) } {
            return false;
        }
        state.current_kv_tokens.clear();
        state.n_past = 0;
        return true;
    }

    if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, n_keep, n_keep + n_discard) } {
        return false;
    }
    unsafe {
        ffi::llama_memory_seq_add(mem, seq_id, n_keep + n_discard, -1, -n_discard);
    }

    if state.current_kv_tokens.len() > n_keep as usize {
        let erase_end = cmp::min((n_keep + n_discard) as usize, state.current_kv_tokens.len());
        state.current_kv_tokens.drain(n_keep as usize..erase_end);
    } else {
        state.current_kv_tokens.clear();
    }
    state.n_past = state.current_kv_tokens.len() as i32;

    if state.n_past + new_tokens_needed <= n_ctx {
        return true;
    }

    if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, 0, -1) } {
        return false;
    }
    state.current_kv_tokens.clear();
    state.n_past = 0;
    true
}

#[allow(clippy::too_many_arguments)]
fn prepare_sequence_for_prompt(
    shared_context: *mut ffi::llama_context,
    primary_model: *mut ffi::llama_model,
    retained_prefix_tokens: i32,
    snapshot_prefix_cache: bool,
    decode_token_reserve: i32,
    model_fingerprint: u64,
    session_store: &SessionStore,
    prefix_state_cache: &mut PrefixStateCache,
    prefix_cache_policy: &mut PrefixCachePolicy,
    context_key: &str,
    prompt_tokens: &[llama_token],
    n_tokens_predict: i32,
    state: &mut SequenceState,
    seq_id: llama_seq_id,
    request: &mut GenerateRequest,
    out_prefill_cursor: &mut usize,
) -> bool {
    *out_prefill_cursor = 0;
    if shared_context.is_null() || primary_model.is_null() || seq_id < 0 || prompt_tokens.is_empty()
    {
        return false;
    }

    let mem = unsafe { ffi::llama_get_memory(shared_context) };
    let has_live_tokens = !state.current_kv_tokens.is_empty();
    let live_match_len = if has_live_tokens {
        session_store.compute_lcp_reuse(state, prompt_tokens)
    } else {
        0
    };
    let mut match_len = live_match_len;
    let mut restored_from_prefix_cache = false;

    // Look up the prefix cache via a handle so we never have to clone the
    // entry's state_bytes (potentially hundreds of MB for real models). The
    // handle releases the immutable borrow on `prefix_state_cache` as soon as
    // the lookup function returns, letting us call `restore_by_handle` and
    // read individual fields without paying for a deep clone.
    if snapshot_prefix_cache {
        if let Some(handle) = prefix_state_cache.find_best_prefix_handle(
            model_fingerprint,
            context_key,
            prompt_tokens,
            prefix_cache_policy,
        ) {
            if handle.token_count > live_match_len
                && prefix_state_cache.restore_by_handle(shared_context, seq_id, handle)
            {
                if let Some(entry) = prefix_state_cache.entry_by_handle(handle) {
                    state.current_kv_tokens.clear();
                    state
                        .current_kv_tokens
                        .extend_from_slice(&entry.prefix_tokens);
                    state.n_past = entry.token_count as i32;
                    match_len = entry.token_count.min(prompt_tokens.len());
                    restored_from_prefix_cache = true;
                }
            }
        }
    }

    if !restored_from_prefix_cache && !has_live_tokens {
        unsafe {
            ffi::llama_memory_seq_rm(mem, seq_id, 0, -1);
        }
        state.current_kv_tokens.clear();
        state.n_past = 0;
        match_len = 0;
    }

    // Re-run LCP after we may have replaced `state.current_kv_tokens` from the
    // prefix cache or cleared it above; the second post-`ensure_context_space`
    // call is redundant because that function only ever truncates KV tokens,
    // never lengthens them, so the LCP cannot grow.
    match_len = match_len.max(session_store.compute_lcp_reuse(state, prompt_tokens));
    let tokens_to_add = (prompt_tokens.len() - match_len) as i32;
    let total_needed = tokens_to_add
        + resolve_initial_decode_context_reservation(n_tokens_predict, decode_token_reserve);

    if !ensure_context_space(
        shared_context,
        retained_prefix_tokens,
        state,
        seq_id,
        total_needed,
    ) {
        return false;
    }

    match_len = match_len.min(session_store.compute_lcp_reuse(state, prompt_tokens));
    let allow_partial_kv = !(unsafe { ffi::llama_model_is_recurrent(primary_model) }
        || unsafe { ffi::llama_model_is_hybrid(primary_model) });

    if match_len < state.current_kv_tokens.len() || state.current_kv_tokens.is_empty() {
        if !allow_partial_kv || state.current_kv_tokens.is_empty() {
            unsafe {
                ffi::llama_memory_seq_rm(mem, seq_id, 0, -1);
            }
            state.current_kv_tokens.clear();
            state.n_past = 0;
            match_len = 0;
        } else {
            if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, match_len as i32, -1) } {
                return false;
            }
            state.current_kv_tokens.truncate(match_len);
            state.n_past = match_len as i32;
        }
    }

    if match_len == prompt_tokens.len() && match_len > 0 {
        if !allow_partial_kv {
            unsafe {
                ffi::llama_memory_seq_rm(mem, seq_id, 0, -1);
            }
            state.current_kv_tokens.clear();
            state.n_past = 0;
            match_len = 0;
        } else {
            if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, match_len as i32 - 1, -1) } {
                return false;
            }
            state.current_kv_tokens.truncate(match_len - 1);
            state.n_past = match_len as i32 - 1;
            match_len -= 1;
        }
    }

    request.cache_hits = match_len as i32;
    *out_prefill_cursor = match_len;
    true
}

fn resolve_initial_decode_context_reservation(max_output_tokens: i32, decode_reserve: i32) -> i32 {
    if max_output_tokens <= 0 {
        0
    } else {
        max_output_tokens.min(decode_reserve.max(1))
    }
}

fn ensure_decode_step_context_space(
    shared_context: *mut ffi::llama_context,
    retained_prefix_tokens: i32,
    slot: &mut SlotState,
) -> bool {
    if shared_context.is_null() || slot.session.is_none() {
        return false;
    }
    if slot.generated_tokens.is_empty() {
        return true;
    }
    if slot
        .request()
        .is_some_and(|request| request.is_multimodal_turn)
        && slot.mirror.n_past + 1 > unsafe { ffi::llama_n_ctx(shared_context) } as i32
    {
        return false;
    }
    ensure_context_space(
        shared_context,
        retained_prefix_tokens,
        &mut slot.mirror,
        slot.seq_id,
        1,
    )
}

fn run_multimodal_prefill(
    mtmd_context: *mut ffi::cogent_mtmd_context,
    shared_context: *mut ffi::llama_context,
    vocab: *const ffi::llama_vocab,
    batch_token_budget: i32,
    request_queue: &mut RequestQueue,
    slot: &mut SlotState,
    piece_scratch: &mut Vec<i8>,
) -> bool {
    if mtmd_context.is_null()
        || shared_context.is_null()
        || vocab.is_null()
        || slot.seq_id < 0
        || slot.sampler().is_none()
        || slot.request().is_none()
    {
        return false;
    }

    let Some(request_snapshot) = slot.request().cloned() else {
        return false;
    };
    let Some(multimodal) = request_snapshot.multimodal.as_ref() else {
        return false;
    };
    let seq_id = slot.seq_id;
    let prefill_cursor = slot.prefill_cursor;
    let add_special = slot.mirror.n_past == 0;

    let mut bitmaps = Vec::with_capacity(multimodal.image_buffers.len());
    for buffer in &multimodal.image_buffers {
        let bitmap = unsafe {
            ffi::cogent_mtmd_bitmap_init_from_buf(mtmd_context, buffer.as_ptr(), buffer.len())
        };
        if bitmap.is_null() {
            clear_multimodal_payload(slot);
            return false;
        }
        bitmaps.push(BitmapGuard(bitmap));
    }

    let mut prompt_text = request_snapshot.original_prompt.clone();
    let marker = unsafe { ffi::cogent_mtmd_default_marker() };
    if !marker.is_null() {
        let marker = unsafe { CStr::from_ptr(marker) }.to_string_lossy();
        if !marker.is_empty() {
            let mut marker_count = prompt_text.matches(marker.as_ref()).count();
            if marker_count > bitmaps.len() {
                clear_multimodal_payload(slot);
                return false;
            }
            while marker_count < bitmaps.len() {
                prompt_text.insert_str(0, marker.as_ref());
                marker_count += 1;
            }
        }
    }

    let prompt_text = match CString::new(prompt_text) {
        Ok(prompt_text) => prompt_text,
        Err(_) => {
            clear_multimodal_payload(slot);
            return false;
        }
    };
    let chunks = ChunksGuard(unsafe { ffi::cogent_mtmd_input_chunks_init() });
    if chunks.0.is_null() {
        clear_multimodal_payload(slot);
        return false;
    }
    let bitmap_ptrs: Vec<*const ffi::cogent_mtmd_bitmap> =
        bitmaps.iter().map(|bitmap| bitmap.0.cast_const()).collect();
    let tokenized = unsafe {
        ffi::cogent_mtmd_tokenize(
            mtmd_context,
            chunks.0,
            prompt_text.as_ptr(),
            add_special,
            true,
            bitmap_ptrs.as_ptr(),
            bitmap_ptrs.len(),
        )
    };
    if !tokenized {
        clear_multimodal_payload(slot);
        return false;
    }

    let memory = unsafe { ffi::llama_get_memory(shared_context) };
    if !unsafe { ffi::llama_memory_seq_rm(memory, seq_id, 0, -1) } {
        clear_multimodal_payload(slot);
        return false;
    }

    let prefill_start = Instant::now();
    let mut new_n_past = 0_i32;
    let eval_status = unsafe {
        ffi::cogent_mtmd_eval_chunks(
            mtmd_context,
            shared_context,
            chunks.0,
            prefill_cursor as i32,
            seq_id,
            batch_token_budget,
            true,
            &mut new_n_past,
        )
    };
    let sync_ok = unsafe { ffi::cogent_llama_synchronize(shared_context) };
    let prefill_end = Instant::now();
    clear_multimodal_payload(slot);
    if eval_status != 0 || !sync_ok {
        return false;
    }

    slot.mirror.n_past = new_n_past;
    slot.mirror
        .current_kv_tokens
        .resize(new_n_past.max(0) as usize, 0);
    let multimodal_prefill_ms = duration_ms(prefill_start, prefill_end);
    let multimodal_token_count = new_n_past.max(0);
    let multimodal_processed_tokens = (multimodal_token_count - prefill_cursor as i32).max(0);

    if let Some(request) = slot.request_mut() {
        request.input_tokens = multimodal_token_count;
        request.prefill_tokens += multimodal_processed_tokens;
        request.prefill_ms += multimodal_prefill_ms;
    }
    slot.prefill_cursor = request_snapshot.prompt_tokens.len();

    let sampler = slot.sampler().expect("checked sampler");
    let next_token =
        unsafe { ffi::cogent_common_sampler_sample(sampler.as_ptr(), shared_context, -1) };
    if next_token == ffi::LLAMA_TOKEN_NULL {
        slot.terminal_error_message = "llama_sampler_sample() failed.".to_string();
        return false;
    }
    unsafe {
        ffi::cogent_common_sampler_accept(sampler.as_ptr(), next_token, true);
    }
    if let Some(request) = slot.request_mut() {
        request.first_sampled_token_id = next_token;
        request.first_token_at.get_or_insert_with(Instant::now);
    }
    if unsafe { ffi::llama_vocab_is_eog(vocab, next_token) } {
        slot.terminal_error_message =
            "Model ended generation immediately after multimodal prefill.".to_string();
        return false;
    }

    slot.generated_tokens.push(next_token);
    append_token_piece_to_slot(vocab, next_token, slot, piece_scratch);
    slot.phase = SlotPhase::Streaming;
    if let Some(request) = slot.request_mut() {
        request.lifecycle = GenerateRequestLifecycle::Streaming;
    }
    SlotScheduler::emit_buffered_token_piece(request_queue, slot);

    if slot
        .request()
        .is_some_and(|request| request.cancel_requested)
    {
        slot.terminal_error_message = "Request cancelled.".to_string();
        slot.phase = SlotPhase::Failed;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Cancelled;
        }
        return true;
    }

    let reached_limit = slot.request().is_some_and(|request| {
        request.max_output_tokens > 0
            && slot.generated_tokens.len() >= request.max_output_tokens as usize
    });
    if reached_limit {
        slot.phase = SlotPhase::Completed;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Completed;
        }
    } else {
        slot.phase = SlotPhase::Decode;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Running;
        }
    }

    true
}

struct BitmapGuard(*mut ffi::cogent_mtmd_bitmap);

impl Drop for BitmapGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                ffi::cogent_mtmd_bitmap_free(self.0);
            }
        }
    }
}

struct ChunksGuard(*mut ffi::cogent_mtmd_input_chunks);

impl Drop for ChunksGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                ffi::cogent_mtmd_input_chunks_free(self.0);
            }
        }
    }
}

fn clear_multimodal_payload(slot: &mut SlotState) {
    if let Some(request) = slot.request_mut() {
        request.multimodal = None;
    }
}

fn normalize_runnable_slot_state(
    slot: &mut SlotState,
    shared_context: *mut ffi::llama_context,
    primary_model: *mut ffi::llama_model,
    retained_prefix_tokens: i32,
) -> bool {
    if slot.request().is_none() {
        return true;
    }

    if slot.phase == SlotPhase::Admitted {
        slot.phase = SlotPhase::Prefill;
    }

    if slot.phase == SlotPhase::Prefill
        && slot
            .request()
            .is_some_and(|request| !request.is_multimodal_turn)
        && slot
            .request()
            .is_some_and(|request| slot.prefill_cursor >= request.prompt_tokens.len())
        && slot.mirror.n_past > 0
    {
        slot.phase = SlotPhase::Decode;
    }

    if slot.phase == SlotPhase::Streaming && slot.buffered_output_text.is_empty() {
        if slot
            .request()
            .is_some_and(|request| request.cancel_requested)
        {
            slot.terminal_error_message = "Request cancelled.".to_string();
            slot.phase = SlotPhase::Failed;
            if let Some(request) = slot.request_mut() {
                request.lifecycle = GenerateRequestLifecycle::Cancelled;
            }
            return true;
        }

        let reached_limit = slot.request().is_some_and(|request| {
            request.max_output_tokens > 0
                && slot.generated_tokens.len() >= request.max_output_tokens as usize
        });
        if reached_limit {
            slot.phase = SlotPhase::Completed;
            if let Some(request) = slot.request_mut() {
                request.lifecycle = GenerateRequestLifecycle::Completed;
            }
            return true;
        }

        slot.phase = if slot.generated_tokens.is_empty() {
            SlotPhase::Prefill
        } else {
            SlotPhase::Decode
        };
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Running;
        }
    }

    if slot.phase == SlotPhase::Decode && slot.generated_tokens.is_empty() {
        return recover_decode_seed_state(
            slot,
            shared_context,
            primary_model,
            retained_prefix_tokens,
        );
    }

    true
}

fn recover_decode_seed_state(
    slot: &mut SlotState,
    shared_context: *mut ffi::llama_context,
    _primary_model: *mut ffi::llama_model,
    _retained_prefix_tokens: i32,
) -> bool {
    if slot.phase != SlotPhase::Decode || !slot.generated_tokens.is_empty() {
        return true;
    }

    let Some(request) = slot.request() else {
        return true;
    };
    let max_output_tokens = request.max_output_tokens;
    let prompt_len = request.prompt_tokens.len();

    if max_output_tokens <= 0 {
        slot.phase = SlotPhase::Completed;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Completed;
        }
        return true;
    }
    if prompt_len == 0 {
        slot.terminal_error_message =
            "Prompt tokenization produced no tokens, so decode had no seed token.".to_string();
        slot.phase = SlotPhase::Failed;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Failed;
        }
        return false;
    }
    if slot.prefill_cursor < prompt_len {
        slot.phase = SlotPhase::Prefill;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Running;
        }
        return true;
    }
    if shared_context.is_null() || slot.seq_id < 0 {
        slot.terminal_error_message =
            "Decode slot lost shared context state before its first sampled token.".to_string();
        slot.phase = SlotPhase::Failed;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Failed;
        }
        return false;
    }
    if slot.mirror.n_past <= 0 || slot.mirror.current_kv_tokens.is_empty() {
        slot.prefill_cursor = 0;
        slot.phase = SlotPhase::Prefill;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Running;
        }
        return true;
    }

    let retained_tokens = cmp::min(
        slot.mirror.current_kv_tokens.len(),
        (slot.mirror.n_past - 1).max(0) as usize,
    );
    slot.mirror.current_kv_tokens.truncate(retained_tokens);
    let mem = unsafe { ffi::llama_get_memory(shared_context) };
    if !reconcile_physical_state(&mut slot.mirror, slot.seq_id, mem) {
        slot.terminal_error_message =
            "Failed to reconcile shared KV state for decode seed recovery.".to_string();
        slot.phase = SlotPhase::Failed;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Failed;
        }
        return false;
    }
    slot.prefill_cursor = cmp::min(prompt_len - 1, retained_tokens);
    slot.phase = SlotPhase::Prefill;
    if let Some(request) = slot.request_mut() {
        request.lifecycle = GenerateRequestLifecycle::Running;
    }
    true
}

fn reconcile_physical_state(
    state: &mut SequenceState,
    seq_id: llama_seq_id,
    mem: ffi::llama_memory_t,
) -> bool {
    if mem.is_null() || seq_id < 0 {
        return false;
    }
    if !unsafe { ffi::llama_memory_seq_rm(mem, seq_id, state.current_kv_tokens.len() as i32, -1) } {
        return false;
    }
    state.n_past = state.current_kv_tokens.len() as i32;
    true
}

#[inline]
fn append_token_piece_to_slot(
    vocab: *const ffi::llama_vocab,
    token: llama_token,
    slot: &mut SlotState,
    piece_scratch: &mut Vec<i8>,
) {
    if !token_to_piece_into(vocab, token, false, piece_scratch) {
        slot.terminal_error_message = "Failed to convert sampled token to text piece.".to_string();
        slot.phase = SlotPhase::Failed;
        if let Some(request) = slot.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Failed;
        }
        return;
    }

    // SAFETY: `piece_scratch` holds the freshly written piece bytes; we
    // reinterpret them as &[u8] without copying. The scratch survives the
    // call, so the slice's provenance is valid for the duration of this
    // function.
    let piece_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(piece_scratch.as_ptr().cast::<u8>(), piece_scratch.len())
    };

    // Append directly to the slot's pending-utf8 buffer; this avoids the
    // mem::take + slice-to-Vec dance the original code did per token.
    slot.pending_utf8_bytes.extend_from_slice(piece_bytes);
    let tail_len = incomplete_utf8_tail_length(&slot.pending_utf8_bytes);
    let complete_len = slot.pending_utf8_bytes.len().saturating_sub(tail_len);
    if complete_len > 0 {
        let complete = String::from_utf8_lossy(&slot.pending_utf8_bytes[..complete_len]);
        slot.pending_emission_text.push_str(&complete);
        slot.pending_utf8_bytes.drain(..complete_len);
    }

    if !slot.pending_emission_text.is_empty() {
        slot.buffered_output_text
            .push_str(&std::mem::take(&mut slot.pending_emission_text));
    }
}

fn apply_stop_sequences_to_slot(slot: &mut SlotState) -> bool {
    let stops = match slot.request() {
        Some(r) if !r.stop.is_empty() => r.stop.clone(),
        _ => return false,
    };
    let output_len = slot.output_text.len();
    let stop_index =
        earliest_stop_index_split(&slot.output_text, &slot.buffered_output_text, &stops);
    let Some(stop_index) = stop_index else {
        hold_stop_suffix(slot, &stops);
        return false;
    };

    if stop_index <= output_len {
        slot.output_text.truncate(stop_index);
        slot.buffered_output_text.clear();
    } else {
        truncate_to_char_boundary(&mut slot.buffered_output_text, stop_index - output_len);
    }
    slot.pending_emission_text.clear();
    slot.pending_utf8_bytes.clear();
    slot.phase = SlotPhase::Completed;
    true
}

fn earliest_stop_index_split(output: &str, buffered: &str, stops: &[String]) -> Option<usize> {
    let output_len = output.len();
    stops
        .iter()
        .filter_map(|stop| {
            if let Some(idx) = output.find(stop) {
                return Some(idx);
            }
            buffered.find(stop).map(|idx| idx + output_len)
        })
        .min()
}

fn hold_stop_suffix(slot: &mut SlotState, stops: &[String]) {
    let max_hold = stops
        .iter()
        .map(|stop| stop.len().saturating_sub(1))
        .max()
        .unwrap_or(0);
    if max_hold == 0 || slot.buffered_output_text.len() <= max_hold {
        return;
    }

    let raw_split = slot.buffered_output_text.len() - max_hold;
    let split = floor_char_boundary(&slot.buffered_output_text, raw_split);
    if split == 0 || split >= slot.buffered_output_text.len() {
        return;
    }

    let held = slot.buffered_output_text.split_off(split);
    slot.pending_emission_text.insert_str(0, &held);
}

fn truncate_to_char_boundary(value: &mut String, max_len: usize) {
    let boundary = floor_char_boundary(value, max_len.min(value.len()));
    value.truncate(boundary);
}

fn floor_char_boundary(value: &str, mut index: usize) -> usize {
    index = index.min(value.len());
    while index > 0 && !value.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn flush_pending_utf8(slot: &mut SlotState) {
    if !slot.pending_emission_text.is_empty() {
        slot.buffered_output_text
            .push_str(&std::mem::take(&mut slot.pending_emission_text));
    }
    if slot.pending_utf8_bytes.is_empty() {
        return;
    }
    let pending = String::from_utf8_lossy(&slot.pending_utf8_bytes);
    slot.buffered_output_text.push_str(&pending);
    slot.pending_utf8_bytes.clear();
}

/// Fills `scratch` with the bytes of `token`'s text piece. Returns `false`
/// on error so callers can act without touching `Result` boxing. The scratch
/// vector is reused across calls — `runtime.scratch_token_piece` holds the
/// allocation, so per-token work is just a `resize` + a `truncate`.
#[inline]
fn token_to_piece_into(
    vocab: *const ffi::llama_vocab,
    token: llama_token,
    special: bool,
    scratch: &mut Vec<i8>,
) -> bool {
    scratch.clear();
    if vocab.is_null() || token < 0 {
        return false;
    }

    if scratch.capacity() < 128 {
        scratch.reserve(128 - scratch.capacity());
    }

    loop {
        let capacity = scratch.capacity() as i32;
        scratch.resize(capacity as usize, 0);
        let result = unsafe {
            ffi::llama_token_to_piece(vocab, token, scratch.as_mut_ptr(), capacity, 0, special)
        };
        if result >= 0 && result <= capacity {
            scratch.truncate(result as usize);
            return true;
        }
        if result == 0 || result == i32::MIN {
            scratch.clear();
            return false;
        }
        let needed = result.saturating_abs();
        if needed <= capacity {
            scratch.clear();
            return false;
        }
        scratch.reserve((needed as usize).saturating_sub(scratch.capacity()));
    }
}

fn incomplete_utf8_tail_length(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }
    let max_lookback = data.len().min(4);
    for offset in 1..=max_lookback {
        let byte = data[data.len() - offset];
        if (byte & 0xC0) == 0x80 {
            continue;
        }
        let expected = if (byte & 0x80) == 0 {
            1
        } else if (byte & 0xE0) == 0xC0 {
            2
        } else if (byte & 0xF0) == 0xE0 {
            3
        } else if (byte & 0xF8) == 0xF0 {
            4
        } else {
            return 0;
        };
        if offset >= expected {
            return 0;
        }
        return offset;
    }
    max_lookback
}

fn duration_ms(start: Instant, end: Instant) -> f64 {
    end.saturating_duration_since(start).as_secs_f64() * 1000.0
}

fn fingerprint_path(path: &std::path::Path) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    hasher.finish()
}

fn c_strings_from_args(args: &[String]) -> Result<Vec<CString>> {
    args.iter()
        .map(|arg| CString::new(arg.as_str()).map_err(Error::from))
        .collect()
}

fn c_ptrs_from_strings(args: &[CString]) -> Vec<*const std::os::raw::c_char> {
    args.iter().map(|arg| arg.as_ptr()).collect()
}

fn runtime_command_from_shim_error(
    value: *mut std::os::raw::c_char,
    fallback: &'static str,
) -> Error {
    if value.is_null() {
        return Error::RuntimeCommand(fallback.to_string());
    }
    let result = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        ffi::cogent_free_string(value);
    }
    Error::RuntimeCommand(result)
}

fn owned_shim_string(value: *mut std::os::raw::c_char, name: &'static str) -> Result<String> {
    if value.is_null() {
        return Err(Error::NullPointer(name));
    }
    let result = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        ffi::cogent_free_string(value);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::config::{SchedulerPolicyConfig, SchedulerPolicyMode};

    fn test_runtime(config: NativeRuntimeConfig) -> InferenceRuntime {
        InferenceRuntime {
            config,
            resolved_limits: ResolvedRuntimeLimits::default(),
            residency_lease: None,
            common_init: std::ptr::null_mut(),
            primary_model: std::ptr::null_mut(),
            shared_context: std::ptr::null_mut(),
            chat_templates: std::ptr::null_mut(),
            mtmd_context: std::ptr::null_mut(),
            last_runtime_observability: RuntimeObservabilityMetrics::default(),
            has_last_runtime_observability: false,
            session_store: SessionStore::default(),
            request_queue: RequestQueue::new(),
            slot_scheduler: SlotScheduler::default(),
            batch_planner: BatchPlanner,
            shared_batch_builder: LlamaBatchBuilder::default(),
            prefix_state_cache: PrefixStateCache::default(),
            prefix_cache_policy: PrefixCachePolicy::default(),
            next_request_id: 1,
            model_fingerprint: 0,
            committed_observability_request_ids: HashSet::new(),
            scratch_decode_ready_slots: Vec::new(),
            scratch_prefill_ready_slots: Vec::new(),
            scratch_logits_contributions: Vec::new(),
            scratch_plan: SharedBatchPlan::default(),
            scratch_token_piece: Vec::new(),
            debug_metrics_enabled: false,
            total_decode_ms: 0.0,
            total_prefill_ms: 0.0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_hits: 0,
            total_prefill_tokens: 0,
            sampler_pool: std::collections::HashMap::new(),
            pending_bookkeeping: None,
        }
    }

    #[test]
    fn native_runtime_config_normalizes_policy_limits() {
        let mut config = NativeRuntimeConfig::default();
        config.context.n_parallel = Some(0);
        config.scheduler.prefill_chunk_size = -1;
        config.scheduler.max_running_requests = Some(0);
        config.cache.max_session_entries = 0;
        config.cache.retained_prefix_tokens = -1;
        config.cache.snapshot_interval_tokens = -1;
        config.cache.max_snapshot_entries = 0;
        config.residency.max_gpu_models_per_device = 0;
        config.observability.backend_profiling = true;

        let config = config.normalize();

        assert_eq!(config.context.n_parallel, Some(1));
        assert_eq!(config.scheduler.prefill_chunk_size, 0);
        assert_eq!(config.scheduler.max_running_requests, Some(1));
        assert_eq!(config.cache.max_session_entries, 1);
        assert_eq!(config.cache.retained_prefix_tokens, 0);
        assert_eq!(config.cache.snapshot_interval_tokens, 0);
        assert_eq!(config.cache.max_snapshot_entries, 1);
        assert_eq!(config.residency.max_gpu_models_per_device, 1);
        assert!(config.observability.effective_runtime_metrics());
    }

    #[test]
    fn detects_incomplete_utf8_tail() {
        assert_eq!(incomplete_utf8_tail_length("a".as_bytes()), 0);
        assert_eq!(incomplete_utf8_tail_length(&[0xE2, 0x82]), 2);
        assert_eq!(incomplete_utf8_tail_length(&[0xE2, 0x82, 0xAC]), 0);
    }

    #[test]
    fn adaptive_prefill_chunk_matches_cpp_fair_share() {
        let mut config = NativeRuntimeConfig::default();
        config.scheduler.prefill_chunk_size = 8;
        config.scheduler.policy = SchedulerPolicyConfig {
            mode: SchedulerPolicyMode::Balanced,
            decode_token_reserve: 1,
            enable_adaptive_prefill_chunking: true,
        };
        let runtime = test_runtime(config);

        let chunk = runtime.resolve_prefill_chunk_size_locked(
            SchedulerTickBudget {
                total_token_budget: 9,
                reserved_decode_tokens: 1,
                reserved_prefill_tokens: 8,
                decode_first: true,
            },
            1,
            4,
        );

        assert_eq!(chunk, 2);
    }

    #[test]
    fn scheduler_loop_reports_invalid_when_runtime_is_not_ready() {
        let mut runtime = test_runtime(NativeRuntimeConfig::default());

        let result = runtime.run_scheduler_loop(1, 0, 0, Duration::ZERO);

        assert_eq!(result.status, RequestStepResult::Invalid);
        assert_eq!(result.ticks_executed, 0);
    }

    #[test]
    fn cancel_request_marks_active_slot_clone() {
        let mut runtime = test_runtime(NativeRuntimeConfig::default());
        let mut request = GenerateRequest::new(7, "ctx");
        request.prompt_tokens = vec![1, 2, 3];
        assert!(runtime.request_queue.push(request.clone()));

        runtime.slot_scheduler.resize(1);
        runtime.slot_scheduler.mutable_slots()[0].attach_request(request, SequenceState::default());
        runtime.slot_scheduler.mutable_slots()[0].phase = SlotPhase::Decode;

        assert!(runtime.cancel_request(7));

        assert!(runtime
            .request_queue
            .find(7)
            .is_some_and(|request| request.cancel_requested));
        assert!(runtime.slot_scheduler.slots()[0]
            .request()
            .is_some_and(|request| request.cancel_requested));
    }
}
