use std::collections::HashSet;
use std::ffi::CString;
use std::ptr::NonNull;

use cogentlm_sys as ffi;

use crate::backend::ensure_backend_initialized;
use crate::error::{Error, Result};
use crate::runtime::config::{NativeRuntimeConfig, ResolvedRuntimeLimits};
use crate::runtime::llama::LlamaBatchBuilder;
use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::request::{GenerateRequestId, RequestQueue};
use crate::runtime::scheduler::{BatchPlanner, SamplerCacheKey, SharedBatchPlan, SlotScheduler};
use crate::runtime::session::{PrefixCachePolicy, PrefixStateCache, SessionStore};

use super::environment::{
    admit_runtime_residency, resolve_batch_token_budget, snapshot_prefix_cache_enabled,
};
use super::native::{
    c_ptrs_from_strings, c_strings_from_args, resolved_runtime_limits,
    runtime_command_from_shim_error,
};
use super::{
    ffi_arg_count_len, fingerprint_path, nonnegative_i32_to_usize, positive_i32_to_usize,
    InferenceRuntime,
};

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
        let common_params = parse_common_params(model_path_string.as_str(), &config)?;
        let residency_lease = match admit_runtime_residency(&config) {
            Ok(lease) => lease,
            Err(error) => {
                unsafe {
                    ffi::cogent_common_params_free(common_params);
                }
                return Err(error);
            }
        };

        let common_init = init_common_runtime(common_params)?;
        let resolved_limits = resolved_runtime_limits(common_init);
        let handles = init_model_handles(common_init, &config, &model_path_string)?;
        let runtime_parts =
            RuntimeParts::new(&config, resolved_limits.clone(), handles.shared_context)?;
        let debug_metrics_enabled = config.observability.effective_runtime_metrics();

        Ok(Self {
            config,
            resolved_limits,
            residency_lease,
            common_init,
            primary_model: handles.primary_model,
            shared_context: handles.shared_context,
            chat_templates: handles.chat_templates,
            mtmd_context: handles.mtmd_context,
            last_runtime_observability: RuntimeObservabilityMetrics::default(),
            has_last_runtime_observability: false,
            session_store: runtime_parts.session_store,
            request_queue: RequestQueue::new(),
            slot_scheduler: runtime_parts.slot_scheduler,
            batch_planner: BatchPlanner,
            shared_batch_builder: runtime_parts.shared_batch_builder,
            prefix_state_cache: PrefixStateCache::new(
                runtime_parts.max_prefix_cache_entries,
                runtime_parts.max_prefix_cache_bytes,
            ),
            prefix_cache_policy: PrefixCachePolicy::new(runtime_parts.prefix_cache_interval_tokens),
            next_request_id: 1,
            model_fingerprint: fingerprint_path(model_path),
            committed_observability_request_ids: HashSet::<GenerateRequestId>::new(),
            scratch_decode_ready_slots: Vec::with_capacity(runtime_parts.max_sequences),
            scratch_prefill_ready_slots: Vec::with_capacity(runtime_parts.max_sequences),
            scratch_logits_contributions: Vec::with_capacity(runtime_parts.scratch_token_capacity),
            scratch_terminal_sequences: Vec::with_capacity(runtime_parts.max_sequences),
            scratch_plan: SharedBatchPlan::with_capacities(
                runtime_parts.scratch_token_capacity,
                runtime_parts.max_sequences,
            ),
            scratch_token_piece: Vec::with_capacity(128),
            debug_metrics_enabled,
            total_decode_ms: 0.0,
            total_prefill_ms: 0.0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_hits: 0,
            total_prefill_tokens: 0,
            sampler_pool: std::collections::HashMap::<
                SamplerCacheKey,
                Vec<NonNull<ffi::cogent_common_sampler>>,
            >::new(),
        })
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
        self.shared_context = std::ptr::null_mut();
        self.primary_model = std::ptr::null_mut();
        drop(self.residency_lease.take());
    }
}

struct ModelHandles {
    primary_model: *mut ffi::llama_model,
    shared_context: *mut ffi::llama_context,
    chat_templates: *mut ffi::cogent_chat_templates,
    mtmd_context: *mut ffi::cogent_mtmd_context,
}

struct RuntimeParts {
    session_store: SessionStore,
    slot_scheduler: SlotScheduler,
    shared_batch_builder: LlamaBatchBuilder,
    max_sequences: usize,
    scratch_token_capacity: usize,
    max_prefix_cache_entries: usize,
    max_prefix_cache_bytes: usize,
    prefix_cache_interval_tokens: usize,
}

impl RuntimeParts {
    fn new(
        config: &NativeRuntimeConfig,
        resolved_limits: ResolvedRuntimeLimits,
        shared_context: *mut ffi::llama_context,
    ) -> Result<Self> {
        let max_cached_sessions = positive_i32_to_usize(config.cache.max_session_entries);
        let resolved_parallel = resolved_limits.n_parallel.max(1);
        let max_sequences = positive_i32_to_usize(resolved_parallel);
        let mut session_store = SessionStore::new(max_cached_sessions, max_sequences);
        session_store.bind_shared_context(shared_context);

        let mut slot_scheduler = SlotScheduler::default();
        slot_scheduler.resize(max_sequences);

        let mut shared_batch_builder = LlamaBatchBuilder::default();
        let batch_token_budget = resolve_batch_token_budget(shared_context, config);
        shared_batch_builder.ensure_capacity(batch_token_budget, resolved_parallel)?;

        Ok(Self {
            session_store,
            slot_scheduler,
            shared_batch_builder,
            max_sequences,
            scratch_token_capacity: positive_i32_to_usize(batch_token_budget),
            max_prefix_cache_entries: positive_i32_to_usize(config.cache.max_snapshot_entries),
            max_prefix_cache_bytes: config.cache.max_snapshot_bytes,
            prefix_cache_interval_tokens: if snapshot_prefix_cache_enabled(config.cache.mode) {
                nonnegative_i32_to_usize(config.cache.snapshot_interval_tokens)
            } else {
                0
            },
        })
    }
}

fn parse_common_params(
    model_path_string: &str,
    config: &NativeRuntimeConfig,
) -> Result<*mut ffi::cogent_common_params> {
    let c_model_path = CString::new(model_path_string)?;
    let common_args = config.llama_common_args();
    let common_arg_cstrings = c_strings_from_args(&common_args)?;
    let common_arg_ptrs = c_ptrs_from_strings(&common_arg_cstrings);
    let common_arg_count = ffi_arg_count_len(common_arg_ptrs.len())?;
    let mut parse_error = std::ptr::null_mut();
    let common_params = unsafe {
        ffi::cogent_common_params_parse_server(
            c_model_path.as_ptr(),
            common_arg_count,
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
    Ok(common_params)
}

fn init_common_runtime(
    common_params: *mut ffi::cogent_common_params,
) -> Result<*mut ffi::cogent_common_init> {
    let mut init_error = std::ptr::null_mut();
    let common_init =
        unsafe { ffi::cogent_common_init_from_params(common_params, &mut init_error) };
    unsafe {
        ffi::cogent_common_params_free(common_params);
    }
    if common_init.is_null() {
        return Err(runtime_command_from_shim_error(
            init_error,
            "failed to initialize llama.cpp common runtime",
        ));
    }
    Ok(common_init)
}

fn init_model_handles(
    common_init: *mut ffi::cogent_common_init,
    config: &NativeRuntimeConfig,
    model_path_string: &str,
) -> Result<ModelHandles> {
    let primary_model = unsafe { ffi::cogent_common_init_model(common_init) };
    let shared_context = unsafe { ffi::cogent_common_init_context(common_init) };
    if primary_model.is_null() || shared_context.is_null() {
        unsafe {
            ffi::cogent_common_init_free(common_init);
        }
        return Err(Error::ModelLoad {
            path: model_path_string.to_string(),
        });
    }

    let vocab = unsafe { ffi::cogent_common_init_vocab(common_init) };
    if vocab.is_null() {
        unsafe {
            ffi::cogent_common_init_free(common_init);
        }
        return Err(Error::NullPointer("llama_model_get_vocab"));
    }

    let chat_templates =
        unsafe { ffi::cogent_chat_templates_init(primary_model, std::ptr::null()) };
    let mtmd_context = init_multimodal_context(config, primary_model, chat_templates, common_init)?;

    Ok(ModelHandles {
        primary_model,
        shared_context,
        chat_templates,
        mtmd_context,
    })
}

fn init_multimodal_context(
    config: &NativeRuntimeConfig,
    primary_model: *mut ffi::llama_model,
    chat_templates: *mut ffi::cogent_chat_templates,
    common_init: *mut ffi::cogent_common_init,
) -> Result<*mut ffi::cogent_mtmd_context> {
    let Some(projector_path) = config.multimodal.projector_path.as_ref() else {
        return Ok(std::ptr::null_mut());
    };

    let c_mmproj_path = CString::new(projector_path.as_str())?;
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
        return Err(Error::NullPointer("cogent_mtmd_init_from_file"));
    }
    Ok(mtmd)
}
