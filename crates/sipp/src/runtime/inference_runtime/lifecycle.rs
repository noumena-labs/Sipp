use std::collections::HashSet;

use crate::shard::inspect_gguf_metadata_path;

use crate::backend::ensure_backend_initialized;
use crate::engine::protocol::{ModelClass, PoolingType};
use crate::error::{Error, Result};
use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::config::{NativeRuntimeConfig, ResolvedRuntimeLimits};
use crate::runtime::llama::LlamaBatchBuilder;
use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::request::{GenerateRequestId, RequestQueue};
use crate::runtime::scheduler::{
    BatchPlanner, SamplerCacheKey, SamplerHandle, SharedBatchPlan, SlotScheduler,
};
use crate::runtime::session::KvCacheManager;

use super::capabilities::RuntimeModelCapabilities;
use super::environment::{admit_runtime_residency, snapshot_prefix_cache_enabled};
use super::{fingerprint_path, nonnegative_i32_to_usize, positive_i32_to_usize, InferenceRuntime};

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

        let mut config = config.normalize();
        let model_class = probe_model_class(model_path, model_path_string.as_str())?;
        apply_model_class_defaults(&mut config, model_class)?;
        let residency_lease = match admit_runtime_residency(&config) {
            Ok(lease) => lease,
            Err(error) => return Err(error),
        };

        let mut native_runtime = load_native_runtime(&model_path_string, &config)?;
        let resolved_limits = native_runtime.resolved_limits();
        init_runtime_extensions(&mut native_runtime, &config, &model_path_string)?;
        let runtime_parts = RuntimeParts::new(&config, resolved_limits.clone())?;
        let capabilities = build_capabilities(&config, &native_runtime)?;

        Ok(Self {
            config,
            resolved_limits,
            capabilities,
            native_runtime,
            _residency_lease: residency_lease,
            last_runtime_observability: RuntimeObservabilityMetrics::default(),
            has_last_runtime_observability: false,
            kv_cache: runtime_parts.kv_cache,
            request_queue: RequestQueue::new(),
            slot_scheduler: runtime_parts.slot_scheduler,
            batch_planner: BatchPlanner,
            shared_batch_builder: runtime_parts.shared_batch_builder,
            next_request_id: 1,
            model_fingerprint: fingerprint_path(model_path),
            committed_observability_request_ids: HashSet::<GenerateRequestId>::new(),
            scratch_decode_ready_slots: Vec::with_capacity(runtime_parts.max_sequences),
            scratch_prefill_ready_slots: Vec::with_capacity(runtime_parts.max_sequences),
            scratch_logits_contributions: Vec::with_capacity(runtime_parts.scratch_token_capacity),
            scratch_embedding_read_slots: Vec::with_capacity(runtime_parts.max_sequences),
            scratch_encoder_slots: Vec::with_capacity(runtime_parts.max_sequences),
            scratch_plan: SharedBatchPlan::with_capacities(
                runtime_parts.scratch_token_capacity,
                runtime_parts.max_sequences,
            ),
            scratch_token_piece: Vec::with_capacity(128),
            total_decode_ms: 0.0,
            total_prefill_ms: 0.0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_hits: 0,
            total_prefill_tokens: 0,
            sampler_pool: std::collections::HashMap::<SamplerCacheKey, Vec<SamplerHandle>>::new(),
            resident_backend_samplers: std::collections::HashMap::new(),
        })
    }
}

impl Drop for InferenceRuntime {
    fn drop(&mut self) {
        self.detach_all_backend_samplers_locked();
        self.slot_scheduler.resize(0, &mut self.kv_cache);
        self.kv_cache.evict_all_active_and_idle();

        self.sampler_pool.clear();
        self.resident_backend_samplers.clear();
    }
}

struct RuntimeParts {
    kv_cache: KvCacheManager,
    slot_scheduler: SlotScheduler,
    shared_batch_builder: LlamaBatchBuilder,
    max_sequences: usize,
    scratch_token_capacity: usize,
}

impl RuntimeParts {
    fn new(config: &NativeRuntimeConfig, resolved_limits: ResolvedRuntimeLimits) -> Result<Self> {
        let resolved_parallel = resolved_limits.n_parallel.max(1);
        let max_sequences = positive_i32_to_usize(resolved_parallel);
        let prefix_cache_interval_tokens = if snapshot_prefix_cache_enabled(config.cache.mode) {
            nonnegative_i32_to_usize(config.cache.snapshot_interval_tokens)
        } else {
            0
        };
        let mut kv_cache = KvCacheManager::with_prefix_cache(
            max_sequences,
            positive_i32_to_usize(config.cache.max_snapshot_entries),
            config.cache.max_snapshot_bytes,
            prefix_cache_interval_tokens,
        );

        let mut slot_scheduler = SlotScheduler::default();
        slot_scheduler.resize(max_sequences, &mut kv_cache);

        let mut shared_batch_builder = LlamaBatchBuilder::default();
        let batch_token_budget = config
            .context
            .n_batch
            .unwrap_or(resolved_limits.n_batch)
            .max(1);
        shared_batch_builder.ensure_capacity(batch_token_budget, resolved_parallel)?;

        Ok(Self {
            kv_cache,
            slot_scheduler,
            shared_batch_builder,
            max_sequences,
            scratch_token_capacity: positive_i32_to_usize(batch_token_budget),
        })
    }
}

fn load_native_runtime(
    model_path_string: &str,
    config: &NativeRuntimeConfig,
) -> Result<NativeRuntimeHandle> {
    let common_args = config.llama_common_args();
    let common_args_json = serde_json::to_string(&common_args)
        .map_err(|error| Error::RuntimeCommand(error.to_string()))?;
    NativeRuntimeHandle::load(model_path_string, &common_args_json)
}

fn init_runtime_extensions(
    native_runtime: &mut NativeRuntimeHandle,
    config: &NativeRuntimeConfig,
    model_path_string: &str,
) -> Result<()> {
    if native_runtime.n_ctx() <= 0 || native_runtime.n_batch() <= 0 {
        return Err(Error::ModelLoad {
            path: model_path_string.to_string(),
        });
    }
    init_multimodal_context(config, native_runtime)?;
    Ok(())
}

fn probe_model_class(model_path: &std::path::Path, model_path_string: &str) -> Result<ModelClass> {
    let metadata = inspect_gguf_metadata_path(model_path).map_err(|_| Error::ModelLoad {
        path: model_path_string.to_string(),
    })?;
    Ok(metadata
        .as_ref()
        .and_then(|metadata| metadata.general_architecture.as_deref())
        .map(ModelClass::from_architecture)
        .unwrap_or(ModelClass::DecoderOnly))
}

/// Apply class-specific defaults to the runtime config and reject
/// combinations llama.cpp cannot satisfy. Runs before `parse_common_params`
/// so the resulting `--embedding` / `--parallel` flags are emitted correctly.
fn apply_model_class_defaults(config: &mut NativeRuntimeConfig, class: ModelClass) -> Result<()> {
    match class {
        ModelClass::DecoderOnly => {}
        ModelClass::EncoderOnly => {
            if config.context.embeddings.is_none() {
                config.context.embeddings = Some(true);
            }
        }
        ModelClass::EncoderDecoder => {
            if config.context.embeddings == Some(true) {
                return Err(Error::UnsupportedOperation {
                    operation: "load",
                    reason: "encoder-decoder models do not support embedding output".to_string(),
                });
            }
            if config.context.n_parallel.unwrap_or(1) > 1 {
                return Err(Error::UnsupportedOperation {
                    operation: "load",
                    reason: "encoder-decoder models require n_parallel=1 (llama.cpp \
                             stores cross-attention state per context)"
                        .to_string(),
                });
            }
        }
    }
    Ok(())
}

fn build_capabilities(
    config: &NativeRuntimeConfig,
    native_runtime: &NativeRuntimeHandle,
) -> Result<RuntimeModelCapabilities> {
    // Probe the raw GGUF metadata, not the common_chat_templates fallback —
    // we want "was the model trained with a chat template" (so chat() is
    // semantically meaningful), not "can llama.cpp synthesize a template".
    let has_chat_template = native_runtime.has_chat_template();
    let pooling_raw = native_runtime.pooling_type();
    let pooling_type =
        PoolingType::from_llama_value(pooling_raw).ok_or_else(|| Error::UnsupportedOperation {
            operation: "load",
            reason: format!("unsupported llama.cpp pooling type {pooling_raw}"),
        })?;
    let decoder_start_token = match native_runtime.decoder_start_token() {
        token if token >= 0 => Some(token),
        _ => None,
    };
    let n_embd_out = native_runtime.n_embd_out();
    let n_cls_out = native_runtime.n_cls_out();
    Ok(RuntimeModelCapabilities {
        class: model_class_from_init(native_runtime)?,
        embedding_dimensions: embedding_dimensions(pooling_type, n_embd_out, n_cls_out),
        pooling_type,
        decoder_start_token,
        has_chat_template,
        embedding_context: config.context.embeddings == Some(true),
    })
}

fn embedding_dimensions(pooling_type: PoolingType, n_embd_out: i32, n_cls_out: i32) -> i32 {
    if pooling_type == PoolingType::Rank {
        n_cls_out
    } else {
        n_embd_out
    }
}

fn model_class_from_init(native_runtime: &NativeRuntimeHandle) -> Result<ModelClass> {
    let has_encoder = native_runtime.has_encoder();
    let has_decoder = native_runtime.has_decoder();
    match (has_encoder, has_decoder) {
        (true, true) => Ok(ModelClass::EncoderDecoder),
        (true, false) => Ok(ModelClass::EncoderOnly),
        (false, true) => Ok(ModelClass::DecoderOnly),
        (false, false) => Err(Error::UnsupportedOperation {
            operation: "load",
            reason: "loaded model exposes neither encoder nor decoder".to_string(),
        }),
    }
}

fn init_multimodal_context(
    config: &NativeRuntimeConfig,
    native_runtime: &mut NativeRuntimeHandle,
) -> Result<()> {
    let Some(projector_path) = config.multimodal.projector_path.as_ref() else {
        return Ok(());
    };

    let use_gpu = config.multimodal.use_gpu.unwrap_or(true);
    let init_ok = native_runtime.init_mtmd(
        projector_path,
        use_gpu,
        config.context.n_threads.unwrap_or(0),
    );
    if !init_ok || !native_runtime.mtmd_support_vision() {
        return Err(Error::NullPointer("sipp_mtmd_init_from_file"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/lifecycle_tests.rs"]
mod lifecycle_tests;
