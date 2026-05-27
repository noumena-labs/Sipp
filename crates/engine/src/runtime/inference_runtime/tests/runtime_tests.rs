use std::collections::HashSet;

use crate::engine::protocol::{ModelClass, PoolingType};
use crate::runtime::config::{NativeRuntimeConfig, ResolvedRuntimeLimits};
use crate::runtime::inference_runtime::capabilities::RuntimeModelCapabilities;
use crate::runtime::llama::LlamaBatchBuilder;
use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::request::RequestQueue;
use crate::runtime::scheduler::{BatchPlanner, SharedBatchPlan, SlotScheduler};
use crate::runtime::session::{PrefixCachePolicy, PrefixStateCache, SessionStore};

use super::super::InferenceRuntime;

pub(crate) fn test_runtime(config: NativeRuntimeConfig) -> InferenceRuntime {
    InferenceRuntime {
        config,
        resolved_limits: ResolvedRuntimeLimits::default(),
        capabilities: RuntimeModelCapabilities {
            class: ModelClass::DecoderOnly,
            embedding_dimensions: 1,
            pooling_type: PoolingType::None,
            decoder_start_token: None,
            has_chat_template: false,
            embedding_context: false,
        },
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
        scratch_terminal_sequences: Vec::new(),
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
    }
}
