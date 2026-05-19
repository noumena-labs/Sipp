//! Unit tests for the parent module.

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
fn runtime_observability_clamps_saturated_total_counters() {
    let mut config = NativeRuntimeConfig::default();
    config.observability.runtime_metrics = true;
    let mut runtime = test_runtime(config);
    runtime.total_input_tokens = usize::MAX;
    runtime.total_output_tokens = usize::MAX;
    runtime.total_cache_hits = usize::MAX;
    runtime.total_prefill_tokens = usize::MAX;

    let metrics = runtime
        .try_get_runtime_observability()
        .expect("runtime metrics");

    assert_eq!(metrics.input_tokens, i32::MAX);
    assert_eq!(metrics.output_tokens, i32::MAX);
    assert_eq!(metrics.cache_hits, i32::MAX);
    assert_eq!(metrics.prefill_tokens, i32::MAX);
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
