//! Tests the `runtime::inference_runtime::slot` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::super::sampler::ResidentBackendSampler;
use super::recovery::normalize_runnable_slot_state;
use super::run_initial_prefill;
use super::sampler_attach::ensure_slot_sampler;
use crate::native_bridge::{NativeRuntimeHandle, SamplerHandle};
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;
use crate::runtime::request::RequestQueue;
use crate::runtime::request::{GenerateRequest, GenerateRequestLifecycle};
use crate::runtime::scheduler::{SamplerCacheKey, SlotPhase, SlotState, TerminalAction};
use crate::runtime::session::KvCacheManager;

fn decode_slot(prompt_tokens: Vec<i32>, max_output_tokens: i32) -> SlotState {
    let mut slot = SlotState::new(0);
    let mut request = GenerateRequest::new(1, "ctx");
    request.prompt_tokens = prompt_tokens;
    request.max_output_tokens = max_output_tokens;
    slot.request = Some(request);
    slot.seq_id = 0;
    slot.phase = SlotPhase::Decode;
    slot.prefill_cursor = slot
        .request()
        .map(|request| request.prompt_tokens.len())
        .unwrap_or_default();
    slot
}

#[test]
fn admitted_slots_transition_to_prefill() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut slot = decode_slot(vec![1, 2], 4);
    slot.phase = SlotPhase::Admitted;
    slot.prefill_cursor = 0;

    assert!(normalize_runnable_slot_state(&mut slot, &mut runtime, 0));

    assert_eq!(slot.phase, SlotPhase::Prefill);
}

#[test]
fn empty_emit_buffer_respects_cancel_requests() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut slot = decode_slot(vec![1, 2], 4);
    slot.phase = SlotPhase::EmitBuffered;
    slot.request_mut().expect("request").cancel_requested = true;

    assert!(normalize_runnable_slot_state(&mut slot, &mut runtime, 0));

    assert_eq!(slot.phase, SlotPhase::Failed);
    assert_eq!(
        slot.request().expect("request").lifecycle,
        GenerateRequestLifecycle::Cancelled
    );
}

#[test]
fn decode_without_seed_completes_when_no_output_is_requested() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut slot = decode_slot(vec![1, 2], 0);

    assert!(normalize_runnable_slot_state(&mut slot, &mut runtime, 0));

    assert_eq!(slot.phase, SlotPhase::Completed);
    assert_eq!(
        slot.request().expect("request").lifecycle,
        GenerateRequestLifecycle::Completed
    );
}

#[test]
fn decode_without_seed_fails_for_empty_prompt() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut slot = decode_slot(Vec::new(), 4);

    assert!(!normalize_runnable_slot_state(&mut slot, &mut runtime, 0));

    assert_eq!(slot.phase, SlotPhase::Failed);
    assert!(slot
        .terminal_error_message
        .contains("Prompt tokenization produced no tokens"));
}

#[test]
fn decode_without_seed_falls_back_to_prefill_when_cursor_is_short() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut slot = decode_slot(vec![1, 2, 3], 4);
    slot.prefill_cursor = 1;

    assert!(normalize_runnable_slot_state(&mut slot, &mut runtime, 0));

    assert_eq!(slot.phase, SlotPhase::Prefill);
    assert_eq!(
        slot.request().expect("request").lifecycle,
        GenerateRequestLifecycle::Running
    );
}

#[test]
fn decode_without_seed_restarts_prefill_when_kv_mirror_is_empty() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut slot = decode_slot(vec![1, 2, 3], 4);
    slot.mirror.n_past = 0;
    slot.mirror.current_kv_tokens.clear();

    assert!(normalize_runnable_slot_state(&mut slot, &mut runtime, 0));

    assert_eq!(slot.phase, SlotPhase::Prefill);
    assert_eq!(slot.prefill_cursor, 0);
}

#[test]
fn decode_without_seed_fails_when_physical_kv_reconcile_fails() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut slot = decode_slot(vec![1, 2, 3], 4);
    slot.mirror.n_past = 3;
    slot.mirror.current_kv_tokens = vec![1, 2, 3];

    assert!(!normalize_runnable_slot_state(&mut slot, &mut runtime, 0));

    assert_eq!(slot.phase, SlotPhase::Failed);
    assert!(slot
        .terminal_error_message
        .contains("Failed to reconcile shared KV state"));
}

#[test]
fn normalize_slots_for_tick_cancels_requested_slots_before_runtime_work() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let mut slot = decode_slot(vec![1, 2], 4);
    slot.phase = SlotPhase::Prefill;
    slot.request_mut().expect("request").cancel_requested = true;
    runtime.slot_scheduler.slots.push(slot);

    runtime.normalize_slots_for_tick();

    let slot = &runtime.slot_scheduler.slots[0];
    assert_eq!(slot.phase, SlotPhase::Failed);
    assert_eq!(
        slot.request().expect("request").lifecycle,
        GenerateRequestLifecycle::Cancelled
    );
}

#[test]
fn embedding_terminal_slots_skip_sampler_creation() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let mut slot = decode_slot(vec![1, 2], 0);
    slot.phase = SlotPhase::Prefill;
    slot.prefill_cursor = 1;
    slot.plan.terminal = TerminalAction::ReadEmbedding;
    runtime.slot_scheduler.slots.push(slot);

    runtime.normalize_slots_for_tick();

    let slot = &runtime.slot_scheduler.slots[0];
    assert!(slot.sampler.is_none());
    assert_eq!(
        slot.request().expect("request").lifecycle,
        GenerateRequestLifecycle::Running
    );
}

#[test]
fn initial_text_prefill_failure_marks_slot_and_request_failed() {
    let mut slot = decode_slot(vec![1, 2, 3], 4);
    slot.phase = SlotPhase::Prefill;
    slot.prefill_cursor = 0;
    let mut native_runtime = NativeRuntimeHandle::empty_for_tests();
    let config = NativeRuntimeConfig::default();
    let mut kv_cache = KvCacheManager::default();
    let mut total_cache_hits = 0;
    let mut request_queue = RequestQueue::new();
    let mut scratch = Vec::new();

    assert!(!run_initial_prefill(
        &mut slot,
        &mut native_runtime,
        &config,
        0,
        &mut kv_cache,
        &mut total_cache_hits,
        &mut request_queue,
        &mut scratch,
    ));

    assert_eq!(slot.phase, SlotPhase::Failed);
    assert!(slot
        .terminal_error_message
        .contains("Failed to prepare sequence for prompt reuse"));
    assert_eq!(
        slot.request().expect("request").lifecycle,
        GenerateRequestLifecycle::Failed
    );
}

#[test]
fn ensure_slot_sampler_reuses_matching_pooled_sampler_without_native_creation() {
    let mut slot = decode_slot(vec![1, 2], 4);
    let mut native_runtime = NativeRuntimeHandle::empty_for_tests();
    let config = NativeRuntimeConfig::default();
    let sampling_json = config
        .try_sampling_json_with_override(None)
        .expect("sampling json");
    let key = SamplerCacheKey {
        sampling_json,
        grammar: String::new(),
        json_schema: String::new(),
    };
    let mut sampler_pool = std::collections::HashMap::new();
    let mut resident_backend_samplers = std::collections::HashMap::new();
    sampler_pool.insert(key.clone(), vec![SamplerHandle::empty_for_tests()]);

    assert!(ensure_slot_sampler(
        &mut slot,
        &mut native_runtime,
        &config,
        &mut sampler_pool,
        &mut resident_backend_samplers
    ));

    assert!(slot.sampler.is_some());
    assert_eq!(slot.sampler_key, Some(key.clone()));
    assert!(sampler_pool.get(&key).is_some_and(Vec::is_empty));
    assert!(!slot.backend_sampler_attached);
}

#[test]
fn ensure_slot_sampler_reuses_matching_resident_sampler_without_native_attach() {
    let mut slot = decode_slot(vec![1, 2], 4);
    let mut native_runtime = NativeRuntimeHandle::empty_for_tests();
    let config = NativeRuntimeConfig::default();
    let sampling_json = config
        .try_sampling_json_with_override(None)
        .expect("sampling json");
    let key = SamplerCacheKey {
        sampling_json,
        grammar: String::new(),
        json_schema: String::new(),
    };
    let mut sampler_pool = std::collections::HashMap::new();
    let mut resident_backend_samplers = std::collections::HashMap::new();
    resident_backend_samplers.insert(
        slot.seq_id,
        ResidentBackendSampler {
            key: key.clone(),
            sampler: SamplerHandle::empty_for_tests(),
        },
    );

    assert!(ensure_slot_sampler(
        &mut slot,
        &mut native_runtime,
        &config,
        &mut sampler_pool,
        &mut resident_backend_samplers
    ));

    assert!(slot.sampler.is_some());
    assert_eq!(slot.sampler_key, Some(key));
    assert!(slot.backend_sampler_attached);
    assert!(resident_backend_samplers.is_empty());
    assert!(sampler_pool.is_empty());
}

#[test]
fn ensure_slot_sampler_drops_mismatched_resident_sampler_before_creation() {
    let mut slot = decode_slot(vec![1, 2], 4);
    let mut native_runtime = NativeRuntimeHandle::empty_for_tests();
    let config = NativeRuntimeConfig::default();
    let mut sampler_pool = std::collections::HashMap::new();
    let mut resident_backend_samplers = std::collections::HashMap::new();
    resident_backend_samplers.insert(
        slot.seq_id,
        ResidentBackendSampler {
            key: SamplerCacheKey {
                sampling_json: r#"{"temperature":0.1}"#.to_string(),
                grammar: String::new(),
                json_schema: String::new(),
            },
            sampler: SamplerHandle::empty_for_tests(),
        },
    );

    assert!(!ensure_slot_sampler(
        &mut slot,
        &mut native_runtime,
        &config,
        &mut sampler_pool,
        &mut resident_backend_samplers
    ));

    assert!(resident_backend_samplers.is_empty());
    assert_eq!(
        slot.terminal_error_message,
        "Failed to create per-slot sampler."
    );
}

#[test]
fn ensure_slot_sampler_reports_plain_creation_failure_without_grammar() {
    let mut slot = decode_slot(vec![1, 2], 4);
    let mut native_runtime = NativeRuntimeHandle::empty_for_tests();
    let config = NativeRuntimeConfig::default();
    let mut sampler_pool = std::collections::HashMap::new();
    let mut resident_backend_samplers = std::collections::HashMap::new();

    assert!(!ensure_slot_sampler(
        &mut slot,
        &mut native_runtime,
        &config,
        &mut sampler_pool,
        &mut resident_backend_samplers
    ));

    assert_eq!(slot.phase, SlotPhase::Failed);
    assert_eq!(
        slot.terminal_error_message,
        "Failed to create per-slot sampler."
    );
}

#[test]
fn ensure_slot_sampler_reports_grammar_creation_failure_with_grammar() {
    let mut slot = decode_slot(vec![1, 2], 4);
    slot.request_mut().expect("request").grammar = "root ::= \"a\"".to_string();
    let mut native_runtime = NativeRuntimeHandle::empty_for_tests();
    let config = NativeRuntimeConfig::default();
    let mut sampler_pool = std::collections::HashMap::new();
    let mut resident_backend_samplers = std::collections::HashMap::new();

    assert!(!ensure_slot_sampler(
        &mut slot,
        &mut native_runtime,
        &config,
        &mut sampler_pool,
        &mut resident_backend_samplers
    ));

    assert_eq!(slot.phase, SlotPhase::Failed);
    assert_eq!(
        slot.terminal_error_message,
        "Failed to create per-slot grammar sampler."
    );
}
