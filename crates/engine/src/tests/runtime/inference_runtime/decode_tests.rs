//! Tests the `runtime::inference_runtime::decode` module in `cogentlm-engine`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use crate::engine::protocol::{EmbedOptions, PoolingType};
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;
use crate::runtime::request::{GenerateRequest, GenerateRequestLifecycle};
use crate::runtime::scheduler::{
    BatchContribution, BatchContributionKind, SharedBatchPlan, SlotPhase, SlotState, TerminalAction,
};

fn request_slot(prompt_tokens: Vec<i32>, phase: SlotPhase) -> SlotState {
    let mut slot = SlotState::new(0);
    let mut request = GenerateRequest::new(7, "ctx");
    request.prompt_tokens = prompt_tokens;
    request.max_output_tokens = 4;
    slot.request_id = request.id;
    slot.request = Some(request);
    slot.seq_id = 0;
    slot.phase = phase;
    slot
}

fn contribution(kind: BatchContributionKind, token: i32) -> BatchContribution {
    BatchContribution {
        slot_index: 0,
        request_id: 7,
        kind,
        token,
        position: 0,
        request_logits: false,
    }
}

#[test]
fn prefill_bookkeeping_advances_slot_and_request_metrics() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime
        .slot_scheduler
        .slots
        .push(request_slot(vec![11], SlotPhase::Prefill));
    let mut plan = SharedBatchPlan::default();
    plan.contributions
        .push(contribution(BatchContributionKind::Prefill, 11));

    runtime.apply_bookkeeping_and_emit(&plan, 1.0, 2.0, 3.0);

    let slot = &runtime.slot_scheduler.slots[0];
    let request = slot.request().expect("request");
    assert_eq!(slot.phase, SlotPhase::Decode);
    assert_eq!(slot.prefill_cursor, 1);
    assert_eq!(slot.mirror.n_past, 1);
    assert_eq!(slot.mirror.current_kv_tokens, vec![11]);
    assert_eq!(request.prefill_tokens, 1);
    assert_eq!(request.prefill_ms, 6.0);
    assert_eq!(request.native_gpu_ms, 1.0);
    assert_eq!(request.native_sync_ms, 2.0);
    assert_eq!(request.native_logic_ms, 3.0);
    assert_eq!(runtime.total_prefill_tokens, 1);
    assert_eq!(runtime.total_prefill_ms, 6.0);
}

#[test]
fn decode_bookkeeping_counts_decode_steps_without_prefill_metrics() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime
        .slot_scheduler
        .slots
        .push(request_slot(vec![11], SlotPhase::Decode));
    let mut plan = SharedBatchPlan::default();
    plan.contributions
        .push(contribution(BatchContributionKind::Decode, 22));

    runtime.apply_bookkeeping_and_emit(&plan, 1.0, 1.0, 1.0);

    let slot = &runtime.slot_scheduler.slots[0];
    let request = slot.request().expect("request");
    assert_eq!(slot.decode_step_count, 1);
    assert_eq!(slot.mirror.current_kv_tokens, vec![22]);
    assert_eq!(request.decode_ms, 3.0);
    assert_eq!(request.prefill_tokens, 0);
    assert_eq!(runtime.total_decode_ms, 3.0);
}

#[test]
fn bookkeeping_fails_slot_when_kv_position_overflows() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let mut slot = request_slot(vec![11], SlotPhase::Decode);
    slot.mirror.n_past = i32::MAX;
    runtime.slot_scheduler.slots.push(slot);
    let mut plan = SharedBatchPlan::default();
    plan.contributions
        .push(contribution(BatchContributionKind::Decode, 22));

    runtime.apply_bookkeeping_and_emit(&plan, 0.0, 0.0, 0.0);

    let slot = &runtime.slot_scheduler.slots[0];
    assert_eq!(slot.phase, SlotPhase::Failed);
    assert_eq!(
        slot.request().expect("request").lifecycle,
        GenerateRequestLifecycle::Failed
    );
    assert!(slot
        .terminal_error_message
        .contains("KV position overflowed"));
}

#[test]
fn bookkeeping_fails_slot_when_prefill_cursor_overflows() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let mut slot = request_slot(vec![11], SlotPhase::Prefill);
    slot.prefill_cursor = usize::MAX;
    runtime.slot_scheduler.slots.push(slot);
    let mut plan = SharedBatchPlan::default();
    plan.contributions
        .push(contribution(BatchContributionKind::Prefill, 11));

    runtime.apply_bookkeeping_and_emit(&plan, 0.0, 0.0, 0.0);

    let slot = &runtime.slot_scheduler.slots[0];
    assert_eq!(slot.phase, SlotPhase::Failed);
    assert!(slot
        .terminal_error_message
        .contains("Prefill cursor overflowed"));
}

#[test]
fn decoder_embedding_read_failure_marks_slot_failed() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.capabilities.embedding_dimensions = 0;
    runtime.capabilities.pooling_type = PoolingType::Mean;
    let mut slot = request_slot(vec![11], SlotPhase::Prefill);
    slot.plan.terminal = TerminalAction::ReadEmbedding;
    slot.request.as_mut().expect("request").embed_options = Some(EmbedOptions::default());
    runtime.slot_scheduler.slots.push(slot);
    let mut plan = SharedBatchPlan::default();
    plan.contributions
        .push(contribution(BatchContributionKind::Prefill, 11));

    runtime.apply_bookkeeping_and_emit(&plan, 0.0, 0.0, 0.0);

    let slot = &runtime.slot_scheduler.slots[0];
    assert_eq!(slot.phase, SlotPhase::Failed);
    assert!(slot
        .terminal_error_message
        .contains("embedding read failed"));
}
