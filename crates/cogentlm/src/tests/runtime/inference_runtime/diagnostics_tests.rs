//! Tests the `runtime::inference_runtime::diagnostics` module in `cogentlm`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::diagnostics::NoProgressCounts;
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;
use crate::runtime::request::{GenerateRequest, GenerateRequestLifecycle};
use crate::runtime::scheduler::{
    BatchContribution, BatchContributionKind, SharedBatchPlan, SlotPhase, SlotState,
};

#[test]
fn no_progress_diagnostic_includes_all_counters() {
    let message = NoProgressCounts {
        active: 1,
        decode_ready: 2,
        prefill_ready: 3,
        decode_without_seed: 4,
        emit_without_buffer: 5,
    }
    .to_message();

    assert!(message.contains("active=1"));
    assert!(message.contains("decode_ready=2"));
    assert!(message.contains("prefill_ready=3"));
    assert!(message.contains("decode_without_seed=4"));
    assert!(message.contains("emit_without_buffer=5"));
}

fn slot(id: u32, phase: SlotPhase, prompt_tokens: Vec<i32>) -> SlotState {
    let mut slot = SlotState::new(id as usize);
    let mut request = GenerateRequest::new(id, "ctx");
    request.prompt_tokens = prompt_tokens;
    request.lifecycle = GenerateRequestLifecycle::Running;
    slot.request_id = id;
    slot.request = Some(request);
    slot.phase = phase;
    slot
}

#[test]
fn no_progress_diagnostic_counts_real_slot_states() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let mut decode_ready = slot(1, SlotPhase::Decode, vec![1]);
    decode_ready.generated_tokens.push(10);
    let mut prefill_ready = slot(2, SlotPhase::Prefill, vec![1, 2]);
    prefill_ready.prefill_cursor = 1;
    let decode_without_seed = slot(3, SlotPhase::Decode, vec![1]);
    let emit_without_buffer = slot(4, SlotPhase::EmitBuffered, vec![1]);
    runtime.slot_scheduler.slots = vec![
        decode_ready,
        prefill_ready,
        decode_without_seed,
        emit_without_buffer,
    ];

    let message = runtime.build_no_progress_diagnostic_locked();

    assert!(message.contains("active=4"));
    assert!(message.contains("decode_ready=1"));
    assert!(message.contains("prefill_ready=1"));
    assert!(message.contains("decode_without_seed=1"));
    assert!(message.contains("emit_without_buffer=1"));
}

#[test]
fn fail_plan_slots_ignores_duplicate_and_missing_slot_contributions() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.slot_scheduler.slots = vec![
        slot(1, SlotPhase::Decode, vec![1]),
        slot(2, SlotPhase::Prefill, vec![1, 2]),
    ];
    let mut plan = SharedBatchPlan::default();
    plan.contributions = vec![
        BatchContribution {
            slot_index: 0,
            request_id: 1,
            kind: BatchContributionKind::Decode,
            token: 1,
            position: 0,
            request_logits: true,
        },
        BatchContribution {
            slot_index: 0,
            request_id: 1,
            kind: BatchContributionKind::Decode,
            token: 2,
            position: 1,
            request_logits: true,
        },
        BatchContribution {
            slot_index: 9,
            request_id: 9,
            kind: BatchContributionKind::Prefill,
            token: 3,
            position: 0,
            request_logits: false,
        },
    ];

    runtime.fail_plan_slots(&plan, "no progress");

    assert_eq!(runtime.slot_scheduler.slots[0].phase, SlotPhase::Failed);
    assert_eq!(
        runtime.slot_scheduler.slots[0].terminal_error_message,
        "no progress"
    );
    assert_eq!(runtime.slot_scheduler.slots[1].phase, SlotPhase::Prefill);
}
