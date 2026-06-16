//! Tests the `runtime::inference_runtime::prefix_snapshots` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use crate::runtime::config::{KvReuseMode, NativeRuntimeConfig};
use crate::runtime::inference_runtime::runtime_tests::test_runtime;
use crate::runtime::request::GenerateRequest;
use crate::runtime::scheduler::{
    BatchContribution, BatchContributionKind, SharedBatchPlan, SlotPhase, SlotState,
};

use super::decode_seed_snapshot_token_count;

#[test]
fn decode_seed_snapshot_requires_at_least_two_prompt_tokens() {
    assert_eq!(decode_seed_snapshot_token_count(0), None);
    assert_eq!(decode_seed_snapshot_token_count(1), None);
    assert_eq!(decode_seed_snapshot_token_count(2), Some(1));
    assert_eq!(decode_seed_snapshot_token_count(19), Some(18));
}

#[test]
fn capture_prefix_snapshots_queues_live_snapshot_under_decode_pressure() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.scratch_decode_ready_slots.push(1);
    runtime.slot_scheduler.slots.push(prefill_slot(vec![1, 2]));
    let mut plan = SharedBatchPlan::default();
    plan.contributions.push(BatchContribution {
        slot_index: 0,
        request_id: 7,
        kind: BatchContributionKind::Prefill,
        token: 2,
        position: 1,
        request_logits: false,
    });

    runtime.capture_prefix_snapshots(&plan);

    assert_eq!(runtime.kv_cache.pending_prefix_snapshot_count(), 1);
}

#[test]
fn capture_prefix_snapshots_queues_from_full_prompt_state() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime
        .slot_scheduler
        .slots
        .push(prefill_slot(vec![1, 2, 3]));
    let mut plan = SharedBatchPlan::default();
    plan.contributions.push(BatchContribution {
        slot_index: 0,
        request_id: 7,
        kind: BatchContributionKind::Prefill,
        token: 3,
        position: 2,
        request_logits: true,
    });

    runtime.capture_prefix_snapshots(&plan);

    assert_eq!(runtime.kv_cache.pending_prefix_snapshot_count(), 1);
}

fn prefill_slot(current_kv_tokens: Vec<i32>) -> SlotState {
    let mut slot = SlotState::new(0);
    let mut request = GenerateRequest::new(7, "ctx");
    request.prompt_tokens = vec![1, 2, 3];
    request.cache_mode = KvReuseMode::LiveSlotAndSnapshot;
    slot.request_id = request.id;
    slot.request = Some(request);
    slot.seq_id = 0;
    slot.phase = SlotPhase::Prefill;
    slot.mirror.n_past = i32::try_from(current_kv_tokens.len()).unwrap();
    slot.mirror.current_kv_tokens = current_kv_tokens;
    slot
}
