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
    runtime.slot_scheduler.slots.push(prefill_slot());
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

fn prefill_slot() -> SlotState {
    let mut slot = SlotState::new(0);
    let mut request = GenerateRequest::new(7, "ctx");
    request.prompt_tokens = vec![1, 2, 3];
    request.cache_mode = KvReuseMode::LiveSlotAndSnapshot;
    slot.request_id = request.id;
    slot.request = Some(request);
    slot.seq_id = 0;
    slot.phase = SlotPhase::Prefill;
    slot.mirror.current_kv_tokens = vec![1, 2];
    slot.mirror.n_past = 2;
    slot
}
