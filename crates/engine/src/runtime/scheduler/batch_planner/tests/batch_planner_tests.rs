//! Unit tests for the parent module.

use crate::runtime::config::{KvReuseMode, SchedulerTickBudget};
use crate::runtime::request::GenerateRequest;
use crate::runtime::scheduler::{
    BatchContributionKind, BatchPlanner, SharedBatchPlan, SlotPhase, SlotState,
};
use crate::runtime::session::KvCacheAdmission;

fn request(id: u32, prompt_tokens: Vec<i32>, max_output_tokens: i32) -> GenerateRequest {
    let mut request = GenerateRequest::new(id, format!("ctx-{id}"));
    request.prompt_tokens = prompt_tokens;
    request.max_output_tokens = max_output_tokens;
    request
}

fn attached_slot(slot_id: usize, request: GenerateRequest) -> SlotState {
    let mut slot = SlotState::new(slot_id);
    slot.attach_request(request, KvCacheAdmission::default());
    slot
}

#[test]
fn returns_empty_plan_when_budget_is_zero() {
    let planner = BatchPlanner;
    let slots = vec![attached_slot(0, request(1, vec![1, 2], 4))];
    let plan = planner.build_policy_batch(&slots, &[], &[0], SchedulerTickBudget::default(), 0);
    assert!(plan.contributions.is_empty());
}

#[test]
fn schedules_decode_contributions_before_prefill() {
    let planner = BatchPlanner;
    let mut decode_slot = attached_slot(0, request(1, vec![1], 4));
    decode_slot.generated_tokens = vec![9];
    decode_slot.mirror.n_past = 3;
    let prefill_slot = attached_slot(1, request(2, vec![4, 5], 4));
    let slots = vec![decode_slot, prefill_slot];
    let budget = SchedulerTickBudget {
        total_token_budget: 3,
        reserved_decode_tokens: 1,
        reserved_prefill_tokens: 2,
        decode_first: true,
    };

    let plan = planner.build_policy_batch(&slots, &[0], &[1], budget, 0);

    assert_eq!(plan.decode_token_count, 1);
    assert_eq!(plan.prefill_token_count, 2);
    assert_eq!(plan.contributions[0].kind, BatchContributionKind::Decode);
    assert_eq!(plan.contributions[0].token, 9);
    assert_eq!(plan.contributions[0].position, 3);
    assert!(plan.contributions[0].request_logits);
    assert_eq!(plan.contributions[1].kind, BatchContributionKind::Prefill);
}

#[test]
fn occupied_slot_count_handles_sparse_high_slot_indexes() {
    let planner = BatchPlanner;
    let mut slots = (0..130).map(SlotState::new).collect::<Vec<_>>();
    for &slot_index in &[65, 65, 129] {
        let mut slot = attached_slot(slot_index, request(slot_index as u32, vec![1], 4));
        slot.generated_tokens = vec![slot_index as i32];
        slots[slot_index] = slot;
    }
    let budget = SchedulerTickBudget {
        total_token_budget: 3,
        reserved_decode_tokens: 3,
        reserved_prefill_tokens: 0,
        decode_first: true,
    };

    let plan = planner.build_policy_batch(&slots, &[65, 65, 129], &[], budget, 0);

    assert_eq!(plan.decode_token_count, 3);
    assert_eq!(plan.occupied_slot_count, 2);
}

#[test]
fn occupied_overflow_scratch_is_reused_across_ticks() {
    let planner = BatchPlanner;
    let mut plan = SharedBatchPlan::default();
    let mut slots = (0..130).map(SlotState::new).collect::<Vec<_>>();
    for &slot_index in &[65, 129] {
        let mut slot = attached_slot(slot_index, request(slot_index as u32, vec![1], 4));
        slot.generated_tokens = vec![slot_index as i32];
        slots[slot_index] = slot;
    }
    let budget = SchedulerTickBudget {
        total_token_budget: 2,
        reserved_decode_tokens: 2,
        reserved_prefill_tokens: 0,
        decode_first: true,
    };

    planner.build_policy_batch_into(&mut plan, &slots, &[65, 129], &[], budget, 0);
    let retained_capacity = plan.occupied_overflow_slots.capacity();
    assert_eq!(plan.occupied_slot_count, 2);
    assert!(retained_capacity >= 2);

    planner.build_policy_batch_into(&mut plan, &slots, &[65, 129], &[], budget, 0);

    assert_eq!(plan.occupied_slot_count, 2);
    assert_eq!(plan.occupied_overflow_slots.capacity(), retained_capacity);
}

#[test]
fn chunked_single_slot_prefill_revisits_without_duplicate_positions() {
    let planner = BatchPlanner;
    let slots = vec![attached_slot(0, request(1, vec![10, 11, 12, 13, 14], 4))];
    let budget = SchedulerTickBudget {
        total_token_budget: 5,
        reserved_decode_tokens: 0,
        reserved_prefill_tokens: 5,
        decode_first: true,
    };

    let plan = planner.build_policy_batch(&slots, &[], &[0], budget, 2);
    let positions: Vec<i32> = plan
        .contributions
        .iter()
        .map(|contribution| contribution.position)
        .collect();

    assert_eq!(positions, vec![0, 1, 2, 3, 4]);
    assert_eq!(plan.prefill_token_count, 5);
    assert_eq!(
        plan.contributions.last().map(|c| c.request_logits),
        Some(true)
    );
    assert!(plan.contributions[..plan.contributions.len() - 1]
        .iter()
        .all(|contribution| !contribution.request_logits));
}

#[test]
fn snapshot_mode_prefill_pauses_at_decode_seed_boundary() {
    let planner = BatchPlanner;
    let mut request = request(1, vec![10, 11, 12, 13], 4);
    request.cache_mode = KvReuseMode::LiveSlotAndSnapshot;
    let slots = vec![attached_slot(0, request)];
    let budget = SchedulerTickBudget {
        total_token_budget: 4,
        reserved_decode_tokens: 0,
        reserved_prefill_tokens: 4,
        decode_first: true,
    };

    let plan = planner.build_policy_batch(&slots, &[], &[0], budget, 0);
    let positions: Vec<i32> = plan
        .contributions
        .iter()
        .map(|contribution| contribution.position)
        .collect();

    assert_eq!(positions, vec![0, 1, 2]);
    assert_eq!(plan.prefill_token_count, 3);
    assert!(plan.contributions.iter().all(|item| !item.request_logits));
}

#[test]
fn live_only_prefill_does_not_pause_at_decode_seed_boundary() {
    let planner = BatchPlanner;
    let mut request = request(1, vec![10, 11, 12, 13], 4);
    request.cache_mode = KvReuseMode::LiveSlotPrefix;
    let slots = vec![attached_slot(0, request)];
    let budget = SchedulerTickBudget {
        total_token_budget: 4,
        reserved_decode_tokens: 0,
        reserved_prefill_tokens: 4,
        decode_first: true,
    };

    let plan = planner.build_policy_batch(&slots, &[], &[0], budget, 0);

    assert_eq!(plan.prefill_token_count, 4);
    assert_eq!(
        plan.contributions.last().map(|item| item.request_logits),
        Some(true)
    );
}

#[test]
fn prefill_fair_share_recomputes_after_each_slice() {
    let planner = BatchPlanner;
    let slots = vec![
        attached_slot(0, request(1, vec![1, 2, 3], 4)),
        attached_slot(1, request(2, vec![4, 5, 6], 4)),
    ];
    let budget = SchedulerTickBudget {
        total_token_budget: 4,
        reserved_decode_tokens: 0,
        reserved_prefill_tokens: 4,
        decode_first: true,
    };

    let plan = planner.build_policy_batch(&slots, &[], &[0, 1], budget, 0);
    let slots_in_order: Vec<usize> = plan
        .contributions
        .iter()
        .map(|contribution| contribution.slot_index)
        .collect();

    assert_eq!(slots_in_order, vec![0, 0, 1, 0]);
}

#[test]
fn apply_decode_results_advances_prefill_cursor_to_decode() {
    let planner = BatchPlanner;
    let mut slots = vec![attached_slot(0, request(1, vec![10, 11], 4))];
    let budget = SchedulerTickBudget {
        total_token_budget: 2,
        reserved_decode_tokens: 0,
        reserved_prefill_tokens: 2,
        decode_first: true,
    };
    let plan = planner.build_policy_batch(&slots, &[], &[0], budget, 0);

    planner.apply_decode_results(&mut slots, &plan);

    assert_eq!(slots[0].prefill_cursor, 2);
    assert_eq!(slots[0].phase, SlotPhase::Decode);
    assert_eq!(slots[0].batch_participation_count, 2);
}

#[test]
fn apply_decode_results_marks_decode_slot_completed_at_limit() {
    let planner = BatchPlanner;
    let mut slot = attached_slot(0, request(1, vec![10], 1));
    slot.generated_tokens = vec![99];
    let mut slots = vec![slot];
    let budget = SchedulerTickBudget {
        total_token_budget: 1,
        reserved_decode_tokens: 1,
        reserved_prefill_tokens: 0,
        decode_first: true,
    };
    let plan = planner.build_policy_batch(&slots, &[0], &[], budget, 0);

    planner.apply_decode_results(&mut slots, &plan);

    assert_eq!(slots[0].decode_step_count, 1);
    assert_eq!(slots[0].phase, SlotPhase::Completed);
}

#[test]
fn apply_decode_results_saturates_long_lived_decode_counters() {
    let planner = BatchPlanner;
    let mut slot = attached_slot(0, request(1, vec![10], 4));
    slot.generated_tokens = vec![99];
    slot.batch_participation_count = usize::MAX;
    slot.decode_step_count = usize::MAX;
    let mut slots = vec![slot];
    let budget = SchedulerTickBudget {
        total_token_budget: 1,
        reserved_decode_tokens: 1,
        reserved_prefill_tokens: 0,
        decode_first: true,
    };
    let plan = planner.build_policy_batch(&slots, &[0], &[], budget, 0);

    planner.apply_decode_results(&mut slots, &plan);

    assert_eq!(slots[0].batch_participation_count, usize::MAX);
    assert_eq!(slots[0].decode_step_count, usize::MAX);
    assert_eq!(slots[0].phase, SlotPhase::Decode);
}
