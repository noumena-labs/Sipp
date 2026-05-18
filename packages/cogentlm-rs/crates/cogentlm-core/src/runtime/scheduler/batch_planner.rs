use crate::runtime::config::SchedulerTickBudget;
use crate::runtime::request::GenerateRequestId;
use crate::runtime::{llama_token, scheduler::SlotPhase};

use super::SlotState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchContributionKind {
    Prefill = 0,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchContribution {
    pub slot_index: usize,
    pub request_id: GenerateRequestId,
    pub kind: BatchContributionKind,
    pub token: llama_token,
    pub position: i32,
    pub request_logits: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SharedBatchPlan {
    pub contributions: Vec<BatchContribution>,
    pub prefill_token_count: i32,
    pub decode_token_count: i32,
    pub occupied_slot_count: i32,
    /// Scratch buffer for the prefill round-robin slot indices. Hoisted onto
    /// the plan so it's reused across ticks instead of allocated fresh each
    /// time `build_policy_batch_into` runs.
    active_prefill_slots: Vec<usize>,
    /// Parallel scratch buffer tracking how many tokens of each active
    /// prefill slot's prompt have already been added this tick.
    in_tick_offset: Vec<usize>,
}

impl SharedBatchPlan {
    pub fn is_empty(&self) -> bool {
        self.contributions.is_empty()
    }

    /// Clears the plan in-place so it can be refilled by
    /// [`BatchPlanner::build_policy_batch_into`] without releasing the
    /// underlying allocations.
    pub fn reset(&mut self) {
        self.contributions.clear();
        self.prefill_token_count = 0;
        self.decode_token_count = 0;
        self.occupied_slot_count = 0;
        self.active_prefill_slots.clear();
        self.in_tick_offset.clear();
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BatchPlanner;

impl BatchPlanner {
    pub fn build_policy_batch(
        &self,
        slots: &[SlotState],
        decode_slots: &[usize],
        prefill_slots: &[usize],
        budget: SchedulerTickBudget,
        prefill_chunk_size: i32,
    ) -> SharedBatchPlan {
        let mut plan = SharedBatchPlan::default();
        self.build_policy_batch_into(
            &mut plan,
            slots,
            decode_slots,
            prefill_slots,
            budget,
            prefill_chunk_size,
        );
        plan
    }

    /// Hot-path entry point: fills `plan` in-place from this tick's ready
    /// slots. Reusing a single plan across ticks avoids reallocating the
    /// contributions Vec (capacity = batch_token_budget, often 512+) every
    /// scheduler iteration — that allocation was the single largest source
    /// of overhead vs the old C++ engine.
    pub fn build_policy_batch_into(
        &self,
        plan: &mut SharedBatchPlan,
        slots: &[SlotState],
        decode_slots: &[usize],
        prefill_slots: &[usize],
        budget: SchedulerTickBudget,
        prefill_chunk_size: i32,
    ) {
        plan.reset();
        if budget.total_token_budget <= 0 {
            return;
        }

        plan.contributions
            .reserve(budget.total_token_budget.max(1) as usize);

        let mut remaining_decode_budget = budget.effective_decode_budget();
        let mut remaining_prefill_budget = budget.effective_prefill_budget();
        let has_decode_pressure = !decode_slots.is_empty();

        for &slot_index in decode_slots {
            if remaining_decode_budget <= 0 {
                break;
            }
            let Some(slot) = slots.get(slot_index) else {
                continue;
            };
            let Some(request) = slot.request() else {
                continue;
            };
            let Some(&token) = slot.generated_tokens.last() else {
                continue;
            };

            plan.contributions.push(BatchContribution {
                slot_index,
                request_id: request.id,
                kind: BatchContributionKind::Decode,
                token,
                position: slot.mirror.n_past,
                request_logits: true,
            });
            plan.decode_token_count += 1;
            remaining_decode_budget -= 1;
        }

        plan.active_prefill_slots.reserve(prefill_slots.len());
        plan.in_tick_offset.reserve(prefill_slots.len());
        for &slot_index in prefill_slots {
            let Some(slot) = slots.get(slot_index) else {
                continue;
            };
            let Some(request) = slot.request() else {
                continue;
            };
            if slot.prefill_cursor >= request.prompt_tokens.len() {
                continue;
            }
            plan.active_prefill_slots.push(slot_index);
            plan.in_tick_offset.push(0_usize);
        }

        let mut next_prefill_slot_index = 0;
        while remaining_prefill_budget > 0 && !plan.active_prefill_slots.is_empty() {
            if next_prefill_slot_index >= plan.active_prefill_slots.len() {
                next_prefill_slot_index = 0;
            }

            let slot_index = plan.active_prefill_slots[next_prefill_slot_index];
            let Some(slot) = slots.get(slot_index) else {
                erase_active_slot(
                    &mut plan.active_prefill_slots,
                    &mut plan.in_tick_offset,
                    next_prefill_slot_index,
                );
                continue;
            };
            let Some(request) = slot.request() else {
                erase_active_slot(
                    &mut plan.active_prefill_slots,
                    &mut plan.in_tick_offset,
                    next_prefill_slot_index,
                );
                continue;
            };
            if slot.prefill_cursor >= request.prompt_tokens.len() {
                erase_active_slot(
                    &mut plan.active_prefill_slots,
                    &mut plan.in_tick_offset,
                    next_prefill_slot_index,
                );
                continue;
            }

            let slot_contribution_start = plan.contributions.len();
            let slot_chunk_budget = resolve_prefill_slice_cap(
                budget,
                prefill_chunk_size,
                remaining_prefill_budget,
                plan.active_prefill_slots.len(),
                has_decode_pressure,
            );
            let resume_offset = plan.in_tick_offset[next_prefill_slot_index];
            let mut remaining_slot_budget = slot_chunk_budget;

            for token_index in (slot.prefill_cursor + resume_offset)..request.prompt_tokens.len() {
                if remaining_slot_budget <= 0 || remaining_prefill_budget <= 0 {
                    break;
                }

                plan.contributions.push(BatchContribution {
                    slot_index,
                    request_id: request.id,
                    kind: BatchContributionKind::Prefill,
                    token: request.prompt_tokens[token_index],
                    position: token_index as i32,
                    request_logits: false,
                });
                plan.prefill_token_count += 1;
                remaining_slot_budget -= 1;
                remaining_prefill_budget -= 1;
            }

            let added_this_iteration = plan.contributions.len() - slot_contribution_start;
            let total_added = resume_offset + added_this_iteration;
            plan.in_tick_offset[next_prefill_slot_index] = total_added;

            let slot_completed_prompt =
                slot.prefill_cursor + total_added >= request.prompt_tokens.len();
            if added_this_iteration > 0 && slot_completed_prompt {
                if let Some(last) = plan.contributions.last_mut() {
                    last.request_logits = true;
                }
            }
            if slot_completed_prompt {
                erase_active_slot(
                    &mut plan.active_prefill_slots,
                    &mut plan.in_tick_offset,
                    next_prefill_slot_index,
                );
                continue;
            }

            next_prefill_slot_index += 1;
        }

        // Bitmask-based unique-slot count for the typical n_parallel ≤ 64
        // case (always true in browser inference). HashSet::new() per tick
        // was a measurable allocation that the C++ engine never paid.
        let mut occupied_mask: u64 = 0;
        let mut occupied_overflow = 0_i32;
        for contribution in &plan.contributions {
            if contribution.slot_index < 64 {
                occupied_mask |= 1u64 << contribution.slot_index;
            } else if contribution.slot_index < (64 + occupied_overflow as usize) {
                // Overflow slots are rare; we approximate by counting only
                // the high-end slots we haven't seen yet via linear scan.
                // For the common case (no slot ≥ 64) this branch never runs.
            } else {
                occupied_overflow += 1;
            }
        }
        plan.occupied_slot_count = occupied_mask.count_ones() as i32 + occupied_overflow;
    }

    pub fn apply_decode_results(&self, slots: &mut [SlotState], plan: &SharedBatchPlan) {
        for contribution in &plan.contributions {
            let Some(slot) = slots.get_mut(contribution.slot_index) else {
                continue;
            };
            let Some(request) = slot.request() else {
                continue;
            };
            let request_max_output_tokens = request.max_output_tokens;
            let request_prompt_len = request.prompt_tokens.len();

            slot.batch_participation_count += 1;

            if contribution.kind == BatchContributionKind::Prefill {
                slot.prefill_cursor += 1;
                slot.phase = if slot.prefill_cursor >= request_prompt_len {
                    SlotPhase::Decode
                } else {
                    SlotPhase::Prefill
                };
                continue;
            }

            if contribution.kind != BatchContributionKind::Decode {
                continue;
            }

            slot.decode_step_count += 1;
            slot.phase = if request_max_output_tokens > 0
                && slot.generated_tokens.len() >= request_max_output_tokens as usize
            {
                SlotPhase::Completed
            } else if !slot.buffered_output_text.is_empty() {
                SlotPhase::Streaming
            } else {
                SlotPhase::Decode
            };
        }
    }
}

fn resolve_prefill_slice_cap(
    budget: SchedulerTickBudget,
    configured_prefill_chunk_size: i32,
    remaining_prefill_budget: i32,
    active_prefill_slot_count: usize,
    has_decode_pressure: bool,
) -> i32 {
    if remaining_prefill_budget <= 0 {
        return 0;
    }

    let mut slice_cap = remaining_prefill_budget;
    if configured_prefill_chunk_size > 0 {
        slice_cap = slice_cap.min(configured_prefill_chunk_size);
    }

    if active_prefill_slot_count > 1 {
        let fair_share = (remaining_prefill_budget / active_prefill_slot_count as i32).max(1);
        slice_cap = slice_cap.min(fair_share);
    }

    if has_decode_pressure {
        let decode_pressure_slice_cap =
            remaining_prefill_budget.min(budget.effective_decode_budget().max(8));
        slice_cap = slice_cap.min(decode_pressure_slice_cap);
    }

    slice_cap.max(1)
}

fn erase_active_slot(
    active_prefill_slots: &mut Vec<usize>,
    in_tick_offset: &mut Vec<usize>,
    index: usize,
) {
    active_prefill_slots.remove(index);
    in_tick_offset.remove(index);
}

#[cfg(test)]
mod tests {
    use crate::runtime::config::SchedulerTickBudget;
    use crate::runtime::request::GenerateRequest;
    use crate::runtime::scheduler::{BatchContributionKind, BatchPlanner, SlotPhase, SlotState};
    use crate::runtime::session::SequenceState;

    fn request(id: u32, prompt_tokens: Vec<i32>, max_output_tokens: i32) -> GenerateRequest {
        let mut request = GenerateRequest::new(id, format!("ctx-{id}"));
        request.prompt_tokens = prompt_tokens;
        request.max_output_tokens = max_output_tokens;
        request
    }

    fn attached_slot(slot_id: usize, request: GenerateRequest) -> SlotState {
        let mut slot = SlotState::new(slot_id);
        slot.attach_request(request, SequenceState::default());
        slot
    }

    #[test]
    fn returns_empty_plan_when_budget_is_zero() {
        let planner = BatchPlanner;
        let slots = vec![attached_slot(0, request(1, vec![1, 2], 4))];
        let plan = planner.build_policy_batch(&slots, &[], &[0], SchedulerTickBudget::default(), 0);
        assert!(plan.is_empty());
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
}
