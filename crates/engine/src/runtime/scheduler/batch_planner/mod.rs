//! Per-tick batch planner: turns the current slots into a flat list of token contributions sized to the scheduler tick budget.

use crate::runtime::config::{KvReuseMode, SchedulerTickBudget};
use crate::runtime::llama_token;
use crate::runtime::numeric::{
    positive_i32_to_usize, saturating_u32_to_i32, saturating_usize_to_i32,
};
use crate::runtime::request::GenerateRequestId;

use super::{SlotState, TerminalAction};

#[cfg(test)]
mod apply_results;
mod helpers;

use helpers::resolve_prefill_slice_cap;

const FAST_OCCUPIED_SLOT_BITS: usize = 64;

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
    /// Scratch buffer for unique occupied slot indexes above the 64-slot
    /// bitmask fast path. Reused so large parallel configurations do not
    /// allocate every scheduler tick.
    occupied_overflow_slots: Vec<usize>,
}

impl SharedBatchPlan {
    pub fn with_capacities(max_contributions: usize, max_slots: usize) -> Self {
        Self {
            contributions: Vec::with_capacity(max_contributions),
            active_prefill_slots: Vec::with_capacity(max_slots),
            in_tick_offset: Vec::with_capacity(max_slots),
            occupied_overflow_slots: Vec::with_capacity(
                max_slots.saturating_sub(FAST_OCCUPIED_SLOT_BITS),
            ),
            ..Self::default()
        }
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
        self.occupied_overflow_slots.clear();
    }

    fn erase_active_prefill_slot(&mut self, index: usize) {
        self.active_prefill_slots.remove(index);
        self.in_tick_offset.remove(index);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BatchPlanner;

impl BatchPlanner {
    #[cfg(test)]
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

        let reserve_capacity = positive_i32_to_usize(budget.total_token_budget).unwrap_or(1);
        plan.contributions.reserve(reserve_capacity);

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

            plan.contributions.push(decode_contribution(
                slot_index,
                request.id,
                token,
                slot.mirror.n_past,
            ));
            plan.decode_token_count = plan.decode_token_count.saturating_add(1);
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
                plan.erase_active_prefill_slot(next_prefill_slot_index);
                continue;
            };
            let Some(request) = slot.request() else {
                plan.erase_active_prefill_slot(next_prefill_slot_index);
                continue;
            };
            if slot.prefill_cursor >= request.prompt_tokens.len() {
                plan.erase_active_prefill_slot(next_prefill_slot_index);
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

            let prompt_end = prefill_stop_exclusive(slot, request.prompt_tokens.len());
            for token_index in (slot.prefill_cursor + resume_offset)..prompt_end {
                if remaining_slot_budget <= 0 || remaining_prefill_budget <= 0 {
                    break;
                }
                let Ok(position) = i32::try_from(token_index) else {
                    break;
                };

                plan.contributions.push(prefill_contribution(
                    slot_index,
                    request.id,
                    request.prompt_tokens[token_index],
                    position,
                ));
                plan.prefill_token_count = plan.prefill_token_count.saturating_add(1);
                remaining_slot_budget -= 1;
                remaining_prefill_budget -= 1;
            }

            let added_this_iteration = plan.contributions.len() - slot_contribution_start;
            let total_added = resume_offset + added_this_iteration;
            plan.in_tick_offset[next_prefill_slot_index] = total_added;

            let slot_reached_tick_stop = slot.prefill_cursor + total_added >= prompt_end;
            let slot_completed_prompt =
                slot.prefill_cursor + total_added >= request.prompt_tokens.len();
            if added_this_iteration > 0 && slot_completed_prompt {
                if let Some(last) = plan.contributions.last_mut() {
                    last.request_logits = true;
                }
            }
            if slot_completed_prompt || slot_reached_tick_stop {
                plan.erase_active_prefill_slot(next_prefill_slot_index);
                continue;
            }

            next_prefill_slot_index += 1;
        }

        // Bitmask unique-slot count for small n_parallel values avoids per-tick HashSet allocation.
        let mut occupied_mask: u64 = 0;
        for contribution in &plan.contributions {
            if contribution.slot_index < FAST_OCCUPIED_SLOT_BITS {
                occupied_mask |= 1u64 << contribution.slot_index;
            } else if !plan
                .occupied_overflow_slots
                .contains(&contribution.slot_index)
            {
                plan.occupied_overflow_slots.push(contribution.slot_index);
            }
        }
        plan.occupied_slot_count = saturating_u32_to_i32(occupied_mask.count_ones())
            .saturating_add(saturating_usize_to_i32(plan.occupied_overflow_slots.len()));
    }

    #[cfg(test)]
    pub fn apply_decode_results(&self, slots: &mut [SlotState], plan: &SharedBatchPlan) {
        apply_results::apply_decode_results(slots, plan);
    }
}

fn decode_contribution(
    slot_index: usize,
    request_id: GenerateRequestId,
    token: llama_token,
    position: i32,
) -> BatchContribution {
    BatchContribution {
        slot_index,
        request_id,
        kind: BatchContributionKind::Decode,
        token,
        position,
        request_logits: true,
    }
}

fn prefill_contribution(
    slot_index: usize,
    request_id: GenerateRequestId,
    token: llama_token,
    position: i32,
) -> BatchContribution {
    BatchContribution {
        slot_index,
        request_id,
        kind: BatchContributionKind::Prefill,
        token,
        position,
        request_logits: false,
    }
}

fn prefill_stop_exclusive(slot: &SlotState, prompt_len: usize) -> usize {
    let Some(request) = slot.request() else {
        return prompt_len;
    };
    if snapshot_reuse_enabled(request.cache_mode)
        && slot.plan.terminal == TerminalAction::SampleTokens
        && slot.prefill_cursor < prompt_len.saturating_sub(1)
    {
        return prompt_len.saturating_sub(1);
    }
    prompt_len
}

fn snapshot_reuse_enabled(mode: KvReuseMode) -> bool {
    matches!(
        mode,
        KvReuseMode::StateSnapshot | KvReuseMode::LiveSlotAndSnapshot
    )
}

#[cfg(test)]
#[path = "../../../tests/runtime/scheduler/batch_planner_tests.rs"]
mod batch_planner_tests;
