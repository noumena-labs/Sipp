//! Per-tick batch planner: turns the current slots into a flat list of token contributions sized to the scheduler tick budget.

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
            occupied_overflow_slots: Vec::with_capacity(max_slots.saturating_sub(64)),
            ..Self::default()
        }
    }

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
        self.occupied_overflow_slots.clear();
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

            plan.contributions.push(BatchContribution {
                slot_index,
                request_id: request.id,
                kind: BatchContributionKind::Decode,
                token,
                position: slot.mirror.n_past,
                request_logits: true,
            });
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
                let Ok(position) = i32::try_from(token_index) else {
                    break;
                };

                plan.contributions.push(BatchContribution {
                    slot_index,
                    request_id: request.id,
                    kind: BatchContributionKind::Prefill,
                    token: request.prompt_tokens[token_index],
                    position,
                    request_logits: false,
                });
                plan.prefill_token_count = plan.prefill_token_count.saturating_add(1);
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

        // Bitmask unique-slot count for n_parallel ≤ 64 — avoids per-tick HashSet alloc.
        let mut occupied_mask: u64 = 0;
        for contribution in &plan.contributions {
            if contribution.slot_index < 64 {
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

            slot.batch_participation_count = slot.batch_participation_count.saturating_add(1);

            if contribution.kind == BatchContributionKind::Prefill {
                slot.prefill_cursor = slot.prefill_cursor.saturating_add(1);
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

            slot.decode_step_count = slot.decode_step_count.saturating_add(1);
            slot.phase =
                if token_limit_reached(slot.generated_tokens.len(), request_max_output_tokens) {
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
        let active_prefill_slot_count = saturating_usize_to_i32(active_prefill_slot_count).max(1);
        let fair_share = (remaining_prefill_budget / active_prefill_slot_count).max(1);
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

fn token_limit_reached(generated_token_count: usize, max_output_tokens: i32) -> bool {
    max_output_tokens > 0
        && generated_token_count >= usize::try_from(max_output_tokens).unwrap_or(usize::MAX)
}

fn saturating_usize_to_i32(value: usize) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

fn saturating_u32_to_i32(value: u32) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

fn positive_i32_to_usize(value: i32) -> Option<usize> {
    usize::try_from(value).ok().filter(|value| *value > 0)
}

#[cfg(test)]
mod tests;
