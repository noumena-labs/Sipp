/////////////////////////////////////////////////////////////////////////////////////////////////
//
// batch_planner.cpp
//
// - Phase 3 shared-batch planning scaffold.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/scheduler/batch_planner.h"

#include <algorithm>
#include <cstddef>

namespace {

int32_t resolve_prefill_slice_cap(
    const noumena::cogentengine::SchedulerTickBudget &budget,
    int32_t configured_prefill_chunk_size,
    int32_t remaining_prefill_budget,
    std::size_t active_prefill_slot_count,
    bool has_decode_pressure) {
  if (remaining_prefill_budget <= 0) {
    return 0;
  }

  int32_t slice_cap = remaining_prefill_budget;

  if (configured_prefill_chunk_size > 0) {
    slice_cap = std::min(slice_cap, configured_prefill_chunk_size);
  }

  if (active_prefill_slot_count > 1) {
    const int32_t fair_share = std::max<int32_t>(
        1, remaining_prefill_budget /
               static_cast<int32_t>(active_prefill_slot_count));
    slice_cap = std::min(slice_cap, fair_share);
  }

  if (has_decode_pressure) {
    const int32_t decode_pressure_slice_cap =
        std::min(remaining_prefill_budget,
                 std::max<int32_t>(8, budget.EffectiveDecodeBudget()));
    slice_cap = std::min(slice_cap, decode_pressure_slice_cap);
  }

  return std::max<int32_t>(1, slice_cap);
}

} // namespace

namespace noumena::cogentengine {

bool SharedBatchPlan::Empty() const { return contributions.empty(); }

SharedBatchPlan BatchPlanner::BuildPolicyBatch(
    const std::vector<SlotState *> &decode_slots,
    const std::vector<SlotState *> &prefill_slots,
    const SchedulerTickBudget &budget,
    int32_t prefill_chunk_size) const {
  SharedBatchPlan plan;
  if (budget.total_token_budget <= 0) {
    return plan;
  }

  plan.contributions.reserve(
      static_cast<std::size_t>(std::max<int32_t>(1, budget.total_token_budget)));

  int32_t remaining_decode_budget = budget.EffectiveDecodeBudget();
  int32_t remaining_prefill_budget = budget.EffectivePrefillBudget();
  const bool has_decode_pressure = !decode_slots.empty();

  // Phase 4 algorithm steps:
  // 1. Spend decode reservation first so decode-ready slots are not delayed
  //    behind long prompt-prefill work.
  // 2. Admit at most one decode contribution per slot for the first Phase 4
  //    policy pass.
  // 3. Spend only the remaining budget on prefill work.
  // 4. Clamp each prefill slot to prefill_chunk_size when chunking is enabled.
  // 5. Keep contribution order explicit; later metrics and fairness analysis
  //    depend on knowing whether decode or prefill consumed the tick.
  for (SlotState *slot : decode_slots) {
    if (remaining_decode_budget <= 0 || slot == nullptr ||
        slot->request == nullptr || slot->generated_tokens.empty()) {
      continue;
    }

    BatchContribution contribution;
    contribution.slot = slot;
    contribution.kind = BatchContributionKind::Decode;
    contribution.token = slot->generated_tokens.back();
    contribution.position =
        slot->session != nullptr
            ? slot->session->n_past
            : static_cast<int32_t>(slot->request->prompt_tokens.size() +
                                   slot->generated_tokens.size() - 1);
    contribution.request_logits = true;
    plan.contributions.push_back(contribution);
    plan.decode_token_count++;
    remaining_decode_budget--;
  }

  std::vector<SlotState *> active_prefill_slots;
  active_prefill_slots.reserve(prefill_slots.size());
  for (SlotState *slot : prefill_slots) {
    if (slot == nullptr || slot->request == nullptr) {
      continue;
    }
    if (slot->prefill_cursor >= slot->request->prompt_tokens.size()) {
      continue;
    }
    active_prefill_slots.push_back(slot);
  }

  std::size_t next_prefill_slot_index = 0;
  while (remaining_prefill_budget > 0 && !active_prefill_slots.empty()) {
    if (next_prefill_slot_index >= active_prefill_slots.size()) {
      next_prefill_slot_index = 0;
    }

    SlotState *slot = active_prefill_slots[next_prefill_slot_index];
    if (slot == nullptr || slot->request == nullptr) {
      active_prefill_slots.erase(
          active_prefill_slots.begin() +
          static_cast<std::ptrdiff_t>(next_prefill_slot_index));
      continue;
    }

    const auto &prompt_tokens = slot->request->prompt_tokens;
    if (slot->prefill_cursor >= prompt_tokens.size()) {
      active_prefill_slots.erase(
          active_prefill_slots.begin() +
          static_cast<std::ptrdiff_t>(next_prefill_slot_index));
      continue;
    }

    const std::size_t slot_contribution_start = plan.contributions.size();
    const int32_t slot_chunk_budget = resolve_prefill_slice_cap(
        budget, prefill_chunk_size, remaining_prefill_budget,
        active_prefill_slots.size(), has_decode_pressure);

    int32_t remaining_slot_budget = slot_chunk_budget;
    for (std::size_t token_index = slot->prefill_cursor;
         token_index < prompt_tokens.size() && remaining_slot_budget > 0;
         ++token_index) {
      BatchContribution contribution;
      contribution.slot = slot;
      contribution.kind = BatchContributionKind::Prefill;
      contribution.token = prompt_tokens[token_index];
      contribution.position = static_cast<int32_t>(token_index);
      contribution.request_logits = false;
      plan.contributions.push_back(contribution);
      plan.prefill_token_count++;
      remaining_slot_budget--;
      remaining_prefill_budget--;
      if (remaining_prefill_budget <= 0) {
        break;
      }
    }

    if (plan.contributions.size() > slot_contribution_start) {
      const std::size_t contributed_count =
          plan.contributions.size() - slot_contribution_start;
      const bool completed_prompt =
          slot->prefill_cursor + contributed_count >= prompt_tokens.size();
      plan.contributions.back().request_logits = completed_prompt;
    }

    const bool slot_completed_prompt =
        slot->prefill_cursor +
            (plan.contributions.size() - slot_contribution_start) >=
        prompt_tokens.size();
    if (slot_completed_prompt) {
      active_prefill_slots.erase(
          active_prefill_slots.begin() +
          static_cast<std::ptrdiff_t>(next_prefill_slot_index));
      continue;
    }

    next_prefill_slot_index++;
  }

  std::vector<const SlotState *> occupied_slots;
  occupied_slots.reserve(static_cast<std::size_t>(
      std::max(0, plan.decode_token_count + plan.prefill_token_count)));
  for (const auto &contribution : plan.contributions) {
    if (contribution.slot == nullptr) {
      continue;
    }
    if (std::find(occupied_slots.begin(), occupied_slots.end(),
                  contribution.slot) == occupied_slots.end()) {
      occupied_slots.push_back(contribution.slot);
    }
  }
  plan.occupied_slot_count = static_cast<int32_t>(occupied_slots.size());

  return plan;
}

void BatchPlanner::ApplyDecodeResults(const SharedBatchPlan &plan) const {
  for (const auto &contribution : plan.contributions) {
    SlotState *slot = contribution.slot;
    if (slot == nullptr || slot->request == nullptr) {
      continue;
    }

    slot->batch_participation_count++;

    if (contribution.kind == BatchContributionKind::Prefill) {
      slot->prefill_cursor++;
      if (slot->prefill_cursor >= slot->request->prompt_tokens.size()) {
        slot->phase = SlotPhase::Decode;
      } else {
        slot->phase = SlotPhase::Prefill;
      }
      continue;
    }

    if (contribution.kind != BatchContributionKind::Decode) {
      continue;
    }

    // - The contribution token for a decode step is the input token fed back
    //   into the shared batch, not the newly sampled output token.
    // - Actual sampled-token ownership must be applied later from the runtime
    //   after logits are read and token pieces are converted with the model
    //   vocab.
    slot->decode_step_count++;

    if (slot->request->max_output_tokens > 0 &&
        slot->generated_tokens.size() >= slot->request->max_output_tokens) {
      slot->phase = SlotPhase::Completed;
    } else if (!slot->buffered_output_text.empty()) {
      slot->phase = SlotPhase::Streaming;
    } else {
      slot->phase = SlotPhase::Decode;
    }
  }
}

} // namespace noumena::cogentengine
