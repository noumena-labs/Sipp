/////////////////////////////////////////////////////////////////////////////////////////////////
//
// batch_planner.cpp
//
// - Phase 3 shared-batch planning scaffold.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/scheduler/batch_planner.h"

#include <algorithm>

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
        static_cast<int32_t>(slot->request->prompt_tokens.size() +
                             slot->generated_tokens.size() - 1);
    contribution.request_logits = true;
    plan.contributions.push_back(contribution);
    plan.decode_token_count++;
    remaining_decode_budget--;
  }

  for (SlotState *slot : prefill_slots) {
    if (remaining_prefill_budget <= 0 || slot == nullptr ||
        slot->request == nullptr) {
      continue;
    }

    const auto &prompt_tokens = slot->request->prompt_tokens;
    if (slot->prefill_cursor >= prompt_tokens.size()) {
      continue;
    }

    const std::size_t slot_contribution_start = plan.contributions.size();
    const int32_t slot_chunk_budget =
        prefill_chunk_size > 0
            ? std::min<int32_t>(prefill_chunk_size, remaining_prefill_budget)
            : remaining_prefill_budget;

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
  }

  std::vector<const SlotState *> occupied_slots;
  occupied_slots.reserve(plan.contributions.size());
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
