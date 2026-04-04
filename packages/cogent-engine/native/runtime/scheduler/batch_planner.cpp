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

SharedBatchPlan
BatchPlanner::BuildSharedBatch(const std::vector<SlotState *> &runnable_slots,
                               int32_t max_batch_tokens) const {
  SharedBatchPlan plan;
  if (max_batch_tokens <= 0 || runnable_slots.empty()) {
    return plan;
  }

  plan.contributions.reserve(
      static_cast<std::size_t>(std::max<int32_t>(1, max_batch_tokens)));

  std::vector<const SlotState *> occupied_slots;
  occupied_slots.reserve(runnable_slots.size());

  const auto mark_slot_occupied = [&](const SlotState *slot) {
    if (slot == nullptr) {
      return;
    }
    if (std::find(occupied_slots.begin(), occupied_slots.end(), slot) ==
        occupied_slots.end()) {
      occupied_slots.push_back(slot);
    }
  };

  int32_t remaining_budget = max_batch_tokens;

  // Pass 1: dense prefill.
  // Consume as many prompt tokens as possible for prefill-ready slots before
  // spending the remaining budget on decode-ready slots.
  for (SlotState *slot : runnable_slots) {
    if (remaining_budget <= 0 || slot == nullptr || slot->request == nullptr) {
      break;
    }
    if (slot->phase != SlotPhase::Prefill) {
      continue;
    }

    const auto &prompt_tokens = slot->request->prompt_tokens;
    if (slot->prefill_cursor >= prompt_tokens.size()) {
      continue;
    }

    const std::size_t slot_contribution_start = plan.contributions.size();
    for (std::size_t token_index = slot->prefill_cursor;
         token_index < prompt_tokens.size() && remaining_budget > 0;
         ++token_index) {
      BatchContribution contribution;
      contribution.slot = slot;
      contribution.kind = BatchContributionKind::Prefill;
      contribution.token = prompt_tokens[token_index];
      contribution.position = static_cast<int32_t>(token_index);
      contribution.request_logits = false;
      plan.contributions.push_back(contribution);
      plan.prefill_token_count++;
      remaining_budget--;
    }

    if (plan.contributions.size() > slot_contribution_start) {
      // Request logits for the last prompt token contributed for this slot in
      // the current tick so decode can continue once the prefill contribution
      // has been applied.
      plan.contributions.back().request_logits = true;
      mark_slot_occupied(slot);
    }
  }

  // Pass 2: decode-ready slots.
  // Keep the first Phase 3 planner deterministic by admitting at most one
  // decode contribution per slot per tick.
  for (SlotState *slot : runnable_slots) {
    if (remaining_budget <= 0 || slot == nullptr || slot->request == nullptr) {
      break;
    }
    if (slot->phase != SlotPhase::Decode &&
        slot->phase != SlotPhase::Streaming) {
      continue;
    }
    if (!slot->buffered_output_text.empty() || slot->generated_tokens.empty()) {
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
    remaining_budget--;
    mark_slot_occupied(slot);
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
