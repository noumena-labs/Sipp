/////////////////////////////////////////////////////////////////////////////////////////////////
//
// batch_planner.h
//
// - Phase 3 shared-batch planning scaffold.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <vector>

#include "llama.h"
#include "runtime/config/scheduler_policy.h"
#include "runtime/scheduler/slot_state.h"

namespace noumena::cogentengine {

enum class BatchContributionKind : std::uint8_t {
  Prefill = 0,
  Decode,
};

struct BatchContribution {
  SlotState *slot = nullptr;
  BatchContributionKind kind = BatchContributionKind::Prefill;
  llama_token token = 0;
  int32_t position = 0;
  bool request_logits = false;
};

struct SharedBatchPlan {
  std::vector<BatchContribution> contributions;
  int32_t prefill_token_count = 0;
  int32_t decode_token_count = 0;
  int32_t occupied_slot_count = 0;

  bool Empty() const;
};

class BatchPlanner {
public:
  SharedBatchPlan BuildSharedBatch(const std::vector<SlotState *> &runnable_slots,
                                   int32_t max_batch_tokens) const;
  SharedBatchPlan BuildPolicyBatch(const std::vector<SlotState *> &decode_slots,
                                   const std::vector<SlotState *> &prefill_slots,
                                   const SchedulerTickBudget &budget,
                                   int32_t prefill_chunk_size) const;
  void ApplyDecodeResults(const SharedBatchPlan &plan) const;
};

} // namespace noumena::cogentengine
