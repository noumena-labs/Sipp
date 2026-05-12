/////////////////////////////////////////////////////////////////////////////////////////////////
//
// slot_scheduler.h
//
// - Single-threaded scheduler.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "runtime/config/scheduler_policy.h"
#include "runtime/request/request_queue.h"
#include "runtime/scheduler/batch_planner.h"
#include "runtime/scheduler/slot_state.h"
#include "runtime/session/session_store.h"

namespace noumena::cogentengine {

class SlotScheduler {
public:
  void Resize(std::size_t slot_count);
  SlotState *FindFirstActiveSlot();
  void SelectDecodeReadySlots(std::vector<SlotState *> &out_slots);
  void SelectPrefillReadySlots(std::vector<SlotState *> &out_slots);
  std::vector<SlotState> &MutableSlots() { return slots_; }
  const std::vector<SlotState> &Slots() const { return slots_; }
  SchedulerTickBudget BuildTickBudget(const SchedulerPolicyConfig &policy,
                                      int32_t decode_ready_count,
                                      int32_t prefill_ready_count,
                                      int32_t max_batch_tokens,
                                      int32_t prefill_chunk_size);
  bool AdmitPendingRequests(RequestQueue &request_queue,
                            SessionStore &session_store);
  void FinalizeCompletedSlots(RequestQueue &request_queue,
                              SessionStore &session_store);
  double EmitBufferedTokenPiece(RequestQueue &request_queue, SlotState &slot);

private:
  std::vector<SlotState> slots_;
};

} // namespace noumena::cogentengine
