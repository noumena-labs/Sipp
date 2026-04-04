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

#include "runtime/request/request_queue.h"
#include "runtime/scheduler/batch_planner.h"
#include "runtime/scheduler/slot_state.h"
#include "runtime/session/session_store.h"

namespace noumena::cogentengine {

class SlotScheduler {
public:
  void Resize(std::size_t slot_count);
  SlotState *FindFirstActiveSlot();
  std::vector<SlotState *> SelectRunnableSlots();
  void Tick(RequestQueue &request_queue, SessionStore &session_store);
  bool AdmitPendingRequests(RequestQueue &request_queue,
                            SessionStore &session_store);
  bool AdvanceActiveSlot();
  void FinalizeCompletedSlots(RequestQueue &request_queue,
                              SessionStore &session_store);
  void EmitBufferedTokenPiece(SlotState &slot);
  void FailActiveRequest(RequestQueue &request_queue, SessionStore &session_store,
                         SlotState &slot,
                         std::string error_message);

private:
  std::vector<SlotState> slots_;
};

} // namespace noumena::cogentengine
