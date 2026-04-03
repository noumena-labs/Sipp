/////////////////////////////////////////////////////////////////////////////////////////////////
//
// slot_scheduler.h
//
// - Single-threaded scheduler.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "runtime/request/request_queue.h"
#include "runtime/scheduler/slot_state.h"
#include "runtime/session/session_store.h"

namespace noumena::cogentengine {

class SlotScheduler {
public:
  using ContextFactory = std::function<llama_context *()>;

  void SetContextFactory(ContextFactory context_factory);
  void Resize(std::size_t slot_count);
  SlotState *FindFirstActiveSlot();
  void Tick(RequestQueue &request_queue, SessionStore &session_store);
  bool AdmitPendingRequests(RequestQueue &request_queue,
                            SessionStore &session_store);
  bool AdvanceActiveSlot();
  void FinalizeCompletedSlots(RequestQueue &request_queue);
  void EmitBufferedTokenPiece(SlotState &slot);
  void FailActiveRequest(RequestQueue &request_queue, SlotState &slot,
                         std::string error_message);

private:
  ContextFactory context_factory_;
  std::vector<SlotState> slots_;
};

} // namespace noumena::cogentengine
