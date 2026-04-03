/////////////////////////////////////////////////////////////////////////////////////////////////
//
// slot_scheduler.cpp
//
// - Single-threaded scheduler.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/scheduler/slot_scheduler.h"

#include <algorithm>
#include <utility>

namespace noumena::cogentengine {

void SlotScheduler::SetContextFactory(ContextFactory context_factory) {
  context_factory_ = std::move(context_factory);
}

void SlotState::ResetToIdle() {
  phase = SlotPhase::Idle;
  request_id = 0;
  request = nullptr;
  session = nullptr;
  prefill_cursor = 0;
  generated_tokens.clear();
  output_text.clear();
  buffered_output_text.clear();
  terminal_error_message.clear();
  sampler = nullptr;
}

void SlotState::AttachRequest(GenerateRequest &request_ref,
                              ContextState &session_ref) {
  request_id = request_ref.id;
  request = &request_ref;
  session = &session_ref;
  phase = SlotPhase::Admitted;
  prefill_cursor = 0;
  generated_tokens.clear();
  output_text.clear();
  buffered_output_text.clear();
  terminal_error_message.clear();
}

void SlotScheduler::Resize(std::size_t slot_count) {
  slots_.resize(slot_count);
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    slots_[i].slot_id = i;
    if (slots_[i].phase == SlotPhase::Idle && slots_[i].request == nullptr) {
      continue;
    }
    slots_[i].ResetToIdle();
    slots_[i].slot_id = i;
  }
}

SlotState *SlotScheduler::FindFirstActiveSlot() {
  auto active_slot_it =
      std::find_if(slots_.begin(), slots_.end(), [](const SlotState &slot) {
        return slot.request != nullptr && slot.phase != SlotPhase::Idle &&
               slot.phase != SlotPhase::Completed &&
               slot.phase != SlotPhase::Failed;
      });
  return active_slot_it == slots_.end() ? nullptr : &(*active_slot_it);
}

void SlotScheduler::Tick(RequestQueue &request_queue,
                         SessionStore &session_store) {
  AdmitPendingRequests(request_queue, session_store);
  AdvanceActiveSlot();
  FinalizeCompletedSlots(request_queue);
}

bool SlotScheduler::AdmitPendingRequests(RequestQueue &request_queue,
                                         SessionStore &session_store) {
  auto idle_slot_it =
      std::find_if(slots_.begin(), slots_.end(), [](const SlotState &slot) {
        return slot.phase == SlotPhase::Idle && slot.request == nullptr;
      });
  if (idle_slot_it == slots_.end()) {
    return false;
  }

  const std::optional<GenerateRequestId> next_request_id =
      request_queue.TryPopNext();
  if (!next_request_id.has_value()) {
    return false;
  }

  GenerateRequest *request = request_queue.FindMutable(*next_request_id);
  if (request == nullptr) {
    return false;
  }

  ContextState &session = session_store.GetOrCreateSession(
      request->context_key, [this]() -> llama_context * {
        return context_factory_ ? context_factory_() : nullptr;
      });
  if (session.ctx == nullptr) {
    session_store.Remove(request->context_key);

    GenerateResponse response;
    response.request_id = request->id;
    response.status = GenerateResponseStatus::Failed;
    response.error_message = "Failed to create or acquire session context.";
    request_queue.MarkCompleted(std::move(response));
    return false;
  }

  idle_slot_it->AttachRequest(*request, session);
  idle_slot_it->phase = SlotPhase::Prefill;
  return true;
}

bool SlotScheduler::AdvanceActiveSlot() {
  SlotState *active_slot = FindFirstActiveSlot();
  if (active_slot == nullptr) {
    return false;
  }

  SlotState &slot = *active_slot;
  GenerateRequest *request = slot.request;
  ContextState *session = slot.session;

  if (request == nullptr || session == nullptr || session->ctx == nullptr) {
    if (request != nullptr) {
      request->lifecycle = GenerateRequestLifecycle::Failed;
    }
    slot.terminal_error_message = "Slot lost request or session state.";
    slot.phase = SlotPhase::Failed;
    return true;
  }

  switch (slot.phase) {
  case SlotPhase::Admitted:
    request->lifecycle = GenerateRequestLifecycle::Running;
    slot.phase = SlotPhase::Prefill;
    return true;

  case SlotPhase::Prefill:
    // - The actual llama prefill work still needs to move here from
    //   InferenceRuntime::Prompt(...).
    // - For now, advance the scheduler-visible state so the slot state machine
    //   is explicit before the decode loop is migrated.
    slot.prefill_cursor = request->prompt_tokens.size();
    request->lifecycle = GenerateRequestLifecycle::Running;
    slot.phase = SlotPhase::Decode;
    return true;

  case SlotPhase::Decode:
    if (!slot.buffered_output_text.empty()) {
      request->lifecycle = GenerateRequestLifecycle::Streaming;
      slot.phase = SlotPhase::Streaming;
      return true;
    }

    if (request->max_output_tokens <= 0 ||
        static_cast<int32_t>(slot.generated_tokens.size()) >=
            request->max_output_tokens) {
      request->lifecycle = GenerateRequestLifecycle::Completed;
      slot.phase = SlotPhase::Completed;
      return true;
    }

    return false;

  case SlotPhase::Streaming:
    if (!slot.buffered_output_text.empty()) {
      return false;
    }

    if (request->max_output_tokens > 0 &&
        static_cast<int32_t>(slot.generated_tokens.size()) <
            request->max_output_tokens) {
      request->lifecycle = GenerateRequestLifecycle::Running;
      slot.phase = SlotPhase::Decode;
      return true;
    }

    request->lifecycle = GenerateRequestLifecycle::Completed;
    slot.phase = SlotPhase::Completed;
    return true;

  case SlotPhase::Idle:
  case SlotPhase::Completed:
  case SlotPhase::Failed:
    return false;
  }

  return false;
}

void SlotScheduler::FinalizeCompletedSlots(RequestQueue &request_queue) {
  for (SlotState &slot : slots_) {
    if (slot.phase != SlotPhase::Completed && slot.phase != SlotPhase::Failed) {
      continue;
    }

    GenerateResponse response;
    response.request_id = slot.request_id;
    response.status = slot.phase == SlotPhase::Completed
                          ? GenerateResponseStatus::Completed
                          : GenerateResponseStatus::Failed;
    response.output_text = std::move(slot.output_text);
    if (slot.phase == SlotPhase::Failed) {
      response.error_message = slot.terminal_error_message.empty()
                                   ? "Request failed."
                                   : slot.terminal_error_message;
    }

    request_queue.MarkCompleted(std::move(response));
    slot.ResetToIdle();
  }
}

void SlotScheduler::EmitBufferedTokenPiece(SlotState &slot) {
  if (slot.buffered_output_text.empty()) {
    return;
  }

  GenerateRequest *request = slot.request;
  slot.output_text.append(slot.buffered_output_text);

  if (request != nullptr && request->on_token_received) {
    request->on_token_received(
        slot.buffered_output_text.c_str(),
        static_cast<int32_t>(slot.buffered_output_text.size()));
  }

  slot.buffered_output_text.clear();
}

void SlotScheduler::FailActiveRequest(RequestQueue &request_queue,
                                      SlotState &slot,
                                      std::string error_message) {
  GenerateResponse response;
  response.request_id = slot.request_id;
  response.status = GenerateResponseStatus::Failed;
  response.error_message = std::move(error_message);
  request_queue.MarkCompleted(std::move(response));
  slot.ResetToIdle();
}

} // namespace noumena::cogentengine
