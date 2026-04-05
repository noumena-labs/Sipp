/////////////////////////////////////////////////////////////////////////////////////////////////
//
// slot_scheduler.cpp
//
// - Single-threaded scheduler.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/scheduler/slot_scheduler.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace {

double duration_ms(std::chrono::steady_clock::time_point start,
                   std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

namespace noumena::cogentengine {

void SlotState::ResetToIdle() {
  if (sampler != nullptr) {
    llama_sampler_free(sampler);
  }
  phase = SlotPhase::Idle;
  seq_id = -1;
  request_id = 0;
  request = nullptr;
  session = nullptr;
  prefill_cursor = 0;
  decode_step_count = 0;
  batch_participation_count = 0;
  scheduler_tick_count = 0;
  generated_tokens.clear();
  output_text.clear();
  buffered_output_text.clear();
  terminal_error_message.clear();
  sampler = nullptr;
}

void SlotState::AttachRequest(GenerateRequest &request_ref,
                              SequenceState &session_ref) {
  seq_id = session_ref.seq_id;
  request_id = request_ref.id;
  request = &request_ref;
  session = &session_ref;
  phase = SlotPhase::Admitted;
  prefill_cursor = 0;
  decode_step_count = 0;
  batch_participation_count = 0;
  scheduler_tick_count = 0;
  generated_tokens.clear();
  output_text.clear();
  buffered_output_text.clear();
  terminal_error_message.clear();
}

void SlotScheduler::Resize(std::size_t slot_count) {
  slots_.resize(slot_count);
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    slots_[i].slot_id = i;
    slots_[i].seq_id = static_cast<llama_seq_id>(i);
    if (slots_[i].phase == SlotPhase::Idle && slots_[i].request == nullptr) {
      continue;
    }
    slots_[i].ResetToIdle();
    slots_[i].slot_id = i;
    slots_[i].seq_id = static_cast<llama_seq_id>(i);
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

std::vector<SlotState *> SlotScheduler::SelectRunnableSlots() {
  std::vector<SlotState *> runnable_slots;
  runnable_slots.reserve(slots_.size());

  for (SlotState &slot : slots_) {
    if (slot.request == nullptr || slot.session == nullptr ||
        slot.session->seq_id < 0) {
      continue;
    }
    if (slot.phase != SlotPhase::Prefill && slot.phase != SlotPhase::Decode &&
        slot.phase != SlotPhase::Streaming) {
      continue;
    }

    // - Dense-prefill ordering and decode reservation policy belong in
    //   BatchPlanner.
    // - This selector stays focused on "which slots are runnable at all".
    runnable_slots.push_back(&slot);
  }

  return runnable_slots;
}

std::vector<SlotState *> SlotScheduler::SelectDecodeReadySlots() {
  std::vector<SlotState *> decode_slots;
  decode_slots.reserve(slots_.size());

  // Phase 4 algorithm steps:
  // 1. Admit only slots that are already decode-ready for this tick.
  // 2. Exclude slots that still have buffered text waiting to be emitted,
  //    because those slots should not consume additional decode budget yet.
  // 3. Keep ordering deterministic by preserving slot order.
  for (SlotState &slot : slots_) {
    if (slot.request == nullptr || slot.session == nullptr) {
      continue;
    }
    if (slot.phase != SlotPhase::Decode) {
      continue;
    }
    if (!slot.buffered_output_text.empty()) {
      continue;
    }
    decode_slots.push_back(&slot);
  }

  return decode_slots;
}

std::vector<SlotState *> SlotScheduler::SelectPrefillReadySlots() {
  std::vector<SlotState *> prefill_slots;
  prefill_slots.reserve(slots_.size());

  // Phase 4 algorithm steps:
  // 1. Admit only slots that still have prompt tokens left to prefill.
  // 2. Keep selection free of fairness heuristics; chunking and reservation
  //    policy belong in the tick budget and batch planner.
  // 3. Preserve slot order so later policy behavior is explainable.
  for (SlotState &slot : slots_) {
    if (slot.request == nullptr || slot.session == nullptr) {
      continue;
    }
    if (slot.phase != SlotPhase::Prefill) {
      continue;
    }
    if (slot.prefill_cursor >= slot.request->prompt_tokens.size()) {
      continue;
    }
    prefill_slots.push_back(&slot);
  }

  return prefill_slots;
}

SchedulerTickBudget
SlotScheduler::BuildTickBudget(const SchedulerPolicyConfig &policy,
                               int32_t decode_ready_count,
                               int32_t prefill_ready_count,
                               int32_t max_batch_tokens,
                               int32_t prefill_chunk_size) {
  SchedulerTickBudget budget;
  budget.total_token_budget = std::max(0, max_batch_tokens);
  budget.decode_first = decode_ready_count > 0;

  // Phase 4 algorithm steps:
  // 1. Start from the shared runtime token budget for a tick.
  // 2. Reserve decode tokens first so short decode-heavy requests are not
  //    starved by one long prompt prefill.
  // 3. Clamp the decode reservation to the actual total budget.
  // 4. Leave adaptive chunk sizing for later; for now, the remaining budget
  //    becomes the prefill budget.
  // 5. If chunking is disabled, the prefill planner may still consume the
  //    remaining budget densely, but decode reservation must remain explicit.
  if (budget.total_token_budget <= 0) {
    return budget;
  }

  const int32_t clamped_decode_ready =
      std::max<int32_t>(0, decode_ready_count);
  const int32_t clamped_prefill_ready =
      std::max<int32_t>(0, prefill_ready_count);

  int32_t reserved_decode_tokens = 0;
  switch (policy.mode) {
  case SchedulerPolicyMode::LatencyFirst:
    reserved_decode_tokens =
        clamped_decode_ready > 0
            ? std::max(policy.decode_token_reserve, clamped_decode_ready)
            : 0;
    break;
  case SchedulerPolicyMode::Balanced:
    reserved_decode_tokens =
        clamped_decode_ready > 0
            ? std::max(policy.decode_token_reserve,
                       std::min(clamped_decode_ready,
                                std::max<int32_t>(1, budget.total_token_budget / 2)))
            : 0;
    break;
  case SchedulerPolicyMode::ThroughputFirst:
    reserved_decode_tokens =
        clamped_decode_ready > 0 ? std::max<int32_t>(1, policy.decode_token_reserve) : 0;
    break;
  }

  budget.reserved_decode_tokens =
      std::clamp(reserved_decode_tokens, 0, budget.total_token_budget);
  budget.reserved_prefill_tokens = clamped_prefill_ready > 0
                                       ? std::max(
                                             0, budget.total_token_budget -
                                                    budget.reserved_decode_tokens)
                                       : 0;

  if (clamped_decode_ready <= 0) {
    budget.reserved_decode_tokens = 0;
  }

  if (clamped_prefill_ready > 0 && budget.reserved_prefill_tokens <= 0 &&
      budget.total_token_budget > 0 && budget.reserved_decode_tokens >= budget.total_token_budget) {
    budget.reserved_decode_tokens = std::max(0, budget.total_token_budget - 1);
    budget.reserved_prefill_tokens = budget.total_token_budget - budget.reserved_decode_tokens;
  }

  if (prefill_chunk_size <= 0 && clamped_prefill_ready > 0) {
    budget.reserved_prefill_tokens =
        std::max(0, budget.total_token_budget - budget.reserved_decode_tokens);
  }

  return budget;
}

void SlotScheduler::Tick(RequestQueue &request_queue,
                         SessionStore &session_store) {
  AdmitPendingRequests(request_queue, session_store);
  AdvanceActiveSlot();
  FinalizeCompletedSlots(request_queue, session_store);
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

  SequenceState &session = session_store.GetOrCreateSession(request->context_key);
  if (session.seq_id < 0) {
    session_store.Remove(request->context_key);

    GenerateResponse response;
    response.request_id = request->id;
    response.status = GenerateResponseStatus::Failed;
    response.error_message = "Failed to create or acquire session context.";
    request_queue.MarkCompleted(std::move(response));
    return false;
  }

  session_store.Pin(session);
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
  slot.scheduler_tick_count++;
  GenerateRequest *request = slot.request;
  SequenceState *session = slot.session;

  if (request == nullptr || session == nullptr || session->seq_id < 0) {
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

void SlotScheduler::FinalizeCompletedSlots(RequestQueue &request_queue,
                                          SessionStore &session_store) {
  for (SlotState &slot : slots_) {
    if (slot.phase != SlotPhase::Completed && slot.phase != SlotPhase::Failed) {
      continue;
    }

    GenerateResponse response;
    response.request_id = slot.request_id;
    response.status =
        slot.request != nullptr && slot.request->cancel_requested
            ? GenerateResponseStatus::Cancelled
            : (slot.phase == SlotPhase::Completed
                   ? GenerateResponseStatus::Completed
                   : GenerateResponseStatus::Failed);
    response.output_text = std::move(slot.output_text);
    if (slot.request != nullptr) {
      GenerateRequest &request = *slot.request;
      request.completed_at = std::chrono::steady_clock::now();
      request.has_completed_at = true;

      response.runtime_observability.queue_delay_ms =
          request.has_admitted_at
              ? duration_ms(request.enqueued_at, request.admitted_at)
              : 0.0;
      response.runtime_observability.ttft_ms =
          request.has_first_token_at
              ? duration_ms(request.enqueued_at, request.first_token_at)
              : 0.0;
      response.runtime_observability.mean_itl_ms =
          request.emitted_token_count > 1
              ? request.accumulated_itl_ms /
                    static_cast<double>(request.emitted_token_count - 1)
              : 0.0;
      response.runtime_observability.tail_itl_ms = request.tail_itl_ms;
      response.runtime_observability.e2e_ms =
          duration_ms(request.enqueued_at, request.completed_at);
      response.runtime_observability.total_ms =
          response.runtime_observability.e2e_ms;
      response.runtime_observability.input_token_count =
          static_cast<int32_t>(request.prompt_tokens.size());
      response.runtime_observability.output_token_count =
          slot.generated_tokens.empty()
              ? request.emitted_token_count
              : static_cast<int32_t>(slot.generated_tokens.size());
      response.runtime_observability.scheduler_tick_count =
          static_cast<int32_t>(slot.scheduler_tick_count);
      response.runtime_observability.batch_participation_count =
          static_cast<int32_t>(slot.batch_participation_count);
      response.runtime_observability.decode_eval_count =
          static_cast<int32_t>(slot.decode_step_count);
      response.runtime_observability.sample_count =
          response.runtime_observability.output_token_count;
      response.runtime_observability.decode_first_tick_count =
          request.decode_first_tick_count;
      response.runtime_observability.chunked_prefill_tick_count =
          request.chunked_prefill_tick_count;
      response.runtime_observability.mixed_workload_tick_count =
          request.mixed_workload_tick_count;
      response.runtime_observability.lcp_reuse_tokens =
          request.lcp_reuse_tokens;
      response.runtime_observability.prefix_cache_restore_tokens =
          request.prefix_cache_restore_tokens;
      response.runtime_observability.prefix_cache_hit_count =
          request.prefix_cache_hit_count;
      response.runtime_observability.prefix_cache_store_count =
          request.prefix_cache_store_count;
    }
    if (response.status == GenerateResponseStatus::Cancelled) {
      response.error_message = "Request cancelled.";
    } else if (slot.phase == SlotPhase::Failed) {
      response.error_message =
          slot.terminal_error_message.empty() ? "Request failed."
                                              : slot.terminal_error_message;
    }

    if (slot.session != nullptr) {
      session_store.Unpin(*slot.session);
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

  if (request != nullptr) {
    const auto now = std::chrono::steady_clock::now();
    if (!request->has_first_token_at) {
      request->first_token_at = now;
      request->has_first_token_at = true;
    } else if (request->has_last_token_at) {
      const double itl_ms = duration_ms(request->last_token_at, now);
      request->accumulated_itl_ms += itl_ms;
      request->tail_itl_ms = std::max(request->tail_itl_ms, itl_ms);
    }

    request->last_token_at = now;
    request->has_last_token_at = true;
    request->emitted_token_count++;
  }

  if (request != nullptr && request->on_token_received) {
    if (!request->on_token_received(
        slot.buffered_output_text.c_str(),
        static_cast<int32_t>(slot.buffered_output_text.size()))) {
      request->cancel_requested = true;
    }
  }

  slot.buffered_output_text.clear();
}

void SlotScheduler::FailActiveRequest(RequestQueue &request_queue,
                                      SessionStore &session_store,
                                      SlotState &slot,
                                      std::string error_message) {
  GenerateResponse response;
  response.request_id = slot.request_id;
  response.status =
      slot.request != nullptr && slot.request->cancel_requested
          ? GenerateResponseStatus::Cancelled
          : GenerateResponseStatus::Failed;
  response.error_message =
      response.status == GenerateResponseStatus::Cancelled
          ? "Request cancelled."
          : std::move(error_message);
  if (slot.request != nullptr) {
    GenerateRequest &request = *slot.request;
    request.completed_at = std::chrono::steady_clock::now();
    request.has_completed_at = true;
    response.runtime_observability.queue_delay_ms =
        request.has_admitted_at
            ? duration_ms(request.enqueued_at, request.admitted_at)
            : 0.0;
    response.runtime_observability.ttft_ms =
        request.has_first_token_at
            ? duration_ms(request.enqueued_at, request.first_token_at)
            : 0.0;
    response.runtime_observability.mean_itl_ms =
        request.emitted_token_count > 1
            ? request.accumulated_itl_ms /
                  static_cast<double>(request.emitted_token_count - 1)
            : 0.0;
    response.runtime_observability.tail_itl_ms = request.tail_itl_ms;
    response.runtime_observability.e2e_ms =
        duration_ms(request.enqueued_at, request.completed_at);
    response.runtime_observability.total_ms =
        response.runtime_observability.e2e_ms;
    response.runtime_observability.input_token_count =
        static_cast<int32_t>(request.prompt_tokens.size());
    response.runtime_observability.output_token_count =
        slot.generated_tokens.empty()
            ? request.emitted_token_count
            : static_cast<int32_t>(slot.generated_tokens.size());
    response.runtime_observability.scheduler_tick_count =
        static_cast<int32_t>(slot.scheduler_tick_count);
    response.runtime_observability.batch_participation_count =
        static_cast<int32_t>(slot.batch_participation_count);
    response.runtime_observability.decode_eval_count =
        static_cast<int32_t>(slot.decode_step_count);
    response.runtime_observability.sample_count =
        response.runtime_observability.output_token_count;
    response.runtime_observability.decode_first_tick_count =
        request.decode_first_tick_count;
    response.runtime_observability.chunked_prefill_tick_count =
        request.chunked_prefill_tick_count;
    response.runtime_observability.mixed_workload_tick_count =
        request.mixed_workload_tick_count;
    response.runtime_observability.lcp_reuse_tokens =
        request.lcp_reuse_tokens;
    response.runtime_observability.prefix_cache_restore_tokens =
        request.prefix_cache_restore_tokens;
    response.runtime_observability.prefix_cache_hit_count =
        request.prefix_cache_hit_count;
    response.runtime_observability.prefix_cache_store_count =
        request.prefix_cache_store_count;
  }
  if (slot.session != nullptr) {
    session_store.Unpin(*slot.session);
  }
  request_queue.MarkCompleted(std::move(response));
  slot.ResetToIdle();
}

} // namespace noumena::cogentengine
