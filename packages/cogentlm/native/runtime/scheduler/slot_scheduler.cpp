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
  generated_tokens.clear();
  output_text.clear();
  buffered_output_text.clear();
  pending_utf8_bytes.clear();
  terminal_error_message.clear();
  sampler = nullptr;
  mirror.current_kv_tokens.clear();
  mirror.n_past = 0;
  mirror.hardware_id = -1;
}

void SlotState::AttachRequest(GenerateRequest &request_ref,
                              SequenceState &session_ref) {
  request_id = request_ref.id;
  request = &request_ref;
  session = &session_ref;
  mirror.current_kv_tokens.clear();
  mirror.n_past = 0;
  mirror.hardware_id = session_ref.hardware_id;
  phase = SlotPhase::Admitted;
  prefill_cursor = 0;
  decode_step_count = 0;
  batch_participation_count = 0;
  generated_tokens.clear();
  if (request_ref.max_output_tokens > 0) {
    generated_tokens.reserve(static_cast<std::size_t>(
        std::max<int32_t>(1, request_ref.max_output_tokens)));
    output_text.reserve(static_cast<std::size_t>(
        std::max<int32_t>(16, request_ref.max_output_tokens * 4)));
  }
  output_text.clear();
  buffered_output_text.clear();
  pending_utf8_bytes.clear();
  terminal_error_message.clear();
}

void SlotScheduler::Resize(std::size_t slot_count) {
  slots_.resize(slot_count);
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    slots_[i].slot_id = i;
    slots_[i].seq_id = -1;
    if (slots_[i].phase == SlotPhase::Idle && slots_[i].request == nullptr) {
      continue;
    }
    slots_[i].ResetToIdle();
    slots_[i].slot_id = i;
    slots_[i].seq_id = -1;
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

void SlotScheduler::SelectDecodeReadySlots(
    std::vector<SlotState *> &out_slots) {
  out_slots.clear();
  out_slots.reserve(slots_.size());

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
    if (slot.generated_tokens.empty()) {
      continue;
    }
    if (!slot.buffered_output_text.empty()) {
      continue;
    }
    out_slots.push_back(&slot);
  }
}

void SlotScheduler::SelectPrefillReadySlots(
    std::vector<SlotState *> &out_slots) {
  out_slots.clear();
  out_slots.reserve(slots_.size());

  // Phase 4 algorithm steps:
  // 1. Admit only slots that still have prompt tokens left to prefill.
  // 2. Keep selection free of fairness heuristics; chunking and reservation
  //    policy belong in the tick budget and batch planner.
  // 3. Preserve slot order so later policy behavior is explainable.
  for (SlotState &slot : slots_) {
    if (slot.request == nullptr || slot.session == nullptr) {
      continue;
    }
    if (slot.phase != SlotPhase::Prefill && slot.phase != SlotPhase::Admitted) {
      continue;
    }
    if (slot.request->is_multimodal_turn &&
        slot.request->multimodal.has_value()) {
      out_slots.push_back(&slot);
      continue;
    }
    if (slot.prefill_cursor >= slot.request->prompt_tokens.size()) {
      continue;
    }
    out_slots.push_back(&slot);
  }
}

SchedulerTickBudget
SlotScheduler::BuildTickBudget(const SchedulerPolicyConfig &policy,
                               int32_t decode_ready_count,
                               int32_t prefill_ready_count,
                               int32_t max_batch_tokens,
                               int32_t prefill_chunk_size) {
  (void)prefill_chunk_size;
  SchedulerTickBudget budget;
  budget.total_token_budget = std::max(0, max_batch_tokens);
  budget.decode_first = decode_ready_count > 0;

  if (budget.total_token_budget <= 0) {
    return budget;
  }

  const int32_t clamped_decode_ready =
      std::max<int32_t>(0, decode_ready_count);
  const int32_t clamped_prefill_ready =
      std::max<int32_t>(0, prefill_ready_count);

  if (clamped_decode_ready == 0) {
    budget.reserved_decode_tokens = 0;
    budget.reserved_prefill_tokens = budget.total_token_budget;
    return budget;
  }

  if (clamped_prefill_ready == 0) {
    budget.reserved_decode_tokens =
        std::min(clamped_decode_ready, budget.total_token_budget);
    budget.reserved_prefill_tokens =
        budget.total_token_budget - budget.reserved_decode_tokens;
    return budget;
  }

  const int32_t requested_decode_reserve =
      policy.decode_token_reserve > 0
          ? std::min(policy.decode_token_reserve, clamped_decode_ready)
          : clamped_decode_ready;
  const int32_t decode_ready_budget =
      std::min(clamped_decode_ready, budget.total_token_budget);

  switch (policy.mode) {
  case SchedulerPolicyMode::LatencyFirst:
    // Decode latency wins. Prefill uses leftover capacity only.
    budget.reserved_decode_tokens =
        policy.decode_token_reserve > 0
            ? std::min(decode_ready_budget, requested_decode_reserve)
            : decode_ready_budget;
    break;
  case SchedulerPolicyMode::ThroughputFirst: {
    // Keep decode alive, but bias the shared batch toward prompt work.
    const int32_t prefill_floor =
        budget.total_token_budget > 1
            ? std::max<int32_t>(1, (budget.total_token_budget * 3) / 4)
            : 0;
    const int32_t decode_ceiling =
        std::max<int32_t>(1, budget.total_token_budget - prefill_floor);
    const int32_t throughput_reserve =
        policy.decode_token_reserve > 0 ? requested_decode_reserve : 1;
    budget.reserved_decode_tokens =
        std::min({decode_ready_budget, decode_ceiling, throughput_reserve});
    break;
  }
  case SchedulerPolicyMode::Balanced:
  default: {
    // Preserve decode responsiveness while leaving room for prefill whenever
    // the batch has more than one token of capacity. With n_batch=1, decode
    // must not be starved by the prefill floor.
    const int32_t prefill_floor = budget.total_token_budget > 1 ? 1 : 0;
    const int32_t decode_ceiling =
        std::max(0, budget.total_token_budget - prefill_floor);
    budget.reserved_decode_tokens =
        std::min(decode_ready_budget, decode_ceiling);
    if (policy.decode_token_reserve > 0) {
      budget.reserved_decode_tokens =
          std::min(budget.reserved_decode_tokens, requested_decode_reserve);
    }
    break;
  }
  }

  budget.reserved_prefill_tokens =
      std::max(0, budget.total_token_budget - budget.reserved_decode_tokens);

  return budget;
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
      request_queue.TryPopNextAdmissible(
          [&session_store](const GenerateRequest &request) {
            return session_store.CanAdmit(request.context_key);
          });
  if (!next_request_id.has_value()) {
    return false;
  }

  GenerateRequest *request = request_queue.FindMutable(*next_request_id);
  if (request == nullptr) {
    return false;
  }

  SequenceState &session = session_store.GetOrCreateSession(request->context_key);
  
  const llama_seq_id leased_seq_id = session_store.AcquireSeqId(-1);
  if (leased_seq_id < 0) {
    GenerateResponse response;
    response.request_id = request->id;
    response.status = GenerateResponseStatus::Failed;
    response.error_message = "Failed to acquire a hardware sequence ID.";
    request_queue.MarkCompleted(std::move(response));
    return false;
  }

  // LEASE FRESH ID: To ensure absolute physical isolation and prevent stale 
  // sequence leakage, we always lease a fresh hardware ID for a new request.
  // The SessionStore will explicitly scrub the hardware clean upon leasing.
  session.hardware_id = leased_seq_id;
  session.current_kv_tokens.clear();
  session.n_past = 0;

  if (request->is_multimodal_turn) {
    session.current_kv_tokens.clear();
    session.n_past = 0;
    // Note: If we had a sticky hit, multimodal still requires a reset.
    // If we didn't have a sticky hit, it's already cleared above.
    if (leased_seq_id >= 0) {
      session_store.ClearSequenceMemory(leased_seq_id);
    }
  }

  session_store.Pin(request->context_key);
  idle_slot_it->AttachRequest(*request, session);
  idle_slot_it->seq_id = leased_seq_id;
  idle_slot_it->phase = SlotPhase::Prefill;
  return true;
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
          request.attributed_total_ms > 0.0
              ? request.attributed_total_ms
              : response.runtime_observability.e2e_ms;
      response.runtime_observability.prompt_eval_ms =
          request.attributed_prompt_eval_ms;
      response.runtime_observability.decode_eval_ms =
          request.attributed_decode_eval_ms;
      response.runtime_observability.sample_ms =
          request.attributed_sample_ms;
      response.runtime_observability.native_policy_prepare_ms =
          request.attributed_native_policy_prepare_ms;
      response.runtime_observability.native_policy_plan_ms =
          request.attributed_native_policy_plan_ms;
      response.runtime_observability.native_batch_build_ms =
          request.attributed_native_batch_build_ms;
      response.runtime_observability.native_llama_decode_wall_ms =
          request.attributed_native_llama_decode_wall_ms;
      response.runtime_observability.native_synchronize_ms =
          request.attributed_native_synchronize_ms;
      response.runtime_observability.native_kv_update_ms =
          request.attributed_native_kv_update_ms;
      response.runtime_observability.native_sampler_wall_ms =
          request.attributed_native_sampler_wall_ms;
      response.runtime_observability.native_token_emit_ms =
          request.attributed_native_token_emit_ms;
      response.runtime_observability.native_prefix_cache_ms =
          request.attributed_native_prefix_cache_ms;
      response.runtime_observability.native_observability_ms =
          request.attributed_native_observability_ms;
      response.runtime_observability.input_token_count =
          static_cast<int32_t>(request.prompt_tokens.size());
      response.runtime_observability.prompt_eval_tokens =
          request.attributed_prompt_eval_tokens;
      response.runtime_observability.output_token_count =
          request.emitted_token_count;
      response.runtime_observability.first_sampled_token_id =
          request.first_sampled_token_id;
      response.runtime_observability.batch_participation_count =
          static_cast<int32_t>(slot.batch_participation_count);
      response.runtime_observability.decode_eval_count =
          request.attributed_decode_eval_count > 0
              ? request.attributed_decode_eval_count
              : static_cast<int32_t>(slot.decode_step_count);
      response.runtime_observability.sample_count =
          request.attributed_sample_count > 0
              ? request.attributed_sample_count
              : static_cast<int32_t>(slot.generated_tokens.size());
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
      response.runtime_observability.native_policy_tick_count =
          request.attributed_native_policy_tick_count;
      response.runtime_observability.native_scheduler_tick_count = 0;
    }
    if (response.status == GenerateResponseStatus::Cancelled) {
      response.error_message = "Request cancelled.";
    } else if (slot.phase == SlotPhase::Failed) {
      response.error_message =
          slot.terminal_error_message.empty() ? "Request failed."
                                              : slot.terminal_error_message;
    }

    if (slot.request != nullptr && slot.session != nullptr) {
      slot.session->current_kv_tokens = slot.mirror.current_kv_tokens;
      slot.session->n_past = slot.mirror.n_past;
      slot.session->hardware_id = slot.mirror.hardware_id;
    }

    if (slot.request != nullptr) {
      const std::string context_key = slot.request->context_key;
      const bool drop_multimodal_session = slot.request->is_multimodal_turn;
      session_store.Unpin(context_key);
      if (drop_multimodal_session) {
        session_store.Remove(context_key);
      }
    }
    
    if (slot.seq_id >= 0) {
      session_store.ReleaseSeqId(slot.seq_id);
      slot.seq_id = -1;
    }
    
    request_queue.MarkCompleted(std::move(response));
    slot.ResetToIdle();
  }
}

void SlotScheduler::EmitBufferedTokenPiece(RequestQueue &request_queue,
                                           SlotState &slot) {
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
    if (request->token_emission_mode ==
        GenerateTokenEmissionMode::RuntimeEvents) {
      request_queue.QueueTokenEvent(request->id, slot.buffered_output_text);
    }
  }

  if (request != nullptr &&
      request->token_emission_mode ==
          GenerateTokenEmissionMode::DirectCallback &&
      request->on_token_received) {
    if (!request->on_token_received(
        slot.buffered_output_text.c_str(),
        static_cast<int32_t>(slot.buffered_output_text.size()))) {
      request->cancel_requested = true;
    }
  }

  slot.buffered_output_text.clear();
}

} // namespace noumena::cogentengine
