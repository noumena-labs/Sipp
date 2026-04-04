/////////////////////////////////////////////////////////////////////////////////////////////////
//
// request_queue.cpp
//
// - Runtime-owned request admission queue.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/request/request_queue.h"

#include <chrono>

namespace noumena::cogentengine {

bool RequestQueue::Push(GenerateRequest request) {
  const GenerateRequestId request_id = request.id;
  if (request_id == 0 || requests_.contains(request_id)) {
    return false;
  }

  request.lifecycle = GenerateRequestLifecycle::Pending;
  request.enqueued_at = std::chrono::steady_clock::now();
  request.has_admitted_at = false;
  request.has_first_token_at = false;
  request.has_last_token_at = false;
  request.has_completed_at = false;
  request.emitted_token_count = 0;
  request.accumulated_itl_ms = 0.0;
  request.tail_itl_ms = 0.0;
  request.decode_first_tick_count = 0;
  request.chunked_prefill_tick_count = 0;
  request.mixed_workload_tick_count = 0;
  request.lcp_reuse_tokens = 0;
  request.prefix_cache_restore_tokens = 0;
  request.prefix_cache_hit_count = 0;
  request.prefix_cache_store_count = 0;
  requests_.emplace(request_id, std::move(request));
  pending_request_ids_.push_back(request_id);
  return true;
}

std::optional<GenerateRequestId> RequestQueue::TryPopNext() {
  if (pending_request_ids_.empty()) {
    return std::nullopt;
  }

  const GenerateRequestId request_id = pending_request_ids_.front();
  pending_request_ids_.pop_front();

  if (GenerateRequest *request = FindMutable(request_id)) {
    request->lifecycle = GenerateRequestLifecycle::Admitted;
    request->admitted_at = std::chrono::steady_clock::now();
    request->has_admitted_at = true;
  }

  return request_id;
}

GenerateRequest *RequestQueue::FindMutable(GenerateRequestId request_id) {
  auto it = requests_.find(request_id);
  return it == requests_.end() ? nullptr : &it->second;
}

void RequestQueue::MarkCompleted(GenerateResponse response) {
  GenerateRequest *request = FindMutable(response.request_id);
  if (request != nullptr) {
    request->lifecycle = response.status == GenerateResponseStatus::Completed
                             ? GenerateRequestLifecycle::Completed
                             : GenerateRequestLifecycle::Failed;
  }

  completed_responses_[response.request_id] = std::move(response);
}

std::optional<GenerateResponse>
RequestQueue::TakeCompletedResponse(GenerateRequestId request_id) {
  auto response_it = completed_responses_.find(request_id);
  if (response_it == completed_responses_.end()) {
    return std::nullopt;
  }

  GenerateResponse response = std::move(response_it->second);
  completed_responses_.erase(response_it);
  requests_.erase(request_id);
  return response;
}

void RequestQueue::Clear() {
  requests_.clear();
  pending_request_ids_.clear();
  completed_responses_.clear();
}

} // namespace noumena::cogentengine
