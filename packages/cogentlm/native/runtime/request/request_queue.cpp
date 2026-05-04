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

void RequestQueue::RemovePendingRequestId(GenerateRequestId request_id) {
  const auto pending_it = pending_request_positions_.find(request_id);
  if (pending_it == pending_request_positions_.end()) {
    return;
  }
  pending_request_ids_.erase(pending_it->second);
  pending_request_positions_.erase(pending_it);
}

void RequestQueue::QueueCompletedResponseId(GenerateRequestId request_id) {
  if (request_id == 0 || !completed_responses_.contains(request_id)) {
    return;
  }
  RuntimeEvent event;
  event.kind = RuntimeEventKind::Terminal;
  event.request_id = request_id;
  event.status = completed_responses_.at(request_id).status;
  runtime_events_.push_back(std::move(event));
}

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
  request.attributed_total_ms = 0.0;
  request.attributed_prompt_eval_ms = 0.0;
  request.attributed_decode_eval_ms = 0.0;
  request.attributed_sample_ms = 0.0;
  request.attributed_prompt_eval_tokens = 0;
  request.attributed_decode_eval_count = 0;
  request.attributed_sample_count = 0;
  request.decode_first_tick_count = 0;
  request.chunked_prefill_tick_count = 0;
  request.mixed_workload_tick_count = 0;
  request.lcp_reuse_tokens = 0;
  request.prefix_cache_restore_tokens = 0;
  request.prefix_cache_hit_count = 0;
  request.prefix_cache_store_count = 0;
  request.cancel_requested = false;
  requests_.emplace(request_id, std::move(request));
  pending_request_ids_.push_back(request_id);
  pending_request_positions_[request_id] = std::prev(pending_request_ids_.end());
  return true;
}

std::optional<GenerateRequestId> RequestQueue::TryPopNext() {
  return TryPopNextAdmissible(
      [](const GenerateRequest &) { return true; });
}

std::optional<GenerateRequestId> RequestQueue::TryPopNextAdmissible(
    const std::function<bool(const GenerateRequest &)> &predicate) {
  auto pending_it = pending_request_ids_.begin();
  while (pending_it != pending_request_ids_.end()) {
    const GenerateRequestId request_id = *pending_it;
    const GenerateRequest *request = Find(request_id);
    if (request == nullptr) {
      pending_request_positions_.erase(request_id);
      pending_it = pending_request_ids_.erase(pending_it);
      continue;
    }
    if (!predicate(*request)) {
      ++pending_it;
      continue;
    }

    pending_request_positions_.erase(request_id);
    pending_request_ids_.erase(pending_it);

    if (GenerateRequest *mutable_request = FindMutable(request_id)) {
      mutable_request->lifecycle = GenerateRequestLifecycle::Admitted;
      mutable_request->admitted_at = std::chrono::steady_clock::now();
      mutable_request->has_admitted_at = true;
    }

    return request_id;
  }

  return std::nullopt;
}

GenerateRequest *RequestQueue::FindMutable(GenerateRequestId request_id) {
  auto it = requests_.find(request_id);
  return it == requests_.end() ? nullptr : &it->second;
}

const GenerateRequest *RequestQueue::Find(GenerateRequestId request_id) const {
  auto it = requests_.find(request_id);
  return it == requests_.end() ? nullptr : &it->second;
}

bool RequestQueue::Contains(GenerateRequestId request_id) const {
  return requests_.contains(request_id);
}

bool RequestQueue::Cancel(GenerateRequestId request_id, std::string error_message) {
  GenerateRequest *request = FindMutable(request_id);
  if (request == nullptr) {
    return false;
  }

  request->cancel_requested = true;
  if (request->lifecycle == GenerateRequestLifecycle::Pending) {
    RemovePendingRequestId(request_id);

    request->lifecycle = GenerateRequestLifecycle::Cancelled;
    request->completed_at = std::chrono::steady_clock::now();
    request->has_completed_at = true;

    GenerateResponse response;
    response.request_id = request_id;
    response.status = GenerateResponseStatus::Cancelled;
    response.error_message = std::move(error_message);
    completed_responses_[request_id] = std::move(response);
    QueueCompletedResponseId(request_id);
    return true;
  }

  return true;
}

void RequestQueue::MarkCompleted(GenerateResponse response) {
  GenerateRequest *request = FindMutable(response.request_id);
  if (request != nullptr) {
    RemovePendingRequestId(response.request_id);
    request->lifecycle =
        response.status == GenerateResponseStatus::Completed
            ? GenerateRequestLifecycle::Completed
            : (response.status == GenerateResponseStatus::Cancelled
                   ? GenerateRequestLifecycle::Cancelled
                   : GenerateRequestLifecycle::Failed);
  }

  completed_responses_[response.request_id] = std::move(response);
  QueueCompletedResponseId(response.request_id);
}

const GenerateResponse *
RequestQueue::PeekCompletedResponse(GenerateRequestId request_id) const {
  auto response_it = completed_responses_.find(request_id);
  return response_it == completed_responses_.end() ? nullptr : &response_it->second;
}

std::vector<GenerateRequestId> RequestQueue::CompletedResponseIds() const {
  std::vector<GenerateRequestId> request_ids;
  request_ids.reserve(completed_responses_.size());
  for (const auto &[request_id, _] : completed_responses_) {
    request_ids.push_back(request_id);
  }
  return request_ids;
}

void RequestQueue::QueueTokenEvent(GenerateRequestId request_id, std::string text) {
  if (request_id == 0 || text.empty()) {
    return;
  }

  RuntimeEvent event;
  event.kind = RuntimeEventKind::Token;
  event.request_id = request_id;
  event.text = std::move(text);
  runtime_events_.push_back(std::move(event));
  total_emitted_token_count_++;
}

std::vector<RuntimeEvent> RequestQueue::DrainRuntimeEvents(std::size_t max_count,
                                                           std::size_t max_text_bytes) {
  std::vector<RuntimeEvent> events;
  const std::size_t drain_limit = max_count == 0 ? runtime_events_.size() : max_count;
  events.reserve(std::min(drain_limit, runtime_events_.size()));

  std::size_t used_text_bytes = 0;
  while (!runtime_events_.empty() && events.size() < drain_limit) {
    RuntimeEvent &event = runtime_events_.front();

    const std::size_t required_text_bytes =
        event.kind == RuntimeEventKind::Token ? event.text.size() + 1 : 0;
    if (required_text_bytes > 0 &&
        used_text_bytes + required_text_bytes > max_text_bytes) {
      break;
    }

    used_text_bytes += required_text_bytes;
    events.push_back(std::move(event));
    runtime_events_.pop_front();
  }

  return events;
}

int32_t RequestQueue::TotalEmittedTokenCount() const {
  return total_emitted_token_count_;
}

bool RequestQueue::ConsumeCompletedResponse(GenerateRequestId request_id) {
  auto response_it = completed_responses_.find(request_id);
  if (response_it == completed_responses_.end()) {
    return false;
  }

  RemovePendingRequestId(request_id);
  completed_responses_.erase(response_it);
  requests_.erase(request_id);
  return true;
}

std::size_t RequestQueue::CompletedResponseCount() const {
  return completed_responses_.size();
}

void RequestQueue::Clear() {
  requests_.clear();
  pending_request_ids_.clear();
  pending_request_positions_.clear();
  completed_responses_.clear();
  runtime_events_.clear();
  total_emitted_token_count_ = 0;
}

} // namespace noumena::cogentengine
