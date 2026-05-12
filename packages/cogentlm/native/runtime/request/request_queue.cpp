/////////////////////////////////////////////////////////////////////////////////////////////////
//
// request_queue.cpp
//
// - Runtime-owned request admission queue.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/request/request_queue.h"

#include <chrono>
#include <cstring>

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
  request.has_admitted_at = false;
  request.has_first_token_at = false;
  request.has_last_token_at = false;
  request.has_completed_at = false;
  request.emitted_token_count = 0;
  request.itl_sum_ms = 0.0;
  request.itl_p99_ms = 0.0;
  request.e2e_ms = 0.0;
  request.prefill_ms = 0.0;
  request.decode_ms = 0.0;
  request.native_logic_ms = 0.0;
  request.inter_decode_js_ms = 0.0;
  request.yield_wait_ms = 0.0;
  request.input_tokens = 0;
  request.output_tokens = 0;
  request.cache_hits = 0;
  request.first_sampled_token_id = -1;
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

GenerateResponse *
RequestQueue::FindMutableCompletedResponse(GenerateRequestId request_id) {
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

void RequestQueue::AppendStreamingToken(GenerateRequestId request_id,
                                        const std::string &text) {
  if (request_id == 0 || text.empty()) {
    return;
  }
  // [u32 LE requestId | u32 LE textLength | bytes...].  Wasm is little-
  // endian; memcpy of u32 fields matches the wire format JS expects.
  const std::size_t text_size = text.size();
  const std::size_t record_size = 8 + text_size;
  if (record_size > streaming_buffer_.size() ||
      static_cast<std::size_t>(streaming_buffer_used_) + record_size >
          streaming_buffer_.size()) {
    streaming_buffer_drop_count_++;
    return;
  }
  uint8_t *dst = streaming_buffer_.data() +
                 static_cast<std::size_t>(streaming_buffer_used_);
  const uint32_t request_id_u32 = static_cast<uint32_t>(request_id);
  const uint32_t text_length_u32 = static_cast<uint32_t>(text_size);
  std::memcpy(dst, &request_id_u32, sizeof(uint32_t));
  std::memcpy(dst + sizeof(uint32_t), &text_length_u32, sizeof(uint32_t));
  if (text_size > 0) {
    std::memcpy(dst + 2 * sizeof(uint32_t), text.data(), text_size);
  }
  streaming_buffer_used_ += static_cast<int32_t>(record_size);
  total_emitted_token_count_++;
}

const uint8_t *RequestQueue::StreamingBufferPointer() const {
  return streaming_buffer_.data();
}

std::size_t RequestQueue::StreamingBufferCapacity() const {
  return streaming_buffer_.size();
}

int32_t *RequestQueue::StreamingBufferUsedAddress() {
  return &streaming_buffer_used_;
}

int32_t *RequestQueue::StreamingBufferDropCountAddress() {
  return &streaming_buffer_drop_count_;
}

void RequestQueue::RemoveStreamingTokenRecordsForRequest(
    GenerateRequestId request_id) {
  if (request_id == 0 || streaming_buffer_used_ == 0) {
    return;
  }
  // Linear scan + in-place compaction.  Runs at settle time, not hot path.
  std::size_t read_offset = 0;
  std::size_t write_offset = 0;
  const std::size_t used = static_cast<std::size_t>(streaming_buffer_used_);
  while (read_offset + 8 <= used) {
    uint32_t record_request_id = 0;
    uint32_t record_text_length = 0;
    std::memcpy(&record_request_id, streaming_buffer_.data() + read_offset,
                sizeof(uint32_t));
    std::memcpy(&record_text_length,
                streaming_buffer_.data() + read_offset + sizeof(uint32_t),
                sizeof(uint32_t));
    const std::size_t record_size = 8 + record_text_length;
    if (read_offset + record_size > used) {
      break;
    }
    if (record_request_id != static_cast<uint32_t>(request_id)) {
      if (write_offset != read_offset) {
        std::memmove(streaming_buffer_.data() + write_offset,
                     streaming_buffer_.data() + read_offset, record_size);
      }
      write_offset += record_size;
    }
    read_offset += record_size;
  }
  streaming_buffer_used_ = static_cast<int32_t>(write_offset);
}

std::vector<RuntimeEvent> RequestQueue::DrainEvents(std::size_t max_count) {
  std::vector<RuntimeEvent> events;
  const std::size_t drain_limit =
      max_count == 0 ? runtime_events_.size() : max_count;
  events.reserve(std::min(drain_limit, runtime_events_.size()));

  while (!runtime_events_.empty() && events.size() < drain_limit) {
    events.push_back(std::move(runtime_events_.front()));
    runtime_events_.pop_front();
  }

  return events;
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

std::size_t RequestQueue::LiveRequestCount() const {
  return requests_.size() - completed_responses_.size();
}

void RequestQueue::Clear() {
  requests_.clear();
  pending_request_ids_.clear();
  pending_request_positions_.clear();
  completed_responses_.clear();
  runtime_events_.clear();
  total_emitted_token_count_ = 0;
  streaming_buffer_used_ = 0;
  streaming_buffer_drop_count_ = 0;
}

} // namespace noumena::cogentengine
