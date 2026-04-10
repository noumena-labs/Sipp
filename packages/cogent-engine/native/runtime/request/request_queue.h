/////////////////////////////////////////////////////////////////////////////////////////////////
//
// request_queue.h
//
// - Runtime-owned request admission queue.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <list>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runtime/request/request_types.h"
#include "runtime/request/response_types.h"

namespace noumena::cogentengine {

enum class RuntimeEventKind : std::uint8_t {
  Token = 1,
  Terminal = 2,
};

struct RuntimeEvent {
  RuntimeEventKind kind = RuntimeEventKind::Token;
  GenerateRequestId request_id = 0;
  GenerateResponseStatus status = GenerateResponseStatus::Pending;
  std::string text;
};

class RequestQueue {
public:
  bool Push(GenerateRequest request);
  std::optional<GenerateRequestId> TryPopNext();
  std::optional<GenerateRequestId>
  TryPopNextAdmissible(const std::function<bool(const GenerateRequest &)> &predicate);
  GenerateRequest *FindMutable(GenerateRequestId request_id);
  const GenerateRequest *Find(GenerateRequestId request_id) const;
  bool Cancel(GenerateRequestId request_id, std::string error_message);
  void MarkCompleted(GenerateResponse response);
  const GenerateResponse *PeekCompletedResponse(GenerateRequestId request_id) const;
  std::vector<GenerateRequestId> CompletedResponseIds() const;
  std::vector<GenerateRequestId> DrainCompletedResponseIds(std::size_t max_count);
  void QueueTokenEvent(GenerateRequestId request_id, std::string text);
  std::vector<RuntimeEvent> DrainRuntimeEvents(std::size_t max_count,
                                               std::size_t max_text_bytes);
  int32_t TotalEmittedTokenCount() const;
  bool ConsumeCompletedResponse(GenerateRequestId request_id);
  std::size_t CompletedResponseCount() const;
  void Clear();

private:
  void RemovePendingRequestId(GenerateRequestId request_id);
  void QueueCompletedResponseId(GenerateRequestId request_id);

  std::unordered_map<GenerateRequestId, GenerateRequest> requests_;
  std::list<GenerateRequestId> pending_request_ids_;
  std::unordered_map<GenerateRequestId, std::list<GenerateRequestId>::iterator>
      pending_request_positions_;
  std::unordered_map<GenerateRequestId, GenerateResponse> completed_responses_;
  std::deque<GenerateRequestId> completed_response_ready_ids_;
  std::deque<RuntimeEvent> runtime_events_;
  std::unordered_set<GenerateRequestId> queued_completed_response_ids_;
};

} // namespace noumena::cogentengine
