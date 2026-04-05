/////////////////////////////////////////////////////////////////////////////////////////////////
//
// request_queue.h
//
// - Runtime-owned request admission queue.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <deque>
#include <functional>
#include <optional>
#include <unordered_map>

#include "runtime/request/request_types.h"
#include "runtime/request/response_types.h"

namespace noumena::cogentengine {

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
  bool ConsumeCompletedResponse(GenerateRequestId request_id);
  void Clear();

private:
  std::unordered_map<GenerateRequestId, GenerateRequest> requests_;
  std::deque<GenerateRequestId> pending_request_ids_;
  std::unordered_map<GenerateRequestId, GenerateResponse> completed_responses_;
};

} // namespace noumena::cogentengine
