/////////////////////////////////////////////////////////////////////////////////////////////////
//
// request_queue.h
//
// - Runtime-owned request admission queue.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <deque>
#include <optional>
#include <unordered_map>

#include "runtime/request/request_types.h"
#include "runtime/request/response_types.h"

namespace noumena::cogentengine {

class RequestQueue {
public:
  bool Push(GenerateRequest request);
  std::optional<GenerateRequestId> TryPopNext();
  GenerateRequest *FindMutable(GenerateRequestId request_id);
  bool Cancel(GenerateRequestId request_id, std::string error_message);
  void MarkCompleted(GenerateResponse response);
  std::optional<GenerateResponse>
  TakeCompletedResponse(GenerateRequestId request_id);
  void Clear();

private:
  std::unordered_map<GenerateRequestId, GenerateRequest> requests_;
  std::deque<GenerateRequestId> pending_request_ids_;
  std::unordered_map<GenerateRequestId, GenerateResponse> completed_responses_;
};

} // namespace noumena::cogentengine
