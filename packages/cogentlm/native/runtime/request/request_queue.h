/////////////////////////////////////////////////////////////////////////////////////////////////
//
// request_queue.h
//
// - Runtime-owned request admission queue.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>
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
  bool Contains(GenerateRequestId request_id) const;
  bool Cancel(GenerateRequestId request_id, std::string error_message);
  void MarkCompleted(GenerateResponse response);
  const GenerateResponse *PeekCompletedResponse(GenerateRequestId request_id) const;
  GenerateResponse *FindMutableCompletedResponse(GenerateRequestId request_id);
  std::vector<GenerateRequestId> CompletedResponseIds() const;
  void QueueTokenEvent(GenerateRequestId request_id, std::string text);
  std::vector<RuntimeEvent> DrainEvents(std::size_t max_count);
  std::vector<RuntimeEvent> DrainRuntimeEvents(std::size_t max_count,
                                               std::size_t max_text_bytes);
  int32_t TotalEmittedTokenCount() const;
  bool ConsumeCompletedResponse(GenerateRequestId request_id);
  std::size_t CompletedResponseCount() const;
  std::size_t LiveRequestCount() const;
  void Clear();

  // Streaming token buffer (StreamingBuffer emission mode).  Single-
  // producer (inference tick) / single-consumer (JS drain inside
  // ce_native_yield while wasm is suspended).  Records are
  // [u32 LE requestId | u32 LE textLength | bytes...].  Overflow bumps
  // `streaming_buffer_drop_count_`; the request's full `output_text` is
  // still preserved by the slot scheduler.
  void AppendStreamingToken(GenerateRequestId request_id,
                            const std::string &text);
  const uint8_t *StreamingBufferPointer() const;
  std::size_t StreamingBufferCapacity() const;
  // Pointers to i32 cells JS reads/writes via HEAP32 directly (no ccall).
  int32_t *StreamingBufferUsedAddress();
  int32_t *StreamingBufferDropCountAddress();
  // Strips not-yet-drained records for a request.  Called on cancel/finalize.
  void RemoveStreamingTokenRecordsForRequest(GenerateRequestId request_id);

private:
  void RemovePendingRequestId(GenerateRequestId request_id);
  void QueueCompletedResponseId(GenerateRequestId request_id);

  std::unordered_map<GenerateRequestId, GenerateRequest> requests_;
  std::list<GenerateRequestId> pending_request_ids_;
  std::unordered_map<GenerateRequestId, std::list<GenerateRequestId>::iterator>
      pending_request_positions_;
  std::unordered_map<GenerateRequestId, GenerateResponse> completed_responses_;
  std::deque<RuntimeEvent> runtime_events_;
  int32_t total_emitted_token_count_ = 0;

  // 64 KB ≈ 25 yield windows of headroom at 200 TPS with 5-byte tokens.
  static constexpr std::size_t kStreamingBufferCapacity = 64 * 1024;
  std::vector<uint8_t> streaming_buffer_ =
      std::vector<uint8_t>(kStreamingBufferCapacity);
  // i32 so JS can read/write them as single HEAP32 slots.
  int32_t streaming_buffer_used_ = 0;
  int32_t streaming_buffer_drop_count_ = 0;
};

} // namespace noumena::cogentengine
