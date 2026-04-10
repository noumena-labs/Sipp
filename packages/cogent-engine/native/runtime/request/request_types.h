/////////////////////////////////////////////////////////////////////////////////////////////////
//
// request_types.h
//
// - Explicit request ownership for the queued runtime.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "llama.h"

namespace noumena::cogentengine {

// Request ids cross the browser FFI boundary through Emscripten `ccall`.
// Keep them 32-bit so multi-argument exported calls preserve argument layout.
using GenerateRequestId = std::uint32_t;
using GenerateTokenCallback = std::function<bool(const char *, int32_t)>;

enum class GenerateRequestLifecycle : std::uint8_t {
  Pending = 0,
  Admitted,
  Running,
  Streaming,
  Completed,
  Cancelled,
  Failed,
};

struct GenerateRequest {
  GenerateRequestId id = 0;
  std::string context_key;
  std::vector<llama_token> prompt_tokens;
  int32_t max_output_tokens = 0;
  GenerateTokenCallback on_token_received;
  GenerateRequestLifecycle lifecycle = GenerateRequestLifecycle::Pending;
  std::chrono::steady_clock::time_point enqueued_at{};
  std::chrono::steady_clock::time_point admitted_at{};
  std::chrono::steady_clock::time_point first_token_at{};
  std::chrono::steady_clock::time_point last_token_at{};
  std::chrono::steady_clock::time_point completed_at{};
  bool has_admitted_at = false;
  bool has_first_token_at = false;
  bool has_last_token_at = false;
  bool has_completed_at = false;
  int32_t emitted_token_count = 0;
  double accumulated_itl_ms = 0.0;
  double tail_itl_ms = 0.0;
  double attributed_total_ms = 0.0;
  double attributed_prompt_eval_ms = 0.0;
  double attributed_decode_eval_ms = 0.0;
  double attributed_sample_ms = 0.0;
  int32_t attributed_prompt_eval_tokens = 0;
  int32_t attributed_decode_eval_count = 0;
  int32_t attributed_sample_count = 0;
  int32_t decode_first_tick_count = 0;
  int32_t chunked_prefill_tick_count = 0;
  int32_t mixed_workload_tick_count = 0;
  int32_t lcp_reuse_tokens = 0;
  int32_t prefix_cache_restore_tokens = 0;
  int32_t prefix_cache_hit_count = 0;
  int32_t prefix_cache_store_count = 0;
  bool cancel_requested = false;
};

} // namespace noumena::cogentengine
