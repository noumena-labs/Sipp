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
#include <optional>
#include <string>
#include <vector>

#include "llama.h"

namespace noumena::cogentengine {

// Request ids cross the browser FFI boundary through Emscripten `ccall`.
// Keep them 32-bit so multi-argument exported calls preserve argument layout.
using GenerateRequestId = std::uint32_t;
using GenerateTokenCallback = std::function<bool(const char *, int32_t)>;

enum class GenerateTokenEmissionMode : std::uint8_t {
  None = 0,
  RuntimeEvents = 1,
  DirectCallback = 2,
  // Native appends UTF-8 bytes to RequestQueue's streaming buffer; JS
  // drains into the SAB ring on each ce_native_yield.
  StreamingBuffer = 3,
};

enum class GenerateRequestLifecycle : std::uint8_t {
  Pending = 0,
  Admitted,
  Running,
  Streaming,
  Completed,
  Cancelled,
  Failed,
};

struct MultimodalPayload {
  std::vector<std::vector<std::uint8_t>> image_buffers;
};

struct GenerateRequest {
  GenerateRequestId id = 0;
  std::string context_key;
  std::string original_prompt;
  // Optional GBNF grammar source. When non-empty the slot sampler is built
  // with llama_sampler_init_grammar prepended to constrain decoding output.
  std::string grammar;
  std::vector<llama_token> prompt_tokens;
  std::optional<MultimodalPayload> multimodal;
  int32_t max_output_tokens = 0;
  GenerateTokenCallback on_token_received;
  GenerateTokenEmissionMode token_emission_mode =
      GenerateTokenEmissionMode::None;
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
  double itl_sum_ms = 0.0;
  double itl_p99_ms = 0.0;
  double e2e_ms = 0.0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double native_sync_ms = 0.0;
  double native_gpu_ms = 0.0;
  double native_logic_ms = 0.0;
  // Cumulative JS-work window between successive syncs, billed to each
  // request participating in the tick.  See observability_metrics.h.
  double inter_decode_js_ms = 0.0;
  double yield_wait_ms = 0.0;
  int32_t input_tokens = 0;
  int32_t output_tokens = 0;
  int32_t cache_hits = 0;
  int32_t first_sampled_token_id = -1;
  bool is_multimodal_turn = false;
  bool cancel_requested = false;
};

} // namespace noumena::cogentengine
