/////////////////////////////////////////////////////////////////////////////////////////////////
//
// request_types.h
//
// - Explicit request ownership for the queued runtime.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "llama.h"

namespace noumena::cogentengine {

using GenerateRequestId = std::uint64_t;
using GenerateTokenCallback = std::function<void(const char *, int32_t)>;

enum class GenerateRequestLifecycle : std::uint8_t {
  Pending = 0,
  Admitted,
  Running,
  Streaming,
  Completed,
  Failed,
};

struct GenerateRequest {
  GenerateRequestId id = 0;
  std::string context_key;
  std::string prompt_text;
  std::vector<llama_token> prompt_tokens;
  int32_t max_output_tokens = 0;
  GenerateTokenCallback on_token_received;
  GenerateRequestLifecycle lifecycle = GenerateRequestLifecycle::Pending;
};

} // namespace noumena::cogentengine
