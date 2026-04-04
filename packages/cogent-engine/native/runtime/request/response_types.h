/////////////////////////////////////////////////////////////////////////////////////////////////
//
// response_types.h
//
// - Explicit response ownership for the queued runtime.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <string>

#include "runtime/metrics/perf_counters.h"
#include "runtime/request/request_types.h"

namespace noumena::cogentengine {

enum class GenerateResponseStatus : std::uint8_t {
  Pending = 0,
  Completed,
  Failed,
};

struct GenerateResponse {
  GenerateRequestId request_id = 0;
  GenerateResponseStatus status = GenerateResponseStatus::Pending;
  std::string output_text;
  std::string error_message;
  PromptPerfStats perf;
};

} // namespace noumena::cogentengine
