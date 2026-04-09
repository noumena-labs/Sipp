/////////////////////////////////////////////////////////////////////////////////////////////////
//
// slot_state.h
//
// - Explicit slot-owned execution state.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"
#include "runtime/request/request_types.h"
#include "runtime/session/session_store.h"

namespace noumena::cogentengine {

enum class SlotPhase : std::uint8_t {
  Idle = 0,
  Admitted,
  Prefill,
  Decode,
  Streaming,
  Completed,
  Failed,
};

struct SlotState {
  std::size_t slot_id = 0;
  llama_seq_id seq_id = -1;
  SlotPhase phase = SlotPhase::Idle;
  GenerateRequestId request_id = 0;
  GenerateRequest *request = nullptr;
  SequenceState *session = nullptr;
  std::size_t prefill_cursor = 0;
  std::size_t decode_step_count = 0;
  std::size_t batch_participation_count = 0;
  std::vector<llama_token> generated_tokens;
  std::string output_text;
  std::string buffered_output_text;
  std::string terminal_error_message;
  llama_sampler *sampler = nullptr;

  void ResetToIdle();
  void AttachRequest(GenerateRequest &request_ref, SequenceState &session_ref);
};

} // namespace noumena::cogentengine
