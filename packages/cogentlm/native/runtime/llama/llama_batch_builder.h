/////////////////////////////////////////////////////////////////////////////////////////////////
//
// llama_batch_builder.h
//
// - Reusable llama_batch ownership for Phase 3 shared batching.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

#include "llama.h"

namespace noumena::cogentengine {

class LlamaBatchBuilder {
public:
  LlamaBatchBuilder() = default;
  ~LlamaBatchBuilder();

  LlamaBatchBuilder(const LlamaBatchBuilder &) = delete;
  LlamaBatchBuilder &operator=(const LlamaBatchBuilder &) = delete;

  void EnsureCapacity(int32_t max_tokens, int32_t max_sequences);
  void Reset();

  bool AddToken(llama_token token, int32_t position, llama_seq_id seq_id,
                bool request_logits);

  bool IsAllocated() const;
  int32_t CapacityTokens() const;
  int32_t CapacitySequences() const;

  llama_batch &Get();
  const llama_batch &Get() const;

private:
  void Free();

  llama_batch batch_ = {};
  int32_t capacity_tokens_ = 0;
  int32_t capacity_sequences_ = 0;
  bool is_allocated_ = false;
};

} // namespace noumena::cogentengine
