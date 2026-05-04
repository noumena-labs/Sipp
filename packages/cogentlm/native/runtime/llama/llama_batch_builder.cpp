/////////////////////////////////////////////////////////////////////////////////////////////////
//
// llama_batch_builder.cpp
//
// - Reusable llama_batch ownership for Phase 3 shared batching.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/llama/llama_batch_builder.h"

#include <algorithm>

#include "runtime/llama/llama_utils.h"

namespace noumena::cogentengine {

LlamaBatchBuilder::~LlamaBatchBuilder() { Free(); }

void LlamaBatchBuilder::EnsureCapacity(int32_t max_tokens,
                                       int32_t max_sequences) {
  max_tokens = std::max<int32_t>(1, max_tokens);
  max_sequences = std::max<int32_t>(1, max_sequences);

  if (is_allocated_ && capacity_tokens_ == max_tokens &&
      capacity_sequences_ == max_sequences) {
    Reset();
    return;
  }

  Free();
  batch_ = llama_batch_init(max_tokens, 0, max_sequences);
  capacity_tokens_ = max_tokens;
  capacity_sequences_ = max_sequences;
  is_allocated_ = true;
}

void LlamaBatchBuilder::Reset() {
  if (!is_allocated_) {
    return;
  }
  llama_utils::BatchClear(batch_);
}

bool LlamaBatchBuilder::AddToken(llama_token token, int32_t position,
                                 llama_seq_id seq_id,
                                 bool request_logits) {
  if (!is_allocated_ || batch_.n_tokens >= capacity_tokens_) {
    return false;
  }

  llama_utils::BatchAdd(batch_, token, position, seq_id, request_logits);
  return true;
}

bool LlamaBatchBuilder::IsAllocated() const { return is_allocated_; }

int32_t LlamaBatchBuilder::CapacityTokens() const { return capacity_tokens_; }

int32_t LlamaBatchBuilder::CapacitySequences() const {
  return capacity_sequences_;
}

llama_batch &LlamaBatchBuilder::Get() { return batch_; }

const llama_batch &LlamaBatchBuilder::Get() const { return batch_; }

void LlamaBatchBuilder::Free() {
  if (!is_allocated_) {
    return;
  }

  llama_batch_free(batch_);
  batch_ = {};
  capacity_tokens_ = 0;
  capacity_sequences_ = 0;
  is_allocated_ = false;
}

} // namespace noumena::cogentengine
