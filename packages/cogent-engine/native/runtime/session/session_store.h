/////////////////////////////////////////////////////////////////////////////////////////////////
//
// session_store.h
//
// - Owns reusable logical sequence state and its LRU eviction state.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"

namespace noumena::cogentengine {

struct SequenceState {
  llama_seq_id seq_id = -1;
  std::vector<llama_token> current_kv_tokens;
  int n_past = 0;
  std::size_t pin_count = 0;
};

class SessionStore {
public:
  explicit SessionStore(size_t max_cached_contexts = 8,
                        size_t max_sequences = 1);
  ~SessionStore();

  void BindSharedContext(struct llama_context *shared_context);

  SequenceState *Find(const std::string &context_key);
  SequenceState &GetOrCreateSession(const std::string &context_key);
  SequenceState &Emplace(const std::string &context_key, SequenceState state);
  void Touch(const std::string &context_key);
  void Pin(SequenceState &sequence_state);
  void Unpin(SequenceState &sequence_state);
  void Remove(const std::string &context_key);
  void EnforceLimitBeforeInsert();
  void Clear();

private:
  void ClearSequenceMemory(llama_seq_id seq_id) const;
  llama_seq_id AcquireSeqId();
  void ReleaseSeqId(llama_seq_id seq_id);

  std::unordered_map<std::string, SequenceState> context_states_;
  std::vector<std::string> context_usage_order_;
  std::deque<llama_seq_id> free_seq_ids_;
  struct llama_context *shared_context_ = nullptr;
  size_t max_cached_contexts_ = 8;
  size_t max_sequences_ = 1;
};

} // namespace noumena::cogentengine
