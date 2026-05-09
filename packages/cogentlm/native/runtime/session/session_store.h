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
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"
#include "runtime/session/prefix_cache_policy.h"

namespace noumena::cogentengine {

struct SequenceState {
  llama_seq_id seq_id = -1;
  std::vector<llama_token> current_kv_tokens;
  int n_past = 0;
  std::size_t pin_count = 0;
  // Rolling FNV-1a hash over current_kv_tokens.  Maintained incrementally on
  // append (the per-tick hot path) and rebuilt only when the sequence is
  // truncated, restored, or replaced — operations that are already O(N) in the
  // token count.  Lets the prefix cache snapshot path read a precomputed hash
  // instead of re-walking the full token vector on every boundary store.
  std::uint64_t prefix_rolling_hash = kPrefixHashSeed;

  // Recompute the rolling hash from scratch.  Call this after any bulk
  // mutation of current_kv_tokens (clear/resize/assign/restore).
  void RebuildRollingHash() {
    std::uint64_t hash = kPrefixHashSeed;
    for (llama_token token : current_kv_tokens) {
      hash = MixPrefixHashToken(hash, token);
    }
    prefix_rolling_hash = hash;
  }
};

class SessionStore {
public:
  explicit SessionStore(size_t max_cached_contexts = 8,
                        size_t max_sequences = 1);
  ~SessionStore();

  void BindSharedContext(struct llama_context *shared_context);

  SequenceState *Find(const std::string &context_key);
  const SequenceState *Find(const std::string &context_key) const;
  size_t ComputeLcpReuse(const SequenceState &sequence_state,
                         const std::vector<llama_token> &incoming_tokens) const;
  bool CanAdmit(const std::string &context_key) const;
  SequenceState &GetOrCreateSession(const std::string &context_key);
  SequenceState &Emplace(const std::string &context_key, SequenceState state);
  void Touch(const std::string &context_key);
  void Pin(const std::string &context_key);
  void Unpin(const std::string &context_key);
  void Remove(const std::string &context_key);
  void EnforceLimitBeforeInsert();
  void Clear();

private:
  struct SessionEntry {
    SequenceState state;
    std::list<std::string>::iterator evictable_it;
    bool is_evictable = false;
  };

  void ClearSequenceMemory(llama_seq_id seq_id) const;
  llama_seq_id AcquireSeqId();
  void ReleaseSeqId(llama_seq_id seq_id);
  void MarkEvictable(const std::string &context_key, SessionEntry &entry);
  void MarkPinned(const std::string &context_key, SessionEntry &entry);
  bool HasEvictableSession() const;

  std::unordered_map<std::string, SessionEntry> context_states_;
  std::list<std::string> evictable_context_keys_;
  std::deque<llama_seq_id> free_seq_ids_;
  std::vector<bool> seq_id_available_;
  struct llama_context *shared_context_ = nullptr;
  size_t max_cached_contexts_ = 8;
  size_t max_sequences_ = 1;
};

} // namespace noumena::cogentengine
