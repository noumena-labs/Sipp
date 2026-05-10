/////////////////////////////////////////////////////////////////////////////////////////////////
//
// session_store.cpp
//
// - Owns reusable logical sequence state and its LRU eviction state.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/session/session_store.h"

#include <algorithm>

namespace noumena::cogentengine {

SessionStore::SessionStore(size_t max_cached_contexts, size_t max_sequences)
    : max_cached_contexts_(std::max<size_t>(1, max_cached_contexts)),
      max_sequences_(std::max<size_t>(1, max_sequences)) {
  seq_id_available_.assign(max_sequences_, true);
  for (size_t i = 0; i < max_sequences_; ++i) {
    free_seq_ids_.push_back(static_cast<llama_seq_id>(i));
  }
}

SessionStore::~SessionStore() { Clear(); }

void SessionStore::BindSharedContext(llama_context *shared_context) {
  shared_context_ = shared_context;
}

SequenceState *SessionStore::Find(const std::string &context_key) {
  auto it = context_states_.find(context_key);
  return it == context_states_.end() ? nullptr : &it->second.state;
}

const SequenceState *SessionStore::Find(const std::string &context_key) const {
  auto it = context_states_.find(context_key);
  return it == context_states_.end() ? nullptr : &it->second.state;
}

size_t SessionStore::ComputeLcpReuse(
    const SequenceState &sequence_state,
    const std::vector<llama_token> &incoming_tokens) const {
  const size_t shared_length = std::min(sequence_state.current_kv_tokens.size(),
                                        incoming_tokens.size());
  size_t match_length = 0;
  for (; match_length < shared_length; ++match_length) {
    if (sequence_state.current_kv_tokens[match_length] !=
        incoming_tokens[match_length]) {
      break;
    }
  }
  return match_length;
}

SequenceState &SessionStore::GetOrCreateSession(const std::string &context_key) {
  if (SequenceState *existing = Find(context_key)) {
    Touch(context_key);
    return *existing;
  }

  EnforceLimitBeforeInsert();

  SequenceState new_state;
  SequenceState &stored_state = Emplace(context_key, std::move(new_state));
  Touch(context_key);
  return stored_state;
}

SequenceState &SessionStore::Emplace(const std::string &context_key,
                                     SequenceState state) {
  auto [it, inserted] = context_states_.emplace(
      context_key, SessionEntry{.state = std::move(state)});
  if (!inserted) {
    it->second.state = std::move(state);
  }
  MarkEvictable(context_key, it->second);
  return it->second.state;
}

void SessionStore::Touch(const std::string &context_key) {
  auto state_it = context_states_.find(context_key);
  if (state_it == context_states_.end()) {
    return;
  }
  if (state_it->second.is_evictable) {
    evictable_context_keys_.splice(evictable_context_keys_.end(),
                                   evictable_context_keys_,
                                   state_it->second.evictable_it);
  }
}

void SessionStore::Pin(const std::string &context_key) {
  auto state_it = context_states_.find(context_key);
  if (state_it == context_states_.end()) {
    return;
  }
  state_it->second.state.pin_count++;
  MarkPinned(context_key, state_it->second);
}

void SessionStore::Unpin(const std::string &context_key) {
  auto state_it = context_states_.find(context_key);
  if (state_it == context_states_.end()) {
    return;
  }

  if (state_it->second.state.pin_count > 0) {
    state_it->second.state.pin_count--;
  }
  if (state_it->second.state.pin_count == 0) {
    MarkEvictable(state_it->first, state_it->second);
  }
}

void SessionStore::Remove(const std::string &context_key) {
  auto state_it = context_states_.find(context_key);
  if (state_it != context_states_.end()) {
    if (state_it->second.is_evictable) {
      evictable_context_keys_.erase(state_it->second.evictable_it);
    }
    context_states_.erase(state_it);
  }
}

void SessionStore::EnforceLimitBeforeInsert() {
  while ((context_states_.size() >= max_cached_contexts_ ||
          free_seq_ids_.empty()) &&
         !evictable_context_keys_.empty()) {
    const std::string evict_key = evictable_context_keys_.front();
    Remove(evict_key);
  }
}

void SessionStore::Clear() {
  context_states_.clear();
  evictable_context_keys_.clear();
}

void SessionStore::ClearSequenceMemory(llama_seq_id seq_id) const {
  if (shared_context_ == nullptr || seq_id < 0) {
    return;
  }

  llama_memory_t mem = llama_get_memory(shared_context_);
  llama_memory_seq_rm(mem, seq_id, 0, -1);
}

llama_seq_id SessionStore::AcquireSeqId(llama_seq_id hint) {
  if (free_seq_ids_.empty()) {
    return -1;
  }

  llama_seq_id seq_id = -1;
  if (hint >= 0 && hint < static_cast<llama_seq_id>(seq_id_available_.size())) {
    auto it = std::find(free_seq_ids_.begin(), free_seq_ids_.end(), hint);
    if (it != free_seq_ids_.end()) {
      seq_id = hint;
      free_seq_ids_.erase(it);
    }
  }

  if (seq_id == -1) {
    seq_id = free_seq_ids_.front();
    free_seq_ids_.pop_front();
  }

  if (seq_id >= 0 &&
      static_cast<size_t>(seq_id) < seq_id_available_.size()) {
    seq_id_available_[static_cast<size_t>(seq_id)] = false;
  }

  // GUARANTEE ISOLATION: Any ID taken from the free pool is stale.
  // We must scrub it to prevent position conflicts (status=-1).
  ClearSequenceMemory(seq_id);

  return seq_id;
}

void SessionStore::ReleaseSeqId(llama_seq_id seq_id) {
  if (seq_id < 0) {
    return;
  }

  if (static_cast<size_t>(seq_id) >= seq_id_available_.size()) {
    return;
  }
  if (seq_id_available_[static_cast<size_t>(seq_id)]) {
    return;
  }
  seq_id_available_[static_cast<size_t>(seq_id)] = true;
  free_seq_ids_.push_back(seq_id);
}

bool SessionStore::CanAdmit(const std::string &context_key) const {
  if (const SequenceState *existing = Find(context_key); existing != nullptr) {
    return existing->pin_count == 0;
  }

  const bool needs_cache_slot = context_states_.size() >= max_cached_contexts_;
  const bool needs_sequence = free_seq_ids_.empty();
  if (!needs_cache_slot && !needs_sequence) {
    return true;
  }

  return HasEvictableSession();
}

void SessionStore::MarkEvictable(const std::string &context_key,
                                 SessionEntry &entry) {
  if (entry.state.pin_count > 0) {
    MarkPinned(context_key, entry);
    return;
  }
  if (entry.is_evictable) {
    evictable_context_keys_.splice(evictable_context_keys_.end(),
                                   evictable_context_keys_,
                                   entry.evictable_it);
    return;
  }

  evictable_context_keys_.push_back(context_key);
  entry.evictable_it = std::prev(evictable_context_keys_.end());
  entry.is_evictable = true;
}

void SessionStore::MarkPinned(const std::string &context_key, SessionEntry &entry) {
  (void)context_key;
  if (!entry.is_evictable) {
    return;
  }
  evictable_context_keys_.erase(entry.evictable_it);
  entry.is_evictable = false;
}

bool SessionStore::HasEvictableSession() const {
  return !evictable_context_keys_.empty();
}

} // namespace noumena::cogentengine
