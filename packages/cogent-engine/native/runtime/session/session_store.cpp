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
  return it == context_states_.end() ? nullptr : &it->second;
}

SequenceState &SessionStore::GetOrCreateSession(const std::string &context_key) {
  if (SequenceState *existing = Find(context_key)) {
    Touch(context_key);
    return *existing;
  }

  EnforceLimitBeforeInsert();

  SequenceState new_state;
  new_state.seq_id = AcquireSeqId();
  SequenceState &stored_state = Emplace(context_key, std::move(new_state));
  Touch(context_key);
  return stored_state;
}

SequenceState &SessionStore::Emplace(const std::string &context_key,
                                     SequenceState state) {
  auto [it, inserted] = context_states_.emplace(context_key, std::move(state));
  if (!inserted) {
    ClearSequenceMemory(it->second.seq_id);
    ReleaseSeqId(it->second.seq_id);
    it->second = std::move(state);
  }
  return it->second;
}

void SessionStore::Touch(const std::string &context_key) {
  auto it = std::find(context_usage_order_.begin(), context_usage_order_.end(),
                      context_key);
  if (it != context_usage_order_.end()) {
    context_usage_order_.erase(it);
  }
  context_usage_order_.push_back(context_key);
}

void SessionStore::Pin(SequenceState &sequence_state) { sequence_state.pin_count++; }

void SessionStore::Unpin(SequenceState &sequence_state) {
  if (sequence_state.pin_count > 0) {
    sequence_state.pin_count--;
  }
}

void SessionStore::Remove(const std::string &context_key) {
  auto state_it = context_states_.find(context_key);
  if (state_it != context_states_.end()) {
    ClearSequenceMemory(state_it->second.seq_id);
    ReleaseSeqId(state_it->second.seq_id);
    context_states_.erase(state_it);
  }

  auto order_it = std::find(context_usage_order_.begin(),
                            context_usage_order_.end(), context_key);
  if (order_it != context_usage_order_.end()) {
    context_usage_order_.erase(order_it);
  }
}

void SessionStore::EnforceLimitBeforeInsert() {
  while ((context_states_.size() >= max_cached_contexts_ ||
          free_seq_ids_.empty()) &&
         !context_usage_order_.empty()) {
    auto order_it = std::find_if(
        context_usage_order_.begin(), context_usage_order_.end(),
        [this](const std::string &candidate_key) {
          auto state_it = context_states_.find(candidate_key);
          return state_it != context_states_.end() &&
                 state_it->second.pin_count == 0;
        });
    if (order_it == context_usage_order_.end()) {
      break;
    }

    const std::string evict_key = *order_it;
    context_usage_order_.erase(order_it);

    auto state_it = context_states_.find(evict_key);
    if (state_it == context_states_.end()) {
      continue;
    }

    ClearSequenceMemory(state_it->second.seq_id);
    ReleaseSeqId(state_it->second.seq_id);
    context_states_.erase(state_it);
  }
}

void SessionStore::Clear() {
  for (auto &[key, state] : context_states_) {
    (void)key;
    ClearSequenceMemory(state.seq_id);
    ReleaseSeqId(state.seq_id);
  }

  context_states_.clear();
  context_usage_order_.clear();
}

void SessionStore::ClearSequenceMemory(llama_seq_id seq_id) const {
  if (shared_context_ == nullptr || seq_id < 0) {
    return;
  }

  llama_memory_t mem = llama_get_memory(shared_context_);
  llama_memory_seq_rm(mem, seq_id, 0, -1);
}

llama_seq_id SessionStore::AcquireSeqId() {
  if (free_seq_ids_.empty()) {
    return -1;
  }

  const llama_seq_id seq_id = free_seq_ids_.front();
  free_seq_ids_.pop_front();
  return seq_id;
}

void SessionStore::ReleaseSeqId(llama_seq_id seq_id) {
  if (seq_id < 0) {
    return;
  }

  if (std::find(free_seq_ids_.begin(), free_seq_ids_.end(), seq_id) ==
      free_seq_ids_.end()) {
    free_seq_ids_.push_back(seq_id);
  }
}

} // namespace noumena::cogentengine
