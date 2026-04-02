/////////////////////////////////////////////////////////////////////////////////////////////////
//
// session_store.cpp
//
// - Owns reusable llama contexts and their LRU eviction state.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/session/session_store.h"

#include <algorithm>

namespace noumena::cogentengine {

SessionStore::SessionStore(size_t max_cached_contexts)
    : max_cached_contexts_(max_cached_contexts) {}

SessionStore::~SessionStore() {
  Clear();
}

ContextState* SessionStore::Find(const std::string& context_key) {
  auto it = context_states_.find(context_key);
  return it == context_states_.end() ? nullptr : &it->second;
}

ContextState& SessionStore::Emplace(const std::string& context_key, ContextState state) {
  auto [it, inserted] =
      context_states_.emplace(context_key, std::move(state));
  if (!inserted) {
    if (it->second.ctx != nullptr) {
      llama_free(it->second.ctx);
    }
    it->second = std::move(state);
  }
  return it->second;
}

void SessionStore::Touch(const std::string& context_key) {
  auto it = std::find(context_usage_order_.begin(), context_usage_order_.end(), context_key);
  if (it != context_usage_order_.end()) {
    context_usage_order_.erase(it);
  }
  context_usage_order_.push_back(context_key);
}

void SessionStore::Remove(const std::string& context_key) {
  auto ctx_it = context_states_.find(context_key);
  if (ctx_it != context_states_.end()) {
    if (ctx_it->second.ctx != nullptr) {
      llama_free(ctx_it->second.ctx);
    }
    context_states_.erase(ctx_it);
  }

  auto order_it = std::find(context_usage_order_.begin(), context_usage_order_.end(), context_key);
  if (order_it != context_usage_order_.end()) {
    context_usage_order_.erase(order_it);
  }
}

void SessionStore::EnforceLimitBeforeInsert() {
  while (context_states_.size() >= max_cached_contexts_ && !context_usage_order_.empty()) {
    const std::string evict_key = context_usage_order_.front();
    context_usage_order_.erase(context_usage_order_.begin());

    auto it = context_states_.find(evict_key);
    if (it == context_states_.end()) {
      continue;
    }

    if (it->second.ctx != nullptr) {
      llama_free(it->second.ctx);
    }
    context_states_.erase(it);
  }
}

void SessionStore::Clear() {
  for (auto& [key, state] : context_states_) {
    (void) key;
    if (state.ctx != nullptr) {
      llama_free(state.ctx);
    }
  }

  context_states_.clear();
  context_usage_order_.clear();
}

}  // namespace noumena::cogentengine
