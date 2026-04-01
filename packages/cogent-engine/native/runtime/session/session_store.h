/////////////////////////////////////////////////////////////////////////////////////////////////
//
// session_store.h
//
// - Owns reusable llama contexts and their LRU eviction state.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"

namespace noumena::cogentengine {

struct ContextState {
  struct llama_context* ctx = nullptr;
  std::vector<llama_token> current_kv_tokens;
  int n_past = 0;
};

class SessionStore {
public:
  explicit SessionStore(size_t max_cached_contexts = 8);
  ~SessionStore();

  ContextState* Find(const std::string& context_key);
  ContextState& Emplace(const std::string& context_key, ContextState state);
  void Touch(const std::string& context_key);
  void Remove(const std::string& context_key);
  void EnforceLimitBeforeInsert();
  void Clear();

private:
  std::unordered_map<std::string, ContextState> context_states_;
  std::vector<std::string> context_usage_order_;
  size_t max_cached_contexts_ = 8;
};

}  // namespace noumena::cogentengine
