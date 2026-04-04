/////////////////////////////////////////////////////////////////////////////////////////////////
//
// prefix_state_cache.h
//
// - In-memory serialized prefix-state cache entries for Phase 5 prefix reuse.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"
#include "runtime/session/prefix_cache_policy.h"

namespace noumena::cogentengine {

struct PrefixCacheEntry {
  std::uint64_t model_fingerprint = 0;
  std::string context_key;
  std::size_t token_count = 0;
  std::uint64_t prefix_hash = 0;
  std::uint64_t retention_priority = 0;
  std::uint64_t hit_count = 0;
  std::size_t approx_bytes = 0;
  std::vector<llama_token> prefix_tokens;
  std::vector<std::uint8_t> state_bytes;
  std::chrono::steady_clock::time_point last_used{};
};

struct PrefixCacheLookupKey {
  std::uint64_t model_fingerprint = 0;
  std::size_t token_count = 0;
  std::uint64_t prefix_hash = 0;

  bool operator==(const PrefixCacheLookupKey &other) const = default;
};

struct PrefixCacheLookupKeyHasher {
  std::size_t operator()(const PrefixCacheLookupKey &key) const noexcept {
    std::size_t hash = static_cast<std::size_t>(key.model_fingerprint);
    hash ^= static_cast<std::size_t>(key.token_count) + 0x9e3779b9u +
            (hash << 6) + (hash >> 2);
    hash ^= static_cast<std::size_t>(key.prefix_hash) + 0x9e3779b9u +
            (hash << 6) + (hash >> 2);
    return hash;
  }
};

class PrefixStateCache {
public:
  explicit PrefixStateCache(std::size_t max_entries = 32);

  void set_max_entries(std::size_t max_entries);

  const PrefixCacheEntry *
  FindBestPrefix(std::uint64_t model_fingerprint, const std::string &context_key,
                 const std::vector<llama_token> &prompt_tokens,
                 PrefixCachePolicy &prefix_cache_policy);

  bool StorePrefixState(struct llama_context *context, llama_seq_id seq_id,
                        std::uint64_t model_fingerprint,
                        const std::string &context_key,
                        const std::vector<llama_token> &tokens,
                        std::size_t token_count, std::uint64_t prefix_hash,
                        std::uint64_t retention_priority = 0);

  void Clear();

private:
  void EnforceLimit();
  void RebuildLookupBuckets();

  std::vector<PrefixCacheEntry> entries_;
  std::unordered_map<PrefixCacheLookupKey, std::vector<std::size_t>,
                     PrefixCacheLookupKeyHasher>
      lookup_buckets_;
  std::size_t max_entries_ = 32;
};

} // namespace noumena::cogentengine
