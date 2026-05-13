/////////////////////////////////////////////////////////////////////////////////////////////////
//
// prefix_cache_policy.h
//
// - Prefix cache boundary and key policy.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "llama.h"

namespace noumena::cogentengine {

// FNV-1a constants exposed so callers (e.g. SequenceState) can maintain a
// rolling hash incrementally and feed the result into the prefix cache without
// re-walking the token array.
inline constexpr std::uint64_t kPrefixHashSeed = 1469598103934665603ull;
inline constexpr std::uint64_t kPrefixHashPrime = 1099511628211ull;

inline std::uint64_t MixPrefixHashToken(std::uint64_t hash,
                                        llama_token token) noexcept {
  hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(token));
  hash *= kPrefixHashPrime;
  return hash;
}

struct PrefixCacheBoundary {
  std::size_t token_count = 0;
  std::uint64_t prefix_hash = 0;
};

struct PrefixCachePolicyStats {
  std::uint64_t lookup_count = 0;
  std::uint64_t hit_count = 0;
  std::uint64_t store_count = 0;
  std::uint64_t restored_token_count = 0;
  std::uint64_t stored_token_count = 0;
};

class PrefixCachePolicy {
public:
  // A zero interval keeps terminal prompt snapshots only.  This avoids hidden
  // KV serialization stalls on workloads that do not reuse intermediate
  // prefix checkpoints.
  explicit PrefixCachePolicy(std::size_t prefix_cache_interval_tokens = 128);

  std::size_t prefix_cache_interval_tokens() const {
    return prefix_cache_interval_tokens_;
  }

  std::size_t minimum_prefix_cache_tokens() const {
    return minimum_prefix_cache_tokens_;
  }

  bool ShouldStoreBoundary(std::size_t token_count,
                           std::size_t terminal_token_count) const;
  std::vector<PrefixCacheBoundary>
  BuildCandidateBoundaries(const std::vector<llama_token> &tokens) const;
  std::uint64_t HashPrefix(const std::vector<llama_token> &tokens,
                           std::size_t token_count) const;

  void RecordLookup();
  void RecordHit(std::size_t token_count);
  void RecordStore(std::size_t token_count);

  const PrefixCachePolicyStats &stats() const { return stats_; }

private:
  std::size_t prefix_cache_interval_tokens_ = 128;
  std::size_t minimum_prefix_cache_tokens_ = 32;
  PrefixCachePolicyStats stats_{};
};

} // namespace noumena::cogentengine
