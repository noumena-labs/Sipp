/////////////////////////////////////////////////////////////////////////////////////////////////
//
// prefix_cache_policy.cpp
//
// - Prefix cache boundary and key policy.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/session/prefix_cache_policy.h"

#include <algorithm>

namespace {

constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

std::uint64_t mix_token_hash(std::uint64_t hash, llama_token token) {
  hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(token));
  hash *= kFnvPrime;
  return hash;
}

} // namespace

namespace noumena::cogentengine {

PrefixCachePolicy::PrefixCachePolicy(std::size_t prefix_cache_interval_tokens)
    : prefix_cache_interval_tokens_(
          std::max<std::size_t>(1, prefix_cache_interval_tokens)),
      minimum_prefix_cache_tokens_(
          std::min<std::size_t>(prefix_cache_interval_tokens_, 32)) {}

bool PrefixCachePolicy::ShouldStoreBoundary(
    std::size_t token_count, std::size_t terminal_token_count) const {
  if (token_count < minimum_prefix_cache_tokens_) {
    return false;
  }
  if (token_count == terminal_token_count) {
    return true;
  }
  return token_count % prefix_cache_interval_tokens_ == 0;
}

std::vector<PrefixCacheBoundary>
PrefixCachePolicy::BuildCandidateBoundaries(
    const std::vector<llama_token> &tokens) const {
  std::vector<PrefixCacheBoundary> boundaries;
  if (tokens.size() < minimum_prefix_cache_tokens_) {
    return boundaries;
  }

  boundaries.reserve(tokens.size() / prefix_cache_interval_tokens_ + 1);

  std::uint64_t rolling_hash = kFnvOffsetBasis;
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    rolling_hash = mix_token_hash(rolling_hash, tokens[index]);
    const std::size_t token_count = index + 1;
    if (!ShouldStoreBoundary(token_count, tokens.size())) {
      continue;
    }

    boundaries.push_back(PrefixCacheBoundary{
        .token_count = token_count,
        .prefix_hash = rolling_hash,
    });
  }

  std::reverse(boundaries.begin(), boundaries.end());
  return boundaries;
}

std::uint64_t PrefixCachePolicy::HashPrefix(
    const std::vector<llama_token> &tokens, std::size_t token_count) const {
  if (token_count == 0 || tokens.empty()) {
    return 0;
  }

  const std::size_t clamped_count = std::min(token_count, tokens.size());
  std::uint64_t rolling_hash = kFnvOffsetBasis;
  for (std::size_t index = 0; index < clamped_count; ++index) {
    rolling_hash = mix_token_hash(rolling_hash, tokens[index]);
  }
  return rolling_hash;
}

void PrefixCachePolicy::RecordLookup() { stats_.lookup_count++; }

void PrefixCachePolicy::RecordHit(std::size_t token_count) {
  stats_.hit_count++;
  stats_.restored_token_count += token_count;
}

void PrefixCachePolicy::RecordStore(std::size_t token_count) {
  stats_.store_count++;
  stats_.stored_token_count += token_count;
}

} // namespace noumena::cogentengine
