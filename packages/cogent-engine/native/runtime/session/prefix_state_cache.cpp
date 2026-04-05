/////////////////////////////////////////////////////////////////////////////////////////////////
//
// prefix_state_cache.cpp
//
// - In-memory serialized prefix-state cache entries for Phase 5 prefix reuse.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/session/prefix_state_cache.h"

#include <algorithm>

namespace noumena::cogentengine {

PrefixStateCache::PrefixStateCache(std::size_t max_entries,
                                   std::size_t max_total_bytes)
    : max_entries_(std::max<std::size_t>(1, max_entries)),
      max_total_bytes_(std::max<std::size_t>(1, max_total_bytes)) {}

void PrefixStateCache::set_max_entries(std::size_t max_entries) {
  max_entries_ = std::max<std::size_t>(1, max_entries);
  EnforceLimit();
}

const PrefixCacheEntry *PrefixStateCache::FindBestPrefix(
    std::uint64_t model_fingerprint, const std::string &context_key,
    const std::vector<llama_token> &prompt_tokens,
    PrefixCachePolicy &prefix_cache_policy) {
  prefix_cache_policy.RecordLookup();

  const std::vector<PrefixCacheBoundary> candidates =
      prefix_cache_policy.BuildCandidateBoundaries(prompt_tokens);
  if (candidates.empty()) {
    return nullptr;
  }

  for (const PrefixCacheBoundary &candidate : candidates) {
    const PrefixCacheLookupKey lookup_key{
        .model_fingerprint = model_fingerprint,
        .token_count = candidate.token_count,
        .prefix_hash = candidate.prefix_hash,
    };

    auto bucket_it = lookup_buckets_.find(lookup_key);
    if (bucket_it == lookup_buckets_.end()) {
      continue;
    }

    PrefixCacheEntry *best_entry = nullptr;
    for (const EntryIterator &entry_it : bucket_it->second) {
      PrefixCacheEntry &entry = *entry_it;
      if (entry.prefix_tokens.size() != candidate.token_count) {
        continue;
      }
      if (!std::equal(entry.prefix_tokens.begin(), entry.prefix_tokens.end(),
                      prompt_tokens.begin())) {
        continue;
      }

      const bool prefer_entry =
          best_entry == nullptr ||
          (entry.context_key == context_key &&
           best_entry->context_key != context_key) ||
          (entry.context_key == best_entry->context_key &&
           entry.retention_priority > best_entry->retention_priority) ||
          (entry.context_key == best_entry->context_key &&
           entry.retention_priority == best_entry->retention_priority &&
           entry.last_used > best_entry->last_used);

      if (prefer_entry) {
        best_entry = &entry;
      }
    }

    if (best_entry != nullptr) {
      best_entry->hit_count++;
      best_entry->last_used = std::chrono::steady_clock::now();
      prefix_cache_policy.RecordHit(best_entry->token_count);
      return best_entry;
    }
  }

  return nullptr;
}

bool PrefixStateCache::StorePrefixState(
    llama_context *context, llama_seq_id seq_id, std::uint64_t model_fingerprint,
    const std::string &context_key, const std::vector<llama_token> &tokens,
    std::size_t token_count, std::uint64_t prefix_hash,
    std::uint64_t retention_priority) {
  if (context == nullptr || seq_id < 0 || token_count == 0 ||
      token_count > tokens.size()) {
    return false;
  }

  const std::size_t prefix_state_size =
      llama_state_seq_get_size(context, seq_id);
  if (prefix_state_size == 0) {
    return false;
  }

  std::vector<std::uint8_t> state_bytes(prefix_state_size);
  const std::size_t copied = llama_state_seq_get_data(context, state_bytes.data(),
                                                      state_bytes.size(), seq_id);
  if (copied != prefix_state_size) {
    return false;
  }

  EntryIterator existing_it = FindExistingEntry(model_fingerprint, context_key,
                                                tokens, token_count,
                                                prefix_hash);

  PrefixCacheEntry entry;
  entry.model_fingerprint = model_fingerprint;
  entry.context_key = context_key;
  entry.token_count = token_count;
  entry.prefix_hash = prefix_hash;
  entry.retention_priority = retention_priority;
  entry.hit_count = existing_it != entries_.end() ? existing_it->hit_count : 0;
  entry.approx_bytes =
      state_bytes.size() + token_count * sizeof(llama_token);
  entry.prefix_tokens.assign(tokens.begin(), tokens.begin() + token_count);
  entry.state_bytes = std::move(state_bytes);
  entry.last_used = std::chrono::steady_clock::now();

  if (existing_it != entries_.end()) {
    total_approx_bytes_ -= existing_it->approx_bytes;
    *existing_it = std::move(entry);
    total_approx_bytes_ += existing_it->approx_bytes;
  } else {
    entries_.push_back(std::move(entry));
    const EntryIterator inserted_it = std::prev(entries_.end());
    total_approx_bytes_ += inserted_it->approx_bytes;
    AddToLookupBucket(inserted_it);
  }

  EnforceLimit();
  return true;
}

void PrefixStateCache::Clear() {
  entries_.clear();
  lookup_buckets_.clear();
  total_approx_bytes_ = 0;
}

void PrefixStateCache::EnforceLimit() {
  while (entries_.size() > max_entries_ ||
         total_approx_bytes_ > max_total_bytes_) {
    const auto evict_it = std::min_element(
        entries_.begin(), entries_.end(),
        [](const PrefixCacheEntry &left, const PrefixCacheEntry &right) {
          if (left.retention_priority != right.retention_priority) {
            return left.retention_priority < right.retention_priority;
          }
          if (left.hit_count != right.hit_count) {
            return left.hit_count < right.hit_count;
          }
          return left.last_used < right.last_used;
        });
    if (evict_it == entries_.end()) {
      break;
    }
    RemoveEntry(evict_it);
  }
}

PrefixStateCache::EntryIterator PrefixStateCache::FindExistingEntry(
    std::uint64_t model_fingerprint, const std::string &context_key,
    const std::vector<llama_token> &tokens, std::size_t token_count,
    std::uint64_t prefix_hash) {
  const PrefixCacheLookupKey lookup_key{
      .model_fingerprint = model_fingerprint,
      .token_count = token_count,
      .prefix_hash = prefix_hash,
  };
  const auto bucket_it = lookup_buckets_.find(lookup_key);
  if (bucket_it == lookup_buckets_.end()) {
    return entries_.end();
  }

  for (const EntryIterator &entry_it : bucket_it->second) {
    if (entry_it->context_key != context_key ||
        entry_it->prefix_tokens.size() != token_count) {
      continue;
    }
    if (std::equal(entry_it->prefix_tokens.begin(), entry_it->prefix_tokens.end(),
                   tokens.begin())) {
      return entry_it;
    }
  }

  return entries_.end();
}

void PrefixStateCache::AddToLookupBucket(const EntryIterator &entry_it) {
  const PrefixCacheLookupKey lookup_key{
      .model_fingerprint = entry_it->model_fingerprint,
      .token_count = entry_it->token_count,
      .prefix_hash = entry_it->prefix_hash,
  };
  lookup_buckets_[lookup_key].push_back(entry_it);
}

void PrefixStateCache::RemoveFromLookupBucket(const EntryIterator &entry_it) {
  const PrefixCacheLookupKey lookup_key{
      .model_fingerprint = entry_it->model_fingerprint,
      .token_count = entry_it->token_count,
      .prefix_hash = entry_it->prefix_hash,
  };
  const auto bucket_it = lookup_buckets_.find(lookup_key);
  if (bucket_it == lookup_buckets_.end()) {
    return;
  }

  auto &bucket = bucket_it->second;
  bucket.erase(
      std::remove_if(bucket.begin(), bucket.end(),
                     [&entry_it](const EntryIterator &candidate) {
                       return candidate == entry_it;
                     }),
      bucket.end());
  if (bucket.empty()) {
    lookup_buckets_.erase(bucket_it);
  }
}

void PrefixStateCache::RemoveEntry(const EntryIterator &entry_it) {
  total_approx_bytes_ -= entry_it->approx_bytes;
  RemoveFromLookupBucket(entry_it);
  entries_.erase(entry_it);
}

} // namespace noumena::cogentengine
