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
#include <deque>
#include <list>
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

// A snapshot enqueued by the inference tick at a boundary moment, to be
// materialized later by `DrainPendingSnapshots` outside the tick.  The cache
// key (`prefix_tokens`, `token_count`, `prefix_hash`) is captured eagerly so
// the entry's identity reflects the boundary state even if the underlying
// GPU KV continues to grow before the drain happens.  Correctness on restore
// is guaranteed by `llama_memory_seq_rm`-truncating any extra tokens past
// `token_count` after `llama_state_seq_set_data`.
struct PendingPrefixSnapshot {
  std::uint64_t model_fingerprint = 0;
  std::string context_key;
  llama_seq_id seq_id = -1;
  std::size_t token_count = 0;
  std::uint64_t prefix_hash = 0;
  std::uint64_t retention_priority = 0;
  std::vector<llama_token> prefix_tokens;
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
  explicit PrefixStateCache(
      std::size_t max_entries = 32,
      std::size_t max_total_bytes = 256ull * 1024ull * 1024ull);

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

  // Defers the expensive `llama_state_seq_get_data` GPU readback off the
  // inference tick.  The boundary-moment cache key is captured eagerly via
  // `prefix_tokens`/`prefix_hash` so the deferred drain can safely run at a
  // later moment when the seq's KV may already hold additional decoded
  // tokens past the boundary; the saved bytes simply represent "at least the
  // requested prefix is reachable from this state", and `RestorePrefixState`
  // callers truncate the seq to `token_count` after `set_data` to recover
  // exact boundary semantics.  Drops snapshots for an already-pending entry
  // with the same lookup identity to avoid unbounded queue growth on long
  // generations.
  void EnqueuePendingSnapshot(PendingPrefixSnapshot snapshot);
  std::size_t PendingSnapshotCount() const { return pending_snapshots_.size(); }
  // Materializes up to `max_to_drain` queued snapshots in FIFO order against
  // the live `context`.  Returns the number of snapshots drained (whether
  // they ultimately stored or were dropped because the seq state was empty).
  // A zero `max_to_drain` drains the entire queue.
  std::size_t DrainPendingSnapshots(struct llama_context *context,
                                    std::size_t max_to_drain);
  // Drops any pending snapshot bound to `seq_id`.  Used when a seq is about
  // to be evicted or repurposed so we don't snapshot a state that no longer
  // matches the recorded prefix tokens.
  void DropPendingSnapshotsForSeq(llama_seq_id seq_id);

  void Clear();

private:
  using EntryList = std::list<PrefixCacheEntry>;
  using EntryIterator = EntryList::iterator;

  void EnforceLimit();
  EntryIterator FindExistingEntry(
      std::uint64_t model_fingerprint, const std::string &context_key,
      const std::vector<llama_token> &tokens, std::size_t token_count,
      std::uint64_t prefix_hash);
  void AddToLookupBucket(const EntryIterator &entry_it);
  void RemoveFromLookupBucket(const EntryIterator &entry_it);
  void RemoveEntry(const EntryIterator &entry_it);

  EntryList entries_;
  std::unordered_map<PrefixCacheLookupKey, std::vector<EntryIterator>,
                     PrefixCacheLookupKeyHasher>
      lookup_buckets_;
  // Pending snapshots awaiting a quiet moment to materialize.  A small bound
  // is enforced via `EnqueuePendingSnapshot`'s same-key dedup so a single
  // long-lived seq's repeated boundary firings cannot accumulate.
  std::deque<PendingPrefixSnapshot> pending_snapshots_;
  std::size_t max_entries_ = 32;
  std::size_t max_total_bytes_ = 256ull * 1024ull * 1024ull;
  std::size_t total_approx_bytes_ = 0;
  // Reusable scratch buffer for KV state serialization.  Avoids repeated
  // large malloc/mmap syscalls on the hot path – the buffer only grows,
  // never shrinks, so subsequent snapshots of equal-or-smaller sequences
  // allocate zero bytes from the system allocator.
  std::vector<std::uint8_t> reusable_snapshot_buffer_;
};

} // namespace noumena::cogentengine
