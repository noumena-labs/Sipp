/////////////////////////////////////////////////////////////////////////////////////////////////
//
// inference_runtime.cpp
//
// - Inference-only runtime over llama.cpp.
// - Owns model lifetime, context reuse, and text generation.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/inference_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chat.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "runtime/config/scheduler_policy.h"
#include "runtime/llama/llama_utils.h"

namespace {

constexpr char kDefaultPromptContextKey[] = "__primary_prompt__";
constexpr int kMaxPredictionTokens = 2048;

using BitmapPtr = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
using InputChunksPtr =
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;

noumena::cogentengine::InferenceRuntimeConfig
normalize_config(noumena::cogentengine::InferenceRuntimeConfig config) {
  config.n_seq_max = std::max<int32_t>(1, config.n_seq_max);
  config.gpu_layers = std::max<int32_t>(-1, config.gpu_layers);
  config.max_cached_sessions = std::max<int32_t>(1, config.max_cached_sessions);
  config.retained_prefix_tokens =
      std::max<int32_t>(0, config.retained_prefix_tokens);
  config.prefill_chunk_size = std::max<int32_t>(0, config.prefill_chunk_size);
  config.prefix_cache_interval_tokens =
      std::max<int32_t>(0, config.prefix_cache_interval_tokens);
  config.max_prefix_cache_entries =
      std::max<int32_t>(1, config.max_prefix_cache_entries);
  config.image_min_tokens = std::max<int32_t>(0, config.image_min_tokens);
  config.image_max_tokens = std::max<int32_t>(0, config.image_max_tokens);
  config.multimodal_use_gpu =
      std::clamp<int32_t>(config.multimodal_use_gpu, -1, 1);
  config.sampling_repeat_last_n =
      std::max<int32_t>(0, config.sampling_repeat_last_n);
  config.sampling_repeat_penalty =
      std::max<float>(0.0f, config.sampling_repeat_penalty);
  config.sampling_top_k = std::max<int32_t>(0, config.sampling_top_k);
  config.sampling_top_p = std::max<float>(0.0f, config.sampling_top_p);
  config.sampling_min_p = std::max<float>(0.0f, config.sampling_min_p);
  config.sampling_temperature =
      std::max<float>(0.0f, config.sampling_temperature);
  config.scheduler_policy.decode_token_reserve =
      std::max<int32_t>(0, config.scheduler_policy.decode_token_reserve);
  config.enable_runtime_observability =
      config.enable_runtime_observability > 0 ? 1 : 0;
  config.enable_backend_profiling = config.enable_backend_profiling > 0 ? 1 : 0;
  if (config.enable_backend_profiling > 0) {
    config.enable_runtime_observability = 1;
  }
  return config;
}

uint32_t resolve_sampling_seed(int32_t seed) {
  if (seed < 0) {
    return LLAMA_DEFAULT_SEED;
  }
  return static_cast<uint32_t>(seed);
}

// Sampler-shape predicates.  A "greedy" configuration collapses the entire
// stochastic chain to a single argmax, which is the hot path for chat with
// temperature 0 (deterministic / tool-calling).  The "neutral" predicates let
// us omit individual stages that would be no-ops for the configured params,
// saving a per-token candidate walk each.
//
// Tolerances are conservative on purpose: only configurations that would
// genuinely produce identical output are short-circuited.
constexpr float kSamplerFloatEpsilon = 1e-6f;

bool sampler_is_greedy(
    const noumena::cogentengine::InferenceRuntimeConfig &cfg) {
  return cfg.sampling_temperature <= kSamplerFloatEpsilon ||
         cfg.sampling_top_k == 1;
}

bool sampler_penalties_neutral(
    const noumena::cogentengine::InferenceRuntimeConfig &cfg) {
  return cfg.sampling_repeat_penalty == 1.0f &&
         cfg.sampling_frequency_penalty == 0.0f &&
         cfg.sampling_presence_penalty == 0.0f;
}

double duration_ms(std::chrono::steady_clock::time_point start,
                   std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool sampler_top_p_identity(
    const noumena::cogentengine::InferenceRuntimeConfig &cfg) {
  return cfg.sampling_top_p >= 1.0f - kSamplerFloatEpsilon;
}

bool sampler_temp_identity(
    const noumena::cogentengine::InferenceRuntimeConfig &cfg) {
  return std::abs(cfg.sampling_temperature - 1.0f) <= kSamplerFloatEpsilon;
}

// Append the runtime-configured sampling stages onto an already-initialized
// chain.  Honors the greedy fast path and skip-neutral predicates so a chain
// only carries stages that actually change the distribution.  Used by both the
// shared sampler and the per-slot grammar sampler so the two paths cannot
// drift.
//
// (Note: an earlier iteration of this function inserted a custom
// "threshold-prefilter" sampler before `top_k` in the hope of bypassing the
// O(N log K) std::partial_sort that llama.cpp uses for npartial <= 128.
// The prefilter was net-negative on every measured scenario — Qwen3's
// raw-logit distribution is peaked enough that the prefilter's safety
// fallback fired most tokens, costing 2 full scans of the 152k-entry array
// without yielding any reduction in `top_k`'s work.  Removed; revisit only if
// we can either (a) acquire empirical per-model logit-spread data to set the
// threshold dynamically or
// (b) move sampling onto the GPU as a fused kernel.)
void append_configured_sampler_stages(
    llama_sampler *chain,
    const noumena::cogentengine::InferenceRuntimeConfig &cfg) {
  if (chain == nullptr) {
    return;
  }
  if (sampler_is_greedy(cfg)) {
    // Pure argmax: no candidate sort, no penalties, no random draw.  This is
    // the dominant chat configuration and previously paid the full chain.
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    return;
  }
  if (cfg.sampling_top_k > 0) {
    llama_sampler_chain_add(chain,
                            llama_sampler_init_top_k(cfg.sampling_top_k));
  }
  if (!sampler_penalties_neutral(cfg)) {
    llama_sampler_chain_add(
        chain, llama_sampler_init_penalties(cfg.sampling_repeat_last_n,
                                            cfg.sampling_repeat_penalty,
                                            cfg.sampling_frequency_penalty,
                                            cfg.sampling_presence_penalty));
  }
  if (!sampler_top_p_identity(cfg)) {
    llama_sampler_chain_add(chain,
                            llama_sampler_init_top_p(cfg.sampling_top_p, 1));
  }
  if (cfg.sampling_min_p > 0.0f) {
    llama_sampler_chain_add(chain,
                            llama_sampler_init_min_p(cfg.sampling_min_p, 1));
  }
  if (!sampler_temp_identity(cfg)) {
    llama_sampler_chain_add(chain,
                            llama_sampler_init_temp(cfg.sampling_temperature));
  }
  llama_sampler_chain_add(
      chain, llama_sampler_init_dist(resolve_sampling_seed(cfg.sampling_seed)));
}

bool token_to_piece_string(const llama_vocab *vocab, llama_token token,
                           bool special, std::string &out_piece) {
  out_piece.clear();
  if (vocab == nullptr || token < 0) {
    return false;
  }

  char stack_buffer[128];
  const int32_t piece_length = llama_token_to_piece(
      vocab, token, stack_buffer, sizeof(stack_buffer), 0, special);
  if (piece_length >= 0) {
    out_piece.assign(stack_buffer, static_cast<std::size_t>(piece_length));
    return true;
  }

  const int32_t required_length = -piece_length;
  if (required_length <= 0) {
    return false;
  }

  out_piece.resize(static_cast<std::size_t>(required_length));
  const int32_t retry_length = llama_token_to_piece(
      vocab, token, out_piece.data(), required_length, 0, special);
  if (retry_length != required_length) {
    out_piece.clear();
    return false;
  }
  return true;
}

bool token_to_piece_buffer(const llama_vocab *vocab, llama_token token,
                           bool special, char *stack_buffer,
                           std::size_t stack_buffer_size,
                           std::string &overflow_piece, const char *&piece_data,
                           std::size_t &piece_size) {
  overflow_piece.clear();
  piece_data = nullptr;
  piece_size = 0;
  if (vocab == nullptr || token < 0 || stack_buffer == nullptr ||
      stack_buffer_size == 0) {
    return false;
  }

  const int32_t piece_length =
      llama_token_to_piece(vocab, token, stack_buffer,
                           static_cast<int32_t>(stack_buffer_size), 0, special);
  if (piece_length >= 0) {
    piece_data = stack_buffer;
    piece_size = static_cast<std::size_t>(piece_length);
    return true;
  }

  const int32_t required_length = -piece_length;
  if (required_length <= 0) {
    return false;
  }

  overflow_piece.resize(static_cast<std::size_t>(required_length));
  const int32_t retry_length = llama_token_to_piece(
      vocab, token, overflow_piece.data(), required_length, 0, special);
  if (retry_length != required_length) {
    overflow_piece.clear();
    return false;
  }
  piece_data = overflow_piece.data();
  piece_size = static_cast<std::size_t>(retry_length);
  return true;
}

// Returns the number of trailing bytes in `data` that belong to an
// incomplete UTF-8 sequence. UTF-8 code points are 1-4 bytes long, so any
// incomplete tail is at most 3 bytes. If the trailing bytes are a complete
// sequence (or the buffer ends mid-ASCII), returns 0.
std::size_t incomplete_utf8_tail_length(const char *data, std::size_t size) {
  if (data == nullptr || size == 0) {
    return 0;
  }
  const auto is_continuation = [](unsigned char b) {
    return (b & 0xC0u) == 0x80u;
  };
  const std::size_t max_lookback = std::min<std::size_t>(size, 4u);
  for (std::size_t offset = 1; offset <= max_lookback; ++offset) {
    const unsigned char byte = static_cast<unsigned char>(data[size - offset]);
    if (is_continuation(byte)) {
      continue;
    }
    std::size_t expected = 0;
    if ((byte & 0x80u) == 0x00u) {
      expected = 1; // ASCII
    } else if ((byte & 0xE0u) == 0xC0u) {
      expected = 2;
    } else if ((byte & 0xF0u) == 0xE0u) {
      expected = 3;
    } else if ((byte & 0xF8u) == 0xF0u) {
      expected = 4;
    } else {
      // Invalid lead byte; drop only this byte as incomplete to avoid
      // emitting garbage, but do not cascade further.
      return 0;
    }
    if (offset >= expected) {
      // The sequence starting at (size - offset) is complete.
      return 0;
    }
    // Missing (expected - offset) continuation bytes.
    return offset;
  }
  // All trailing bytes are continuations with no lead byte in reach.
  return max_lookback;
}
} // namespace

namespace noumena::cogentengine {

bool InferenceRuntime::EnsureContextSpace(SequenceState &state,
                                          llama_seq_id seq_id,
                                          int new_tokens_needed, int n_ctx) {
  if (shared_context_ == nullptr || seq_id < 0 || n_ctx <= 0) {
    return false;
  }

  if (new_tokens_needed <= 0) {
    return true;
  }
  if (new_tokens_needed > n_ctx) {
    fprintf(stderr, "Input too large for context size!\n");
    return false;
  }

  llama_memory_t mem = llama_get_memory(shared_context_);
  if (state.n_past + new_tokens_needed <= n_ctx) {
    return true;
  }

  const int n_keep = std::min(config_.retained_prefix_tokens, state.n_past);
  const int required_discard = state.n_past + new_tokens_needed - n_ctx;
  const int max_discard = std::max(0, state.n_past - n_keep);
  const int n_discard = std::clamp(required_discard, 0, max_discard);

  if (n_discard <= 0) {
    if (!llama_memory_seq_rm(mem, seq_id, 0, -1)) {
      return false;
    }
    state.current_kv_tokens.clear();
    state.n_past = 0;
    return true;
  }

  if (!llama_memory_seq_rm(mem, seq_id, n_keep, n_keep + n_discard)) {
    return false;
  }

  llama_memory_seq_add(mem, seq_id, n_keep + n_discard, -1, -n_discard);

  if (static_cast<int>(state.current_kv_tokens.size()) > n_keep) {
    const int erase_end =
        std::min<int>(n_keep + n_discard, state.current_kv_tokens.size());
    const auto it_start = state.current_kv_tokens.begin() + n_keep;
    const auto it_end = state.current_kv_tokens.begin() + erase_end;
    state.current_kv_tokens.erase(it_start, it_end);
  } else {
    state.current_kv_tokens.clear();
  }

  if (state.n_past + new_tokens_needed <= n_ctx) {
    return true;
  }

  if (!llama_memory_seq_rm(mem, seq_id, 0, -1)) {
    return false;
  }
  state.current_kv_tokens.clear();
  state.n_past = 0;
  return true;
}

bool InferenceRuntime::ReconcilePhysicalState(SequenceState &state,
                                              llama_seq_id seq_id,
                                              llama_memory_t mem) {
  if (mem == nullptr || seq_id < 0) {
    return false;
  }

  // FORCE PHYSICAL SYNC: Ensure llama.cpp's internal state matches our mirror.
  if (!llama_memory_seq_rm(mem, seq_id,
                           static_cast<int32_t>(state.current_kv_tokens.size()),
                           -1)) {
    return false;
  }
  const int old_n_past = state.n_past;
  state.n_past = static_cast<int>(state.current_kv_tokens.size());
  return true;
}

int32_t InferenceRuntime::ResolveInitialDecodeContextReservationLocked(
    int32_t max_output_tokens) const {
  if (max_output_tokens <= 0) {
    return 0;
  }

  const int32_t configured_headroom =
      std::max<int32_t>(1, config_.scheduler_policy.decode_token_reserve);
  return std::min(max_output_tokens, configured_headroom);
}

bool InferenceRuntime::EnsureDecodeStepContextSpaceLocked(SlotState &slot) {
  if (shared_context_ == nullptr || slot.session == nullptr) {
    return false;
  }

  if (slot.generated_tokens.empty()) {
    return true;
  }

  const int n_ctx = llama_n_ctx(shared_context_);
  if (slot.request != nullptr && slot.request->is_multimodal_turn &&
      slot.session->n_past + 1 > n_ctx) {
    return false;
  }
  return EnsureContextSpace(*slot.session, slot.seq_id, 1, n_ctx);
}

bool InferenceRuntime::PrepareSequenceForPromptLocked(
    const std::string &context_key,
    const std::vector<llama_token> &prompt_tokens, int n_tokens_predict,
    SequenceState &state, llama_seq_id seq_id, GenerateRequest *request,
    std::size_t &out_prefill_cursor) {
  out_prefill_cursor = 0;
  if (shared_context_ == nullptr || seq_id < 0) {
    return false;
  }

  llama_memory_t mem = llama_get_memory(shared_context_);

  const bool has_live_tokens = !state.current_kv_tokens.empty();
  const std::size_t live_match_len =
      has_live_tokens ? session_store_.ComputeLcpReuse(state, prompt_tokens)
                      : 0;
  std::size_t match_len = live_match_len;
  bool restored_from_prefix_cache = false;

  if (!has_live_tokens && !prompt_tokens.empty()) {
    if (const PrefixCacheEntry *cached_prefix =
            prefix_state_cache_.FindBestPrefix(model_fingerprint_, context_key,
                                               prompt_tokens,
                                               prefix_cache_policy_);
        cached_prefix != nullptr) {
      const std::size_t restored = llama_state_seq_set_data(
          shared_context_, cached_prefix->state_bytes.data(),
          cached_prefix->state_bytes.size(), seq_id);
      if (restored == cached_prefix->state_bytes.size()) {
        // The cached `state_bytes` may correspond to a deferred snapshot
        // taken *after* the boundary moment (i.e. the seq had already
        // decoded a few tokens past `token_count` by the time the drain
        // ran).
        state.current_kv_tokens = cached_prefix->prefix_tokens;
        state.n_past = static_cast<int>(cached_prefix->token_count);
        match_len = std::min(cached_prefix->token_count, prompt_tokens.size());
        restored_from_prefix_cache = true;
      } else {
        llama_memory_seq_rm(mem, seq_id, 0, -1);
        state.current_kv_tokens.clear();
        state.n_past = 0;
      }
    }
  }

  // Ensure we have an authoritative match length before we check space.
  match_len = session_store_.ComputeLcpReuse(state, prompt_tokens);

  const int n_ctx = llama_n_ctx(shared_context_);
  const int tokens_to_add = static_cast<int>(prompt_tokens.size() - match_len);
  const int total_needed =
      tokens_to_add +
      ResolveInitialDecodeContextReservationLocked(n_tokens_predict);

  if (!EnsureContextSpace(state, seq_id, total_needed, n_ctx)) {
    return false;
  }

  // Final LCP check after potential eviction.
  match_len = session_store_.ComputeLcpReuse(state, prompt_tokens);

  const bool is_recurrent = llama_model_is_recurrent(primary_model_);
  const bool is_hybrid = llama_model_is_hybrid(primary_model_);
  const bool allow_partial_kv = !(is_recurrent || is_hybrid);

  // If the current match is shorter than the physical KV cache, truncate the
  // tail. CRITICAL: If state.current_kv_tokens is empty, we MUST scrub the
  // entire physical sequence to ensure isolation from previous users of this
  // seq_id (status=-1 fix).
  if (match_len < state.current_kv_tokens.size() ||
      state.current_kv_tokens.empty()) {
    llama_memory_t mem = llama_get_memory(shared_context_);
    if (!allow_partial_kv || state.current_kv_tokens.empty()) {
      llama_memory_seq_rm(mem, seq_id, 0, -1);
      state.current_kv_tokens.clear();
      state.n_past = 0;
      match_len = 0;
    } else {
      if (!llama_memory_seq_rm(mem, seq_id, static_cast<int32_t>(match_len),
                               -1)) {
        return false;
      }
      state.current_kv_tokens.resize(match_len);
      state.n_past = static_cast<int>(match_len);
    }
  }

  // Edge case: if we matched the entire prompt, we must still re-decode the
  // last token to trigger the logits generation required for the next token
  // sampling.
  if (match_len == prompt_tokens.size() && match_len > 0) {
    llama_memory_t mem = llama_get_memory(shared_context_);
    if (!allow_partial_kv) {
      llama_memory_seq_rm(mem, seq_id, 0, -1);
      state.current_kv_tokens.clear();
      state.n_past = 0;
      match_len = 0;
    } else {
      if (!llama_memory_seq_rm(mem, seq_id, static_cast<int32_t>(match_len - 1),
                               -1)) {
        return false;
      }
      state.current_kv_tokens.resize(match_len - 1);
      state.n_past = static_cast<int>(match_len - 1);
      match_len--;
    }
  }

  if (request != nullptr) {
    request->cache_hits = static_cast<int32_t>(match_len);
    total_cache_hits_ += match_len;
  }

  out_prefill_cursor = match_len;
  return true;
}

void InferenceRuntime::MaybeStorePrefixCacheEntryLocked(
    const std::string &context_key, const SequenceState &state,
    llama_seq_id seq_id, std::size_t token_count,
    std::size_t terminal_token_count, GenerateRequest *request) {
  if (request != nullptr && request->is_multimodal_turn) {
    return;
  }
  if (shared_context_ == nullptr || seq_id < 0 || token_count == 0 ||
      token_count > state.current_kv_tokens.size()) {
    return;
  }
  if (!prefix_cache_policy_.ShouldStoreBoundary(token_count,
                                                terminal_token_count)) {
    return;
  }

  // Capture the boundary's identity (tokens + hash) eagerly, but defer the
  // expensive `llama_state_seq_get_data` GPU readback to a quieter moment
  // (`DrainPendingSnapshots` from the burst's Waiting tail or completion
  // path).  Synchronous storage was the source of multi-hundred-millisecond
  // mid-decode tail-ITL spikes whenever an interval boundary landed inside
  // an active streaming response.
  PendingPrefixSnapshot pending;
  pending.model_fingerprint = model_fingerprint_;
  pending.context_key = context_key;
  pending.seq_id = seq_id;
  pending.token_count = token_count;
  pending.prefix_hash =
      prefix_cache_policy_.HashPrefix(state.current_kv_tokens, token_count);
  pending.retention_priority = token_count;
  pending.prefix_tokens.assign(state.current_kv_tokens.begin(),
                               state.current_kv_tokens.begin() +
                                   static_cast<std::ptrdiff_t>(token_count));
  prefix_state_cache_.EnqueuePendingSnapshot(std::move(pending));

  prefix_cache_policy_.RecordStore(token_count);
}

bool InferenceRuntime::RunMultimodalPrefillLocked(SlotState &slot,
                                                  const llama_vocab *vocab) {
  if (shared_context_ == nullptr || mtmd_ctx_ == nullptr || vocab == nullptr ||
      slot.request == nullptr || slot.session == nullptr ||
      slot.sampler == nullptr) {
    return false;
  }

  GenerateRequest &request = *slot.request;
  SequenceState &mirror = slot.mirror;
  if (!request.multimodal.has_value()) {
    return false;
  }

  const MultimodalPayload &multimodal = *request.multimodal;
  std::vector<BitmapPtr> bitmaps;
  bitmaps.reserve(multimodal.image_buffers.size());
  std::vector<const mtmd_bitmap *> bitmap_ptrs;
  bitmap_ptrs.reserve(multimodal.image_buffers.size());
  for (const std::vector<std::uint8_t> &buffer : multimodal.image_buffers) {
    mtmd_bitmap *bitmap = mtmd_helper_bitmap_init_from_buf(
        mtmd_ctx_, buffer.data(), buffer.size());
    if (bitmap == nullptr) {
      request.multimodal.reset();
      return false;
    }
    bitmaps.emplace_back(bitmap, &mtmd_bitmap_free);
    bitmap_ptrs.push_back(bitmap);
  }

  std::string prompt_text = request.original_prompt;
  const char *media_marker = mtmd_default_marker();
  if (media_marker != nullptr && media_marker[0] != '\0') {
    const std::string marker(media_marker);
    std::size_t marker_count = 0;
    std::size_t search_pos = 0;
    while ((search_pos = prompt_text.find(marker, search_pos)) !=
           std::string::npos) {
      marker_count++;
      search_pos += marker.size();
    }
    if (marker_count > bitmap_ptrs.size()) {
      request.multimodal.reset();
      return false;
    }
    while (marker_count < bitmap_ptrs.size()) {
      prompt_text.insert(0, marker);
      marker_count++;
    }
  }

  mtmd_input_text text_input{};
  text_input.text = prompt_text.c_str();
  text_input.add_special = mirror.n_past == 0;
  text_input.parse_special = true;

  InputChunksPtr chunks(mtmd_input_chunks_init(), &mtmd_input_chunks_free);
  if (!chunks || mtmd_tokenize(mtmd_ctx_, chunks.get(), &text_input,
                               bitmap_ptrs.data(), bitmap_ptrs.size()) != 0) {
    request.multimodal.reset();
    return false;
  }

  llama_memory_t memory = llama_get_memory(shared_context_);
  if (!llama_memory_seq_rm(memory, slot.seq_id, 0, -1)) {
    request.multimodal.reset();
    return false;
  }
  mirror.current_kv_tokens.clear();
  mirror.n_past = 0;

  const auto prefill_start = std::chrono::steady_clock::now();
  llama_pos new_n_past = 0;
  const int32_t eval_status = mtmd_helper_eval_chunks(
      mtmd_ctx_, shared_context_, chunks.get(), 0, slot.seq_id,
      ResolveBatchTokenBudgetLocked(), true, &new_n_past);
  const auto prefill_end = std::chrono::steady_clock::now();
  request.multimodal.reset();
  if (eval_status != 0) {
    return false;
  }

  mirror.n_past = static_cast<int>(new_n_past);
  mirror.current_kv_tokens.resize(static_cast<std::size_t>(new_n_past));
  const double multimodal_prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start)
          .count();
  request.input_tokens += static_cast<int32_t>(new_n_past);
  request.prefill_ms += multimodal_prefill_ms;
  total_input_tokens_ += static_cast<std::size_t>(new_n_past);
  total_prefill_ms_ += multimodal_prefill_ms;
  // request.e2e_ms is finalized at completion (enqueued_at → completed_at)
  // by FinalizeCompletedSlots; we don't accumulate ticks into it here.
  slot.prefill_cursor = request.prompt_tokens.size();

  // The multimodal prefill path runs on the same async backends as normal
  // decode, so force completion before reading logits for the first sample.
  llama_synchronize(shared_context_);

  const llama_token next_token =
      llama_sampler_sample(slot.sampler, shared_context_, -1);
  request.first_sampled_token_id = static_cast<int32_t>(next_token);
  if (llama_vocab_is_eog(vocab, next_token)) {
    slot.terminal_error_message =
        "Model ended generation immediately after multimodal prefill "
        "(first sampled token was EOG).";
    return false;
  }

  char piece_buffer[128];
  std::string piece_overflow;
  const char *piece_data = nullptr;
  std::size_t piece_size = 0;
  if (!token_to_piece_buffer(vocab, next_token, false, piece_buffer,
                             sizeof(piece_buffer), piece_overflow, piece_data,
                             piece_size)) {
    slot.terminal_error_message =
        "Failed to convert the first multimodal sampled token to text.";
    return false;
  }
  if (piece_size == 0) {
    slot.terminal_error_message =
        "First multimodal sampled token decoded to an empty text piece.";
    return false;
  }

  slot.generated_tokens.push_back(next_token);
  // Stitch any pending UTF-8 continuation bytes in front of this piece so
  // multi-byte codepoints that span sampled tokens are emitted cleanly.
  std::string stitched = std::move(slot.pending_utf8_bytes);
  slot.pending_utf8_bytes.clear();
  stitched.append(piece_data, piece_size);
  const std::size_t tail_len =
      incomplete_utf8_tail_length(stitched.data(), stitched.size());
  if (tail_len > 0) {
    slot.pending_utf8_bytes.assign(stitched.end() - tail_len, stitched.end());
    stitched.resize(stitched.size() - tail_len);
  }
  if (!stitched.empty()) {
    slot.buffered_output_text.append(stitched);
  }
  slot.phase = SlotPhase::Streaming;
  request.lifecycle = GenerateRequestLifecycle::Streaming;
  slot_scheduler_.EmitBufferedTokenPiece(request_queue_, slot);

  if (request.cancel_requested) {
    slot.terminal_error_message = "Request cancelled.";
    slot.phase = SlotPhase::Failed;
    request.lifecycle = GenerateRequestLifecycle::Cancelled;
    return true;
  }

  if (request.max_output_tokens > 0 &&
      static_cast<int32_t>(slot.generated_tokens.size()) >=
          request.max_output_tokens) {
    slot.phase = SlotPhase::Completed;
    request.lifecycle = GenerateRequestLifecycle::Completed;
  } else {
    slot.phase = SlotPhase::Decode;
    request.lifecycle = GenerateRequestLifecycle::Running;
  }

  return true;
}

bool InferenceRuntime::RecoverDecodeSeedStateLocked(SlotState &slot,
                                                    GenerateRequest &request,
                                                    SequenceState &mirror) {
  if (slot.phase != SlotPhase::Decode || !slot.generated_tokens.empty()) {
    return true;
  }

  if (request.max_output_tokens <= 0) {
    slot.phase = SlotPhase::Completed;
    request.lifecycle = GenerateRequestLifecycle::Completed;
    return true;
  }

  if (request.prompt_tokens.empty()) {
    slot.terminal_error_message =
        "Prompt tokenization produced no tokens, so decode had no seed token.";
    slot.phase = SlotPhase::Failed;
    request.lifecycle = GenerateRequestLifecycle::Failed;
    return false;
  }

  if (slot.prefill_cursor < request.prompt_tokens.size()) {
    slot.phase = SlotPhase::Prefill;
    request.lifecycle = GenerateRequestLifecycle::Running;
    return true;
  }

  if (shared_context_ == nullptr || slot.seq_id < 0) {
    slot.terminal_error_message =
        "Decode slot lost shared context state before its first sampled token.";
    slot.phase = SlotPhase::Failed;
    request.lifecycle = GenerateRequestLifecycle::Failed;
    return false;
  }

  if (mirror.n_past <= 0 || mirror.current_kv_tokens.empty()) {
    slot.prefill_cursor = 0;
    slot.phase = SlotPhase::Prefill;
    request.lifecycle = GenerateRequestLifecycle::Running;
    return true;
  }

  llama_memory_t mem = llama_get_memory(shared_context_);
  const int32_t rewind_position = std::max(0, mirror.n_past - 1);

  // When recovering a decode slot (e.g. after a session reload or failure),
  // we must ensure the physical KV cache is rolled back to match our
  // logical mirror. Legacy code only rewound by 1 token; we now perform
  // a full reconciliation to handle mixed-load session reuse correctly.
  const std::size_t retained_tokens = std::min<std::size_t>(
      mirror.current_kv_tokens.size(),
      static_cast<std::size_t>(std::max(0, rewind_position)));
  mirror.current_kv_tokens.resize(retained_tokens);

  if (!ReconcilePhysicalState(mirror, slot.seq_id, mem)) {
    slot.terminal_error_message = "Failed to reconcile shared KV state for a "
                                  "decode slot during seed recovery.";
    slot.phase = SlotPhase::Failed;
    request.lifecycle = GenerateRequestLifecycle::Failed;
    return false;
  }
  slot.prefill_cursor =
      std::min<std::size_t>(request.prompt_tokens.size() - 1, retained_tokens);
  slot.phase = SlotPhase::Prefill;
  request.lifecycle = GenerateRequestLifecycle::Running;
  return true;
}

bool InferenceRuntime::NormalizeRunnableSlotStateLocked(SlotState &slot) {
  if (slot.request == nullptr) {
    return true;
  }

  GenerateRequest &request = *slot.request;

  if (slot.phase == SlotPhase::Admitted) {
    slot.phase = SlotPhase::Prefill;
  }

  if (slot.phase == SlotPhase::Prefill && !request.is_multimodal_turn &&
      slot.prefill_cursor >= request.prompt_tokens.size() &&
      slot.mirror.n_past > 0) {
    slot.phase = SlotPhase::Decode;
  }

  if (slot.phase == SlotPhase::Streaming && slot.buffered_output_text.empty()) {
    if (request.cancel_requested) {
      slot.terminal_error_message = "Request cancelled.";
      slot.phase = SlotPhase::Failed;
      request.lifecycle = GenerateRequestLifecycle::Cancelled;
      return true;
    }

    if (request.max_output_tokens > 0 &&
        static_cast<int32_t>(slot.generated_tokens.size()) >=
            request.max_output_tokens) {
      slot.phase = SlotPhase::Completed;
      request.lifecycle = GenerateRequestLifecycle::Completed;
      return true;
    }

    slot.phase =
        slot.generated_tokens.empty() ? SlotPhase::Prefill : SlotPhase::Decode;
    request.lifecycle = GenerateRequestLifecycle::Running;
  }

  // Defense-in-depth: a slot that lands in Decode with no sampled token
  // (because, e.g., a state-machine bug advanced `prefill_cursor` past
  // `prompt_tokens.size()` without populating `mirror.n_past` or
  // `generated_tokens`) is otherwise unrunnable — `SelectDecodeReadySlots`
  // filters it out for empty `generated_tokens`, and
  // `SelectPrefillReadySlots` filters it out because phase is no longer
  // Prefill.  `RecoverDecodeSeedStateLocked` rolls the slot back to the
  // last prefill position so the next tick can re-seed it; without this
  // call any such slot becomes a permanent "no progress" hang and trips
  // the FatalNoProgress diagnostic.
  if (slot.phase == SlotPhase::Decode && slot.generated_tokens.empty()) {
    return RecoverDecodeSeedStateLocked(slot, request, slot.mirror);
  }

  return true;
}

std::string InferenceRuntime::BuildNoProgressDiagnosticLocked() const {
  auto phase_name = [](SlotPhase phase) {
    switch (phase) {
    case SlotPhase::Idle:
      return "Idle";
    case SlotPhase::Admitted:
      return "Admitted";
    case SlotPhase::Prefill:
      return "Prefill";
    case SlotPhase::Decode:
      return "Decode";
    case SlotPhase::Streaming:
      return "Streaming";
    case SlotPhase::Completed:
      return "Completed";
    case SlotPhase::Failed:
      return "Failed";
    }
    return "Unknown";
  };

  auto lifecycle_name = [](GenerateRequestLifecycle lifecycle) {
    switch (lifecycle) {
    case GenerateRequestLifecycle::Pending:
      return "Pending";
    case GenerateRequestLifecycle::Admitted:
      return "Admitted";
    case GenerateRequestLifecycle::Running:
      return "Running";
    case GenerateRequestLifecycle::Streaming:
      return "Streaming";
    case GenerateRequestLifecycle::Completed:
      return "Completed";
    case GenerateRequestLifecycle::Cancelled:
      return "Cancelled";
    case GenerateRequestLifecycle::Failed:
      return "Failed";
    }
    return "Unknown";
  };

  int32_t active_count = 0;
  int32_t decode_ready_count = 0;
  int32_t prefill_ready_count = 0;
  int32_t decode_without_seed_count = 0;
  int32_t streaming_without_buffer_count = 0;
  std::ostringstream stream;

  for (const SlotState &slot : slot_scheduler_.Slots()) {
    if (slot.request == nullptr) {
      continue;
    }
    if (slot.phase != SlotPhase::Idle && slot.phase != SlotPhase::Completed &&
        slot.phase != SlotPhase::Failed) {
      active_count++;
    }
    if (slot.phase == SlotPhase::Decode && slot.buffered_output_text.empty() &&
        !slot.generated_tokens.empty()) {
      decode_ready_count++;
    }
    if (slot.phase == SlotPhase::Prefill &&
        (slot.request->is_multimodal_turn ||
         slot.prefill_cursor < slot.request->prompt_tokens.size())) {
      prefill_ready_count++;
    }
    if (slot.phase == SlotPhase::Decode && slot.generated_tokens.empty()) {
      decode_without_seed_count++;
    }
    if (slot.phase == SlotPhase::Streaming &&
        slot.buffered_output_text.empty()) {
      streaming_without_buffer_count++;
    }
  }

  stream << "Shared batch tick could not make progress"
         << " (active=" << active_count
         << ", decode_ready=" << decode_ready_count
         << ", prefill_ready=" << prefill_ready_count
         << ", decode_without_seed=" << decode_without_seed_count
         << ", streaming_without_buffer=" << streaming_without_buffer_count
         << ").";

  int32_t detailed_slots = 0;
  for (const SlotState &slot : slot_scheduler_.Slots()) {
    if (slot.request == nullptr || slot.phase == SlotPhase::Idle) {
      continue;
    }
    if (detailed_slots >= 4) {
      stream << " ...";
      break;
    }

    stream << " slot#" << slot.slot_id << "{phase=" << phase_name(slot.phase)
           << ", request=" << slot.request_id
           << ", lifecycle=" << lifecycle_name(slot.request->lifecycle)
           << ", prefill=" << slot.prefill_cursor << "/"
           << slot.request->prompt_tokens.size()
           << ", generated=" << slot.generated_tokens.size()
           << ", buffered=" << slot.buffered_output_text.size() << ", nPast="
           << (slot.session != nullptr ? slot.session->n_past : -1)
           << ", contextKey=" << slot.request->context_key << "}";
    detailed_slots++;
  }

  return stream.str();
}

void InferenceRuntime::CompletePendingBookkeepingLocked() {
  if (!has_pending_bookkeeping_) {
    return;
  }

  const bool collect_observability = config_.enable_runtime_observability > 0;
  const auto &pb = pending_bookkeeping_;
  const struct llama_model *model = llama_get_model(shared_context_);
  const struct llama_vocab *vocab = llama_model_get_vocab(model);

  // 0. Token-count bookkeeping + per-request "did this tick include a
  // prefill / decode contribution from THIS request?" classification.
  // Cross-mode ticks (one request prefilling while another decodes) must
  // attribute tick wall-time to decode_ms / prefill_ms per request based on
  // that request's own contributions, not the union across the whole batch.
  scratch_tick_requests_.clear();
  bool tick_had_prefill = false;
  bool tick_had_decode = false;
  std::unordered_map<GenerateRequest *, std::uint8_t> per_request_kinds;
  constexpr std::uint8_t kKindPrefill = 1u << 0;
  constexpr std::uint8_t kKindDecode = 1u << 1;

  for (const BatchContribution &contribution : pb.plan.contributions) {
    if (contribution.request == nullptr) {
      continue;
    }
    GenerateRequest &request = *contribution.request;
    auto [it, inserted] = per_request_kinds.try_emplace(&request, 0u);
    if (inserted) {
      scratch_tick_requests_.push_back(&request);
    }
    if (contribution.kind == BatchContributionKind::Prefill) {
      request.input_tokens++;
      total_input_tokens_++;
      tick_had_prefill = true;
      it->second |= kKindPrefill;
    } else {
      tick_had_decode = true;
      request.output_tokens++;
      request.emitted_token_count++;
      total_output_tokens_++;
      it->second |= kKindDecode;
      // first_token_at is set at the sampling site inside RunPolicyBatchTickLocked
      // so TTFT doesn't include the post-decode yield + next-tick prefix.
    }
  }

  // Wall-clock phase time billed to each participating request.  Event-loop
  // contention inside llama_synchronize counts here — it's wall time the
  // model actually took.  Each request's decode_ms / prefill_ms reflects
  // only the kinds *that request* contributed to this tick.
  const double tick_phase_ms = pb.native_gpu_ms + pb.native_logic_ms;
  for (GenerateRequest *request_ptr : scratch_tick_requests_) {
    if (request_ptr == nullptr) {
      continue;
    }
    GenerateRequest &request = *request_ptr;
    const std::uint8_t kinds = per_request_kinds[&request];
    request.native_sync_ms += pb.native_sync_ms;
    request.native_gpu_ms += pb.native_gpu_ms;
    request.native_logic_ms += pb.native_logic_ms;
    if (kinds & kKindDecode) {
      request.decode_ms += tick_phase_ms;
    }
    if (kinds & kKindPrefill) {
      request.prefill_ms += tick_phase_ms;
    }
  }

  // System-wide totals: a mixed tick counts once toward decode (its
  // user-observable phase) when any contribution was decode; otherwise toward
  // prefill if any contribution was prefill.
  if (tick_had_decode) {
    total_decode_ms_ += tick_phase_ms;
  } else if (tick_had_prefill) {
    total_prefill_ms_ += tick_phase_ms;
  }

  // 1. Convert sampled tokens to text pieces. This is native CPU work.
  for (const auto &logits : pb.logits_contributions) {
    if (logits.contribution_index < pb.plan.contributions.size()) {
      const BatchContribution &contribution =
          pb.plan.contributions[logits.contribution_index];
      if (contribution.slot != nullptr) {
        SlotState &slot = *contribution.slot;

        if (logits.sampled_token >= 0 && slot.phase != SlotPhase::Failed) {
          char piece_buffer[128];
          std::string piece_overflow;
          const char *piece_data = nullptr;
          std::size_t piece_size = 0;
          if (token_to_piece_buffer(vocab, logits.sampled_token, false,
                                    piece_buffer, sizeof(piece_buffer),
                                    piece_overflow, piece_data, piece_size)) {
            std::string stitched = std::move(slot.pending_utf8_bytes);
            slot.pending_utf8_bytes.clear();
            stitched.append(piece_data, piece_size);

            const std::size_t tail_len =
                incomplete_utf8_tail_length(stitched.data(), stitched.size());
            if (tail_len > 0) {
              slot.pending_utf8_bytes.assign(stitched.end() - tail_len,
                                             stitched.end());
              stitched.resize(stitched.size() - tail_len);
            }

            if (!stitched.empty()) {
              slot.pending_emission_text.append(stitched);
            }
          } else {
            slot.terminal_error_message =
                "Failed to convert sampled token to text piece.";
            slot.phase = SlotPhase::Failed;
          }
        }

        if (slot.phase == SlotPhase::Completed ||
            slot.phase == SlotPhase::Failed) {
          if (!slot.pending_utf8_bytes.empty()) {
            slot.pending_emission_text.append(slot.pending_utf8_bytes);
            slot.pending_utf8_bytes.clear();
          }
        }

        if (!slot.pending_emission_text.empty()) {
          slot.buffered_output_text.append(slot.pending_emission_text);
          slot.pending_emission_text.clear();
        }
      }
    }
  }

  // 2. Perform deferred Prefix Cache storage.
  for (const auto &entry : pb.prefix_cache_entries) {
    SlotState *slot = entry.first;
    std::size_t token_count = entry.second;
    if (slot != nullptr && slot->request != nullptr &&
        slot->session != nullptr) {
      MaybeStorePrefixCacheEntryLocked(
          slot->request->context_key, *slot->session, slot->seq_id, token_count,
          slot->request->prompt_tokens.size(), slot->request);
    }
  }
}

void InferenceRuntime::EmitPendingTokensLocked() {
  if (!has_pending_bookkeeping_) {
    return;
  }

  for (const BatchContribution &contribution : pending_bookkeeping_.plan.contributions) {
    if (contribution.slot != nullptr) {
      SlotState &slot = *contribution.slot;
      if (!slot.buffered_output_text.empty()) {
        std::lock_guard<std::mutex> lock(request_queue_mutex_);
        slot_scheduler_.EmitBufferedTokenPiece(request_queue_, slot);
      }
    }
  }
  has_pending_bookkeeping_ = false;
}

void InferenceRuntime::FlushAllPendingBookkeepingLocked() {
  CompletePendingBookkeepingLocked();
  EmitPendingTokensLocked();
}

bool InferenceRuntime::RunPolicyBatchTickLocked() {
  if (primary_model_ == nullptr || shared_context_ == nullptr ||
      sampler_ == nullptr) {
    return false;
  }

  // Commit results from the previous tick before starting a new batch.
  if (has_pending_bookkeeping_) {
    FlushAllPendingBookkeepingLocked();
  }

  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  if (vocab == nullptr) {
    return false;
  }

  auto combine_slots = [](std::vector<SlotState *> &out,
                          const std::vector<SlotState *> &left,
                          const std::vector<SlotState *> &right) {
    out.clear();
    out.reserve(left.size() + right.size());
    for (SlotState *slot : left) {
      if (slot != nullptr) {
        out.push_back(slot);
      }
    }
    for (SlotState *slot : right) {
      if (slot != nullptr) {
        out.push_back(slot);
      }
    }
  };

  // Phase 1: Global State Normalization & Sequence Preparation.
  // We perform this pass over ALL mutable slots to ensure that newly admitted
  // or recovered requests are correctly transitioned (e.g. Admitted -> Prefill)
  // and their KV caches are physically sanitized before we select the tick's
  // batch.
  for (SlotState &slot : slot_scheduler_.MutableSlots()) {
    if (slot.request == nullptr || slot.session == nullptr || slot.seq_id < 0) {
      continue;
    }

    // 1. Core State Machine Transitions (Admitted -> Prefill, Seed Recovery,
    // etc.)
    if (!NormalizeRunnableSlotStateLocked(slot)) {
      continue;
    }

    // 2. Sampler Lifecycle Management
    if (slot.sampler == nullptr) {
      if (!slot.request->grammar.empty()) {
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = config_.enable_runtime_observability == 0;
        slot.sampler = llama_sampler_chain_init(sparams);
        if (slot.sampler != nullptr) {
          const llama_vocab *grammar_vocab =
              llama_model_get_vocab(primary_model_);
          llama_sampler *grammar_sampler = llama_sampler_init_grammar(
              grammar_vocab, slot.request->grammar.c_str(), "root");
          if (grammar_sampler == nullptr) {
            llama_sampler_free(slot.sampler);
            slot.sampler = nullptr;
          } else {
            llama_sampler_chain_add(slot.sampler, grammar_sampler);
            append_configured_sampler_stages(slot.sampler, config_);
          }
        }
      } else {
        slot.sampler = llama_sampler_clone(sampler_);
      }
      if (slot.sampler == nullptr) {
        slot.terminal_error_message =
            slot.request->grammar.empty()
                ? "Failed to clone per-slot sampler."
                : "Failed to build per-slot grammar sampler.";
        slot.phase = SlotPhase::Failed;
        slot.request->lifecycle = GenerateRequestLifecycle::Failed;
        continue;
      }
    }

    // 3. Prompt Sequence Preparation (KV Cache Sanitization & LCP Reuse)
    if (slot.phase == SlotPhase::Prefill && slot.prefill_cursor == 0) {
      if (slot.request->is_multimodal_turn) {
        if (!RunMultimodalPrefillLocked(slot, vocab)) {
          if (slot.terminal_error_message.empty()) {
            slot.terminal_error_message =
                "Failed to evaluate multimodal prompt.";
          }
          slot.phase = SlotPhase::Failed;
          slot.request->lifecycle = GenerateRequestLifecycle::Failed;
          slot.request->multimodal.reset();
        }
        continue;
      }

      std::size_t prefill_cursor = 0;
      if (!PrepareSequenceForPromptLocked(
              slot.request->context_key, slot.request->prompt_tokens,
              slot.request->max_output_tokens, slot.mirror, slot.seq_id,
              slot.request, prefill_cursor)) {
        slot.terminal_error_message =
            "Failed to prepare sequence for prompt reuse.";
        slot.phase = SlotPhase::Failed;
        slot.request->lifecycle = GenerateRequestLifecycle::Failed;
        continue;
      }

      slot.prefill_cursor = prefill_cursor;
      slot.phase = slot.prefill_cursor >= slot.request->prompt_tokens.size()
                       ? SlotPhase::Decode
                       : SlotPhase::Prefill;
    }

    // 4. Context Headroom Extension (for Decode slots)
    if (slot.phase == SlotPhase::Decode) {
      if (!EnsureDecodeStepContextSpaceLocked(slot)) {
        slot.terminal_error_message =
            "Failed to extend decode context headroom.";
        slot.phase = SlotPhase::Failed;
        slot.request->lifecycle = GenerateRequestLifecycle::Failed;
        continue;
      }
    }

    slot.request->lifecycle = GenerateRequestLifecycle::Running;
  }

  // Phase 2: Batch Selection.
  // Now that all slots are normalized and their logical mirrors reflect the
  // actual intent (e.g. truncated after a sticky hit), we select the best
  // subset for the hardware batch.
  slot_scheduler_.SelectDecodeReadySlots(scratch_live_decode_ready_slots_);
  slot_scheduler_.SelectPrefillReadySlots(scratch_live_prefill_ready_slots_);
  combine_slots(scratch_live_runnable_slots_, scratch_live_decode_ready_slots_,
                scratch_live_prefill_ready_slots_);

  if (scratch_live_runnable_slots_.empty()) {
    return false;
  }

  const int32_t batch_token_budget = ResolveBatchTokenBudgetLocked();
  const SchedulerTickBudget tick_budget = slot_scheduler_.BuildTickBudget(
      config_.scheduler_policy,
      static_cast<int32_t>(scratch_live_decode_ready_slots_.size()),
      static_cast<int32_t>(scratch_live_prefill_ready_slots_.size()),
      batch_token_budget, config_.prefill_chunk_size);
  const int32_t effective_prefill_chunk_size = ResolvePrefillChunkSizeLocked(
      tick_budget,
      static_cast<int32_t>(scratch_live_decode_ready_slots_.size()),
      static_cast<int32_t>(scratch_live_prefill_ready_slots_.size()));
  const auto tick_start = std::chrono::steady_clock::now();
  SharedBatchPlan plan = batch_planner_.BuildPolicyBatch(
      scratch_live_decode_ready_slots_, scratch_live_prefill_ready_slots_,
      tick_budget, effective_prefill_chunk_size);
  if (plan.Empty()) {
    return false;
  }

  shared_batch_builder_.EnsureCapacity(batch_token_budget,
                                       std::max<int32_t>(1, config_.n_seq_max));
  shared_batch_builder_.Reset();

  // Reuse the persistent scratch (capacity stable across ticks); clear() keeps
  // the existing allocation so the inference hot path performs no heap work.
  scratch_logits_contributions_.clear();

  int32_t batch_token_index = 0;

  for (std::size_t i = 0; i < plan.contributions.size(); ++i) {
    const BatchContribution &contribution = plan.contributions[i];
    if (contribution.slot == nullptr || contribution.slot->seq_id < 0) {
      continue;
    }

    const bool added = shared_batch_builder_.AddToken(
        contribution.token, contribution.position, contribution.slot->seq_id,
        contribution.request_logits);
    if (!added) {
      if (contribution.slot != nullptr) {
        contribution.slot->terminal_error_message =
            "Shared batch builder capacity was exceeded.";
        contribution.slot->phase = SlotPhase::Failed;
      }
      return false;
    }

    if (contribution.request_logits) {
      scratch_logits_contributions_.push_back(PendingLogitsContribution{
          i, contribution.request, batch_token_index});
    }

    batch_token_index++;
  }
  const auto tick_start_pre_decode = std::chrono::steady_clock::now();
  const double planning_ms = duration_ms(tick_start, tick_start_pre_decode);

  const auto gpu_start = std::chrono::steady_clock::now();
  const int32_t decode_status =
      llama_decode(shared_context_, shared_batch_builder_.Get());
  llama_synchronize(shared_context_);
  const auto gpu_end = std::chrono::steady_clock::now();

  // Raw wall-clock decode+sync window.  In WebGPU+wasm this includes any
  // event-loop wait inside llama_synchronize (the GPU-completion microtask
  // queuing behind other worker JS work).
  const double raw_gpu_ms = duration_ms(gpu_start, gpu_end);

  if (decode_status != 0) {
    // ... (rest of failure logic)
    const llama_batch &failing_batch = shared_batch_builder_.Get();
    std::ostringstream diag;
    diag << "llama_decode() failed in shared tick (status=" << decode_status
         << ", n_tokens=" << failing_batch.n_tokens
         << ", n_seq_max=" << config_.n_seq_max
         << ", n_ctx=" << llama_n_ctx(shared_context_) << ", contributions={";
    const int32_t kPreviewLimit = 16;
    const int32_t preview_count =
        std::min(failing_batch.n_tokens, kPreviewLimit);
    for (int32_t i = 0; i < preview_count; ++i) {
      diag << (i == 0 ? "" : ",") << "(seq=" << failing_batch.seq_id[i][0]
           << ",pos=" << failing_batch.pos[i]
           << ",log=" << static_cast<int>(failing_batch.logits[i] != 0) << ")";
    }
    if (failing_batch.n_tokens > kPreviewLimit) {
      diag << ",...+" << (failing_batch.n_tokens - kPreviewLimit);
    }
    diag << "}, slots={";
    bool first_slot = true;
    for (const SlotState *slot : scratch_live_runnable_slots_) {
      if (slot == nullptr) {
        continue;
      }
      if (!first_slot) {
        diag << ",";
      }
      first_slot = false;
      const int32_t n_past = slot->session != nullptr
                                 ? static_cast<int32_t>(slot->mirror.n_past)
                                 : -1;
      const std::size_t kv_tokens =
          slot->session != nullptr ? slot->mirror.current_kv_tokens.size() : 0;
      diag << "(slot=" << slot->slot_id << ",seq=" << slot->seq_id
           << ",phase=" << static_cast<int>(slot->phase)
           << ",cursor=" << slot->prefill_cursor << "/"
           << (slot->request != nullptr ? slot->request->prompt_tokens.size()
                                        : 0)
           << ",n_past=" << n_past << ",kv=" << kv_tokens
           << ",gen=" << slot->generated_tokens.size() << ")";
    }
    diag << "})";
    const std::string diagnostic = diag.str();

    for (SlotState *slot : scratch_live_runnable_slots_) {
      if (slot == nullptr) {
        continue;
      }
      slot->terminal_error_message = diagnostic;
      slot->phase = SlotPhase::Failed;
      if (slot->request != nullptr) {
        slot->request->lifecycle = GenerateRequestLifecycle::Failed;
      }
    }
    FlushAllPendingBookkeepingLocked();
    return false;
  }

  const auto bookkeeping_start = std::chrono::steady_clock::now();
  CompletePendingBookkeepingLocked();
  const auto bookkeeping_end = std::chrono::steady_clock::now();
  const double bookkeeping_ms = duration_ms(bookkeeping_start, bookkeeping_end);

  const double native_logic_ms = planning_ms + bookkeeping_ms;

  // CPU state management (logical context progress)
  for (const BatchContribution &contribution : plan.contributions) {
    if (contribution.slot == nullptr || contribution.slot->request == nullptr) {
      continue;
    }

    SequenceState &mirror = contribution.slot->mirror;
    mirror.current_kv_tokens.push_back(contribution.token);
    mirror.n_past++;
  }

  batch_planner_.ApplyDecodeResults(plan);

  const bool has_decode_pressure = !scratch_live_decode_ready_slots_.empty();
  std::vector<std::pair<SlotState *, std::size_t>> next_prefix_cache_entries;

  if (!has_decode_pressure) {
    if (scratch_prefix_cache_seen_.size() < slot_scheduler_.Slots().size()) {
      scratch_prefix_cache_seen_.assign(slot_scheduler_.Slots().size(), 0);
    }
    for (const BatchContribution &contribution : plan.contributions) {
      if (contribution.kind != BatchContributionKind::Prefill ||
          contribution.slot == nullptr ||
          contribution.slot->request == nullptr ||
          contribution.slot->session == nullptr) {
        continue;
      }
      const std::size_t slot_id = contribution.slot->slot_id;
      if (slot_id >= scratch_prefix_cache_seen_.size() ||
          scratch_prefix_cache_seen_[slot_id]) {
        continue;
      }
      const std::size_t kv_size =
          contribution.slot->mirror.current_kv_tokens.size();
      const std::size_t terminal_size =
          contribution.slot->request->prompt_tokens.size();
      if (!prefix_cache_policy_.ShouldStoreBoundary(kv_size, terminal_size)) {
        continue;
      }
      scratch_prefix_cache_seen_[slot_id] = 1;
      next_prefix_cache_entries.push_back({contribution.slot, kv_size});
    }
    // Clear flags immediately so they are ready for the next tick.
    for (const auto &entry : next_prefix_cache_entries) {
      scratch_prefix_cache_seen_[entry.first->slot_id] = 0;
    }
  }

  for (PendingLogitsContribution &pending_logits :
       scratch_logits_contributions_) {
    const BatchContribution &logit_contribution =
        plan.contributions[pending_logits.contribution_index];
    if (logit_contribution.slot == nullptr ||
        logit_contribution.slot->sampler == nullptr ||
        pending_logits.batch_token_index < 0) {
      continue;
    }

    SlotState &slot = *logit_contribution.slot;
    GenerateRequest &slot_request = *pending_logits.request;
    const llama_token next_token = llama_sampler_sample(
        slot.sampler, shared_context_, pending_logits.batch_token_index);

    pending_logits.sampled_token = next_token;

    // TTFT clock: the moment the first sampled token exists.  Marked here
    // (not at start of the next tick) so the metric reflects model-produce
    // latency without inflating it by the post-tick yield + the next tick's
    // bookkeeping prefix.
    if (!slot_request.has_first_token_at) {
      slot_request.first_token_at = std::chrono::steady_clock::now();
      slot_request.has_first_token_at = true;
    }

    if (llama_vocab_is_eog(vocab, next_token)) {
      if (!slot.pending_utf8_bytes.empty()) {
        slot.pending_emission_text.append(slot.pending_utf8_bytes);
        slot.pending_utf8_bytes.clear();
      }
      slot.phase = SlotPhase::Completed;
      slot_request.lifecycle = GenerateRequestLifecycle::Completed;
      continue;
    }

    slot.generated_tokens.push_back(next_token);

    if (slot_request.max_output_tokens > 0 &&
        static_cast<int32_t>(slot.generated_tokens.size()) >=
            slot_request.max_output_tokens) {
      slot.phase = SlotPhase::Completed;
      slot_request.lifecycle = GenerateRequestLifecycle::Completed;
    } else {
      slot.phase = SlotPhase::Streaming;
      slot_request.lifecycle = GenerateRequestLifecycle::Streaming;
    }
  }

  // Flush any tokens already buffered into the runtime's streaming buffer
  // so they're visible to the next ce_native_yield drain.
  EmitPendingTokensLocked();

  const auto tick_end = std::chrono::steady_clock::now();
  const double total_tick_ms = duration_ms(tick_start, tick_end);

  // Prepare bookkeeping for the NEXT tick's GPU window.
  // We use direct assignment to avoid aggregate initialization errors in some C++ environments.
  pending_bookkeeping_.plan = std::move(plan);
  pending_bookkeeping_.logits_contributions = std::move(scratch_logits_contributions_);
  pending_bookkeeping_.prefix_cache_entries = std::move(next_prefix_cache_entries);
  pending_bookkeeping_.effective_prefill_chunk_size = effective_prefill_chunk_size;
  pending_bookkeeping_.tick_budget = tick_budget;
  pending_bookkeeping_.tick_time_ms = raw_gpu_ms;
  pending_bookkeeping_.native_gpu_ms = raw_gpu_ms;
  pending_bookkeeping_.native_sync_ms = 0.0;
  pending_bookkeeping_.native_logic_ms = native_logic_ms;
  pending_bookkeeping_.total_tick_ms = total_tick_ms;

  has_pending_bookkeeping_ = true;

  // If any slot was marked Completed or Failed this tick, we MUST flush
  // immediately so that the FinalizeCompletedSlots() call following this
  // tick sees the final tokens and accurate request metrics.
  bool must_flush = false;
  for (const SlotState &slot : slot_scheduler_.Slots()) {
    if (slot.phase == SlotPhase::Completed || slot.phase == SlotPhase::Failed) {
      must_flush = true;
      break;
    }
  }

  if (must_flush) {
    FlushAllPendingBookkeepingLocked();
  }

  return true;
}

int32_t InferenceRuntime::ResolvePrefillChunkSizeLocked(
    const SchedulerTickBudget &tick_budget, int32_t decode_ready_count,
    int32_t prefill_ready_count) const {
  const int32_t configured_chunk_size = std::max(0, config_.prefill_chunk_size);
  if (!config_.scheduler_policy.enable_adaptive_prefill_chunking ||
      prefill_ready_count <= 0) {
    return configured_chunk_size;
  }

  if (decode_ready_count <= 0 && configured_chunk_size <= 0) {
    return 0;
  }

  const int32_t prefill_budget = tick_budget.EffectivePrefillBudget();
  if (prefill_budget <= 0) {
    return configured_chunk_size;
  }

  const int32_t fair_share = std::max<int32_t>(
      1, prefill_budget / std::max<int32_t>(1, prefill_ready_count));
  if (configured_chunk_size > 0) {
    return std::min(configured_chunk_size, fair_share);
  }
  return fair_share;
}

void InferenceRuntime::CommitCompletedObservabilityLocked(
    GenerateRequestId request_id, const GenerateResponse &response) {
  if (request_id == 0 ||
      committed_observability_request_ids_.contains(request_id)) {
    return;
  }
  committed_observability_request_ids_.insert(request_id);

  if (config_.enable_runtime_observability == 0) {
    return;
  }

  const RuntimeObservabilityMetrics request_metrics =
      response.runtime_observability;

  // Latency (User Experience)
  last_runtime_observability_.ttft_ms = request_metrics.ttft_ms;
  last_runtime_observability_.itl_avg_ms = request_metrics.itl_avg_ms;
  last_runtime_observability_.itl_p99_ms = request_metrics.itl_p99_ms;
  last_runtime_observability_.e2e_ms = request_metrics.e2e_ms;

  // Phases (Compute)
  last_runtime_observability_.prefill_ms = request_metrics.prefill_ms;
  last_runtime_observability_.decode_ms = request_metrics.decode_ms;

  // Native (Hardware & Engine)
  last_runtime_observability_.native_gpu_ms = request_metrics.native_gpu_ms;
  last_runtime_observability_.native_sync_ms = request_metrics.native_sync_ms;
  last_runtime_observability_.native_logic_ms = request_metrics.native_logic_ms;

  // Counts
  last_runtime_observability_.input_tokens = request_metrics.input_tokens;
  last_runtime_observability_.output_tokens = request_metrics.output_tokens;
  last_runtime_observability_.cache_hits = request_metrics.cache_hits;

  has_last_runtime_observability_ = true;
}

void InferenceRuntime::CommitNewCompletedResponsesObservabilityLocked() {
  std::vector<GenerateRequestId> completed_request_ids;
  {
    std::lock_guard<std::mutex> lock(request_queue_mutex_);
    if (committed_observability_request_ids_.size() >=
        request_queue_.CompletedResponseCount()) {
      return;
    }
    completed_request_ids = request_queue_.CompletedResponseIds();
  }

  if (completed_request_ids.empty()) {
    return;
  }

  std::sort(completed_request_ids.begin(), completed_request_ids.end());
  for (GenerateRequestId request_id : completed_request_ids) {
    if (request_id == 0 ||
        committed_observability_request_ids_.contains(request_id)) {
      continue;
    }

    std::lock_guard<std::mutex> lock(request_queue_mutex_);
    const GenerateResponse *completed =
        request_queue_.PeekCompletedResponse(request_id);
    if (completed == nullptr) {
      continue;
    }
    CommitCompletedObservabilityLocked(request_id, *completed);
  }
}

int32_t InferenceRuntime::ResolveBatchTokenBudgetLocked() const {
  if (shared_context_ != nullptr) {
    const auto n_batch = static_cast<int32_t>(llama_n_batch(shared_context_));
    return std::max<int32_t>(1, n_batch);
  }

  if (config_.n_batch > 0) {
    return std::max<int32_t>(1, config_.n_batch);
  }

  const llama_context_params default_params = llama_context_default_params();
  return std::max<int32_t>(1, static_cast<int32_t>(default_params.n_batch));
}

InferenceRuntime::InferenceRuntime(std::string model_path,
                                   InferenceRuntimeConfig config)
    : config_(normalize_config(config)),
      session_store_(
          static_cast<size_t>(config_.max_cached_sessions),
          static_cast<size_t>(std::max<int32_t>(1, config_.n_seq_max))),
      prefix_state_cache_(static_cast<std::size_t>(
          std::max<int32_t>(1, config_.max_prefix_cache_entries))),
      prefix_cache_policy_(
          static_cast<std::size_t>(config_.prefix_cache_interval_tokens)),
      model_fingerprint_(
          static_cast<std::uint64_t>(std::hash<std::string>{}(model_path))),
      total_decode_ms_(0.0),
      total_prefill_ms_(0.0),
      total_input_tokens_(0),
      total_output_tokens_(0),
      total_cache_hits_(0) {
  if (model_path.empty()) {
    fprintf(stderr, "%s: error: model path is required\n", __func__);
    return;
  }

#if defined(NDEBUG) || defined(CE_SUPPRESS_LLAMA_LOGS)
  llama_log_set(llama_utils::LogCallbackDefault, nullptr);
#endif

  ggml_backend_load_all();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = config_.gpu_layers;
  model_params.use_mlock = false;
#if defined(__EMSCRIPTEN__)
  model_params.use_mmap = false;
#else
  model_params.use_mmap = true;
#endif

  ggml_backend_dev_t cpu_only_devices[2] = {nullptr, nullptr};
  if (config_.gpu_layers == 0) {
    cpu_only_devices[0] =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_only_devices[0] != nullptr) {
      model_params.devices = cpu_only_devices;
    }
  }

  primary_model_ = llama_model_load_from_file(model_path.c_str(), model_params);
  if (primary_model_ == nullptr) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return;
  }

  if (const char *tmpl = llama_model_chat_template(primary_model_, nullptr);
      tmpl != nullptr && tmpl[0] != '\0') {
    try {
      chat_templates_ = common_chat_templates_init(primary_model_, "");
    } catch (const std::exception &error) {
      fprintf(stderr,
              "%s: warning: failed to initialize common chat template: %s\n",
              __func__, error.what());
    }
  }

  shared_context_ = CreateContext();
  if (shared_context_ == nullptr) {
    fprintf(stderr, "%s: error: failed to create shared context\n", __func__);
    return;
  }
  session_store_.BindSharedContext(shared_context_);

  if (!config_.mmproj_path.empty()) {
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = config_.multimodal_use_gpu >= 0
                              ? config_.multimodal_use_gpu != 0
                              : config_.gpu_layers != 0;
    mtmd_params.print_timings = false;
    mtmd_params.n_threads = config_.n_threads > 0
                                ? config_.n_threads
                                : llama_utils::DefaultThreadCount();
    if (config_.flash_attention >= 0) {
      mtmd_params.flash_attn_type =
          static_cast<llama_flash_attn_type>(config_.flash_attention);
    }
    if (config_.image_min_tokens > 0) {
      mtmd_params.image_min_tokens = config_.image_min_tokens;
    }
    if (config_.image_max_tokens > 0) {
      mtmd_params.image_max_tokens = config_.image_max_tokens;
    }
    mtmd_ctx_ = mtmd_init_from_file(config_.mmproj_path.c_str(), primary_model_,
                                    mtmd_params);
    if (mtmd_ctx_ == nullptr) {
      fprintf(stderr,
              "%s: error: failed to initialize multimodal projector from %s\n",
              __func__, config_.mmproj_path.c_str());
      return;
    }
    if (!mtmd_support_vision(mtmd_ctx_)) {
      fprintf(
          stderr,
          "%s: error: multimodal projector does not expose vision support\n",
          __func__);
      mtmd_free(mtmd_ctx_);
      mtmd_ctx_ = nullptr;
      return;
    }
  }

  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = config_.enable_runtime_observability == 0;
  sampler_ = llama_sampler_chain_init(sparams);
  if (!sampler_) {
    return;
  }

  // Stage selection is delegated to append_configured_sampler_stages so the
  // greedy fast path and the skip-neutral gating live in one place and the
  // grammar sampler chain (built per-slot) cannot drift.  Top-K still runs
  // before penalties for the configured stochastic chain.
  append_configured_sampler_stages(sampler_, config_);

  slot_scheduler_.Resize(
      static_cast<std::size_t>(std::max<int32_t>(1, config_.n_seq_max)));
  shared_batch_builder_.EnsureCapacity(ResolveBatchTokenBudgetLocked(),
                                       std::max<int32_t>(1, config_.n_seq_max));
}

llama_context *InferenceRuntime::CreateContext() const {
  if (primary_model_ == nullptr) {
    return nullptr;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx =
      config_.n_ctx > 0
          ? static_cast<uint32_t>(config_.n_ctx)
          : static_cast<uint32_t>(
                std::min(4096 * 2, llama_model_n_ctx_train(primary_model_)));
  if (config_.n_batch > 0) {
    ctx_params.n_batch = static_cast<uint32_t>(config_.n_batch);
  }
  if (config_.n_ubatch > 0) {
    ctx_params.n_ubatch = static_cast<uint32_t>(config_.n_ubatch);
  } else if (ctx_params.n_ubatch > ctx_params.n_batch) {
    ctx_params.n_ubatch = ctx_params.n_batch;
  }
  ctx_params.n_seq_max = static_cast<uint32_t>(config_.n_seq_max);
  ctx_params.n_threads = config_.n_threads > 0
                             ? config_.n_threads
                             : llama_utils::DefaultThreadCount();
  ctx_params.n_threads_batch = config_.n_threads_batch > 0
                                   ? config_.n_threads_batch
                                   : ctx_params.n_threads;
  ctx_params.no_perf = config_.enable_runtime_observability == 0;

  if (config_.flash_attention >= 0) {
    ctx_params.flash_attn_type =
        static_cast<llama_flash_attn_type>(config_.flash_attention);
  }
  if (config_.kv_unified >= 0) {
    ctx_params.kv_unified = config_.kv_unified != 0;
  }

  return llama_init_from_model(primary_model_, ctx_params);
}

InferenceRuntime::~InferenceRuntime() {
  if (sampler_ != nullptr) {
    llama_sampler_free(sampler_);
  }

  session_store_.Clear();

  if (mtmd_ctx_ != nullptr) {
    mtmd_free(mtmd_ctx_);
  }

  if (shared_context_ != nullptr) {
    llama_free(shared_context_);
  }

  if (primary_model_ != nullptr) {
    llama_model_free(primary_model_);
  }
}

bool InferenceRuntime::IsReady() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (primary_model_ == nullptr || shared_context_ == nullptr ||
      sampler_ == nullptr) {
    return false;
  }
  if (!config_.mmproj_path.empty()) {
    return mtmd_ctx_ != nullptr && mtmd_support_vision(mtmd_ctx_);
  }
  return true;
}

bool InferenceRuntime::TryGetRuntimeObservability(
    RuntimeObservabilityMetrics &out) const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (config_.enable_runtime_observability == 0) {
    return false;
  }

  RuntimeObservabilityMetrics metrics = {};
  
  // Cumulative Counters (The Source of Truth for System TPS)
  metrics.input_tokens = static_cast<int32_t>(total_input_tokens_);
  metrics.output_tokens = static_cast<int32_t>(total_output_tokens_);
  metrics.cache_hits = static_cast<int32_t>(total_cache_hits_);
  metrics.prefill_ms = total_prefill_ms_;
  metrics.decode_ms = total_decode_ms_;

  bool any_active = false;
  for (const auto &slot : slot_scheduler_.Slots()) {
    if (slot.request != nullptr) {
      any_active = true;
      // Report TTFT of the first active request in the session.
      if (metrics.ttft_ms == 0.0 && slot.request->has_first_token_at) {
        metrics.ttft_ms = duration_ms(slot.request->enqueued_at, slot.request->first_token_at);
      }
    }
  }

  // If nothing is active, fall back to latency metrics from the last completed request
  // (to avoid zeros in the UI after a query finishes).
  if (!any_active && has_last_runtime_observability_) {
    metrics.ttft_ms = last_runtime_observability_.ttft_ms;
    metrics.itl_avg_ms = last_runtime_observability_.itl_avg_ms;
    metrics.itl_p99_ms = last_runtime_observability_.itl_p99_ms;
    metrics.e2e_ms = last_runtime_observability_.e2e_ms;
  }

  out = metrics;
  return true;
}

bool InferenceRuntime::RuntimeObservabilityEnabled() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return config_.enable_runtime_observability > 0;
}

bool InferenceRuntime::BackendProfilingEnabled() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return config_.enable_backend_profiling > 0;
}

GenerateRequestId InferenceRuntime::EnqueueRequest(
    std::string context_key, std::string prompt, int n_tokens_predict,
    std::string grammar,
    GenerateTokenEmissionMode token_emission_mode) {
  const auto enqueued_at = std::chrono::steady_clock::now();
  // Fast-fail without lock (model pointer is immutable after construction).
  if (primary_model_ == nullptr || sampler_ == nullptr) {
    return 0;
  }
  if (n_tokens_predict <= 0 || n_tokens_predict > kMaxPredictionTokens) {
    return 0;
  }
  if (context_key.empty()) {
    context_key = kDefaultPromptContextKey;
  }

  // Tokenize OUTSIDE the lock – this is the expensive part and does not
  // mutate any runtime state.  primary_model_ is write-once (set in the
  // constructor, cleared only in the destructor) so the vocab read is safe.
  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  auto prompt_tokens = llama_utils::Tokenize(vocab, prompt, true, true);

  // Lock only for the brief queue mutation.
  std::lock_guard<std::mutex> lock(operation_mutex_);

  // Re-check under lock in case of concurrent shutdown.
  if (primary_model_ == nullptr || sampler_ == nullptr) {
    return 0;
  }

  GenerateRequest request;
  request.id = next_request_id_++;
  request.enqueued_at = enqueued_at;
  request.context_key = std::move(context_key);
  request.original_prompt = std::move(prompt);
  request.max_output_tokens = n_tokens_predict;
  request.token_emission_mode = token_emission_mode;
  request.prompt_tokens = std::move(prompt_tokens);
  request.grammar = std::move(grammar);

  if (!request_queue_.Push(std::move(request))) {
    return 0;
  }

  return next_request_id_ - 1;
}

GenerateRequestId InferenceRuntime::EnqueueMultimodalRequest(
    std::string context_key, std::string prompt, int n_tokens_predict,
    std::vector<std::pair<const std::uint8_t *, std::size_t>> image_views,
    std::string grammar,
    GenerateTokenEmissionMode token_emission_mode) {
  const auto enqueued_at = std::chrono::steady_clock::now();
  if (primary_model_ == nullptr || sampler_ == nullptr ||
      mtmd_ctx_ == nullptr || !mtmd_support_vision(mtmd_ctx_)) {
    return 0;
  }
  if (n_tokens_predict <= 0 || n_tokens_predict > kMaxPredictionTokens) {
    return 0;
  }
  if (image_views.empty()) {
    return 0;
  }
  if (context_key.empty()) {
    context_key = kDefaultPromptContextKey;
  }

  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  auto prompt_tokens = llama_utils::Tokenize(vocab, prompt, false, true);
  MultimodalPayload payload;
  payload.image_buffers.reserve(image_views.size());
  for (const auto &[image_data, image_size] : image_views) {
    if (image_data == nullptr || image_size == 0) {
      return 0;
    }
    payload.image_buffers.emplace_back(image_data, image_data + image_size);
  }

  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (primary_model_ == nullptr || sampler_ == nullptr ||
      mtmd_ctx_ == nullptr || !mtmd_support_vision(mtmd_ctx_)) {
    return 0;
  }

  GenerateRequest request;
  request.id = next_request_id_++;
  request.enqueued_at = enqueued_at;
  request.context_key = std::move(context_key);
  request.original_prompt = std::move(prompt);
  request.prompt_tokens = std::move(prompt_tokens);
  request.multimodal = std::move(payload);
  request.max_output_tokens = n_tokens_predict;
  request.token_emission_mode = token_emission_mode;
  request.is_multimodal_turn = true;
  request.grammar = std::move(grammar);

  if (!request_queue_.Push(std::move(request))) {
    return 0;
  }

  return next_request_id_ - 1;
}

bool InferenceRuntime::CancelRequest(GenerateRequestId request_id) {
  std::lock_guard<std::mutex> lock(request_queue_mutex_);
  if (request_id == 0) {
    return false;
  }
  return request_queue_.Cancel(request_id, "Request cancelled.");
}

RequestStepResult InferenceRuntime::RunSchedulerTick() {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return RunSchedulerTickLocked();
}

SchedulerBurstResult InferenceRuntime::RunSchedulerBurst(
    int32_t max_ticks, int32_t max_completed_responses,
    int32_t max_emitted_tokens, int32_t max_duration_us) {
  std::lock_guard<std::mutex> lock(operation_mutex_);

  SchedulerBurstResult burst_result;
  if (max_ticks <= 0 || primary_model_ == nullptr ||
      shared_context_ == nullptr || sampler_ == nullptr) {
    burst_result.status = RequestStepResult::Invalid;
    return burst_result;
  }

  const int32_t clamped_max_completed =
      std::max<int32_t>(0, max_completed_responses);
  const int32_t clamped_max_emitted = std::max<int32_t>(0, max_emitted_tokens);
  const bool has_duration_deadline = max_duration_us > 0;
  const auto deadline = has_duration_deadline
                            ? std::chrono::steady_clock::now() +
                                  std::chrono::microseconds(max_duration_us)
                            : std::chrono::steady_clock::time_point::max();

  for (int32_t tick_index = 0; tick_index < max_ticks; ++tick_index) {
    const std::size_t completed_before =
        request_queue_.CompletedResponseCount();
    const int32_t emitted_before = request_queue_.TotalEmittedTokenCount();
    const RequestStepResult step_result = RunSchedulerTickLocked();
    const std::size_t completed_after = request_queue_.CompletedResponseCount();
    const int32_t emitted_after = request_queue_.TotalEmittedTokenCount();

    burst_result.ticks_executed++;
    if (completed_after > completed_before) {
      burst_result.completed_response_count +=
          static_cast<int32_t>(completed_after - completed_before);
    }
    if (emitted_after > emitted_before) {
      burst_result.emitted_token_count += emitted_after - emitted_before;
    }
    if (step_result == RequestStepResult::Progressed ||
        step_result == RequestStepResult::Terminal) {
      burst_result.progressed_ticks++;
    }

    if (step_result == RequestStepResult::Invalid ||
        step_result == RequestStepResult::FatalNoProgress) {
      burst_result.status = step_result;
      return burst_result;
    }

    // A snapshot drain is "free latency" only at moments when no streaming
    // tokens are about to be observed by the user.  Two such moments exist:
    //
    //   1. `Waiting` — the tick had nothing to schedule.  Pure idle.
    //   2. `completed_response_count > 0` at burst exit — a request just
    //      finished inside this burst.  JS will settle the request, then
    //      enqueue the next one (or yield to the user).  The cost lands
    //      between requests where neither wallMs nor ITL observes it, and
    //      crucially BEFORE the next request's `PrepareSequenceForPrompt`
    //      cache lookup — without this, EMIT_LIMIT=1 streaming bursts
    //      always exit via the emit-limit branch (never `Waiting`) and the
    //      pending snapshot from the just-completed request stays queued
    //      until *its successor* has already missed the cache.
    //
    // During active streaming `completed_response_count == 0`, so this
    // gate keeps the drain off the per-token critical path.

    if (step_result == RequestStepResult::Waiting) {
      prefix_state_cache_.DrainPendingSnapshots(shared_context_, 2);
      burst_result.status = burst_result.progressed_ticks > 0 ||
                                    burst_result.completed_response_count > 0
                                ? RequestStepResult::Progressed
                                : RequestStepResult::Waiting;
      return burst_result;
    }

    const auto drain_if_request_just_completed = [this, &burst_result]() {
      if (burst_result.completed_response_count > 0) {
        prefix_state_cache_.DrainPendingSnapshots(shared_context_, 2);
      }
    };

    if (clamped_max_completed > 0 &&
        burst_result.completed_response_count >= clamped_max_completed) {
      drain_if_request_just_completed();
      burst_result.status = RequestStepResult::Progressed;
      return burst_result;
    }
    if (clamped_max_emitted > 0 &&
        burst_result.emitted_token_count >= clamped_max_emitted) {
      drain_if_request_just_completed();
      burst_result.status = RequestStepResult::Progressed;
      return burst_result;
    }
    if (has_duration_deadline && std::chrono::steady_clock::now() >= deadline) {
      drain_if_request_just_completed();
      burst_result.status = burst_result.progressed_ticks > 0 ||
                                    burst_result.completed_response_count > 0
                                ? RequestStepResult::Progressed
                                : RequestStepResult::Waiting;
      return burst_result;
    }
  }

  FlushAllPendingBookkeepingLocked();

  if (burst_result.completed_response_count > 0) {
    prefix_state_cache_.DrainPendingSnapshots(shared_context_, 2);
  }
  burst_result.status = burst_result.progressed_ticks > 0 ||
                                burst_result.completed_response_count > 0
                            ? RequestStepResult::Progressed
                            : RequestStepResult::Waiting;
  return burst_result;
}

const uint8_t *InferenceRuntime::StreamingBufferPointer() {
  std::lock_guard<std::mutex> lock(request_queue_mutex_);
  return request_queue_.StreamingBufferPointer();
}

std::size_t InferenceRuntime::StreamingBufferCapacity() {
  std::lock_guard<std::mutex> lock(request_queue_mutex_);
  return request_queue_.StreamingBufferCapacity();
}

int32_t *InferenceRuntime::StreamingBufferUsedAddress() {
  std::lock_guard<std::mutex> lock(request_queue_mutex_);
  return request_queue_.StreamingBufferUsedAddress();
}

int32_t *InferenceRuntime::StreamingBufferDropCountAddress() {
  std::lock_guard<std::mutex> lock(request_queue_mutex_);
  return request_queue_.StreamingBufferDropCountAddress();
}

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

extern "C" {
// JSPI yield.  Uses scheduler.yield() when present, otherwise a reusable
// MessageChannel (NOT setTimeout — clamped to 4 ms after 5 nesting levels).
// After the yield resolves and before wasm resumes, invokes the optional
// `Module._ce_yield_drain` hook so the host can drain accumulated streaming
// bytes once per yield window.
// clang-format off
EM_ASYNC_JS(void, ce_native_yield, (), {
  if (typeof scheduler !== 'undefined' && typeof scheduler.yield === 'function') {
    await scheduler.yield();
  } else {
    if (!Module._ce_yield_channel) {
      const channel = new MessageChannel();
      const queue = [];
      channel.port1.onmessage = () => {
        const resolve = queue.shift();
        if (resolve) resolve();
      };
      Module._ce_yield_channel = channel;
      Module._ce_yield_queue = queue;
    }
    await new Promise(resolve => {
      Module._ce_yield_queue.push(resolve);
      Module._ce_yield_channel.port2.postMessage(0);
    });
  }
  if (typeof Module._ce_yield_drain === 'function') {
    try { Module._ce_yield_drain(); }
    catch (e) { if (typeof console !== 'undefined') console.error('ce_yield_drain threw:', e); }
  }
});
} // extern "C"
// clang-format on
#endif

SchedulerBurstResult InferenceRuntime::RunSchedulerLoop(
    int32_t max_ticks, int32_t max_completed_responses,
    int32_t max_emitted_tokens, int32_t max_duration_us) {
  std::unique_lock<std::mutex> lock(operation_mutex_);
  SchedulerBurstResult loop_result;

  if (primary_model_ == nullptr || shared_context_ == nullptr ||
      sampler_ == nullptr) {
    loop_result.status = RequestStepResult::Invalid;
    return loop_result;
  }

  const int32_t yield_threshold_us = 16000; // 16ms
  const auto loop_start = std::chrono::steady_clock::now();
  auto last_yield_time = loop_start;

  while (true) {
    // Check if we have enqueued requests. If not, we can exit the loop.
    if (request_queue_.LiveRequestCount() == 0 && !has_pending_bookkeeping_) {
      loop_result.status = RequestStepResult::Waiting;
      break;
    }

    const std::size_t completed_before = [this]() {
      std::lock_guard<std::mutex> lock(request_queue_mutex_);
      return request_queue_.CompletedResponseCount();
    }();
    const int32_t emitted_before = [this]() {
      std::lock_guard<std::mutex> lock(request_queue_mutex_);
      return request_queue_.TotalEmittedTokenCount();
    }();

    const RequestStepResult step_result = RunSchedulerTickLocked();

    const std::size_t completed_after = [this]() {
      std::lock_guard<std::mutex> lock(request_queue_mutex_);
      return request_queue_.CompletedResponseCount();
    }();
    const int32_t emitted_after = [this]() {
      std::lock_guard<std::mutex> lock(request_queue_mutex_);
      return request_queue_.TotalEmittedTokenCount();
    }();

    loop_result.ticks_executed++;
    if (completed_after > completed_before) {
      loop_result.completed_response_count +=
          static_cast<int32_t>(completed_after - completed_before);
    }
    if (emitted_after > emitted_before) {
      loop_result.emitted_token_count += emitted_after - emitted_before;
    }
    if (step_result == RequestStepResult::Progressed ||
        step_result == RequestStepResult::Terminal) {
      loop_result.progressed_ticks++;
    }

    if (step_result == RequestStepResult::Invalid ||
        step_result == RequestStepResult::FatalNoProgress) {
      loop_result.status = step_result;
      break;
    }

    // Check if we need to yield to the browser for responsiveness.
#ifdef __EMSCRIPTEN__
    const auto now = std::chrono::steady_clock::now();

    // Check if we reached the absolute duration limit.
    if (max_duration_us > 0) {
      const auto elapsed_us =
          std::chrono::duration_cast<std::chrono::microseconds>(now -
                                                                loop_start)
              .count();
      if (elapsed_us >= max_duration_us) {
        loop_result.status = RequestStepResult::Progressed;
        break;
      }
    }

    if (std::chrono::duration_cast<std::chrono::microseconds>(now -
                                                              last_yield_time)
            .count() >= yield_threshold_us) {
      // Must flush bookkeeping before yielding so JS sees the latest tokens.
      FlushAllPendingBookkeepingLocked();

      // Unlock while suspended so other JS calls can potentially interact
      // with the runtime (though they should ideally wait).
      lock.unlock();
      ce_native_yield();
      lock.lock();

      last_yield_time = std::chrono::steady_clock::now();
    }
#endif

    // Termination conditions
    if (max_ticks > 0 && loop_result.ticks_executed >= max_ticks) {
      loop_result.status = RequestStepResult::Progressed;
      break;
    }
    if (max_completed_responses > 0 &&
        loop_result.completed_response_count >= max_completed_responses) {
      loop_result.status = RequestStepResult::Progressed;
      break;
    }
    if (max_emitted_tokens > 0 &&
        loop_result.emitted_token_count >= max_emitted_tokens) {
      loop_result.status = RequestStepResult::Progressed;
      break;
    }
  }

  FlushAllPendingBookkeepingLocked();
  return loop_result;
}

RequestStepResult InferenceRuntime::RunSchedulerTickLocked() {
  if (primary_model_ == nullptr || shared_context_ == nullptr ||
      sampler_ == nullptr) {
    return RequestStepResult::Invalid;
  }

  const std::size_t completed_before = [this]() {
    std::lock_guard<std::mutex> lock(request_queue_mutex_);
    return request_queue_.CompletedResponseCount();
  }();

  bool admitted_any = false;
  while (true) {
    std::lock_guard<std::mutex> lock(request_queue_mutex_);
    if (!slot_scheduler_.AdmitPendingRequests(request_queue_, session_store_)) {
      break;
    }
    admitted_any = true;
  }

  const bool tick_executed = RunPolicyBatchTickLocked();

  // Only flush on no-progress.  Success path keeps pending bookkeeping for
  // the next tick's CPU/GPU overlap window; must_flush inside
  // RunPolicyBatchTickLocked already handles terminating slots.
  if (!tick_executed) {
    FlushAllPendingBookkeepingLocked();
  }

  {
    std::lock_guard<std::mutex> lock(request_queue_mutex_);
    slot_scheduler_.FinalizeCompletedSlots(request_queue_, session_store_);
  }

  CommitNewCompletedResponsesObservabilityLocked();

  const std::size_t completed_after = [this]() {
    std::lock_guard<std::mutex> lock(request_queue_mutex_);
    return request_queue_.CompletedResponseCount();
  }();

  if (completed_after > completed_before) {
    return RequestStepResult::Progressed;
  }

  if (!tick_executed) {
    SlotState *active_slot = slot_scheduler_.FindFirstActiveSlot();
    if (active_slot == nullptr) {
      return admitted_any ? RequestStepResult::Progressed
                          : RequestStepResult::Waiting;
    }

    if (active_slot->phase != SlotPhase::Failed &&
        active_slot->phase != SlotPhase::Completed) {
      active_slot->terminal_error_message = BuildNoProgressDiagnosticLocked();
      active_slot->phase = SlotPhase::Failed;
      slot_scheduler_.FinalizeCompletedSlots(request_queue_, session_store_);
      CommitNewCompletedResponsesObservabilityLocked();
      if (request_queue_.CompletedResponseCount() > completed_before) {
        return RequestStepResult::Progressed;
      }
      return RequestStepResult::FatalNoProgress;
    }
  }

  return (tick_executed || admitted_any) ? RequestStepResult::Progressed
                                         : RequestStepResult::Waiting;
}

bool InferenceRuntime::TryPeekCompletedResponse(
    GenerateRequestId request_id, GenerateResponse &out_response) const {
  std::lock_guard<std::mutex> lock(request_queue_mutex_);
  const GenerateResponse *response =
      request_queue_.PeekCompletedResponse(request_id);
  if (response == nullptr) {
    return false;
  }
  out_response = *response;
  return true;
}

bool InferenceRuntime::HasRequest(GenerateRequestId request_id) const {
  std::lock_guard<std::mutex> lock(request_queue_mutex_);
  return request_queue_.Contains(request_id);
}

bool InferenceRuntime::ConsumeCompletedResponse(GenerateRequestId request_id) {
  std::lock_guard<std::mutex> lock(request_queue_mutex_);
  committed_observability_request_ids_.erase(request_id);
  return request_queue_.ConsumeCompletedResponse(request_id);
}

const char *InferenceRuntime::GetMediaMarker() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (mtmd_ctx_ == nullptr || !mtmd_support_vision(mtmd_ctx_)) {
    return nullptr;
  }
  return mtmd_default_marker();
}

const char *InferenceRuntime::GetChatTemplate() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (primary_model_ == nullptr) {
    return nullptr;
  }
  const char *tmpl = llama_model_chat_template(primary_model_, nullptr);
  return tmpl != nullptr && tmpl[0] != '\0' ? tmpl : nullptr;
}

std::string InferenceRuntime::GetBosText() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (primary_model_ == nullptr) {
    return {};
  }
  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  if (vocab == nullptr) {
    return {};
  }
  const llama_token bos = llama_vocab_bos(vocab);
  if (bos == LLAMA_TOKEN_NULL) {
    return {};
  }
  std::string piece;
  if (!token_to_piece_string(vocab, bos, true, piece) || piece.empty()) {
    return {};
  }
  return piece;
}

std::string InferenceRuntime::GetEosText() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (primary_model_ == nullptr) {
    return {};
  }
  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  if (vocab == nullptr) {
    return {};
  }
  const llama_token eos = llama_vocab_eos(vocab);
  if (eos == LLAMA_TOKEN_NULL) {
    return {};
  }
  std::string piece;
  if (!token_to_piece_string(vocab, eos, true, piece) || piece.empty()) {
    return {};
  }
  return piece;
}

std::string InferenceRuntime::ApplyChatTemplate(
    const std::vector<common_chat_msg> &messages, bool add_assistant) const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (primary_model_ == nullptr || messages.empty()) {
    return {};
  }

  const char *tmpl = llama_model_chat_template(primary_model_, nullptr);
  if (tmpl == nullptr || tmpl[0] == '\0' || chat_templates_ == nullptr) {
    return {};
  }

  try {
    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.add_generation_prompt = add_assistant;
    inputs.use_jinja = true;
    return common_chat_templates_apply(chat_templates_.get(), inputs).prompt;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s: warning: failed to apply common chat template: %s\n",
            __func__, error.what());
    return {};
  }
}

} // namespace noumena::cogentengine
