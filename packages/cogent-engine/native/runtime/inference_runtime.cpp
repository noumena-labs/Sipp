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
#include <exception>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
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
constexpr std::size_t kDefaultPrefixCacheIntervalTokens = 128;

using BitmapPtr = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
using InputChunksPtr =
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;

noumena::cogentengine::InferenceRuntimeConfig
normalize_config(noumena::cogentengine::InferenceRuntimeConfig config) {
  config.n_seq_max = std::max<int32_t>(1, config.n_seq_max);
  config.gpu_layers = std::max<int32_t>(0, config.gpu_layers);
  config.max_cached_sessions = std::max<int32_t>(1, config.max_cached_sessions);
  config.retained_prefix_tokens =
      std::max<int32_t>(0, config.retained_prefix_tokens);
  config.prefill_chunk_size = std::max<int32_t>(0, config.prefill_chunk_size);
  config.prefix_cache_interval_tokens =
      std::max<int32_t>(1, config.prefix_cache_interval_tokens);
  config.max_prefix_cache_entries =
      std::max<int32_t>(1, config.max_prefix_cache_entries);
  config.image_min_tokens = std::max<int32_t>(0, config.image_min_tokens);
  config.image_max_tokens = std::max<int32_t>(0, config.image_max_tokens);
  config.multimodal_use_gpu =
      std::clamp<int32_t>(config.multimodal_use_gpu, -1, 1);
  config.debug_compare_multimodal_embeddings =
      config.debug_compare_multimodal_embeddings > 0 ? 1 : 0;
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

double proportional_share(double total, int32_t part, int32_t whole) {
  if (total <= 0.0 || part <= 0 || whole <= 0) {
    return 0.0;
  }
  return total * (static_cast<double>(part) / static_cast<double>(whole));
}

uint32_t resolve_sampling_seed(int32_t seed) {
  if (seed < 0) {
    return LLAMA_DEFAULT_SEED;
  }
  return static_cast<uint32_t>(seed);
}

} // namespace

namespace noumena::cogentengine {

bool InferenceRuntime::EnsureContextSpace(SequenceState &state,
                                          int new_tokens_needed, int n_ctx) {
  if (shared_context_ == nullptr || state.seq_id < 0 || n_ctx <= 0) {
    return false;
  }

  if (new_tokens_needed <= 0) {
    return true;
  }

  if (new_tokens_needed > n_ctx) {
    fprintf(stderr, "Input too large for context size!\n");
    return false;
  }

  if (state.n_past + new_tokens_needed <= n_ctx) {
    return true;
  }

  const int n_keep = std::min(config_.retained_prefix_tokens, state.n_past);
  const int required_discard = state.n_past + new_tokens_needed - n_ctx;
  const int max_discard = std::max(0, state.n_past - n_keep);
  const int n_discard = std::clamp(required_discard, 0, max_discard);

  llama_memory_t mem = llama_get_memory(shared_context_);

  if (n_discard <= 0) {
    if (!llama_memory_seq_rm(mem, state.seq_id, 0, -1)) {
      return false;
    }
    state.current_kv_tokens.clear();
    state.n_past = 0;
    return true;
  }

  if (!llama_memory_seq_rm(mem, state.seq_id, n_keep, n_keep + n_discard)) {
    return false;
  }

  llama_memory_seq_add(mem, state.seq_id, n_keep + n_discard, -1, -n_discard);

  if (static_cast<int>(state.current_kv_tokens.size()) > n_keep) {
    const int erase_end =
        std::min<int>(n_keep + n_discard, state.current_kv_tokens.size());
    const auto it_start = state.current_kv_tokens.begin() + n_keep;
    const auto it_end = state.current_kv_tokens.begin() + erase_end;
    state.current_kv_tokens.erase(it_start, it_end);
  } else {
    state.current_kv_tokens.clear();
  }

  state.n_past = std::max(0, state.n_past - n_discard);

  if (state.n_past + new_tokens_needed <= n_ctx) {
    return true;
  }

  if (!llama_memory_seq_rm(mem, state.seq_id, 0, -1)) {
    return false;
  }
  state.current_kv_tokens.clear();
  state.n_past = 0;

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
  return EnsureContextSpace(*slot.session, 1, n_ctx);
}

bool InferenceRuntime::PrepareSequenceForPromptLocked(
    const std::string &context_key,
    const std::vector<llama_token> &prompt_tokens, int n_tokens_predict,
    SequenceState &state, GenerateRequest *request,
    std::size_t &out_prefill_cursor) {
  out_prefill_cursor = 0;
  if (shared_context_ == nullptr || state.seq_id < 0) {
    return false;
  }

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
      llama_memory_t mem = llama_get_memory(shared_context_);
      llama_memory_seq_rm(mem, state.seq_id, 0, -1);

      const std::size_t restored = llama_state_seq_set_data(
          shared_context_, cached_prefix->state_bytes.data(),
          cached_prefix->state_bytes.size(), state.seq_id);
      if (restored == cached_prefix->state_bytes.size()) {
        state.current_kv_tokens = cached_prefix->prefix_tokens;
        state.n_past = static_cast<int>(cached_prefix->token_count);
        match_len = cached_prefix->token_count;
        restored_from_prefix_cache = true;
        if (request != nullptr) {
          request->prefix_cache_restore_tokens +=
              static_cast<int32_t>(cached_prefix->token_count);
          request->prefix_cache_hit_count++;
        }
      } else {
        llama_memory_seq_rm(mem, state.seq_id, 0, -1);
        state.current_kv_tokens.clear();
        state.n_past = 0;
      }
    }
  }

  if (request != nullptr) {
    request->lcp_reuse_tokens = static_cast<int32_t>(live_match_len);
  }

  const int n_ctx = llama_n_ctx(shared_context_);
  const int tokens_to_add = static_cast<int>(prompt_tokens.size() - match_len);
  const int initial_decode_headroom =
      ResolveInitialDecodeContextReservationLocked(n_tokens_predict);
  const int total_needed = tokens_to_add + initial_decode_headroom;
  if (!EnsureContextSpace(state, total_needed, n_ctx)) {
    return false;
  }

  // EnsureContextSpace may have shrunk the KV cache (tail truncation) or evict
  // tokens from the middle (shifting the sequence). Either action invalidates
  // our previously calculated match_len if the mutated state no longer matches
  // the prompt. Re-compute the true longest common prefix length to guarantee
  // that we don't accidentally skip prefilling tokens that were just evicted.
  match_len = 0;
  for (std::size_t i = 0;
       i < std::min(state.current_kv_tokens.size(), prompt_tokens.size());
       ++i) {
    if (state.current_kv_tokens[i] == prompt_tokens[i]) {
      match_len++;
    } else {
      break;
    }
  }

  // Sync diagnostic counters with the post-eviction match_len so that
  // observability metrics reflect the actual reuse, not the stale
  // pre-eviction value.
  if (request != nullptr) {
    if (restored_from_prefix_cache) {
      request->prefix_cache_restore_tokens = static_cast<int32_t>(match_len);
    } else {
      request->lcp_reuse_tokens = static_cast<int32_t>(match_len);
    }
  }

  llama_memory_t mem = llama_get_memory(shared_context_);
  const bool is_recurrent = llama_model_is_recurrent(primary_model_);
  const bool is_hybrid = llama_model_is_hybrid(primary_model_);
  const bool allow_partial_kv = !(is_recurrent || is_hybrid);

  if (match_len < state.current_kv_tokens.size()) {
    if (!allow_partial_kv) {
      llama_memory_seq_rm(mem, state.seq_id, 0, -1);
      state.current_kv_tokens.clear();
      state.n_past = 0;
      match_len = 0;
      if (request != nullptr) {
        request->lcp_reuse_tokens = 0;
        if (restored_from_prefix_cache) {
          request->prefix_cache_restore_tokens = 0;
        }
      }
    } else {
      if (!llama_memory_seq_rm(mem, state.seq_id,
                               static_cast<int32_t>(match_len), -1)) {
        return false;
      }
      state.current_kv_tokens.resize(match_len);
      state.n_past = static_cast<int>(match_len);
    }
  }

  if (match_len == prompt_tokens.size() && match_len > 0) {
    if (!allow_partial_kv) {
      llama_memory_seq_rm(mem, state.seq_id, 0, -1);
      state.current_kv_tokens.clear();
      state.n_past = 0;
      match_len = 0;
      if (request != nullptr) {
        request->lcp_reuse_tokens = 0;
      }
    } else {
      if (!llama_memory_seq_rm(mem, state.seq_id,
                               static_cast<int32_t>(match_len - 1), -1)) {
        return false;
      }
      state.current_kv_tokens.resize(match_len - 1);
      state.n_past = static_cast<int>(match_len - 1);
      match_len--;
      if (request != nullptr) {
        if (restored_from_prefix_cache) {
          request->prefix_cache_restore_tokens =
              static_cast<int32_t>(std::max<std::size_t>(match_len, 0));
        } else {
          request->lcp_reuse_tokens =
              static_cast<int32_t>(std::max<std::size_t>(match_len, 0));
        }
      }
    }
  }

  out_prefill_cursor = match_len;
  return true;
}

void InferenceRuntime::MaybeStorePrefixCacheEntryLocked(
    const std::string &context_key, const SequenceState &state,
    std::size_t token_count, std::size_t terminal_token_count,
    GenerateRequest *request) {
  if (request != nullptr && request->is_multimodal_turn) {
    return;
  }
  if (shared_context_ == nullptr || state.seq_id < 0 || token_count == 0 ||
      token_count > state.current_kv_tokens.size()) {
    return;
  }
  if (!prefix_cache_policy_.ShouldStoreBoundary(token_count,
                                                terminal_token_count)) {
    return;
  }

  const std::uint64_t prefix_hash =
      prefix_cache_policy_.HashPrefix(state.current_kv_tokens, token_count);
  if (!prefix_state_cache_.StorePrefixState(
          shared_context_, state.seq_id, model_fingerprint_, context_key,
          state.current_kv_tokens, token_count, prefix_hash, token_count)) {
    return;
  }

  prefix_cache_policy_.RecordStore(token_count);
  if (request != nullptr) {
    request->prefix_cache_store_count++;
  }
}

bool InferenceRuntime::RunMultimodalPrefillLocked(SlotState &slot,
                                                  const llama_vocab *vocab) {
  if (shared_context_ == nullptr || mtmd_ctx_ == nullptr || vocab == nullptr ||
      slot.request == nullptr || slot.session == nullptr ||
      slot.sampler == nullptr) {
    return false;
  }

  GenerateRequest &request = *slot.request;
  SequenceState &session = *slot.session;
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
  if (media_marker != nullptr && media_marker[0] != '\0' &&
      prompt_text.find(media_marker) == std::string::npos) {
    for (std::size_t index = 0; index < bitmap_ptrs.size(); ++index) {
      prompt_text.insert(0, media_marker);
    }
  }

  mtmd_input_text text_input{};
  text_input.text = prompt_text.c_str();
  text_input.add_special = session.n_past == 0;
  text_input.parse_special = true;

  InputChunksPtr chunks(mtmd_input_chunks_init(), &mtmd_input_chunks_free);
  if (!chunks || mtmd_tokenize(mtmd_ctx_, chunks.get(), &text_input,
                               bitmap_ptrs.data(), bitmap_ptrs.size()) != 0) {
    request.multimodal.reset();
    return false;
  }

  llama_memory_t memory = llama_get_memory(shared_context_);
  if (!llama_memory_seq_rm(memory, session.seq_id, 0, -1)) {
    request.multimodal.reset();
    return false;
  }
  session.current_kv_tokens.clear();
  session.n_past = 0;

  const auto prefill_start = std::chrono::steady_clock::now();
  llama_pos new_n_past = 0;
  const int32_t eval_status = mtmd_helper_eval_chunks(
      mtmd_ctx_, shared_context_, chunks.get(), 0, session.seq_id,
      config_.n_batch > 0 ? config_.n_batch : 256, true, &new_n_past);
  const auto prefill_end = std::chrono::steady_clock::now();
  request.multimodal.reset();
  if (eval_status != 0) {
    return false;
  }

  session.n_past = static_cast<int>(new_n_past);
  const double multimodal_prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start)
          .count();
  request.attributed_prompt_eval_tokens += session.n_past;
  request.attributed_prompt_eval_ms += multimodal_prefill_ms;
  request.attributed_total_ms += multimodal_prefill_ms;
  slot.prefill_cursor = request.prompt_tokens.size();

  // The multimodal prefill path runs on the same async backends as normal
  // decode, so force completion before reading logits for the first sample.
  llama_synchronize(shared_context_);

  const llama_token next_token =
      llama_sampler_sample(slot.sampler, shared_context_, -1);
  llama_sampler_accept(slot.sampler, next_token);
  request.attributed_sample_count++;
  request.first_sampled_token_id = static_cast<int32_t>(next_token);
  if (llama_vocab_is_eog(vocab, next_token)) {
    slot.terminal_error_message =
        "Model ended generation immediately after multimodal prefill "
        "(first sampled token was EOG).";
    return false;
  }

  char piece_buffer[128];
  const int piece_length = llama_token_to_piece(vocab, next_token, piece_buffer,
                                                sizeof(piece_buffer), 0, true);
  if (piece_length < 0) {
    slot.terminal_error_message =
        "Failed to convert the first multimodal sampled token to text.";
    return false;
  }
  if (piece_length == 0) {
    slot.terminal_error_message =
        "First multimodal sampled token decoded to an empty text piece.";
    return false;
  }

  slot.generated_tokens.push_back(next_token);
  slot.buffered_output_text.append(piece_buffer,
                                   static_cast<std::size_t>(piece_length));
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

bool InferenceRuntime::RunPolicyBatchTickLocked() {
  if (primary_model_ == nullptr || shared_context_ == nullptr ||
      sampler_ == nullptr) {
    return false;
  }

  auto combine_slots = [](const std::vector<SlotState *> &left,
                          const std::vector<SlotState *> &right) {
    std::vector<SlotState *> combined;
    combined.reserve(left.size() + right.size());
    std::unordered_set<SlotState *> seen;
    seen.reserve(left.size() + right.size());
    for (SlotState *slot : left) {
      if (slot != nullptr && seen.insert(slot).second) {
        combined.push_back(slot);
      }
    }
    for (SlotState *slot : right) {
      if (slot != nullptr && seen.insert(slot).second) {
        combined.push_back(slot);
      }
    }
    return combined;
  };

  std::vector<SlotState *> decode_ready_slots =
      slot_scheduler_.SelectDecodeReadySlots();
  std::vector<SlotState *> prefill_ready_slots =
      slot_scheduler_.SelectPrefillReadySlots();
  std::vector<SlotState *> runnable_slots =
      combine_slots(decode_ready_slots, prefill_ready_slots);
  if (runnable_slots.empty()) {
    return false;
  }

  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  if (vocab == nullptr) {
    return false;
  }

  for (SlotState *slot : runnable_slots) {
    if (slot == nullptr || slot->request == nullptr ||
        slot->session == nullptr || slot->seq_id < 0) {
      if (slot != nullptr) {
        slot->terminal_error_message =
            "Runnable slot lost request or sequence state.";
        slot->phase = SlotPhase::Failed;
        if (slot->request != nullptr) {
          slot->request->lifecycle = GenerateRequestLifecycle::Failed;
        }
      }
      continue;
    }

    if (slot->sampler == nullptr) {
      slot->sampler = llama_sampler_clone(sampler_);
      if (slot->sampler == nullptr) {
        slot->terminal_error_message = "Failed to clone per-slot sampler.";
        slot->phase = SlotPhase::Failed;
        slot->request->lifecycle = GenerateRequestLifecycle::Failed;
        continue;
      }
    }

    GenerateRequest &request = *slot->request;
    SequenceState &session = *slot->session;

    if (slot->phase == SlotPhase::Prefill && slot->prefill_cursor == 0) {
      if (request.is_multimodal_turn) {
        if (!RunMultimodalPrefillLocked(*slot, vocab)) {
          if (slot->terminal_error_message.empty()) {
            slot->terminal_error_message =
                "Failed to evaluate multimodal prompt.";
          }
          slot->phase = SlotPhase::Failed;
          request.lifecycle = GenerateRequestLifecycle::Failed;
          request.multimodal.reset();
        }
        continue;
      }

      std::size_t prefill_cursor = 0;
      if (!PrepareSequenceForPromptLocked(
              request.context_key, request.prompt_tokens,
              request.max_output_tokens, session, &request, prefill_cursor)) {
        slot->terminal_error_message =
            "Failed to prepare sequence for prompt reuse.";
        slot->phase = SlotPhase::Failed;
        request.lifecycle = GenerateRequestLifecycle::Failed;
        continue;
      }

      slot->prefill_cursor = prefill_cursor;
      slot->phase = slot->prefill_cursor >= request.prompt_tokens.size()
                        ? SlotPhase::Decode
                        : SlotPhase::Prefill;
    }

    request.lifecycle = GenerateRequestLifecycle::Running;
    if (config_.enable_runtime_observability > 0) {
      llama_perf_sampler_reset(slot->sampler);
    }
  }

  std::vector<SlotState *> live_decode_ready_slots =
      slot_scheduler_.SelectDecodeReadySlots();
  for (SlotState *slot : live_decode_ready_slots) {
    if (slot == nullptr || slot->request == nullptr ||
        slot->session == nullptr) {
      continue;
    }

    if (!EnsureDecodeStepContextSpaceLocked(*slot)) {
      slot->terminal_error_message =
          "Failed to extend decode context headroom.";
      slot->phase = SlotPhase::Failed;
      slot->request->lifecycle = GenerateRequestLifecycle::Failed;
    }
  }
  live_decode_ready_slots = slot_scheduler_.SelectDecodeReadySlots();
  std::vector<SlotState *> live_prefill_ready_slots =
      slot_scheduler_.SelectPrefillReadySlots();
  std::vector<SlotState *> live_runnable_slots =
      combine_slots(live_decode_ready_slots, live_prefill_ready_slots);
  if (live_runnable_slots.empty()) {
    return false;
  }

  const int32_t batch_token_budget =
      config_.n_batch > 0 ? config_.n_batch : 256;
  const SchedulerTickBudget tick_budget = slot_scheduler_.BuildTickBudget(
      config_.scheduler_policy,
      static_cast<int32_t>(live_decode_ready_slots.size()),
      static_cast<int32_t>(live_prefill_ready_slots.size()), batch_token_budget,
      config_.prefill_chunk_size);
  const int32_t effective_prefill_chunk_size = ResolvePrefillChunkSizeLocked(
      tick_budget, static_cast<int32_t>(live_decode_ready_slots.size()),
      static_cast<int32_t>(live_prefill_ready_slots.size()));
  SharedBatchPlan plan = batch_planner_.BuildPolicyBatch(
      live_decode_ready_slots, live_prefill_ready_slots, tick_budget,
      effective_prefill_chunk_size);
  if (plan.Empty()) {
    return false;
  }

  {
    std::vector<GenerateRequest *> tick_requests;
    tick_requests.reserve(plan.contributions.size());
    std::vector<GenerateRequest *> decode_requests;
    decode_requests.reserve(plan.contributions.size());
    std::vector<GenerateRequest *> prefill_requests;
    prefill_requests.reserve(plan.contributions.size());
    std::unordered_set<GenerateRequest *> tick_request_set;
    tick_request_set.reserve(plan.contributions.size());
    std::unordered_set<GenerateRequest *> decode_request_set;
    decode_request_set.reserve(plan.contributions.size());
    std::unordered_set<GenerateRequest *> prefill_request_set;
    prefill_request_set.reserve(plan.contributions.size());

    const auto mark_request = [](std::vector<GenerateRequest *> &requests,
                                 std::unordered_set<GenerateRequest *> &seen,
                                 GenerateRequest *request) {
      if (request == nullptr || !seen.insert(request).second) {
        return;
      }
      requests.push_back(request);
    };

    for (const BatchContribution &contribution : plan.contributions) {
      if (contribution.slot == nullptr ||
          contribution.slot->request == nullptr) {
        continue;
      }
      mark_request(tick_requests, tick_request_set, contribution.slot->request);
      if (contribution.kind == BatchContributionKind::Decode) {
        mark_request(decode_requests, decode_request_set,
                     contribution.slot->request);
      } else if (contribution.kind == BatchContributionKind::Prefill) {
        mark_request(prefill_requests, prefill_request_set,
                     contribution.slot->request);
      }
    }

    if (plan.prefill_token_count > 0 && plan.decode_token_count > 0) {
      for (GenerateRequest *request : tick_requests) {
        request->mixed_workload_tick_count++;
      }
    }
    if (tick_budget.EffectiveDecodeBudget() > 0) {
      for (GenerateRequest *request : decode_requests) {
        request->decode_first_tick_count++;
      }
    }
    if (effective_prefill_chunk_size > 0) {
      for (GenerateRequest *request : prefill_requests) {
        request->chunked_prefill_tick_count++;
      }
    }
  }

  shared_batch_builder_.EnsureCapacity(batch_token_budget,
                                       std::max<int32_t>(1, config_.n_seq_max));
  shared_batch_builder_.Reset();

  struct PendingLogitsContribution {
    const BatchContribution *contribution = nullptr;
    int32_t batch_token_index = -1;
  };

  std::vector<PendingLogitsContribution> logits_contributions;
  logits_contributions.reserve(plan.contributions.size());

  int32_t batch_token_index = 0;

  for (const BatchContribution &contribution : plan.contributions) {
    if (contribution.slot == nullptr || contribution.slot->seq_id < 0) {
      continue;
    }

    const bool added =
        contribution.kind == BatchContributionKind::Prefill
            ? shared_batch_builder_.AddPrefillToken(
                  contribution.token, contribution.position,
                  contribution.slot->seq_id, contribution.request_logits)
            : shared_batch_builder_.AddDecodeToken(
                  contribution.token, contribution.position,
                  contribution.slot->seq_id, contribution.request_logits);
    if (!added) {
      if (contribution.slot != nullptr) {
        contribution.slot->terminal_error_message =
            "Shared batch builder capacity was exceeded.";
        contribution.slot->phase = SlotPhase::Failed;
      }
      return false;
    }

    if (contribution.request_logits) {
      logits_contributions.push_back(
          PendingLogitsContribution{&contribution, batch_token_index});
    }

    batch_token_index++;
  }

  llama_perf_context_reset(shared_context_);
  const auto tick_start = std::chrono::steady_clock::now();

  if (llama_decode(shared_context_, shared_batch_builder_.Get()) != 0) {
    for (SlotState *slot : live_runnable_slots) {
      if (slot == nullptr) {
        continue;
      }
      slot->terminal_error_message = "llama_decode() failed in shared tick.";
      slot->phase = SlotPhase::Failed;
      if (slot->request != nullptr) {
        slot->request->lifecycle = GenerateRequestLifecycle::Failed;
      }
    }
    return false;
  }

  llama_synchronize(shared_context_);

  for (const BatchContribution &contribution : plan.contributions) {
    if (contribution.slot == nullptr || contribution.slot->session == nullptr) {
      continue;
    }

    SequenceState &session = *contribution.slot->session;
    session.current_kv_tokens.push_back(contribution.token);
    session.n_past++;
  }

  batch_planner_.ApplyDecodeResults(plan);

  // Stall-free prefix caching: only store snapshots when there are no
  // active decode slots competing for the lock.  When decode and prefill
  // are interleaved in the same tick, the KV state serialization would
  // stall token generation for all active users.  By deferring the
  // snapshot, we trade a potential cache miss on a future request for
  // smooth, uninterrupted decode latency now.
  //
  // Reference: Sarathi-Serve (OSDI '24, arXiv:2403.02310) establishes
  // that protecting decode latency over prefill efficiency is the correct
  // trade-off in user-facing LLM serving systems.
  const bool has_decode_pressure = !live_decode_ready_slots.empty();

  if (!has_decode_pressure) {
    std::vector<SlotState *> prefix_cache_slots;
    prefix_cache_slots.reserve(plan.contributions.size());
    std::unordered_set<SlotState *> prefix_cache_slot_set;
    prefix_cache_slot_set.reserve(plan.contributions.size());
    for (const BatchContribution &contribution : plan.contributions) {
      if (contribution.kind != BatchContributionKind::Prefill ||
          contribution.slot == nullptr ||
          contribution.slot->request == nullptr ||
          contribution.slot->session == nullptr) {
        continue;
      }
      if (!prefix_cache_slot_set.insert(contribution.slot).second) {
        continue;
      }
      prefix_cache_slots.push_back(contribution.slot);
    }

    for (SlotState *slot : prefix_cache_slots) {
      MaybeStorePrefixCacheEntryLocked(
          slot->request->context_key, *slot->session,
          slot->session->current_kv_tokens.size(),
          slot->request->prompt_tokens.size(), slot->request);
    }
  }

  for (const PendingLogitsContribution &pending_logits : logits_contributions) {
    const BatchContribution *logit_contribution = pending_logits.contribution;
    if (logit_contribution == nullptr || logit_contribution->slot == nullptr ||
        logit_contribution->slot->request == nullptr ||
        logit_contribution->slot->sampler == nullptr ||
        pending_logits.batch_token_index < 0) {
      continue;
    }

    SlotState &slot = *logit_contribution->slot;
    GenerateRequest &slot_request = *slot.request;
    const llama_token next_token = llama_sampler_sample(
        slot.sampler, shared_context_, pending_logits.batch_token_index);
    llama_sampler_accept(slot.sampler, next_token);
    if (slot_request.first_sampled_token_id < 0) {
      slot_request.first_sampled_token_id = static_cast<int32_t>(next_token);
    }

    if (llama_vocab_is_eog(vocab, next_token)) {
      slot.phase = SlotPhase::Completed;
      slot_request.lifecycle = GenerateRequestLifecycle::Completed;
      continue;
    }

    char piece_buffer[128];
    const int piece_length = llama_token_to_piece(
        vocab, next_token, piece_buffer, sizeof(piece_buffer), 0, true);
    if (piece_length < 0) {
      slot.terminal_error_message =
          "Failed to convert sampled token to text piece.";
      slot.phase = SlotPhase::Failed;
      slot_request.lifecycle = GenerateRequestLifecycle::Failed;
      continue;
    }
    if (piece_length == 0 && slot_request.emitted_token_count == 0) {
      slot.terminal_error_message =
          "Leading sampled token decoded to an empty text piece.";
      slot.phase = SlotPhase::Failed;
      slot_request.lifecycle = GenerateRequestLifecycle::Failed;
      continue;
    }

    slot.generated_tokens.push_back(next_token);
    slot.buffered_output_text.append(piece_buffer,
                                     static_cast<std::size_t>(piece_length));
    slot.phase = SlotPhase::Streaming;
    slot_request.lifecycle = GenerateRequestLifecycle::Streaming;
    slot_scheduler_.EmitBufferedTokenPiece(request_queue_, slot);

    if (slot_request.cancel_requested) {
      slot.terminal_error_message = "Request cancelled.";
      slot.phase = SlotPhase::Failed;
      slot_request.lifecycle = GenerateRequestLifecycle::Cancelled;
      continue;
    }

    if (slot_request.max_output_tokens > 0 &&
        static_cast<int32_t>(slot.generated_tokens.size()) >=
            slot_request.max_output_tokens) {
      slot.phase = SlotPhase::Completed;
      slot_request.lifecycle = GenerateRequestLifecycle::Completed;
    } else if (slot.phase != SlotPhase::Failed) {
      slot.phase = SlotPhase::Decode;
      slot_request.lifecycle = GenerateRequestLifecycle::Running;
    }
  }

  const auto tick_end = std::chrono::steady_clock::now();
  if (config_.enable_runtime_observability > 0) {
    const auto ctx_perf = llama_perf_context(shared_context_);
    const double tick_total_ms =
        std::chrono::duration<double, std::milli>(tick_end - tick_start)
            .count();

    std::vector<GenerateRequest *> attributed_requests;
    attributed_requests.reserve(plan.contributions.size());
    std::unordered_set<GenerateRequest *> attributed_request_set;
    attributed_request_set.reserve(plan.contributions.size());
    std::unordered_map<GenerateRequest *, int32_t> prefill_token_counts;
    std::unordered_map<GenerateRequest *, int32_t> decode_token_counts;
    std::unordered_map<GenerateRequest *, int32_t> sample_counts;
    std::unordered_map<GenerateRequest *, double> sample_ms_by_request;
    prefill_token_counts.reserve(plan.contributions.size());
    decode_token_counts.reserve(plan.contributions.size());
    sample_counts.reserve(logits_contributions.size());
    sample_ms_by_request.reserve(logits_contributions.size());

    const auto ensure_attributed_request = [&](GenerateRequest *request) {
      if (request == nullptr ||
          !attributed_request_set.insert(request).second) {
        return;
      }
      attributed_requests.push_back(request);
    };

    for (const BatchContribution &contribution : plan.contributions) {
      if (contribution.slot == nullptr ||
          contribution.slot->request == nullptr) {
        continue;
      }
      GenerateRequest *request = contribution.slot->request;
      ensure_attributed_request(request);
      if (contribution.kind == BatchContributionKind::Prefill) {
        prefill_token_counts[request]++;
      } else if (contribution.kind == BatchContributionKind::Decode) {
        decode_token_counts[request]++;
      }
    }

    for (const PendingLogitsContribution &pending_logits :
         logits_contributions) {
      const BatchContribution *contribution = pending_logits.contribution;
      if (contribution == nullptr || contribution->slot == nullptr ||
          contribution->slot->request == nullptr) {
        continue;
      }

      GenerateRequest *request = contribution->slot->request;
      ensure_attributed_request(request);
      sample_counts[request]++;
      if (contribution->slot->sampler != nullptr) {
        sample_ms_by_request[request] =
            llama_perf_sampler(contribution->slot->sampler).t_sample_ms;
      }
    }

    double tick_sample_ms = 0.0;
    for (const auto &[_, request_sample_ms] : sample_ms_by_request) {
      tick_sample_ms += request_sample_ms;
    }

    const int32_t total_prefill_tokens = plan.prefill_token_count;
    const int32_t total_decode_tokens = plan.decode_token_count;
    const int32_t total_sample_count =
        static_cast<int32_t>(logits_contributions.size());
    const int32_t total_work_units =
        total_prefill_tokens + total_decode_tokens + total_sample_count;
    const double tick_overhead_ms =
        std::max(0.0, tick_total_ms - ctx_perf.t_p_eval_ms -
                          ctx_perf.t_eval_ms - tick_sample_ms);

    for (GenerateRequest *request : attributed_requests) {
      if (request == nullptr) {
        continue;
      }

      const int32_t request_prefill_tokens =
          prefill_token_counts.contains(request) ? prefill_token_counts[request]
                                                 : 0;
      const int32_t request_decode_tokens =
          decode_token_counts.contains(request) ? decode_token_counts[request]
                                                : 0;
      const int32_t request_sample_count =
          sample_counts.contains(request) ? sample_counts[request] : 0;
      const double request_sample_ms = sample_ms_by_request.contains(request)
                                           ? sample_ms_by_request[request]
                                           : 0.0;
      const int32_t request_work_units =
          request_prefill_tokens + request_decode_tokens + request_sample_count;

      const double prompt_share_ms = proportional_share(
          ctx_perf.t_p_eval_ms, request_prefill_tokens, total_prefill_tokens);
      const double decode_share_ms = proportional_share(
          ctx_perf.t_eval_ms, request_decode_tokens, total_decode_tokens);
      const double overhead_share_ms = proportional_share(
          tick_overhead_ms, request_work_units, total_work_units);

      request->attributed_prompt_eval_ms += prompt_share_ms;
      request->attributed_decode_eval_ms += decode_share_ms;
      request->attributed_sample_ms += request_sample_ms;
      request->attributed_total_ms += prompt_share_ms + decode_share_ms +
                                      request_sample_ms + overhead_share_ms;
      request->attributed_prompt_eval_tokens += request_prefill_tokens;
      request->attributed_decode_eval_count += request_decode_tokens;
      request->attributed_sample_count += request_sample_count;
    }

    if (!has_last_runtime_observability_) {
      last_runtime_observability_ = {};
      for (SlotState *slot : live_runnable_slots) {
        if (slot != nullptr && slot->request != nullptr) {
          last_runtime_observability_.input_token_count +=
              static_cast<int32_t>(slot->request->prompt_tokens.size());
        }
      }
      has_last_runtime_observability_ = true;
    }

    last_runtime_observability_.total_ms += tick_total_ms;
    last_runtime_observability_.prompt_eval_ms += ctx_perf.t_p_eval_ms;
    last_runtime_observability_.decode_eval_ms += ctx_perf.t_eval_ms;
    last_runtime_observability_.prompt_eval_tokens += plan.prefill_token_count;
    last_runtime_observability_.decode_eval_count += plan.decode_token_count;
    last_runtime_observability_.sample_count +=
        static_cast<int32_t>(logits_contributions.size());
    last_runtime_observability_.output_token_count = 0;
    last_runtime_observability_.first_sampled_token_id = -1;
    last_runtime_observability_.lcp_reuse_tokens = 0;
    last_runtime_observability_.prefix_cache_restore_tokens = 0;
    last_runtime_observability_.prefix_cache_hit_count = 0;
    last_runtime_observability_.prefix_cache_store_count = 0;
    for (SlotState *slot : live_runnable_slots) {
      if (slot != nullptr && slot->request != nullptr) {
        last_runtime_observability_.output_token_count +=
            slot->request->emitted_token_count;
        if (last_runtime_observability_.first_sampled_token_id < 0 &&
            slot->request->first_sampled_token_id >= 0) {
          last_runtime_observability_.first_sampled_token_id =
              slot->request->first_sampled_token_id;
        }
      }
      if (slot != nullptr && slot->request != nullptr) {
        last_runtime_observability_.lcp_reuse_tokens +=
            slot->request->lcp_reuse_tokens;
        last_runtime_observability_.prefix_cache_restore_tokens +=
            slot->request->prefix_cache_restore_tokens;
        last_runtime_observability_.prefix_cache_hit_count +=
            slot->request->prefix_cache_hit_count;
        last_runtime_observability_.prefix_cache_store_count +=
            slot->request->prefix_cache_store_count;
      }
    }
    last_runtime_observability_.sample_ms += tick_sample_ms;

    UpdateSharedBatchMetricsLocked(plan);
    UpdateSchedulerObservabilityLocked(plan, tick_budget,
                                       effective_prefill_chunk_size);
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

void InferenceRuntime::UpdateSharedBatchMetricsLocked(
    const SharedBatchPlan &plan) {
  if (plan.Empty()) {
    return;
  }

  shared_batch_observability_.tick_count++;
  shared_batch_observability_.total_occupied_slots +=
      static_cast<std::uint64_t>(std::max(0, plan.occupied_slot_count));
  shared_batch_observability_.total_prefill_tokens +=
      static_cast<std::uint64_t>(std::max(0, plan.prefill_token_count));
  shared_batch_observability_.total_decode_tokens +=
      static_cast<std::uint64_t>(std::max(0, plan.decode_token_count));
}

void InferenceRuntime::UpdateSchedulerObservabilityLocked(
    const SharedBatchPlan &plan, const SchedulerTickBudget &budget,
    int32_t effective_prefill_chunk_size) {
  // Phase 4 algorithm steps:
  // 1. Record whether this tick used explicit decode reservation.
  // 2. Record whether chunked prefill was active.
  // 3. Record whether the tick mixed decode and prefill contributions.
  // 4. Later, attach real queue delay, TTFT, ITL, and tail ITL once the
  //    request lifecycle carries precise timestamps.
  scheduler_observability_.tick_count++;
  if (budget.EffectiveDecodeBudget() > 0 && plan.decode_token_count > 0) {
    scheduler_observability_.decode_first_tick_count++;
  }
  if (effective_prefill_chunk_size > 0 && plan.prefill_token_count > 0) {
    scheduler_observability_.chunked_prefill_tick_count++;
  }
  if (plan.decode_token_count > 0 && plan.prefill_token_count > 0) {
    scheduler_observability_.mixed_workload_tick_count++;
  }
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

  const RuntimeObservabilityMetrics accumulated_runtime_observability =
      last_runtime_observability_;
  last_runtime_observability_ = response.runtime_observability;
  last_runtime_observability_.total_ms =
      accumulated_runtime_observability.total_ms > 0.0
          ? accumulated_runtime_observability.total_ms
          : std::max(response.runtime_observability.total_ms,
                     response.runtime_observability.e2e_ms);
  last_runtime_observability_.prompt_eval_ms =
      accumulated_runtime_observability.prompt_eval_ms;
  last_runtime_observability_.decode_eval_ms =
      accumulated_runtime_observability.decode_eval_ms;
  last_runtime_observability_.sample_ms =
      accumulated_runtime_observability.sample_ms;
  last_runtime_observability_.prompt_eval_tokens =
      accumulated_runtime_observability.prompt_eval_tokens;
  last_runtime_observability_.decode_eval_count =
      std::max(last_runtime_observability_.decode_eval_count,
               accumulated_runtime_observability.decode_eval_count);
  last_runtime_observability_.sample_count =
      std::max(last_runtime_observability_.sample_count,
               accumulated_runtime_observability.sample_count);
  last_runtime_observability_.output_token_count =
      std::max(last_runtime_observability_.output_token_count,
               accumulated_runtime_observability.output_token_count);
  has_last_runtime_observability_ = true;

  scheduler_observability_.accumulated_queue_delay_ms +=
      response.runtime_observability.queue_delay_ms;
  scheduler_observability_.accumulated_ttft_ms +=
      response.runtime_observability.ttft_ms;
  scheduler_observability_.max_tail_itl_ms =
      std::max(scheduler_observability_.max_tail_itl_ms,
               response.runtime_observability.tail_itl_ms);
}

void InferenceRuntime::CommitNewCompletedResponsesObservabilityLocked() {
  std::vector<GenerateRequestId> completed_request_ids =
      request_queue_.CompletedResponseIds();
  if (completed_request_ids.empty()) {
    return;
  }

  std::sort(completed_request_ids.begin(), completed_request_ids.end());
  for (GenerateRequestId request_id : completed_request_ids) {
    if (request_id == 0 ||
        committed_observability_request_ids_.contains(request_id)) {
      continue;
    }

    const GenerateResponse *completed =
        request_queue_.PeekCompletedResponse(request_id);
    if (completed == nullptr) {
      continue;
    }
    CommitCompletedObservabilityLocked(request_id, *completed);
  }
}

InferenceRuntime::InferenceRuntime(std::string model_path,
                                   InferenceRuntimeConfig config)
    : config_(normalize_config(config)),
      session_store_(
          static_cast<size_t>(config_.max_cached_sessions),
          static_cast<size_t>(std::max<int32_t>(1, config_.n_seq_max))),
      prefix_state_cache_(static_cast<std::size_t>(
          std::max<int32_t>(1, config_.max_prefix_cache_entries))),
      prefix_cache_policy_(static_cast<std::size_t>(
          config_.prefix_cache_interval_tokens > 0
              ? config_.prefix_cache_interval_tokens
              : static_cast<int32_t>(kDefaultPrefixCacheIntervalTokens))),
      model_fingerprint_(
          static_cast<std::uint64_t>(std::hash<std::string>{}(model_path))) {
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
    mtmd_params.debug_compare_embeddings =
        config_.debug_compare_multimodal_embeddings > 0;
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
  }

  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = config_.enable_runtime_observability == 0;
  sampler_ = llama_sampler_chain_init(sparams);
  if (!sampler_) {
    return;
  }

  llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(
                                        config_.sampling_repeat_last_n,
                                        config_.sampling_repeat_penalty,
                                        config_.sampling_frequency_penalty,
                                        config_.sampling_presence_penalty));
  llama_sampler_chain_add(sampler_,
                          llama_sampler_init_top_k(config_.sampling_top_k));
  llama_sampler_chain_add(sampler_,
                          llama_sampler_init_top_p(config_.sampling_top_p, 1));
  if (config_.sampling_min_p > 0.0f) {
    llama_sampler_chain_add(
        sampler_, llama_sampler_init_min_p(config_.sampling_min_p, 1));
  }
  llama_sampler_chain_add(
      sampler_, llama_sampler_init_temp(config_.sampling_temperature));
  llama_sampler_chain_add(
      sampler_,
      llama_sampler_init_dist(resolve_sampling_seed(config_.sampling_seed)));

  slot_scheduler_.Resize(
      static_cast<std::size_t>(std::max<int32_t>(1, config_.n_seq_max)));
  shared_batch_builder_.EnsureCapacity(config_.n_batch > 0 ? config_.n_batch
                                                           : 256,
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
  ctx_params.n_batch =
      config_.n_batch > 0 ? static_cast<uint32_t>(config_.n_batch) : 256u;
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

  llama_backend_free();
}

bool InferenceRuntime::IsReady() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return primary_model_ != nullptr && shared_context_ != nullptr &&
         sampler_ != nullptr;
}

bool InferenceRuntime::TryGetRuntimeObservability(
    RuntimeObservabilityMetrics &out) const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (config_.enable_runtime_observability == 0 ||
      !has_last_runtime_observability_) {
    return false;
  }

  out = last_runtime_observability_;
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

void InferenceRuntime::ResetRuntimeObservability() {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  last_runtime_observability_ = {};
  has_last_runtime_observability_ = false;
  shared_batch_observability_ = {};
  scheduler_observability_ = {};
  committed_observability_request_ids_.clear();
}

GenerateRequestId
InferenceRuntime::EnqueueRequest(std::string context_key, std::string prompt,
                                 int n_tokens_predict,
                                 TokenCallback on_token_received) {
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
  auto prompt_tokens = llama_utils::Tokenize(vocab, prompt, false, true);

  // Lock only for the brief queue mutation.
  std::lock_guard<std::mutex> lock(operation_mutex_);

  // Re-check under lock in case of concurrent shutdown.
  if (primary_model_ == nullptr || sampler_ == nullptr) {
    return 0;
  }

  GenerateRequest request;
  request.id = next_request_id_++;
  request.context_key = std::move(context_key);
  request.original_prompt = std::move(prompt);
  request.max_output_tokens = n_tokens_predict;
  request.on_token_received = std::move(on_token_received);
  request.prompt_tokens = std::move(prompt_tokens);

  if (!request_queue_.Push(std::move(request))) {
    return 0;
  }

  return next_request_id_ - 1;
}

GenerateRequestId InferenceRuntime::EnqueueMultimodalRequest(
    std::string context_key, std::string prompt, int n_tokens_predict,
    std::vector<std::pair<const std::uint8_t *, std::size_t>> image_views,
    TokenCallback on_token_received) {
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
  request.context_key = std::move(context_key);
  request.original_prompt = std::move(prompt);
  request.prompt_tokens = std::move(prompt_tokens);
  request.multimodal = std::move(payload);
  request.max_output_tokens = n_tokens_predict;
  request.on_token_received = std::move(on_token_received);
  request.is_multimodal_turn = true;

  if (!request_queue_.Push(std::move(request))) {
    return 0;
  }

  return next_request_id_ - 1;
}

bool InferenceRuntime::CancelRequest(GenerateRequestId request_id) {
  std::lock_guard<std::mutex> lock(operation_mutex_);
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

    if (step_result == RequestStepResult::Waiting) {
      burst_result.status = burst_result.progressed_ticks > 0 ||
                                    burst_result.completed_response_count > 0
                                ? RequestStepResult::Progressed
                                : RequestStepResult::Waiting;
      return burst_result;
    }

    if (clamped_max_completed > 0 &&
        burst_result.completed_response_count >= clamped_max_completed) {
      burst_result.status = RequestStepResult::Progressed;
      return burst_result;
    }
    if (clamped_max_emitted > 0 &&
        burst_result.emitted_token_count >= clamped_max_emitted) {
      burst_result.status = RequestStepResult::Progressed;
      return burst_result;
    }
    if (has_duration_deadline && std::chrono::steady_clock::now() >= deadline) {
      burst_result.status = burst_result.progressed_ticks > 0 ||
                                    burst_result.completed_response_count > 0
                                ? RequestStepResult::Progressed
                                : RequestStepResult::Waiting;
      return burst_result;
    }
  }

  burst_result.status = burst_result.progressed_ticks > 0 ||
                                burst_result.completed_response_count > 0
                            ? RequestStepResult::Progressed
                            : RequestStepResult::Waiting;
  return burst_result;
}

RequestStepResult InferenceRuntime::RunSchedulerTickLocked() {

  if (primary_model_ == nullptr || shared_context_ == nullptr ||
      sampler_ == nullptr) {
    return RequestStepResult::Invalid;
  }

  const std::size_t completed_before = request_queue_.CompletedResponseCount();
  bool admitted_any = false;
  while (slot_scheduler_.AdmitPendingRequests(request_queue_, session_store_)) {
    admitted_any = true;
  }

  const bool tick_executed = RunPolicyBatchTickLocked();

  slot_scheduler_.FinalizeCompletedSlots(request_queue_, session_store_);
  CommitNewCompletedResponsesObservabilityLocked();
  if (request_queue_.CompletedResponseCount() > completed_before) {
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
      active_slot->terminal_error_message =
          "Shared batch tick could not make progress.";
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

RequestStepResult
InferenceRuntime::RunRequestStep(GenerateRequestId request_id) {
  std::lock_guard<std::mutex> lock(operation_mutex_);

  if (request_id == 0 || primary_model_ == nullptr ||
      shared_context_ == nullptr || sampler_ == nullptr) {
    return RequestStepResult::Invalid;
  }

  if (const GenerateResponse *completed =
          request_queue_.PeekCompletedResponse(request_id);
      completed != nullptr) {
    CommitCompletedObservabilityLocked(request_id, *completed);
    return RequestStepResult::Terminal;
  }

  GenerateRequest *target_request = request_queue_.FindMutable(request_id);
  if (target_request == nullptr) {
    return RequestStepResult::Invalid;
  }

  bool admitted_any = false;
  while (slot_scheduler_.AdmitPendingRequests(request_queue_, session_store_)) {
    admitted_any = true;
  }

  if (const GenerateResponse *completed =
          request_queue_.PeekCompletedResponse(request_id);
      completed != nullptr) {
    CommitCompletedObservabilityLocked(request_id, *completed);
    return RequestStepResult::Terminal;
  }

  const bool tick_executed = RunPolicyBatchTickLocked();

  // Ensure terminal slots (Completed/Failed) are always moved to the
  // request_queue, especially if RunPolicyBatchTickLocked failed early due to
  // slot setup errors.
  slot_scheduler_.FinalizeCompletedSlots(request_queue_, session_store_);
  CommitNewCompletedResponsesObservabilityLocked();

  if (!tick_executed) {
    // If the request we are tracking just finished (possibly failed), return
    // Terminal.
    if (const GenerateResponse *completed =
            request_queue_.PeekCompletedResponse(request_id);
        completed != nullptr) {
      CommitCompletedObservabilityLocked(request_id, *completed);
      return RequestStepResult::Terminal;
    }

    if (target_request->lifecycle == GenerateRequestLifecycle::Pending ||
        target_request->lifecycle == GenerateRequestLifecycle::Admitted) {
      return admitted_any ? RequestStepResult::Progressed
                          : RequestStepResult::Waiting;
    }

    SlotState *active_slot = slot_scheduler_.FindFirstActiveSlot();
    if (active_slot == nullptr) {
      return RequestStepResult::FatalNoProgress;
    }
    if (active_slot->phase != SlotPhase::Failed &&
        active_slot->phase != SlotPhase::Completed) {
      active_slot->terminal_error_message =
          "Shared batch tick could not make progress.";
      active_slot->phase = SlotPhase::Failed;
    }

    // Finalize again in case the cleanup logic above marked a slot as Failed.
    slot_scheduler_.FinalizeCompletedSlots(request_queue_, session_store_);
    CommitNewCompletedResponsesObservabilityLocked();
  }

  if (const GenerateResponse *completed =
          request_queue_.PeekCompletedResponse(request_id);
      completed != nullptr) {
    CommitCompletedObservabilityLocked(request_id, *completed);
    return RequestStepResult::Terminal;
  }

  return (tick_executed || admitted_any) ? RequestStepResult::Progressed
                                         : RequestStepResult::Waiting;
}

std::vector<GenerateRequestId>
InferenceRuntime::DrainCompletedResponseIds(int32_t max_count) {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (max_count < 0) {
    return {};
  }

  std::vector<GenerateRequestId> request_ids =
      request_queue_.DrainCompletedResponseIds(
          static_cast<std::size_t>(max_count));
  std::sort(request_ids.begin(), request_ids.end());
  return request_ids;
}

std::vector<RuntimeEvent>
InferenceRuntime::DrainRuntimeEvents(int32_t max_count,
                                     int32_t max_text_bytes) {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  const std::size_t clamped_max_count =
      max_count <= 0 ? 0 : static_cast<std::size_t>(max_count);
  const std::size_t clamped_max_text_bytes =
      max_text_bytes <= 0 ? 0 : static_cast<std::size_t>(max_text_bytes);
  return request_queue_.DrainRuntimeEvents(clamped_max_count,
                                           clamped_max_text_bytes);
}

bool InferenceRuntime::TryPeekCompletedResponse(
    GenerateRequestId request_id, GenerateResponse &out_response) const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  const GenerateResponse *response =
      request_queue_.PeekCompletedResponse(request_id);
  if (response == nullptr) {
    return false;
  }
  out_response = *response;
  return true;
}

bool InferenceRuntime::ConsumeCompletedResponse(GenerateRequestId request_id) {
  std::lock_guard<std::mutex> lock(operation_mutex_);
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

  const std::size_t split_index = messages.size() - 1;
  std::vector<common_chat_msg> past_messages;
  past_messages.reserve(split_index);
  for (std::size_t index = 0; index < split_index; ++index) {
    past_messages.push_back(messages[index]);
  }

  try {
    return common_chat_format_single(chat_templates_.get(), past_messages,
                                     messages.back(), add_assistant,
                                     /* use_jinja = */ true);
  } catch (const std::exception &error) {
    fprintf(stderr, "%s: warning: failed to apply common chat template: %s\n",
            __func__, error.what());
    return {};
  }
}

} // namespace noumena::cogentengine
