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

#include "runtime/llama/llama_utils.h"

namespace {

constexpr char kDefaultPromptContextKey[] = "__primary_prompt__";
constexpr int kMaxPredictionTokens = 2048;

noumena::cogentengine::InferenceRuntimeConfig
normalize_config(noumena::cogentengine::InferenceRuntimeConfig config) {
  config.n_seq_max = std::max<int32_t>(1, config.n_seq_max);
  config.gpu_layers = std::max<int32_t>(0, config.gpu_layers);
  config.max_cached_sessions = std::max<int32_t>(1, config.max_cached_sessions);
  config.retained_prefix_tokens =
      std::max<int32_t>(0, config.retained_prefix_tokens);
  return config;
}

} // namespace

namespace noumena::cogentengine {

bool InferenceRuntime::EnsureContextSpace(ContextState &state,
                                          int new_tokens_needed, int n_ctx) {
  if (state.ctx == nullptr || n_ctx <= 0) {
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

  llama_memory_t mem = llama_get_memory(state.ctx);

  if (n_discard <= 0) {
    if (!llama_memory_seq_rm(mem, 0, 0, -1)) {
      return false;
    }
    state.current_kv_tokens.clear();
    state.n_past = 0;
    return true;
  }

  if (!llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard)) {
    return false;
  }

  llama_memory_seq_add(mem, 0, n_keep + n_discard, -1, -n_discard);

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

  if (!llama_memory_seq_rm(mem, 0, 0, -1)) {
    return false;
  }
  state.current_kv_tokens.clear();
  state.n_past = 0;

  return true;
}

bool InferenceRuntime::ExecutePromptTokensLocked(
    const std::string &context_key,
    const std::vector<llama_token> &prompt_tokens, int n_tokens_predict,
    TokenCallback on_token_received) {
  if (primary_model_ == nullptr || sampler_ == nullptr) {
    return false;
  }
  if (n_tokens_predict <= 0 || n_tokens_predict > kMaxPredictionTokens) {
    return false;
  }

  std::string model_context_key =
      context_key.empty() ? kDefaultPromptContextKey : context_key;
  if (prompt_tokens.empty()) {
    return true;
  }

  ContextState *state = session_store_.Find(model_context_key);
  if (state == nullptr) {
    session_store_.EnforceLimitBeforeInsert();

    ContextState new_state;
    new_state.ctx = CreateContext();
    if (new_state.ctx == nullptr) {
      return false;
    }
    new_state.n_past = 0;
    state = &session_store_.Emplace(model_context_key, std::move(new_state));
  }

  session_store_.Touch(model_context_key);
  llama_context *ctx = state->ctx;
  if (ctx == nullptr) {
    session_store_.Remove(model_context_key);
    return false;
  }

  const std::vector<llama_token> &new_tokens = prompt_tokens;
  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  llama_memory_t mem = llama_get_memory(ctx);
  const bool is_recurrent = llama_model_is_recurrent(primary_model_);
  const bool is_hybrid = llama_model_is_hybrid(primary_model_);
  const bool allow_partial_kv = !(is_recurrent || is_hybrid);

  size_t match_len = 0;
  const size_t min_len =
      std::min(state->current_kv_tokens.size(), new_tokens.size());
  for (size_t i = 0; i < min_len; ++i) {
    if (state->current_kv_tokens[i] != new_tokens[i]) {
      break;
    }
    match_len++;
  }

  const int n_ctx = llama_n_ctx(ctx);
  const int tokens_to_add = static_cast<int>(new_tokens.size() - match_len);
  const int total_needed = tokens_to_add + n_tokens_predict;

  if (!EnsureContextSpace(*state, total_needed, n_ctx)) {
    return false;
  }

  if (match_len < state->current_kv_tokens.size()) {
    if (!allow_partial_kv) {
      llama_memory_seq_rm(mem, 0, 0, -1);
      state->current_kv_tokens.clear();
      state->n_past = 0;
      match_len = 0;
    } else {
      if (!llama_memory_seq_rm(mem, 0, match_len, -1)) {
        fprintf(stderr, "failed to remove tokens from memory\n");
        return false;
      }
      state->current_kv_tokens.resize(match_len);
      state->n_past = static_cast<int>(match_len);
    }
  }

  llama_perf_context_reset(ctx);
  llama_sampler_reset(sampler_);
  llama_perf_sampler_reset(sampler_);
  const auto total_start = std::chrono::steady_clock::now();

  const int n_batch = static_cast<int>(llama_n_batch(ctx));
  llama_batch batch = llama_batch_init(
      n_batch, 0,
      static_cast<int32_t>(std::max<uint32_t>(1, llama_n_seq_max(ctx))));

  if (match_len == new_tokens.size() && match_len > 0) {
    if (!allow_partial_kv) {
      llama_memory_seq_rm(mem, 0, 0, -1);
      state->current_kv_tokens.clear();
      state->n_past = 0;
      match_len = 0;
    } else {
      if (!llama_memory_seq_rm(mem, 0, match_len - 1, -1)) {
        fprintf(stderr,
                "failed to remove last token from memory for re-evaluation\n");
        llama_batch_free(batch);
        return false;
      }
      state->current_kv_tokens.resize(match_len - 1);
      state->n_past = static_cast<int>(match_len - 1);
      match_len--;
    }
  }

  for (size_t i = match_len; i < new_tokens.size(); ++i) {
    const int batch_pos = static_cast<int>(i);
    const bool logits = (i == new_tokens.size() - 1);

    llama_utils::BatchAdd(batch, new_tokens[i], batch_pos, 0, logits);

    if (batch.n_tokens >= n_batch) {
      if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s : failed to eval prompt\n", __func__);
        llama_batch_free(batch);
        return false;
      }
      state->n_past += batch.n_tokens;
      llama_utils::BatchClear(batch);
    }
  }

  if (batch.n_tokens > 0) {
    if (llama_decode(ctx, batch) != 0) {
      fprintf(stderr, "%s : failed to eval prompt final\n", __func__);
      llama_batch_free(batch);
      return false;
    }
    state->n_past += batch.n_tokens;
  }

  state->current_kv_tokens = new_tokens;

  llama_utils::BatchClear(batch);
  int output_token_count = 0;

  for (int i = 0; i < n_tokens_predict; ++i) {
    const llama_token tok = llama_sampler_sample(sampler_, ctx, -1);

    if (llama_vocab_is_eog(vocab, tok)) {
      break;
    }

    char buf[128];
    const int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    if (n < 0) {
      break;
    }
    output_token_count++;

    if (on_token_received) {
      on_token_received(buf, n);
    }

    llama_utils::BatchClear(batch);
    llama_utils::BatchAdd(batch, tok, state->n_past, 0, true);

    if (llama_decode(ctx, batch) != 0) {
      break;
    }

    state->n_past++;
    state->current_kv_tokens.push_back(tok);
  }

  llama_batch_free(batch);

  const auto total_end = std::chrono::steady_clock::now();
  const auto ctx_perf = llama_perf_context(ctx);
  const auto sampler_perf = llama_perf_sampler(sampler_);

  last_prompt_perf_ = PromptPerfStats{
      .total_ms =
          std::chrono::duration<double, std::milli>(total_end - total_start)
              .count(),
      .prompt_eval_ms = ctx_perf.t_p_eval_ms,
      .decode_eval_ms = ctx_perf.t_eval_ms,
      .sample_ms = sampler_perf.t_sample_ms,
      .input_token_count = static_cast<int32_t>(new_tokens.size()),
      .prompt_eval_tokens = ctx_perf.n_p_eval,
      .decode_eval_count = ctx_perf.n_eval,
      .sample_count = sampler_perf.n_sample,
      .output_token_count = output_token_count,
  };
  has_last_prompt_perf_ = true;

  return true;
}

InferenceRuntime::InferenceRuntime(std::string model_path,
                                   InferenceRuntimeConfig config)
    : config_(normalize_config(config)),
      session_store_(static_cast<size_t>(config_.max_cached_sessions)) {
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

  primary_model_ = llama_model_load_from_file(model_path.c_str(), model_params);
  if (primary_model_ == nullptr) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return;
  }

  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = false;
  sampler_ = llama_sampler_chain_init(sparams);

  llama_sampler_chain_add(sampler_,
                          llama_sampler_init_penalties(64, 1.05f, 0.0f, 0.0f));
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(40));
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(0.8f, 1));
  llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.7f));
  llama_sampler_chain_add(sampler_,
                          llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  slot_scheduler_.SetContextFactory(
      [this]() -> llama_context * { return CreateContext(); });
  slot_scheduler_.Resize(
      static_cast<std::size_t>(std::max<int32_t>(1, config_.n_seq_max)));
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
  ctx_params.no_perf = false;

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

  if (primary_model_ != nullptr) {
    llama_model_free(primary_model_);
  }

  llama_backend_free();
}

bool InferenceRuntime::IsReady() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return primary_model_ != nullptr && sampler_ != nullptr;
}

bool InferenceRuntime::TryGetLastPromptPerf(PromptPerfStats &out) const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (!has_last_prompt_perf_) {
    return false;
  }

  out = last_prompt_perf_;
  return true;
}

GenerateRequestId
InferenceRuntime::EnqueueRequest(std::string context_key, std::string prompt,
                                 int n_tokens_predict,
                                 TokenCallback on_token_received) {
  std::lock_guard<std::mutex> lock(operation_mutex_);

  if (primary_model_ == nullptr || sampler_ == nullptr) {
    return 0;
  }
  if (n_tokens_predict <= 0 || n_tokens_predict > kMaxPredictionTokens) {
    return 0;
  }
  if (context_key.empty()) {
    context_key = kDefaultPromptContextKey;
  }

  GenerateRequest request;
  request.id = next_request_id_++;
  request.context_key = std::move(context_key);
  request.prompt_text = std::move(prompt);
  request.max_output_tokens = n_tokens_predict;
  request.on_token_received = std::move(on_token_received);

  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  request.prompt_tokens =
      llama_utils::Tokenize(vocab, request.prompt_text, false, true);

  if (!request_queue_.Push(std::move(request))) {
    return 0;
  }

  return next_request_id_ - 1;
}

bool InferenceRuntime::RunUntilRequestCompletes(
    GenerateRequestId request_id, GenerateResponse &out_response) {
  std::lock_guard<std::mutex> lock(operation_mutex_);

  out_response = {};
  has_last_prompt_perf_ = false;

  if (request_id == 0 || primary_model_ == nullptr || sampler_ == nullptr) {
    return false;
  }

  while (true) {
    if (auto completed = request_queue_.TakeCompletedResponse(request_id);
        completed.has_value()) {
      out_response = std::move(*completed);
      return out_response.status == GenerateResponseStatus::Completed;
    }

    GenerateRequest *target_request = request_queue_.FindMutable(request_id);
    if (target_request == nullptr) {
      return false;
    }

    slot_scheduler_.Tick(request_queue_, session_store_);

    if (auto completed = request_queue_.TakeCompletedResponse(request_id);
        completed.has_value()) {
      out_response = std::move(*completed);
      return out_response.status == GenerateResponseStatus::Completed;
    }

    SlotState *active_slot = slot_scheduler_.FindFirstActiveSlot();
    if (active_slot == nullptr) {
      return false;
    }

    GenerateRequest *active_request = active_slot->request;
    if (active_request == nullptr) {
      active_slot->phase = SlotPhase::Failed;
      active_slot->terminal_error_message = "Active slot lost request state.";
      slot_scheduler_.FinalizeCompletedSlots(request_queue_);
      continue;
    }

    const bool success = ExecutePromptTokensLocked(
        active_request->context_key, active_request->prompt_tokens,
        active_request->max_output_tokens,
        [this, active_slot](const char *token_piece, int32_t token_length) {
          if (token_piece == nullptr || token_length <= 0) {
            return;
          }
          active_slot->buffered_output_text.append(
              token_piece, static_cast<std::size_t>(token_length));
          slot_scheduler_.EmitBufferedTokenPiece(*active_slot);
        });

    if (success) {
      active_request->lifecycle = GenerateRequestLifecycle::Completed;
      active_slot->generated_tokens.resize(static_cast<std::size_t>(
          std::max(0, last_prompt_perf_.output_token_count)));
      active_slot->phase = SlotPhase::Completed;
    } else {
      active_request->lifecycle = GenerateRequestLifecycle::Failed;
      active_slot->terminal_error_message = "Queued request execution failed.";
      active_slot->phase = SlotPhase::Failed;
    }

    slot_scheduler_.FinalizeCompletedSlots(request_queue_);
  }
}

bool InferenceRuntime::Prompt(std::string model_context_key, std::string prompt,
                              int n_tokens_predict,
                              TokenCallback on_token_received) {
  // Phase 2 note:
  // - This is still the live Phase 1 execution path.
  // - Once request queue ownership is implemented, keep Prompt(...) as the
  //   synchronous wrapper that enqueues a request and waits for completion.
  std::lock_guard<std::mutex> lock(operation_mutex_);
  has_last_prompt_perf_ = false;

  if (primary_model_ == nullptr || sampler_ == nullptr) {
    return false;
  }
  if (n_tokens_predict <= 0 || n_tokens_predict > kMaxPredictionTokens) {
    return false;
  }
  if (model_context_key.empty()) {
    model_context_key = kDefaultPromptContextKey;
  }

  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  std::vector<llama_token> new_tokens =
      llama_utils::Tokenize(vocab, prompt, false, true);
  return ExecutePromptTokensLocked(model_context_key, new_tokens,
                                   n_tokens_predict, on_token_received);
}

} // namespace noumena::cogentengine
