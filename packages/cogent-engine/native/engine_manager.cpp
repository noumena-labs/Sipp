/////////////////////////////////////////////////////////////////////////////////////////////////
//
// engine_manager.cpp
//
// - Inference-only runtime over llama.cpp.
// - Owns model lifetime, context reuse, and text generation.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "engine_manager.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <thread>

namespace {

constexpr int kDefaultRetainedPromptTokens = 100;
constexpr char kDefaultPromptContextKey[] = "__primary_prompt__";
constexpr int kMaxPredictionTokens = 2048;

std::vector<llama_token> common_tokenize(
    const struct llama_vocab* vocab,
    const std::string& text,
    bool add_special,
    bool parse_special) {
  int n_tokens = static_cast<int>(text.length()) + 2 * static_cast<int>(add_special);
  std::vector<llama_token> result(n_tokens);
  n_tokens = llama_tokenize(
      vocab,
      text.data(),
      text.length(),
      result.data(),
      result.size(),
      add_special,
      parse_special);
  if (n_tokens < 0) {
    result.resize(-n_tokens);
    const int check = llama_tokenize(
        vocab,
        text.data(),
        text.length(),
        result.data(),
        result.size(),
        add_special,
        parse_special);
    GGML_ASSERT(check == -n_tokens);
  } else {
    result.resize(n_tokens);
  }
  return result;
}

void common_batch_clear(struct llama_batch& batch) {
  batch.n_tokens = 0;
}

void common_batch_add(
    struct llama_batch& batch,
    llama_token id,
    llama_pos pos,
    const std::vector<llama_seq_id>& seq_ids,
    bool logits) {
  GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

  batch.token[batch.n_tokens] = id;
  batch.pos[batch.n_tokens] = pos;
  batch.n_seq_id[batch.n_tokens] = seq_ids.size();
  for (size_t i = 0; i < seq_ids.size(); ++i) {
    batch.seq_id[batch.n_tokens][i] = seq_ids[i];
  }
  batch.logits[batch.n_tokens] = logits;
  batch.n_tokens++;
}

void log_callback_default(enum ggml_log_level level, const char* text, void* user_data) {
  (void) text;
  (void) level;
  (void) user_data;
}

int default_thread_count() {
#if defined(__EMSCRIPTEN__)
  #if defined(GGML_PTHREADS) && GGML_PTHREADS
    #if defined(CE_WASM_PTHREAD_POOL_SIZE)
      return std::clamp(CE_WASM_PTHREAD_POOL_SIZE, 1, 8);
    #else
      return 2;
    #endif
  #else
    return 1;
  #endif
#else
  return std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, 8);
#endif
}

}  // namespace

bool noumena::cogentengine::CogentEngineManager::EnsureContextSpace(
    ContextState& state,
    int new_tokens_needed,
    int n_ctx) {
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

  const int n_keep = std::min(kDefaultRetainedPromptTokens, state.n_past);
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
    const int erase_end = std::min<int>(n_keep + n_discard, state.current_kv_tokens.size());
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

noumena::cogentengine::CogentEngineManager::CogentEngineManager(
    std::string model_path,
    int gpu_layers_n) {
  if (model_path.empty()) {
    fprintf(stderr, "%s: error: model path is required\n", __func__);
    return;
  }

#if defined(NDEBUG) || defined(CE_SUPPRESS_LLAMA_LOGS)
  llama_log_set(log_callback_default, nullptr);
#endif

  ggml_backend_load_all();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = gpu_layers_n;
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

  llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(64, 1.05f, 0.0f, 0.0f));
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(40));
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(0.8f, 1));
  llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.7f));
  llama_sampler_chain_add(sampler_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

noumena::cogentengine::CogentEngineManager::~CogentEngineManager() {
  if (sampler_ != nullptr) {
    llama_sampler_free(sampler_);
  }

  for (auto const& [key, state] : context_states_) {
    (void) key;
    llama_free(state.ctx);
  }

  if (primary_model_ != nullptr) {
    llama_model_free(primary_model_);
  }

  llama_backend_free();
}

bool noumena::cogentengine::CogentEngineManager::IsReady() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return primary_model_ != nullptr && sampler_ != nullptr;
}

bool noumena::cogentengine::CogentEngineManager::TryGetLastPromptPerf(
    PromptPerfStats& out) const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (!has_last_prompt_perf_) {
    return false;
  }

  out = last_prompt_perf_;
  return true;
}

void noumena::cogentengine::CogentEngineManager::TouchContextKey(
    const std::string& context_key) {
  auto it = std::find(context_usage_order_.begin(), context_usage_order_.end(), context_key);
  if (it != context_usage_order_.end()) {
    context_usage_order_.erase(it);
  }
  context_usage_order_.push_back(context_key);
}

void noumena::cogentengine::CogentEngineManager::ReleaseContextState(
    const std::string& context_key) {
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

void noumena::cogentengine::CogentEngineManager::EnforceContextLimit() {
  while (context_states_.size() >= kMaxCachedContexts && !context_usage_order_.empty()) {
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

std::string noumena::cogentengine::CogentEngineManager::Prompt(
    std::string model_context_key,
    std::string prompt,
    int n_tokens_predict,
    std::function<void(std::string)> onTokenReceived) {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  has_last_prompt_perf_ = false;

  if (primary_model_ == nullptr || sampler_ == nullptr) {
    return "";
  }
  if (n_tokens_predict <= 0 || n_tokens_predict > kMaxPredictionTokens) {
    return "";
  }
  if (model_context_key.empty()) {
    model_context_key = kDefaultPromptContextKey;
  }

  const llama_vocab* vocab = llama_model_get_vocab(primary_model_);

  std::string normalized_prompt = prompt;
  size_t pos = 0;
  while ((pos = normalized_prompt.find("\r\n", pos)) != std::string::npos) {
    normalized_prompt.replace(pos, 2, "\n");
    pos += 1;
  }
  while ((pos = normalized_prompt.find('\r')) != std::string::npos) {
    normalized_prompt.replace(pos, 1, "\n");
  }

  auto ltrim = [](std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
      ++start;
    }
    if (start > 0) {
      value.erase(0, start);
    }
  };

  std::string formatted_prompt = normalized_prompt;
  ltrim(formatted_prompt);

  const bool is_chat_formatted = formatted_prompt.starts_with("<|im_start|>") ||
                                 formatted_prompt.starts_with("<|startoftext|>") ||
                                 formatted_prompt.starts_with("<|begin_of_text|>");

  if (!is_chat_formatted) {
    formatted_prompt = "<|im_start|>user\n" + formatted_prompt +
                       "\n<|im_end|>\n<|im_start|>assistant\n";
  }
  std::vector<llama_token> new_tokens = common_tokenize(vocab, formatted_prompt, false, true);

  auto context_it = context_states_.find(model_context_key);
  if (context_it == context_states_.end()) {
    EnforceContextLimit();

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = std::min(4096 * 2, llama_model_n_ctx_train(primary_model_));
    ctx_params.n_batch = 256;
    ctx_params.no_perf = false;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = default_thread_count();

    ContextState state;
    state.ctx = llama_init_from_model(primary_model_, ctx_params);
    if (state.ctx == nullptr) {
      return "";
    }
    state.n_past = 0;
    context_it = context_states_.emplace(model_context_key, std::move(state)).first;
  }

  TouchContextKey(model_context_key);
  ContextState& state = context_it->second;
  llama_context* ctx = state.ctx;
  if (ctx == nullptr) {
    ReleaseContextState(model_context_key);
    return "";
  }

  llama_memory_t mem = llama_get_memory(ctx);
  const bool is_recurrent = llama_model_is_recurrent(primary_model_);
  const bool is_hybrid = llama_model_is_hybrid(primary_model_);
  const bool allow_partial_kv = !(is_recurrent || is_hybrid);

  size_t match_len = 0;
  const size_t min_len = std::min(state.current_kv_tokens.size(), new_tokens.size());
  for (size_t i = 0; i < min_len; ++i) {
    if (state.current_kv_tokens[i] != new_tokens[i]) {
      break;
    }
    match_len++;
  }

  const int n_ctx = llama_n_ctx(ctx);
  const int tokens_to_add = static_cast<int>(new_tokens.size() - match_len);
  const int total_needed = tokens_to_add + n_tokens_predict;

  if (!EnsureContextSpace(state, total_needed, n_ctx)) {
    return "Error: Context full and input too large to shift.";
  }

  if (match_len < state.current_kv_tokens.size()) {
    if (!allow_partial_kv) {
      llama_memory_seq_rm(mem, 0, 0, -1);
      state.current_kv_tokens.clear();
      state.n_past = 0;
      match_len = 0;
    } else {
      if (!llama_memory_seq_rm(mem, 0, match_len, -1)) {
        fprintf(stderr, "failed to remove tokens from memory\n");
        return "";
      }
      state.current_kv_tokens.resize(match_len);
      state.n_past = static_cast<int>(match_len);
    }
  }

  llama_perf_context_reset(ctx);
  llama_sampler_reset(sampler_);
  llama_perf_sampler_reset(sampler_);
  const auto total_start = std::chrono::steady_clock::now();

  const int n_batch = llama_n_batch(ctx);
  llama_batch batch = llama_batch_init(n_batch, 0, 1);

  if (match_len == new_tokens.size() && match_len > 0) {
    if (!allow_partial_kv) {
      llama_memory_seq_rm(mem, 0, 0, -1);
      state.current_kv_tokens.clear();
      state.n_past = 0;
      match_len = 0;
    } else {
      if (!llama_memory_seq_rm(mem, 0, match_len - 1, -1)) {
        fprintf(stderr, "failed to remove last token from memory for re-evaluation\n");
        llama_batch_free(batch);
        return "";
      }
      state.current_kv_tokens.resize(match_len - 1);
      state.n_past = static_cast<int>(match_len - 1);
      match_len--;
    }
  }

  for (size_t i = match_len; i < new_tokens.size(); ++i) {
    const int batch_pos = static_cast<int>(i);
    const bool logits = (i == new_tokens.size() - 1);

    common_batch_add(batch, new_tokens[i], batch_pos, {0}, logits);

    if (batch.n_tokens >= n_batch) {
      if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s : failed to eval prompt\n", __func__);
        llama_batch_free(batch);
        return "";
      }
      state.n_past += batch.n_tokens;
      common_batch_clear(batch);
    }
  }

  if (batch.n_tokens > 0) {
    if (llama_decode(ctx, batch) != 0) {
      fprintf(stderr, "%s : failed to eval prompt final\n", __func__);
      llama_batch_free(batch);
      return "";
    }
    state.n_past += batch.n_tokens;
  }

  state.current_kv_tokens = new_tokens;

  std::string response;
  response.reserve(n_tokens_predict * 4);
  common_batch_clear(batch);
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
    response.append(buf, n);
    output_token_count++;

    if (onTokenReceived) {
      onTokenReceived(std::string(buf, n));
    }

    common_batch_clear(batch);
    common_batch_add(batch, tok, state.n_past, {0}, true);

    if (llama_decode(ctx, batch) != 0) {
      break;
    }

    state.n_past++;
    state.current_kv_tokens.push_back(tok);
  }

  llama_batch_free(batch);

  const auto total_end = std::chrono::steady_clock::now();
  const auto ctx_perf = llama_perf_context(ctx);
  const auto sampler_perf = llama_perf_sampler(sampler_);

  last_prompt_perf_ = PromptPerfStats{
      .total_ms =
          std::chrono::duration<double, std::milli>(total_end - total_start).count(),
      .prompt_eval_ms = ctx_perf.t_p_eval_ms,
      .decode_eval_ms = ctx_perf.t_eval_ms,
      .sample_ms = sampler_perf.t_sample_ms,
      .prompt_eval_tokens = ctx_perf.n_p_eval,
      .decode_eval_count = ctx_perf.n_eval,
      .sample_count = sampler_perf.n_sample,
      .output_token_count = output_token_count,
  };
  has_last_prompt_perf_ = true;

  return response;
}
