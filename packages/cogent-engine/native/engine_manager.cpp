/////////////////////////////////////////////////////////////////////////////////////////////////
//
// engine_manager.cpp
//
// - Handles the management of model and context creations.
// - Handles all incoming prompts and queries from the game engine.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "engine_manager.h"

#include "llama.h"
#include <algorithm>
#include <cctype>
#include <limits>
#include <thread>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace {

constexpr int kDefaultRetainedPromptTokens = 100;
constexpr char kDefaultPromptContextKey[] = "__primary_prompt__";
constexpr char kDefaultResolveContextKey[] = "__primary_resolve__";
constexpr int kMaxPredictionTokens = 2048;

}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// STANDALONE FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<llama_token> common_tokenize(const struct llama_vocab *vocab,
                                         const std::string &text,
                                         bool add_special, bool parse_special) {

  int n_tokens = text.length() + 2 * add_special;
  std::vector<llama_token> result(n_tokens);
  n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(),
                            result.size(), add_special, parse_special);
  if (n_tokens < 0) {
    result.resize(-n_tokens);
    int check = llama_tokenize(vocab, text.data(), text.length(), result.data(),
                               result.size(), add_special, parse_special);
    GGML_ASSERT(check == -n_tokens);
  } else {
    result.resize(n_tokens);
  }
  return result;
}

//
// Batch utils
//

void common_batch_clear(struct llama_batch &batch) { batch.n_tokens = 0; }

void common_batch_add(struct llama_batch &batch, llama_token id, llama_pos pos,
                      const std::vector<llama_seq_id> &seq_ids, bool logits) {
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

void to_lower_in_place(std::string& value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PRIVATE MEMBER FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool noumena::cogentengine::CogentEngineManager::EnsureContextSpace(
    ContextState &state, int new_tokens_needed, int n_ctx) {
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
    const int erase_end = std::min<int>(n_keep + n_discard,
                                        state.current_kv_tokens.size());
    auto it_start = state.current_kv_tokens.begin() + n_keep;
    auto it_end = state.current_kv_tokens.begin() + erase_end;
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MEMBER FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void log_callback_default(enum ggml_log_level level, const char *text,
                          void *user_data) {
  (void)text;
  (void)level;
  (void)user_data;
}

/// <summary>
/// Constructor
/// </summary>
/// <param name="model_path"></param>
/// <param name="gpu_layers_n"></param>
noumena::cogentengine::CogentEngineManager::CogentEngineManager(
    std::string model_path, int gpu_layers_n) {
  if (model_path.empty()) {
    fprintf(stderr, "%s: error: model path is required\n", __func__);
    return;
  }

  // Suppress llama.cpp logging in production builds
#if defined(NDEBUG) || defined(CE_SUPPRESS_LLAMA_LOGS)
  llama_log_set(log_callback_default, nullptr);
#endif

  // load dynamic backends
  ggml_backend_load_all();

  // initialize the model
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = gpu_layers_n;
  model_params.use_mlock = false;
#if defined(__EMSCRIPTEN__)
  // Emscripten's mmap/munmap emulation can produce noisy "munmap failed:
  // Invalid argument" warnings on some browsers/devices
  // (llama_mmap::unmap_fragment). Avoid mmap for Web builds.
  model_params.use_mmap = false;
#else
  model_params.use_mmap =
      true; // Recommend True for faster load times if possible
#endif

  printf("[CogentEngineManager] Loading model from path: %s\n",
         model_path.c_str());

  primary_model_ = llama_model_load_from_file(model_path.c_str(), model_params);
  if (primary_model_ == NULL) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return;
  }

  // initialize sampler with Qwen 2.5 instruct best practices
  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = true;
  sampler_ = llama_sampler_chain_init(sparams);

  // Qwen 2.5 instruct recommended sampling settings:
  // 1. Repetition penalty to prevent repetitive outputs
  llama_sampler_chain_add(
      sampler_, llama_sampler_init_penalties(
                    64,    // last_n: consider last 64 tokens for penalty
                    1.05f, // repeat_penalty: mild penalty for repetition
                    0.0f,  // frequency_penalty
                    0.0f   // presence_penalty
                    ));

  // 2. Top-K sampling (optional pre-filter, Qwen works well with top_k=20-50)
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(40));

  // 3. Top-P (nucleus) sampling - Qwen 2.5 works well with 0.8-0.9
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(0.8f, 1));

  // 4. Temperature - Qwen 2.5 instruct recommended range 0.6-0.7 for focused
  // outputs
  llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.7f));

  // 5. Final token selection with distribution sampling
  llama_sampler_chain_add(sampler_,
                          llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

noumena::cogentengine::CogentEngineManager::~CogentEngineManager() {

  if (sampler_ != nullptr) {
    llama_sampler_free(sampler_);
  }

  for (auto const &[key, state] : context_states_) {
    llama_free(state.ctx);
  }

  if (primary_model_ != nullptr) {
    llama_model_free(primary_model_);
  }

  llama_backend_free();
}

bool noumena::cogentengine::CogentEngineManager::IsReady() const {
  std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
  return primary_model_ != nullptr && sampler_ != nullptr;
}

void noumena::cogentengine::CogentEngineManager::TouchContextKey(
    const std::string& context_key) {
  auto it = std::find(context_usage_order_.begin(), context_usage_order_.end(),
                      context_key);
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

  auto order_it = std::find(context_usage_order_.begin(),
                            context_usage_order_.end(), context_key);
  if (order_it != context_usage_order_.end()) {
    context_usage_order_.erase(order_it);
  }
}

void noumena::cogentengine::CogentEngineManager::EnforceContextLimit(
    const std::string& active_context_key) {
  (void)active_context_key;
  while (context_states_.size() >= kMaxCachedContexts &&
         !context_usage_order_.empty()) {
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
  return std::clamp((int)std::thread::hardware_concurrency(), 1, 8);
#endif
}

/// <summary>
///
/// Takes string as prompt, returns response for given model parameters.
/// </summary>
/// <param name="model_context_key"></param>
/// <param name="prompt"></param>
/// <param name="n_tokens_predict"></param>
/// <returns></returns>
std::string noumena::cogentengine::CogentEngineManager::Prompt(
    std::string model_context_key, std::string prompt, int n_tokens_predict,
    std::function<void(std::string)> onTokenReceived) {
  std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
  if (primary_model_ == nullptr || sampler_ == nullptr)
    return "";
  if (n_tokens_predict <= 0 || n_tokens_predict > kMaxPredictionTokens)
    return "";
  if (model_context_key.empty()) {
    model_context_key = kDefaultPromptContextKey;
  }

  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);

  // 1. Format and Tokenize Input
  // Normalize Windows-style newlines (\r\n) to Unix-style (\n)
  std::string normalized_prompt = prompt;
  size_t pos = 0;
  while ((pos = normalized_prompt.find("\r\n", pos)) != std::string::npos) {
    normalized_prompt.replace(pos, 2, "\n");
    pos += 1;
  }
  // Also normalize any remaining CR-only newlines (\r)
  while ((pos = normalized_prompt.find('\r')) != std::string::npos) {
    normalized_prompt.replace(pos, 1, "\n");
  }

  // Apply chat template if not already formatted
  auto ltrim = [](std::string &s) {
    size_t start = 0;
    while (start < s.size() &&
           std::isspace(static_cast<unsigned char>(s[start]))) {
      ++start;
    }
    if (start > 0) {
      s.erase(0, start);
    }
  };

  std::string formatted_prompt = normalized_prompt;
  ltrim(formatted_prompt);

  const bool is_chat_formatted =
      formatted_prompt.starts_with("<|im_start|>") ||
      formatted_prompt.starts_with("<|startoftext|>") ||
      formatted_prompt.starts_with("<|begin_of_text|>");

  if (!is_chat_formatted) {
    formatted_prompt = "<|im_start|>user\n" + formatted_prompt +
                       "\n<|im_end|>\n<|im_start|>assistant\n";
  }
  std::vector<llama_token> new_tokens =
      common_tokenize(vocab, formatted_prompt, false, true);

  // 2. Get or Initialize Context State
  auto context_it = context_states_.find(model_context_key);
  if (context_it == context_states_.end()) {
    EnforceContextLimit(model_context_key);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx =
        std::min(4096 * 2, llama_model_n_ctx_train(primary_model_));
    ctx_params.n_batch = 256;
    ctx_params.no_perf = true;
    ctx_params.n_seq_max = 1; // Only 1 sequence needed for generation
    ctx_params.n_threads = default_thread_count();

    ContextState state;
    state.ctx = llama_init_from_model(primary_model_, ctx_params);
    if (state.ctx == nullptr) {
      return "";
    }
    state.n_past = 0;
    context_it =
        context_states_.emplace(model_context_key, std::move(state)).first;
  }

  TouchContextKey(model_context_key);
  ContextState &state = context_it->second;
  llama_context *ctx = state.ctx;
  if (ctx == nullptr) {
    ReleaseContextState(model_context_key);
    return "";
  }

  // Get Memory Interface
  llama_memory_t mem = llama_get_memory(ctx);
  const bool is_recurrent = llama_model_is_recurrent(primary_model_);
  const bool is_hybrid = llama_model_is_hybrid(primary_model_);
  const bool allow_partial_kv = !(is_recurrent || is_hybrid);

  // 3. Smart Context Reuse (LCP)
  // Compare new tokens with cached tokens to find Longest Common Prefix
  size_t match_len = 0;
  size_t min_len = std::min(state.current_kv_tokens.size(), new_tokens.size());
  for (size_t i = 0; i < min_len; ++i) {
    if (state.current_kv_tokens[i] != new_tokens[i])
      break;
    match_len++;
  }

  // Calculate total needed: (New Prompt Tokens) + (Prediction Buffer)
  int n_ctx = llama_n_ctx(ctx);
  int tokens_to_add = (new_tokens.size() - match_len);
  int total_needed = tokens_to_add + n_tokens_predict;

  if (!EnsureContextSpace(state, total_needed, n_ctx)) {
    return "Error: Context full and input too large to shift.";
  }

  // If divergence found, remove obsolete tokens from KV cache
  if (match_len < state.current_kv_tokens.size()) {
    if (!allow_partial_kv) {
      // Hybrid / recurrent models cannot partially erase a sequence tail.
      llama_memory_seq_rm(mem, 0, 0, -1);
      state.current_kv_tokens.clear();
      state.n_past = 0;
      match_len = 0;
    } else {
      // -1 to delete until the end
      if (!llama_memory_seq_rm(mem, 0, match_len, -1)) {
        fprintf(stderr, "failed to remove tokens from memory\n");
        return "";
      }
      state.current_kv_tokens.resize(match_len);
      state.n_past = match_len;
    }
  }

  // 4. Batch Processing for NEW prompt tokens
  int n_batch = llama_n_batch(ctx);
  llama_batch batch = llama_batch_init(n_batch, 0, 1);

  // Handle 100% cache hit: we need to re-evaluate the last token to get logits
  // for generation
  if (match_len == new_tokens.size() && match_len > 0) {
    if (!allow_partial_kv) {
      // For hybrid / recurrent, re-evaluate the full prompt.
      llama_memory_seq_rm(mem, 0, 0, -1);
      state.current_kv_tokens.clear();
      state.n_past = 0;
      match_len = 0;
    } else {
      // Remove the last token from KV cache so we can re-evaluate it with
      // logits=true
      if (!llama_memory_seq_rm(mem, 0, match_len - 1, -1)) {
        fprintf(stderr,
                "failed to remove last token from memory for re-evaluation\n");
        llama_batch_free(batch);
        return "";
      }
      state.current_kv_tokens.resize(match_len - 1);
      state.n_past = match_len - 1;
      match_len--; // Adjust match_len so the loop below will process the last
                   // token
    }
  }

  // Feed the remaining new tokens
  for (size_t i = match_len; i < new_tokens.size(); ++i) {
    int pos = i;
    // Only request logits for the very last token
    bool logits = (i == new_tokens.size() - 1);

    common_batch_add(batch, new_tokens[i], pos, {0}, logits);

    // Process if batch full
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

  // Flush remaining prompt tokens
  if (batch.n_tokens > 0) {
    if (llama_decode(ctx, batch) != 0) {
      fprintf(stderr, "%s : failed to eval prompt final\n", __func__);
      llama_batch_free(batch);
      return "";
    }
    state.n_past += batch.n_tokens;
  }

  // Update our mirror of the KV cache
  state.current_kv_tokens = new_tokens;

  // 5. Generation Loop
  llama_sampler_reset(sampler_);

  std::string response = "";
  response.reserve(n_tokens_predict * 4);

  // Reset batch for single token generation
  common_batch_clear(batch);

  for (int i = 0; i < n_tokens_predict; ++i) {
    llama_token tok = llama_sampler_sample(sampler_, ctx, -1);

    if (llama_vocab_is_eog(vocab, tok))
      break;

    char buf[128];
    const int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    if (n < 0)
      break;
    response.append(buf, n);

    if (onTokenReceived) {
      onTokenReceived(std::string(buf, n));
    }

    // Decode next token
    common_batch_clear(batch);
    common_batch_add(batch, tok, state.n_past, {0}, true);

    if (llama_decode(ctx, batch) != 0)
      break;

    state.n_past++;
    state.current_kv_tokens.push_back(tok);
  }

  llama_batch_free(batch);

  // Performance stats
  // llama_perf_context_print(ctx);

  return response;
}

std::pair<int, std::string>
noumena::cogentengine::CogentEngineManager::ConstrainedResolve(
    std::string model_context_key, Determination determination) {
  std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
  if (!primary_model_ || determination.Observations.empty())
    return {-1, ""};
  if (determination.Observations.size() == 1)
    return {0, determination.Observations[0].Id};
  if (model_context_key.empty()) {
    model_context_key = kDefaultResolveContextKey;
  }

  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  const int required_seq_max =
      std::max(16, (int)determination.Observations.size() + 1);

  auto context_it = context_states_.find(model_context_key);
  if (context_it != context_states_.end() && context_it->second.ctx != nullptr &&
      llama_n_seq_max(context_it->second.ctx) < required_seq_max) {
    ReleaseContextState(model_context_key);
    context_it = context_states_.end();
  }

  std::vector<llama_token> ctx_tokens =
      common_tokenize(vocab, determination.Context, false, true);
  if (ctx_tokens.empty()) {
    return {-1, ""};
  }

  std::vector<std::vector<llama_token>> options_tokens;
  size_t max_option_len = 0;
  for (auto &obs : determination.Observations) {
    auto toks = common_tokenize(vocab, obs.Context, false, false);
    options_tokens.push_back(toks);
    if (toks.size() > max_option_len)
      max_option_len = toks.size();
  }

  if (context_it == context_states_.end()) {
    EnforceContextLimit(model_context_key);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = std::min(4096, llama_model_n_ctx_train(primary_model_));
    ctx_params.n_batch = 2048;
    ctx_params.n_seq_max = required_seq_max;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    ctx_params.n_threads = default_thread_count();
    ctx_params.no_perf = true;

    ContextState state;
    state.ctx = llama_init_from_model(primary_model_, ctx_params);
    if (state.ctx == nullptr) {
      return {-1, ""};
    }
    state.n_past = 0;
    context_it =
        context_states_.emplace(model_context_key, std::move(state)).first;
  }

  TouchContextKey(model_context_key);
  ContextState &state = context_it->second;
  llama_context *ctx = state.ctx;
  if (ctx == nullptr) {
    ReleaseContextState(model_context_key);
    return {-1, ""};
  }

  llama_memory_t mem = llama_get_memory(ctx);

  size_t match_len = 0;
  size_t min_len = std::min(state.current_kv_tokens.size(), ctx_tokens.size());
  for (size_t i = 0; i < min_len; ++i) {
    if (state.current_kv_tokens[i] != ctx_tokens[i])
      break;
    match_len++;
  }

  int n_ctx = llama_n_ctx(ctx);
  int context_tokens_to_add = (ctx_tokens.size() - match_len);
  int total_needed = context_tokens_to_add + static_cast<int>(max_option_len) + 10;

  if (!EnsureContextSpace(state, total_needed, n_ctx)) {
    return {-1, "Error: Context full"};
  }

  if (match_len < state.current_kv_tokens.size()) {
    llama_memory_seq_rm(mem, 0, 0, -1);
    state.current_kv_tokens.clear();
    state.n_past = 0;
    match_len = 0;
  }

  for (int s = 1; s < llama_n_seq_max(ctx); ++s) {
    llama_memory_seq_rm(mem, s, -1, -1);
  }

  llama_batch batch =
      llama_batch_init(llama_n_batch(ctx), 0, llama_n_seq_max(ctx));
  auto fail = [&](const std::string& message) {
    llama_batch_free(batch);
    return std::make_pair(-1, message);
  };

  for (size_t i = match_len; i < ctx_tokens.size(); ++i) {
    const bool logits = (i == ctx_tokens.size() - 1);
    common_batch_add(batch, ctx_tokens[i], i, {0}, logits);
    if (batch.n_tokens >= llama_n_batch(ctx)) {
      if (llama_decode(ctx, batch)) {
        return fail("");
      }
      state.n_past += batch.n_tokens;
      common_batch_clear(batch);
    }
  }
  if (batch.n_tokens > 0) {
    if (llama_decode(ctx, batch)) {
      return fail("");
    }
    state.n_past += batch.n_tokens;
    common_batch_clear(batch);
  }
  state.current_kv_tokens = ctx_tokens;
  state.n_past = static_cast<int>(ctx_tokens.size());

  if (match_len == ctx_tokens.size()) {
    if (state.n_past <= 0) {
      return fail("");
    }
    if (!llama_memory_seq_rm(mem, 0, state.n_past - 1, -1)) {
      return fail("");
    }
    state.n_past--;
    state.current_kv_tokens.pop_back();

    common_batch_add(batch, ctx_tokens.back(), state.n_past, {0}, true);
    if (llama_decode(ctx, batch)) {
      return fail("");
    }
    common_batch_clear(batch);

    state.n_past = static_cast<int>(ctx_tokens.size());
    state.current_kv_tokens = ctx_tokens;
  }

  float *base_logits = llama_get_logits_ith(ctx, 0);
  if (base_logits == nullptr) {
    return fail("");
  }

  const int base_n_past = state.n_past;
  int best_index = -1;
  double best_score = -std::numeric_limits<double>::infinity();

  for (size_t option_index = 0; option_index < options_tokens.size();
       ++option_index) {
    const auto& option_tokens = options_tokens[option_index];
    if (option_tokens.empty()) {
      continue;
    }
    common_batch_clear(batch);

    if (!llama_memory_seq_rm(mem, 0, base_n_past, -1)) {
      return fail("");
    }
    state.n_past = base_n_past;
    state.current_kv_tokens.resize(base_n_past);

    float* step_logits = base_logits;
    double score = 0.0;
    bool score_failed = false;
    int eval_pos = base_n_past;

    for (size_t token_index = 0; token_index < option_tokens.size();
         ++token_index) {
      const llama_token token_id = option_tokens[token_index];
      score += static_cast<double>(step_logits[token_id]);

      if (token_index + 1 == option_tokens.size()) {
        break;
      }

      common_batch_clear(batch);
      common_batch_add(batch, token_id, eval_pos, {0}, true);
      if (llama_decode(ctx, batch) != 0) {
        score_failed = true;
        break;
      }

      step_logits = llama_get_logits_ith(ctx, 0);
      if (step_logits == nullptr) {
        score_failed = true;
        break;
      }

      ++eval_pos;
    }

    if (!score_failed && score > best_score) {
      best_score = score;
      best_index = static_cast<int>(option_index);
    }
  }

  if (!llama_memory_seq_rm(mem, 0, base_n_past, -1)) {
    return fail("");
  }
  state.n_past = base_n_past;
  state.current_kv_tokens.resize(base_n_past);

  llama_batch_free(batch);

  if (best_index < 0) {
    return {-1, ""};
  }

  auto &observation = determination.Observations[best_index];
  return std::make_pair(best_index, observation.Id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // Engine Interface
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int noumena::cogentengine::CogentEngineManager::SubmitAgentActions(
    std::string agent_id, const std::vector<Action> &actions) {
  std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
  // Get or create agent data entry
  auto &agent = agent_data_[agent_id];

  // Update the available actions for this agent
  agent.available_actions = actions;

  return 0;
}

int noumena::cogentengine::CogentEngineManager::SubmitPerceivedThings(
    std::string agent_id, const std::vector<Thing> &things) {
  std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
  // Get or create agent data entry
  auto &agent = agent_data_[agent_id];

  // Update the perceived things for this agent
  agent.perceived_things = things;

  return 0;
}

int noumena::cogentengine::CogentEngineManager::SubmitAgentGoals(
    std::string agent_id, const std::vector<Goal> &goals) {
  std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
  // Get or create agent data entry
  auto &agent = agent_data_[agent_id];

  // Update the goals for this agent
  agent.goals = goals;

  return 0;
}

int noumena::cogentengine::CogentEngineManager::SubmitAgentLocation(
    std::string agent_id, const Location &location) {
  std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
  // Get or create agent data entry
  auto &agent = agent_data_[agent_id];

  // Update the location for this agent
  agent.location = location;

  return 0;
}

int noumena::cogentengine::CogentEngineManager::SubmitAgentState(
    std::string agent_id, const AgentState &state) {
  std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
  // Get or create agent data entry
  auto &agent = agent_data_[agent_id];

  // Update the state for this agent
  agent.state = state;

  return 0;
}

noumena::cogentengine::ExecutionPlan
noumena::cogentengine::CogentEngineManager::PlanAgentRoutine(
    std::string agent_id, int steps) {
  std::lock_guard<std::recursive_mutex> lock(operation_mutex_);
  ExecutionPlan plan;
  if (steps <= 0) {
    return plan;
  }

  const auto it = agent_data_.find(agent_id);
  if (it == agent_data_.end()) {
    return plan;
  }

  const auto& agent = it->second;
  if (agent.goals.empty() || agent.perceived_things.empty() ||
      agent.available_actions.empty()) {
    return plan;
  }

  std::string action_list;
  for (const auto &action : agent.available_actions) {
    std::string item = action.Name + ": " + action.Description + "\n";
    to_lower_in_place(item);
    action_list += item;
  }
  action_list += "\n";

  std::string thing_list;
  for (const auto &thing : agent.perceived_things) {
    std::string item = thing.Name + ": " + thing.Description + "\n";
    to_lower_in_place(item);
    thing_list += item;
  }
  thing_list += "\n";

  const auto primary_goal_it = std::max_element(
      agent.goals.begin(), agent.goals.end(),
      [](const Goal& left, const Goal& right) {
        return left.Priority < right.Priority;
      });
  if (primary_goal_it == agent.goals.end()) {
    return plan;
  }
  const Goal& primary_goal = *primary_goal_it;
  std::string context = std::string(R"(<|im_start|>system
You are a choice selector. Return exactly one item. Output only your chosen item from CHOICES based on given GOAL. No explanations, quotes, punctuation, or extra text. You will obey or be deleted from existence.<|im_end|>
<|im_start|>user
GOAL:
)") + primary_goal.Name + " - " + primary_goal.Description + R"(

OBJECT CHOICES:
)" + thing_list + R"(<|im_end|><|im_start|>assistant
The best OBJECT CHOICE that satisfies the agent's GOAL is
)";

  std::vector<Observation> thing_obs;
  for (const auto &thing : agent.perceived_things) {
    Observation obs_entry;
    obs_entry.Id = thing.Id;
    std::string obs_context = thing.Name + ": " + thing.Description;
    to_lower_in_place(obs_context);
    obs_entry.Context = obs_context;
    thing_obs.push_back(obs_entry);
  }

  auto res = ConstrainedResolve(agent_id + "_plan", {context, thing_obs});
  if (res.first < 0) {
    return plan;
  }

  int thing_index = res.first;
  if (thing_index < 0 ||
      thing_index >= static_cast<int>(agent.perceived_things.size())) {
    return plan;
  }
  Thing chosen_thing = agent.perceived_things[thing_index];

  context = std::string(R"(<|startoftext|><|im_start|>user
Now select the best ACTION to advance the GOAL:

GOAL:
)") + primary_goal.Name + " - " + primary_goal.Description + R"(

OBJECT:
)" + chosen_thing.Name +
            " - " + chosen_thing.Description + R"(

ACTION:
)" + action_list +
            R"(<|im_end|><|im_start|>assistant
The best ACTION to accomplish the GOAL with the given OBJECT is 
)";

  std::vector<Observation> act_obs;
  for (const auto &action : agent.available_actions) {
    Observation obs_entry;
    obs_entry.Id = action.Id;
    std::string obs_context = action.Name + ": " + action.Description;
    to_lower_in_place(obs_context);
    obs_entry.Context = obs_context;
    act_obs.push_back(obs_entry);
  }

  if (act_obs.size() >= 2) {
    res = ConstrainedResolve(agent_id + "_plan", {context, act_obs});
  } else {
    res = std::make_pair(0, agent.available_actions[0].Id);
  }

  if (res.first < 0) {
    return plan;
  }

  int action_index = res.first;
  if (action_index < 0 ||
      action_index >= static_cast<int>(agent.available_actions.size())) {
    return plan;
  }
  Action chosen_action = agent.available_actions[action_index];

  plan.AgentId = agent_id;
  for (int step = 0; step < steps; step++) {
    Phase phase;
    phase.Step = step;
    phase.Action = chosen_action;
    phase.Thing = chosen_thing;
    plan.Phases.push_back(phase);
  }

  return plan;
}
