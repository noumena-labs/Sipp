#include <emscripten/emscripten.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

#include "engine_bridge.h"

namespace {

constexpr int kStatusFailure = -1;
constexpr int kStatusInvalidArguments = -2;
constexpr int kStatusNotInitialized = -3;
constexpr int kMaxPromptTokens = 2048;

bool is_valid_prediction_tokens(int token_count) {
  return token_count > 0 && token_count <= kMaxPromptTokens;
}

bool g_isEngineInitialized = false;
std::mutex g_apiMutex;

char *duplicate_heap_string(const std::string &value) {
  char *out = static_cast<char *>(std::malloc(value.size() + 1));
  if (!out) {
    return nullptr;
  }
  std::memcpy(out, value.c_str(), value.size() + 1);
  return out;
}

std::string prompt_perf_to_json(const CE_PromptPerfMetrics &metrics) {
  std::ostringstream out;
  out << "{"
      << "\"totalMs\":" << metrics.total_ms << ","
      << "\"promptEvalMs\":" << metrics.prompt_eval_ms << ","
      << "\"decodeEvalMs\":" << metrics.decode_eval_ms << ","
      << "\"sampleMs\":" << metrics.sample_ms << ","
      << "\"inputTokenCount\":" << metrics.input_token_count << ","
      << "\"promptEvalTokens\":" << metrics.prompt_eval_tokens << ","
      << "\"decodeEvalCount\":" << metrics.decode_eval_count << ","
      << "\"sampleCount\":" << metrics.sample_count << ","
      << "\"outputTokenCount\":" << metrics.output_token_count << "}";
  return out.str();
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
int CE_Init(const char *model_path, int n_ctx, int n_batch, int n_ubatch,
            int n_seq_max, int n_threads, int n_threads_batch, int gpu_layers,
            int flash_attention, int kv_unified, int max_cached_sessions,
            int retained_prefix_tokens) {
  std::lock_guard<std::mutex> lock(g_apiMutex);

  if (!model_path || std::strlen(model_path) == 0) {
    return kStatusInvalidArguments;
  }

  if (g_isEngineInitialized) {
    CE_ClosePlugin();
    g_isEngineInitialized = false;
  }

  const CE_InitConfig config{
      .n_ctx = n_ctx,
      .n_batch = n_batch,
      .n_ubatch = n_ubatch,
      .n_seq_max = n_seq_max,
      .n_threads = n_threads,
      .n_threads_batch = n_threads_batch,
      .gpu_layers = gpu_layers,
      .flash_attention = flash_attention,
      .kv_unified = kv_unified,
      .max_cached_sessions = max_cached_sessions,
      .retained_prefix_tokens = retained_prefix_tokens,
  };

  const int init_status = CE_InitPlugin(model_path, &config);
  if (init_status != 0) {
    return init_status;
  }

  g_isEngineInitialized = true;
  return 0;
}

EMSCRIPTEN_KEEPALIVE
void CE_Close() {
  std::lock_guard<std::mutex> lock(g_apiMutex);

  if (!g_isEngineInitialized) {
    return;
  }

  CE_ClosePlugin();
  g_isEngineInitialized = false;
}

EMSCRIPTEN_KEEPALIVE
char *CE_GetLastPromptPerfJson() {
  std::lock_guard<std::mutex> lock(g_apiMutex);

  if (!g_isEngineInitialized) {
    return nullptr;
  }

  CE_PromptPerfMetrics metrics{};
  if (CE_GetLastPromptPerf(&metrics) != 0) {
    return nullptr;
  }

  return duplicate_heap_string(prompt_perf_to_json(metrics));
}

EMSCRIPTEN_KEEPALIVE
char *CE_GetBackendInfoJson() {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  return duplicate_heap_string(CE_GetBackendInfoJsonString());
}

EMSCRIPTEN_KEEPALIVE
int CE_StreamPrompt(const char *context_key, const char *prompt, int n_tokens,
                    CE_TokenCallback on_token) {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  if (!g_isEngineInitialized) {
    return kStatusNotInitialized;
  }
  if (prompt == nullptr || !is_valid_prediction_tokens(n_tokens) ||
      on_token == nullptr) {
    return kStatusInvalidArguments;
  }
  return CE_StreamPromptQuery(context_key, prompt, n_tokens, on_token);
}

EMSCRIPTEN_KEEPALIVE
void CE_FreeString(char *str) {
  if (str) {
    std::free(str);
  }
}

} // extern "C"
