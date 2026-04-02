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
constexpr int kMaxPromptTokens = 2048;

bool is_valid_prediction_tokens(int token_count) {
  return token_count > 0 && token_count <= kMaxPromptTokens;
}

bool g_isEngineInitialized = false;
std::mutex g_apiMutex;

char* duplicate_heap_string(const std::string& value) {
  char* out = static_cast<char*>(std::malloc(value.size() + 1));
  if (!out) {
    return nullptr;
  }
  std::memcpy(out, value.c_str(), value.size() + 1);
  return out;
}

std::string prompt_perf_to_json(const CE_PromptPerfMetrics& metrics) {
  std::ostringstream out;
  out << "{"
      << "\"totalMs\":" << metrics.total_ms << ","
      << "\"promptEvalMs\":" << metrics.prompt_eval_ms << ","
      << "\"decodeEvalMs\":" << metrics.decode_eval_ms << ","
      << "\"sampleMs\":" << metrics.sample_ms << ","
      << "\"promptEvalTokens\":" << metrics.prompt_eval_tokens << ","
      << "\"decodeEvalCount\":" << metrics.decode_eval_count << ","
      << "\"sampleCount\":" << metrics.sample_count << ","
      << "\"outputTokenCount\":" << metrics.output_token_count
      << "}";
  return out.str();
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
int CE_Init(const char* model_path) {
  std::lock_guard<std::mutex> lock(g_apiMutex);

  if (g_isEngineInitialized) {
    return kStatusFailure;
  }

  if (!model_path || std::strlen(model_path) == 0) {
    return kStatusInvalidArguments;
  }

  const int init_status = CE_InitPlugin(model_path);
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
char* CE_GetLastPromptPerfJson() {
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
char* CE_Prompt(const char* context_key, const char* prompt, int n_tokens) {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  if (!g_isEngineInitialized || !is_valid_prediction_tokens(n_tokens)) {
    return nullptr;
  }

  return duplicate_heap_string(CE_ProcessPromptQuery(context_key, prompt, n_tokens));
}

EMSCRIPTEN_KEEPALIVE
void CE_FreeString(char* str) {
  if (str) {
    std::free(str);
  }
}

}  // extern "C"
