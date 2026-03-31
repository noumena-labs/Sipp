#include "engine_bridge.h"
#include "engine_manager.h"

#include <memory>
#include <mutex>
#include <string>

using noumena::cogentengine::CogentEngineManager;

namespace {

constexpr int kDefaultGpuLayers = 99;
constexpr int kStatusError = -1;

std::mutex g_engineMutex;
std::shared_ptr<CogentEngineManager> g_engineManager;

std::shared_ptr<CogentEngineManager> acquire_engine_manager() {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  return g_engineManager;
}

}  // namespace

int CE_InitPlugin(const char* model_path) {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  if (model_path == nullptr || model_path[0] == '\0' || g_engineManager) {
    return kStatusError;
  }

  auto manager =
      std::make_shared<CogentEngineManager>(model_path, kDefaultGpuLayers);
  if (!manager || !manager->IsReady()) {
    return kStatusError;
  }

  g_engineManager = std::move(manager);
  return 0;
}

void CE_ClosePlugin() {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  g_engineManager.reset();
}

int CE_GetLastPromptPerf(CE_PromptPerfMetrics* out_metrics) {
  if (out_metrics == nullptr) {
    return kStatusError;
  }

  auto manager = acquire_engine_manager();
  if (!manager) {
    return kStatusError;
  }

  noumena::cogentengine::PromptPerfStats perf_stats;
  if (!manager->TryGetLastPromptPerf(perf_stats)) {
    return kStatusError;
  }

  out_metrics->total_ms = perf_stats.total_ms;
  out_metrics->prompt_eval_ms = perf_stats.prompt_eval_ms;
  out_metrics->decode_eval_ms = perf_stats.decode_eval_ms;
  out_metrics->sample_ms = perf_stats.sample_ms;
  out_metrics->prompt_eval_tokens = perf_stats.prompt_eval_tokens;
  out_metrics->decode_eval_count = perf_stats.decode_eval_count;
  out_metrics->sample_count = perf_stats.sample_count;
  out_metrics->output_token_count = perf_stats.output_token_count;
  return 0;
}

std::string CE_ProcessPromptQuery(
    const char* context_key,
    const char* prompt,
    int n_tokens_predict) {
  auto manager = acquire_engine_manager();
  if (!manager) {
    return {};
  }

  return manager->Prompt(
      context_key ? context_key : "",
      prompt ? prompt : "",
      n_tokens_predict);
}
