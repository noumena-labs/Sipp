#include "engine_bridge.h"

#include <memory>
#include <mutex>
#include <string>

#include "runtime/inference_runtime.h"

using noumena::cogentengine::InferenceRuntime;

namespace {

constexpr int kDefaultGpuLayers = 99;
constexpr int kStatusError = -1;

std::mutex g_engineMutex;
std::shared_ptr<InferenceRuntime> g_engineRuntime;

std::shared_ptr<InferenceRuntime> acquire_engine_runtime() {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  return g_engineRuntime;
}

}  // namespace

int CE_InitPlugin(const char* model_path) {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  if (model_path == nullptr || model_path[0] == '\0' || g_engineRuntime) {
    return kStatusError;
  }

  auto runtime =
      std::make_shared<InferenceRuntime>(model_path, kDefaultGpuLayers);
  if (!runtime || !runtime->IsReady()) {
    return kStatusError;
  }

  g_engineRuntime = std::move(runtime);
  return 0;
}

void CE_ClosePlugin() {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  g_engineRuntime.reset();
}

int CE_GetLastPromptPerf(CE_PromptPerfMetrics* out_metrics) {
  if (out_metrics == nullptr) {
    return kStatusError;
  }

  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return kStatusError;
  }

  noumena::cogentengine::PromptPerfStats perf_stats;
  if (!runtime->TryGetLastPromptPerf(perf_stats)) {
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
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return {};
  }

  return runtime->Prompt(
      context_key ? context_key : "",
      prompt ? prompt : "",
      n_tokens_predict);
}
