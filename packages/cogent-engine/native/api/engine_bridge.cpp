#include "engine_bridge.h"

#include <memory>
#include <mutex>
#include <string>

#include "runtime/inference_runtime.h"

using noumena::cogentengine::InferenceRuntime;

namespace {

constexpr int kStatusError = -1;

std::mutex g_engineMutex;
std::shared_ptr<InferenceRuntime> g_engineRuntime;

std::shared_ptr<InferenceRuntime> acquire_engine_runtime() {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  return g_engineRuntime;
}

}  // namespace

int CE_InitPlugin(const char* model_path, const CE_InitConfig* config) {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  if (model_path == nullptr || model_path[0] == '\0' || g_engineRuntime) {
    return kStatusError;
  }

  noumena::cogentengine::InferenceRuntimeConfig runtime_config{};
  if (config != nullptr) {
    runtime_config.n_ctx = config->n_ctx;
    runtime_config.n_batch = config->n_batch;
    runtime_config.n_ubatch = config->n_ubatch;
    runtime_config.n_seq_max = config->n_seq_max;
    runtime_config.n_threads = config->n_threads;
    runtime_config.n_threads_batch = config->n_threads_batch;
    runtime_config.gpu_layers = config->gpu_layers >= 0 ? config->gpu_layers : 99;
    runtime_config.flash_attention = config->flash_attention;
    runtime_config.kv_unified = config->kv_unified;
    runtime_config.max_cached_sessions =
        config->max_cached_sessions > 0 ? config->max_cached_sessions : 8;
    runtime_config.retained_prefix_tokens =
        config->retained_prefix_tokens >= 0 ? config->retained_prefix_tokens : 100;
  }

  auto runtime = std::make_shared<InferenceRuntime>(model_path, runtime_config);
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
  out_metrics->input_token_count = perf_stats.input_token_count;
  out_metrics->prompt_eval_tokens = perf_stats.prompt_eval_tokens;
  out_metrics->decode_eval_count = perf_stats.decode_eval_count;
  out_metrics->sample_count = perf_stats.sample_count;
  out_metrics->output_token_count = perf_stats.output_token_count;
  return 0;
}

int CE_StreamPromptQuery(
    const char* context_key,
    const char* prompt,
    int n_tokens_predict,
    CE_TokenCallback on_token) {
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return kStatusError;
  }

  const bool success = runtime->Prompt(
      context_key ? context_key : "",
      prompt ? prompt : "",
      n_tokens_predict,
      [on_token](const char* token_piece, int32_t token_length) {
        if (on_token != nullptr) {
          on_token(token_piece, token_length);
        }
      });
  return success ? 0 : kStatusError;
}
