#include "engine_bridge.h"

#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "ggml-backend.h"
#include "ggml-webgpu.h"
#include "llama.h"
#include "runtime/inference_runtime.h"

using noumena::cogentengine::InferenceRuntime;

namespace {

constexpr int kStatusError = -1;

std::mutex g_engineMutex;
std::shared_ptr<InferenceRuntime> g_engineRuntime;

const char *backend_dev_type_name(enum ggml_backend_dev_type type) {
  switch (type) {
  case GGML_BACKEND_DEVICE_TYPE_CPU:
    return "cpu";
  case GGML_BACKEND_DEVICE_TYPE_GPU:
    return "gpu";
  case GGML_BACKEND_DEVICE_TYPE_IGPU:
    return "igpu";
  case GGML_BACKEND_DEVICE_TYPE_ACCEL:
    return "accel";
  default:
    return "unknown";
  }
}

std::string json_escape(const char *value) {
  const std::string input = value ? value : "";
  std::string escaped;
  escaped.reserve(input.size());

  for (const char ch : input) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped += ch;
      break;
    }
  }

  return escaped;
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

std::string prompt_perf_to_json(
    const noumena::cogentengine::PromptPerfStats &perf) {
  std::ostringstream out;
  out << "{"
      << "\"totalMs\":" << perf.total_ms << ","
      << "\"promptEvalMs\":" << perf.prompt_eval_ms << ","
      << "\"decodeEvalMs\":" << perf.decode_eval_ms << ","
      << "\"sampleMs\":" << perf.sample_ms << ","
      << "\"queueDelayMs\":" << perf.queue_delay_ms << ","
      << "\"ttftMs\":" << perf.ttft_ms << ","
      << "\"meanItlMs\":" << perf.mean_itl_ms << ","
      << "\"tailItlMs\":" << perf.tail_itl_ms << ","
      << "\"e2elMs\":" << perf.e2e_ms << ","
      << "\"inputTokenCount\":" << perf.input_token_count << ","
      << "\"promptEvalTokens\":" << perf.prompt_eval_tokens << ","
      << "\"decodeEvalCount\":" << perf.decode_eval_count << ","
      << "\"sampleCount\":" << perf.sample_count << ","
      << "\"outputTokenCount\":" << perf.output_token_count << ","
      << "\"schedulerTickCount\":" << perf.scheduler_tick_count << ","
      << "\"batchParticipationCount\":" << perf.batch_participation_count << ","
      << "\"decodeFirstTickCount\":" << perf.decode_first_tick_count << ","
      << "\"chunkedPrefillTickCount\":" << perf.chunked_prefill_tick_count
      << ","
      << "\"mixedWorkloadTickCount\":" << perf.mixed_workload_tick_count
      << ","
      << "\"lcpReuseTokens\":" << perf.lcp_reuse_tokens << ","
      << "\"prefixCacheRestoreTokens\":" << perf.prefix_cache_restore_tokens
      << ","
      << "\"prefixCacheHitCount\":" << perf.prefix_cache_hit_count << ","
      << "\"prefixCacheStoreCount\":" << perf.prefix_cache_store_count
      << "}";
  return out.str();
}

std::string generate_response_to_json(
    const noumena::cogentengine::GenerateResponse &response) {
  std::ostringstream out;
  out << "{"
      << "\"requestId\":" << response.request_id << ","
      << "\"completed\":"
      << json_bool(response.status ==
                   noumena::cogentengine::GenerateResponseStatus::Completed)
      << ","
      << "\"failed\":"
      << json_bool(response.status ==
                   noumena::cogentengine::GenerateResponseStatus::Failed)
      << ","
      << "\"cancelled\":"
      << json_bool(response.status ==
                   noumena::cogentengine::GenerateResponseStatus::Cancelled)
      << ","
      << "\"outputText\":\"" << json_escape(response.output_text.c_str())
      << "\","
      << "\"errorMessage\":";
  if (!response.error_message.empty()) {
    out << "\"" << json_escape(response.error_message.c_str()) << "\"";
  } else {
    out << "null";
  }
  out << ",\"perf\":" << prompt_perf_to_json(response.perf) << "}";
  return out.str();
}

std::shared_ptr<InferenceRuntime> acquire_engine_runtime() {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  return g_engineRuntime;
}

} // namespace

int CE_InitPlugin(const char *model_path, const CE_InitConfig *config) {
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
    runtime_config.gpu_layers =
        config->gpu_layers >= 0 ? config->gpu_layers : 99;
    runtime_config.flash_attention = config->flash_attention;
    runtime_config.kv_unified = config->kv_unified;
    runtime_config.max_cached_sessions =
        config->max_cached_sessions > 0 ? config->max_cached_sessions : 8;
    runtime_config.retained_prefix_tokens = config->retained_prefix_tokens >= 0
                                                ? config->retained_prefix_tokens
                                                : 100;
    runtime_config.prefill_chunk_size =
        config->prefill_chunk_size >= 0 ? config->prefill_chunk_size : 0;
    runtime_config.prefix_cache_interval_tokens =
        config->prefix_cache_interval_tokens > 0
            ? config->prefix_cache_interval_tokens
            : 128;
    runtime_config.max_prefix_cache_entries =
        config->max_prefix_cache_entries > 0 ? config->max_prefix_cache_entries : 32;
    runtime_config.scheduler_policy.mode =
        config->scheduler_policy <= 0
            ? noumena::cogentengine::SchedulerPolicyMode::LatencyFirst
            : (config->scheduler_policy == 2
                   ? noumena::cogentengine::SchedulerPolicyMode::
                         ThroughputFirst
                   : noumena::cogentengine::SchedulerPolicyMode::Balanced);
    runtime_config.scheduler_policy.decode_token_reserve =
        config->decode_token_reserve >= 0 ? config->decode_token_reserve : 1;
    runtime_config.scheduler_policy.enable_adaptive_prefill_chunking =
        config->adaptive_prefill_chunking > 0;
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

int CE_GetLastPromptPerf(CE_PromptPerfMetrics *out_metrics) {
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
  out_metrics->queue_delay_ms = perf_stats.queue_delay_ms;
  out_metrics->ttft_ms = perf_stats.ttft_ms;
  out_metrics->mean_itl_ms = perf_stats.mean_itl_ms;
  out_metrics->tail_itl_ms = perf_stats.tail_itl_ms;
  out_metrics->e2e_ms = perf_stats.e2e_ms;
  out_metrics->input_token_count = perf_stats.input_token_count;
  out_metrics->prompt_eval_tokens = perf_stats.prompt_eval_tokens;
  out_metrics->decode_eval_count = perf_stats.decode_eval_count;
  out_metrics->sample_count = perf_stats.sample_count;
  out_metrics->output_token_count = perf_stats.output_token_count;
  out_metrics->scheduler_tick_count = perf_stats.scheduler_tick_count;
  out_metrics->batch_participation_count = perf_stats.batch_participation_count;
  out_metrics->decode_first_tick_count = perf_stats.decode_first_tick_count;
  out_metrics->chunked_prefill_tick_count =
      perf_stats.chunked_prefill_tick_count;
  out_metrics->mixed_workload_tick_count =
      perf_stats.mixed_workload_tick_count;
  out_metrics->lcp_reuse_tokens = perf_stats.lcp_reuse_tokens;
  out_metrics->prefix_cache_restore_tokens =
      perf_stats.prefix_cache_restore_tokens;
  out_metrics->prefix_cache_hit_count = perf_stats.prefix_cache_hit_count;
  out_metrics->prefix_cache_store_count = perf_stats.prefix_cache_store_count;
  return 0;
}

const char *CE_GetBackendInfoJsonString() {
  static std::string info_json;

  std::ostringstream out;
  out << "{";
#ifdef GGML_USE_WEBGPU
  out << "\"webgpuCompiled\":true,";
#else
  out << "\"webgpuCompiled\":false,";
#endif
  out << "\"webgpuRegistered\":";
  const ggml_backend_reg_t webgpu_reg =
      ggml_backend_reg_by_name(GGML_WEBGPU_NAME);
  out << (webgpu_reg != nullptr ? "true" : "false") << ",";
  out << "\"webgpuDeviceCount\":";
  out << (webgpu_reg != nullptr ? ggml_backend_reg_dev_count(webgpu_reg) : 0)
      << ",";
  out << "\"gpuOffloadSupported\":";
  out << (llama_supports_gpu_offload() ? "true" : "false") << ",";
  out << "\"engineInitialized\":";
  out << (acquire_engine_runtime() ? "true" : "false") << ",";
  out << "\"availableBackends\":[";

  const size_t backend_count = ggml_backend_reg_count();
  for (size_t i = 0; i < backend_count; ++i) {
    if (i > 0) {
      out << ",";
    }

    ggml_backend_reg_t reg = ggml_backend_reg_get(i);
    out << "{"
        << "\"name\":\"" << json_escape(ggml_backend_reg_name(reg)) << "\","
        << "\"deviceCount\":" << ggml_backend_reg_dev_count(reg) << "}";
  }

  out << "],";
  out << "\"devices\":[";

  const size_t device_count = ggml_backend_dev_count();
  for (size_t i = 0; i < device_count; ++i) {
    if (i > 0) {
      out << ",";
    }

    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(dev, &props);
    const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

    out << "{"
        << "\"name\":\"" << json_escape(props.name) << "\","
        << "\"description\":\"" << json_escape(props.description) << "\","
        << "\"type\":\"" << backend_dev_type_name(props.type) << "\","
        << "\"backendName\":\""
        << json_escape(reg ? ggml_backend_reg_name(reg) : "") << "\","
        << "\"deviceId\":";
    if (props.device_id != nullptr && props.device_id[0] != '\0') {
      out << "\"" << json_escape(props.device_id) << "\"";
    } else {
      out << "null";
    }
    out << ","
        << "\"memoryFreeBytes\":" << props.memory_free << ","
        << "\"memoryTotalBytes\":" << props.memory_total << ","
        << "\"capabilities\":{"
        << "\"async\":" << json_bool(props.caps.async) << ","
        << "\"hostBuffer\":" << json_bool(props.caps.host_buffer) << ","
        << "\"bufferFromHostPtr\":"
        << json_bool(props.caps.buffer_from_host_ptr) << ","
        << "\"events\":" << json_bool(props.caps.events) << "}"
        << "}";
  }

  out << "]}";
  info_json = out.str();
  return info_json.c_str();
}

CE_RequestId CE_EnqueuePromptQuery(const char *context_key, const char *prompt,
                                   int n_tokens_predict,
                                   CE_TokenCallback on_token) {
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return 0;
  }

  return runtime->EnqueueRequest(
      context_key ? context_key : "", prompt ? prompt : "", n_tokens_predict,
      [on_token](const char *token_piece, int32_t token_length) {
        if (on_token != nullptr) {
          return on_token(token_piece, token_length) == 0;
        }
        return true;
      });
}

int CE_CancelQueuedPromptQuery(CE_RequestId request_id) {
  auto runtime = acquire_engine_runtime();
  if (!runtime || request_id == 0) {
    return 0;
  }
  return runtime->CancelRequest(request_id) ? 1 : 0;
}

std::string CE_RunQueuedRequestJsonString(CE_RequestId request_id) {
  noumena::cogentengine::GenerateResponse response{};
  response.request_id = request_id;

  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    response.status = noumena::cogentengine::GenerateResponseStatus::Failed;
    response.error_message = "Engine is not initialized.";
    return generate_response_to_json(response);
  }

  const bool success = runtime->RunUntilRequestCompletes(request_id, response);
  if (!success &&
      response.status != noumena::cogentengine::GenerateResponseStatus::Failed &&
      response.status !=
          noumena::cogentengine::GenerateResponseStatus::Completed) {
    response.status = noumena::cogentengine::GenerateResponseStatus::Failed;
    response.error_message = "Queued request execution failed.";
  }

  return generate_response_to_json(response);
}
