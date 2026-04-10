#include "engine_bridge.h"

#include <cstring>
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
constexpr int kCompletedRequestStatusPending = 0;
constexpr int kCompletedRequestStatusCompleted = 1;
constexpr int kCompletedRequestStatusCancelled = 2;
constexpr int kCompletedRequestStatusFailed = 3;

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


std::shared_ptr<InferenceRuntime> acquire_engine_runtime() {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  return g_engineRuntime;
}

int completed_status_to_code(
    noumena::cogentengine::GenerateResponseStatus status) {
  switch (status) {
  case noumena::cogentengine::GenerateResponseStatus::Pending:
    return kCompletedRequestStatusPending;
  case noumena::cogentengine::GenerateResponseStatus::Completed:
    return kCompletedRequestStatusCompleted;
  case noumena::cogentengine::GenerateResponseStatus::Cancelled:
    return kCompletedRequestStatusCancelled;
  case noumena::cogentengine::GenerateResponseStatus::Failed:
    return kCompletedRequestStatusFailed;
  }
  return kCompletedRequestStatusPending;
}

bool try_get_completed_response(CE_RequestId request_id,
                                noumena::cogentengine::GenerateResponse &out) {
  auto runtime = acquire_engine_runtime();
  if (!runtime || request_id == 0) {
    return false;
  }
  return runtime->TryPeekCompletedResponse(request_id, out);
}

int copy_completed_field(char *buffer, int32_t capacity,
                         const std::string &value) {
  if (buffer == nullptr || capacity <= 0) {
    return kStatusError;
  }

  const std::size_t byte_count = value.size();
  if (static_cast<std::size_t>(capacity) <= byte_count) {
    return kStatusError;
  }

  if (byte_count > 0) {
    std::memcpy(buffer, value.data(), byte_count);
  }
  buffer[byte_count] = '\0';
  return static_cast<int>(byte_count);
}

void copy_runtime_observability(
    const noumena::cogentengine::RuntimeObservabilityMetrics &runtime_observability,
    CE_RuntimeObservabilityMetrics *out_metrics) {
  out_metrics->total_ms = runtime_observability.total_ms;
  out_metrics->prompt_eval_ms = runtime_observability.prompt_eval_ms;
  out_metrics->decode_eval_ms = runtime_observability.decode_eval_ms;
  out_metrics->sample_ms = runtime_observability.sample_ms;
  out_metrics->queue_delay_ms = runtime_observability.queue_delay_ms;
  out_metrics->ttft_ms = runtime_observability.ttft_ms;
  out_metrics->mean_itl_ms = runtime_observability.mean_itl_ms;
  out_metrics->tail_itl_ms = runtime_observability.tail_itl_ms;
  out_metrics->e2e_ms = runtime_observability.e2e_ms;
  out_metrics->input_token_count = runtime_observability.input_token_count;
  out_metrics->prompt_eval_tokens = runtime_observability.prompt_eval_tokens;
  out_metrics->decode_eval_count = runtime_observability.decode_eval_count;
  out_metrics->sample_count = runtime_observability.sample_count;
  out_metrics->output_token_count = runtime_observability.output_token_count;
  out_metrics->batch_participation_count =
      runtime_observability.batch_participation_count;
  out_metrics->decode_first_tick_count =
      runtime_observability.decode_first_tick_count;
  out_metrics->chunked_prefill_tick_count =
      runtime_observability.chunked_prefill_tick_count;
  out_metrics->mixed_workload_tick_count =
      runtime_observability.mixed_workload_tick_count;
  out_metrics->lcp_reuse_tokens = runtime_observability.lcp_reuse_tokens;
  out_metrics->prefix_cache_restore_tokens =
      runtime_observability.prefix_cache_restore_tokens;
  out_metrics->prefix_cache_hit_count =
      runtime_observability.prefix_cache_hit_count;
  out_metrics->prefix_cache_store_count =
      runtime_observability.prefix_cache_store_count;
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
    runtime_config.enable_runtime_observability =
        config->enable_runtime_observability > 0 ? 1 : 0;
    runtime_config.enable_backend_profiling =
        config->enable_backend_profiling > 0 ? 1 : 0;
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

int CE_GetRuntimeObservability(CE_RuntimeObservabilityMetrics *out_metrics) {
  if (out_metrics == nullptr) {
    return kStatusError;
  }

  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return kStatusError;
  }

  noumena::cogentengine::RuntimeObservabilityMetrics runtime_observability;
  if (!runtime->TryGetRuntimeObservability(runtime_observability)) {
    return kStatusError;
  }

  copy_runtime_observability(runtime_observability, out_metrics);
  return 0;
}

int CE_ResetRuntimeObservability() {
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return kStatusError;
  }

  runtime->ResetRuntimeObservability();
  return 0;
}

int CE_RunSchedulerTick() {
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return static_cast<int>(
        noumena::cogentengine::RequestStepResult::Invalid);
  }

  return static_cast<int>(runtime->RunSchedulerTick());
}

int CE_RunRequestStep(CE_RequestId request_id) {
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return static_cast<int>(
        noumena::cogentengine::RequestStepResult::Invalid);
  }

  return static_cast<int>(runtime->RunRequestStep(request_id));
}

int CE_GetCompletedRequestStatus(CE_RequestId request_id) {
  noumena::cogentengine::GenerateResponse response{};
  if (!try_get_completed_response(request_id, response)) {
    return kCompletedRequestStatusPending;
  }

  return completed_status_to_code(response.status);
}

int CE_GetCompletedRequestOutputSize(CE_RequestId request_id) {
  noumena::cogentengine::GenerateResponse response{};
  if (!try_get_completed_response(request_id, response)) {
    return kStatusError;
  }
  return static_cast<int>(response.output_text.size());
}

int CE_CopyCompletedRequestOutput(CE_RequestId request_id, char *buffer,
                                  int32_t capacity) {
  noumena::cogentengine::GenerateResponse response{};
  if (!try_get_completed_response(request_id, response)) {
    return kStatusError;
  }
  return copy_completed_field(buffer, capacity, response.output_text);
}

int CE_GetCompletedRequestErrorSize(CE_RequestId request_id) {
  noumena::cogentengine::GenerateResponse response{};
  if (!try_get_completed_response(request_id, response)) {
    return kStatusError;
  }
  return static_cast<int>(response.error_message.size());
}

int CE_CopyCompletedRequestError(CE_RequestId request_id, char *buffer,
                                 int32_t capacity) {
  noumena::cogentengine::GenerateResponse response{};
  if (!try_get_completed_response(request_id, response)) {
    return kStatusError;
  }
  return copy_completed_field(buffer, capacity, response.error_message);
}

int CE_GetCompletedRequestRuntimeObservability(
    CE_RequestId request_id, CE_RuntimeObservabilityMetrics *out_metrics) {
  if (out_metrics == nullptr) {
    return kStatusError;
  }

  noumena::cogentengine::GenerateResponse response{};
  if (!try_get_completed_response(request_id, response)) {
    return kStatusError;
  }

  copy_runtime_observability(response.runtime_observability, out_metrics);
  return 0;
}

int CE_ConsumeCompletedRequest(CE_RequestId request_id) {
  auto runtime = acquire_engine_runtime();
  if (!runtime || request_id == 0) {
    return 0;
  }
  return runtime->ConsumeCompletedResponse(request_id) ? 1 : 0;
}

const char *CE_GetBackendObservabilityJsonString() {
  static std::string info_json;
  const auto runtime = acquire_engine_runtime();
  const bool profiling_enabled =
      runtime != nullptr && runtime->BackendProfilingEnabled();

  std::ostringstream out;
  out << "{";
  out << "\"profilingEnabled\":" << json_bool(profiling_enabled) << ",";
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
  out << (runtime ? "true" : "false") << ",";
  out << "\"availableBackends\":[";

  if (profiling_enabled) {
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
  }

  out << "],";
  out << "\"devices\":[";

  if (profiling_enabled) {
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
