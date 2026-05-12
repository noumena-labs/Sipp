#include "engine_bridge.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"
#include "llama.h"
#include "runtime/inference_runtime.h"

using noumena::cogentengine::GenerateTokenEmissionMode;
using noumena::cogentengine::InferenceRuntime;

namespace {

constexpr int kStatusError = -1;
constexpr int kCompletedRequestStatusPending = 0;
constexpr int kCompletedRequestStatusCompleted = 1;
constexpr int kCompletedRequestStatusCancelled = 2;
constexpr int kCompletedRequestStatusFailed = 3;
constexpr int kCompletedRequestStatusUnknown = 4;

std::mutex g_engineMutex;
std::shared_ptr<InferenceRuntime> g_engineRuntime;
using json = nlohmann::json;

bool is_valid_prediction_tokens(int token_count) {
  return token_count > 0;
}

bool map_token_emission_mode(CE_TokenEmissionMode mode,
                             GenerateTokenEmissionMode &out_mode) {
  switch (mode) {
  case CE_TOKEN_EMISSION_NONE:
    out_mode = GenerateTokenEmissionMode::None;
    return true;
  case CE_TOKEN_EMISSION_STREAMING_BUFFER:
    out_mode = GenerateTokenEmissionMode::StreamingBuffer;
    return true;
  default:
    return false;
  }
}

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

const char *empty_c_string() {
  static thread_local std::string empty;
  empty.clear();
  return empty.c_str();
}

bool parse_chat_messages_json(const char *messages_json,
                              std::vector<common_chat_msg> &out_messages) {
  out_messages.clear();
  if (messages_json == nullptr || messages_json[0] == '\0') {
    return false;
  }

  const json parsed = json::parse(messages_json, nullptr, false);
  if (parsed.is_discarded() || !parsed.is_array()) {
    return false;
  }
  try {
    out_messages = common_chat_msgs_parse_oaicompat(parsed);
  } catch (const std::exception &) {
    return false;
  }

  return true;
}

bool validate_media_buffers(int32_t n_images, const uint8_t *images_flat_buffer,
                            const int32_t *image_sizes,
                            std::size_t &out_total_bytes) {
  out_total_bytes = 0;
  if (n_images < 0) {
    return false;
  }
  if (n_images == 0) {
    return true;
  }
  if (images_flat_buffer == nullptr || image_sizes == nullptr) {
    return false;
  }

  std::size_t total_bytes = 0;
  for (int32_t index = 0; index < n_images; ++index) {
    const int32_t image_size = image_sizes[index];
    if (image_size <= 0) {
      return false;
    }

    const std::size_t size = static_cast<std::size_t>(image_size);
    if (size > std::numeric_limits<std::size_t>::max() - total_bytes) {
      return false;
    }
    total_bytes += size;
  }

  out_total_bytes = total_bytes;
  return true;
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
  if (out_metrics == nullptr) {
    return;
  }

  // Latency (User Experience)
  out_metrics->ttft_ms = runtime_observability.ttft_ms;
  out_metrics->itl_avg_ms = runtime_observability.itl_avg_ms;
  out_metrics->itl_p99_ms = runtime_observability.itl_p99_ms;
  out_metrics->e2e_ms = runtime_observability.e2e_ms;

  // Phases (Compute)
  out_metrics->prefill_ms = runtime_observability.prefill_ms;
  out_metrics->decode_ms = runtime_observability.decode_ms;

  // Native (Hardware & Engine)
  out_metrics->native_gpu_ms = runtime_observability.native_gpu_ms;
  out_metrics->native_sync_ms = runtime_observability.native_sync_ms;
  out_metrics->native_logic_ms = runtime_observability.native_logic_ms;

  // Counts
  out_metrics->input_tokens = runtime_observability.input_tokens;
  out_metrics->output_tokens = runtime_observability.output_tokens;
  out_metrics->cache_hits = runtime_observability.cache_hits;
}

void copy_scheduler_loop_result(
    const noumena::cogentengine::SchedulerBurstResult &burst_result,
    CE_SchedulerLoopResult *out_result) {
  if (out_result == nullptr) {
    return;
  }
  out_result->ticks_executed = burst_result.ticks_executed;
  out_result->progressed_ticks = burst_result.progressed_ticks;
  out_result->completed_response_count = burst_result.completed_response_count;
  out_result->emitted_token_count = burst_result.emitted_token_count;
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
    runtime_config.gpu_layers = config->gpu_layers;
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
            : 0;
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
    runtime_config.mmproj_path =
        config->mmproj_path != nullptr ? config->mmproj_path : "";
    runtime_config.multimodal_use_gpu = std::clamp(
        config->multimodal_use_gpu,
        static_cast<int32_t>(-1),
        static_cast<int32_t>(1));
    runtime_config.image_min_tokens =
        config->image_min_tokens > 0 ? config->image_min_tokens : 0;
    runtime_config.image_max_tokens =
        config->image_max_tokens > 0 ? config->image_max_tokens : 0;
    runtime_config.sampling_repeat_last_n =
        config->sampling_repeat_last_n >= 0 ? config->sampling_repeat_last_n : 64;
    runtime_config.sampling_repeat_penalty =
        config->sampling_repeat_penalty >= 0.0f
            ? config->sampling_repeat_penalty
            : 1.05f;
    runtime_config.sampling_frequency_penalty =
        config->sampling_frequency_penalty;
    runtime_config.sampling_presence_penalty =
        config->sampling_presence_penalty;
    runtime_config.sampling_top_k =
        config->sampling_top_k >= 0 ? config->sampling_top_k : 40;
    runtime_config.sampling_top_p =
        config->sampling_top_p >= 0.0f ? config->sampling_top_p : 0.8f;
    runtime_config.sampling_min_p =
        config->sampling_min_p >= 0.0f ? config->sampling_min_p : 0.0f;
    runtime_config.sampling_temperature =
        config->sampling_temperature >= 0.0f
            ? config->sampling_temperature
            : 0.7f;
    runtime_config.sampling_seed = config->sampling_seed;
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

int CE_RunSchedulerBurst(int32_t max_ticks, int32_t max_completed_responses,
                         int32_t max_emitted_tokens,
                         CE_SchedulerLoopResult *out_result) {
  if (out_result == nullptr) {
    return static_cast<int>(
        noumena::cogentengine::RequestStepResult::Invalid);
  }

  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    out_result->ticks_executed = 0;
    out_result->progressed_ticks = 0;
    out_result->completed_response_count = 0;
    out_result->emitted_token_count = 0;
    return static_cast<int>(
        noumena::cogentengine::RequestStepResult::Invalid);
  }

  const noumena::cogentengine::SchedulerBurstResult burst_result =
      runtime->RunSchedulerBurst(max_ticks, max_completed_responses,
                                 max_emitted_tokens);
  copy_scheduler_loop_result(burst_result, out_result);
  return static_cast<int>(burst_result.status);
}

int CE_RunSchedulerBurstWithDeadline(int32_t max_ticks,
                                     int32_t max_completed_responses,
                                     int32_t max_emitted_tokens,
                                     int32_t max_duration_us,
                                     CE_SchedulerLoopResult *out_result) {
  if (out_result == nullptr) {
    return static_cast<int>(
        noumena::cogentengine::RequestStepResult::Invalid);
  }

  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    out_result->ticks_executed = 0;
    out_result->progressed_ticks = 0;
    out_result->completed_response_count = 0;
    out_result->emitted_token_count = 0;
    return static_cast<int>(
        noumena::cogentengine::RequestStepResult::Invalid);
  }

  const noumena::cogentengine::SchedulerBurstResult burst_result =
      runtime->RunSchedulerBurst(max_ticks, max_completed_responses,
                                 max_emitted_tokens, max_duration_us);
  copy_scheduler_loop_result(burst_result, out_result);
  return static_cast<int>(burst_result.status);
}

int CE_GetCompletedRequestStatus(CE_RequestId request_id) {
  auto runtime = acquire_engine_runtime();
  if (!runtime || request_id == 0) {
    return kCompletedRequestStatusUnknown;
  }
  noumena::cogentengine::GenerateResponse response{};
  if (runtime->TryPeekCompletedResponse(request_id, response)) {
    return completed_status_to_code(response.status);
  }

  return runtime->HasRequest(request_id) ? kCompletedRequestStatusPending
                                         : kCompletedRequestStatusUnknown;
}

int CE_RunSchedulerLoop(int32_t max_ticks, int32_t max_completed_responses,
                        int32_t max_emitted_tokens, int32_t max_duration_us,
                        CE_SchedulerLoopResult *out_result) {
  if (out_result == nullptr) {
    return static_cast<int>(noumena::cogentengine::RequestStepResult::Invalid);
  }

  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    out_result->ticks_executed = 0;
    out_result->progressed_ticks = 0;
    out_result->completed_response_count = 0;
    out_result->emitted_token_count = 0;
    return static_cast<int>(noumena::cogentengine::RequestStepResult::Invalid);
  }

  const noumena::cogentengine::SchedulerBurstResult burst_result =
      runtime->RunSchedulerLoop(max_ticks, max_completed_responses,
                                max_emitted_tokens, max_duration_us);
  copy_scheduler_loop_result(burst_result, out_result);
  return static_cast<int>(burst_result.status);
}

const uint8_t *CE_GetStreamingBufferPointer() {
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return nullptr;
  }
  return runtime->StreamingBufferPointer();
}

int32_t CE_GetStreamingBufferCapacity() {
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return 0;
  }
  return static_cast<int32_t>(runtime->StreamingBufferCapacity());
}

int32_t *CE_GetStreamingBufferUsedAddress() {
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return nullptr;
  }
  return runtime->StreamingBufferUsedAddress();
}

int32_t *CE_GetStreamingBufferDropCountAddress() {
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return nullptr;
  }
  return runtime->StreamingBufferDropCountAddress();
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

CE_RequestId CE_StartPromptRequestWithTokenEmissionMode(
    const char *context_key, const char *prompt, int n_tokens_predict,
    CE_TokenEmissionMode token_emission_mode, const char *grammar) {
  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return 0;
  }

  GenerateTokenEmissionMode native_emission_mode;
  if (!map_token_emission_mode(token_emission_mode, native_emission_mode)) {
    return 0;
  }

  return runtime->EnqueueRequest(
      context_key ? context_key : "", prompt ? prompt : "", n_tokens_predict,
      grammar ? std::string(grammar) : std::string(), native_emission_mode);
}

CE_RequestId CE_StartPromptWithMediaRequestWithTokenEmissionMode(
    const char *context_key, const char *prompt, int n_tokens_predict,
    int32_t n_images, const uint8_t *images_flat_buffer,
    const int32_t *image_sizes, CE_TokenEmissionMode token_emission_mode,
    const char *grammar) {
  if (prompt == nullptr || !is_valid_prediction_tokens(n_tokens_predict)) {
    return 0;
  }
  if (n_images < 0) {
    return 0;
  }
  if (n_images == 0) {
    return CE_StartPromptRequestWithTokenEmissionMode(
        context_key, prompt, n_tokens_predict, token_emission_mode, grammar);
  }

  GenerateTokenEmissionMode native_emission_mode;
  if (!map_token_emission_mode(token_emission_mode, native_emission_mode)) {
    return 0;
  }

  std::size_t total_media_bytes = 0;
  if (!validate_media_buffers(n_images, images_flat_buffer, image_sizes,
                              total_media_bytes)) {
    return 0;
  }
  (void)total_media_bytes;

  auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return 0;
  }

  std::vector<std::pair<const std::uint8_t *, std::size_t>> image_views;
  image_views.reserve(static_cast<std::size_t>(n_images));
  std::size_t byte_offset = 0;
  for (int32_t index = 0; index < n_images; ++index) {
    const std::size_t image_size = static_cast<std::size_t>(image_sizes[index]);
    image_views.emplace_back(images_flat_buffer + byte_offset, image_size);
    byte_offset += image_size;
  }

  return runtime->EnqueueMultimodalRequest(
      context_key ? context_key : "", prompt, n_tokens_predict,
      std::move(image_views),
      grammar ? std::string(grammar) : std::string(), native_emission_mode);
}

const char *CE_GetMediaMarkerString() {
  const auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return nullptr;
  }
  return runtime->GetMediaMarker();
}

const char *CE_GetChatTemplateString() {
  const auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return nullptr;
  }
  return runtime->GetChatTemplate();
}

const char *CE_GetBosTextString() {
  static thread_local std::string cached;
  const auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return empty_c_string();
  }
  cached = runtime->GetBosText();
  return cached.c_str();
}

const char *CE_GetEosTextString() {
  static thread_local std::string cached;
  const auto runtime = acquire_engine_runtime();
  if (!runtime) {
    return empty_c_string();
  }
  cached = runtime->GetEosText();
  return cached.c_str();
}

const char *CE_ApplyChatTemplateString(const char *messages_json,
                                       int add_assistant) {
  static thread_local std::string formatted_prompt;
  std::vector<common_chat_msg> messages;
  if (!parse_chat_messages_json(messages_json, messages)) {
    formatted_prompt.clear();
    return empty_c_string();
  }

  const auto runtime = acquire_engine_runtime();
  if (!runtime) {
    formatted_prompt.clear();
    return empty_c_string();
  }

  formatted_prompt =
      runtime->ApplyChatTemplate(messages, add_assistant != 0);
  return formatted_prompt.c_str();
}

int CE_CancelPromptRequest(CE_RequestId request_id) {
  auto runtime = acquire_engine_runtime();
  if (!runtime || request_id == 0) {
    return 0;
  }
  return runtime->CancelRequest(request_id) ? 1 : 0;
}
