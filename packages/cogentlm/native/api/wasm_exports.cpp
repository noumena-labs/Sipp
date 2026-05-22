#include <emscripten/emscripten.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "ffi_types.h"

#include "ggml-backend.h"
#include "ggml-webgpu.h"
#include "llama.h"

extern "C" {
std::uint32_t cogentlm_browser_engine_abi_version();
void *cogentlm_browser_engine_create();
std::uint32_t cogentlm_browser_engine_id(void *engine);
int cogentlm_browser_engine_load(void *engine, const char *model_path,
                                 const char *runtime_config_json);
int cogentlm_browser_engine_last_error_size(const void *engine);
int cogentlm_browser_engine_copy_last_error(const void *engine,
                                            std::uint8_t *buffer,
                                            std::uintptr_t buffer_len);
int cogentlm_browser_engine_close(void *engine);
CE_RequestId cogentlm_browser_engine_start_text_request(
    void *engine, const char *context_key, const char *prompt, int max_tokens,
    int token_emission_mode, const char *grammar);
CE_RequestId cogentlm_browser_engine_start_media_request(
    void *engine, const char *context_key, const char *prompt, int max_tokens,
    int image_count, const std::uint8_t *images_flat_buffer,
    const std::int32_t *image_sizes, int token_emission_mode,
    const char *grammar);
CE_RequestId cogentlm_browser_engine_start_chat_request(
    void *engine, const char *context_key, const char *messages_json,
    int max_tokens, int image_count, const std::uint8_t *images_flat_buffer,
    const std::int32_t *image_sizes, int token_emission_mode,
    const char *grammar);
int cogentlm_browser_engine_cancel_request(void *engine,
                                           CE_RequestId request_id);
int cogentlm_browser_engine_run_scheduler_loop(
    void *engine, int max_ticks, int max_completed_responses,
    int max_emitted_tokens, int max_duration_us, int streaming_active,
    CE_SchedulerLoopResult *out_result);
int cogentlm_browser_engine_completed_request_status(
    const void *engine, CE_RequestId request_id);
int cogentlm_browser_engine_completed_request_output_size(
    const void *engine, CE_RequestId request_id);
int cogentlm_browser_engine_copy_completed_request_output(
    const void *engine, CE_RequestId request_id, std::uint8_t *buffer,
    std::uintptr_t buffer_len);
int cogentlm_browser_engine_completed_request_error_size(
    const void *engine, CE_RequestId request_id);
int cogentlm_browser_engine_copy_completed_request_error(
    const void *engine, CE_RequestId request_id, std::uint8_t *buffer,
    std::uintptr_t buffer_len);
int cogentlm_browser_engine_consume_completed_request(
    void *engine, CE_RequestId request_id);
int cogentlm_browser_engine_runtime_observability(
    const void *engine, CE_RuntimeObservabilityMetrics *out_metrics);
int cogentlm_browser_engine_completed_runtime_observability(
    const void *engine, CE_RequestId request_id,
    CE_RuntimeObservabilityMetrics *out_metrics);
std::uint8_t *cogentlm_browser_engine_streaming_buffer_pointer(void *engine);
std::int32_t *cogentlm_browser_engine_streaming_buffer_used_address(
    void *engine);
std::int32_t *cogentlm_browser_engine_streaming_buffer_drop_count_address(
    void *engine);
char *cogentlm_browser_engine_media_marker(const void *engine);
char *cogentlm_browser_engine_chat_template(const void *engine);
char *cogentlm_browser_engine_bos_text(const void *engine);
char *cogentlm_browser_engine_eos_text(const void *engine);
char *cogentlm_browser_engine_probe_chat_boundary_info(const void *engine);
void cogentlm_browser_engine_free_string(char *value);
char *cogentlm_detect_model_from_gguf_bytes_json(const char *name,
                                                 const std::uint8_t *bytes,
                                                 std::uintptr_t bytes_len);
char *cogentlm_pairing_validate_json(const char *classified_json,
                                     const char *explicit_projector_id);
void *cogentlm_sha256_create();
int cogentlm_sha256_update(void *hasher, const std::uint8_t *bytes,
                           std::uintptr_t bytes_len);
char *cogentlm_sha256_finalize(void *hasher);
int cogentlm_sha256_close(void *hasher);
char *cogentlm_model_service_create_json(const char *config_json);
int cogentlm_model_service_close(void *service);
char *cogentlm_model_service_list_json(void *service);
char *cogentlm_model_service_current_json(void *service);
char *cogentlm_model_service_manifest_json(void *service);
char *cogentlm_model_service_prepare_load_json(void *service,
                                               const char *source_json,
                                               const char *options_json);
char *cogentlm_model_service_commit_load_json(void *service,
                                              const char *commit_json);
char *cogentlm_model_service_abort_load_json(void *service,
                                             const char *error_json);
char *cogentlm_model_service_remove_json(void *service,
                                         const char *model_id);
char *cogentlm_model_service_unload_json(void *service);
char *cogentlm_model_service_snapshot_json(void *service);
char *cogentlm_model_service_drain_events_json(void *service);
char *cogentlm_model_service_record_event_json(void *service,
                                               const char *event_type,
                                               const char *patch_json);
}

namespace {

constexpr int kStatusInvalidArguments = -2;
constexpr double kMaxExactInteger = 9007199254740991.0;

bool is_valid_prediction_tokens(int token_count) {
  return token_count > 0;
}

bool g_isEngineInitialized = false;
std::string g_lastEngineError;

char *duplicate_heap_string(const char *value) {
  const char *source = value != nullptr ? value : "";
  const std::size_t length = std::strlen(source);
  char *out = static_cast<char *>(std::malloc(length + 1));
  if (!out) {
    return nullptr;
  }
  std::memcpy(out, source, length + 1);
  return out;
}

int copy_string_with_nul(const std::string &value, char *buffer,
                         std::int32_t capacity) {
  if (buffer == nullptr || capacity <= 0 ||
      static_cast<std::size_t>(capacity) <= value.size()) {
    return kStatusInvalidArguments;
  }
  std::memcpy(buffer, value.data(), value.size());
  buffer[value.size()] = '\0';
  return static_cast<int>(value.size());
}

bool read_size_arg(double value, std::uint64_t *out) {
  if (!std::isfinite(value) || value < 0 || value > kMaxExactInteger) {
    return false;
  }
  *out = static_cast<std::uint64_t>(value);
  return true;
}

void *g_rustBrowserEngine = nullptr;
std::string g_mediaMarkerCache;
std::string g_chatTemplateCache;

const char *json_bool(bool value) {
  return value ? "true" : "false";
}

std::string json_escape(const char *value) {
  std::ostringstream out;
  const char *cursor = value != nullptr ? value : "";
  while (*cursor) {
    const unsigned char ch = static_cast<unsigned char>(*cursor++);
    switch (ch) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (ch < 0x20) {
        out << "\\u";
        constexpr char kHex[] = "0123456789abcdef";
        out << '0' << '0' << kHex[(ch >> 4) & 0x0f] << kHex[ch & 0x0f];
      } else {
        out << static_cast<char>(ch);
      }
      break;
    }
  }
  return out.str();
}

const char *backend_dev_type_name(enum ggml_backend_dev_type type) {
  switch (type) {
  case GGML_BACKEND_DEVICE_TYPE_CPU:
    return "CPU";
  case GGML_BACKEND_DEVICE_TYPE_GPU:
    return "GPU";
  case GGML_BACKEND_DEVICE_TYPE_IGPU:
    return "IGPU";
  case GGML_BACKEND_DEVICE_TYPE_ACCEL:
    return "ACCEL";
  case GGML_BACKEND_DEVICE_TYPE_META:
    return "META";
  default:
    return "UNKNOWN";
  }
}

std::string take_rust_string(char *value);
std::string read_rust_engine_last_error_locked(const void *engine);
void close_rust_engine_locked();
void ensure_llama_cache_env_locked();

std::string take_rust_string(char *value) {
  if (value == nullptr) {
    return "";
  }
  std::string out(value);
  cogentlm_browser_engine_free_string(value);
  return out;
}

void close_rust_engine_locked() {
  if (g_rustBrowserEngine != nullptr) {
    cogentlm_browser_engine_close(g_rustBrowserEngine);
    g_rustBrowserEngine = nullptr;
  }
  g_isEngineInitialized = false;
}

std::string read_rust_engine_last_error_locked(const void *engine) {
  if (engine == nullptr) {
    return "";
  }
  const int byte_length = cogentlm_browser_engine_last_error_size(engine);
  if (byte_length <= 0) {
    return "";
  }
  std::string error(static_cast<std::size_t>(byte_length), '\0');
  const int copied = cogentlm_browser_engine_copy_last_error(
      engine, reinterpret_cast<std::uint8_t *>(error.data()),
      static_cast<std::uintptr_t>(error.size() + 1));
  if (copied != byte_length) {
    return "";
  }
  return error;
}

void ensure_llama_cache_env_locked() {
  if (std::getenv("LLAMA_CACHE") != nullptr) {
    return;
  }
  constexpr const char *kCacheDir = "/tmp/cogentlm-llama-cache";
  mkdir("/tmp", 0777);
  mkdir(kCacheDir, 0777);
  setenv("LLAMA_CACHE", kCacheDir, 0);
}

std::string backend_observability_json_locked() {
  const bool profiling_enabled = false;
  std::ostringstream out;
  out << "{";
  out << "\"profilingEnabled\":" << json_bool(profiling_enabled) << ",";
  out << "\"webgpuCompiled\":true,";
  ggml_backend_reg_t webgpu_reg = ggml_backend_reg_by_name(GGML_WEBGPU_NAME);
  out << "\"webgpuRegistered\":" << json_bool(webgpu_reg != nullptr) << ",";
  out << "\"webgpuDeviceCount\":"
      << (webgpu_reg != nullptr ? ggml_backend_reg_dev_count(webgpu_reg) : 0)
      << ",";
  out << "\"gpuOffloadSupported\":"
      << json_bool(llama_supports_gpu_offload()) << ",";
  out << "\"engineInitialized\":" << json_bool(g_isEngineInitialized) << ",";
  out << "\"availableBackends\":[";

  if (profiling_enabled) {
    const size_t backend_count = ggml_backend_reg_count();
    for (size_t i = 0; i < backend_count; ++i) {
      if (i > 0) {
        out << ",";
      }
      ggml_backend_reg_t reg = ggml_backend_reg_get(i);
      out << "{\"name\":\"" << json_escape(ggml_backend_reg_name(reg))
          << "\",\"deviceCount\":" << ggml_backend_reg_dev_count(reg) << "}";
    }
  }

  out << "],\"devices\":[";
  if (profiling_enabled) {
    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count; ++i) {
      if (i > 0) {
        out << ",";
      }
      ggml_backend_dev_t dev = ggml_backend_dev_get(i);
      ggml_backend_dev_props props{};
      ggml_backend_dev_get_props(dev, &props);
      ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

      out << "{\"name\":\"" << json_escape(props.name)
          << "\",\"description\":\"" << json_escape(props.description)
          << "\",\"type\":\"" << backend_dev_type_name(props.type)
          << "\",\"backendName\":\""
          << json_escape(reg ? ggml_backend_reg_name(reg) : "")
          << "\",\"deviceId\":";
      if (props.device_id != nullptr && props.device_id[0] != '\0') {
        out << "\"" << json_escape(props.device_id) << "\"";
      } else {
        out << "null";
      }
      out << ",\"memoryFreeBytes\":" << props.memory_free
          << ",\"memoryTotalBytes\":" << props.memory_total
          << ",\"capabilities\":{\"async\":" << json_bool(props.caps.async)
          << ",\"hostBuffer\":" << json_bool(props.caps.host_buffer)
          << ",\"bufferFromHostPtr\":"
          << json_bool(props.caps.buffer_from_host_ptr)
          << ",\"events\":" << json_bool(props.caps.events) << "}}";
    }
  }
  out << "]}";
  return out.str();
}

int init_engine_locked(const char *model_path,
                       const char *runtime_config_json) {
  g_lastEngineError.clear();
  if (!model_path || std::strlen(model_path) == 0) {
    g_lastEngineError = "model path is empty";
    return kStatusInvalidArguments;
  }
  if (!runtime_config_json || std::strlen(runtime_config_json) == 0) {
    g_lastEngineError = "runtime config JSON is empty";
    return kStatusInvalidArguments;
  }

  close_rust_engine_locked();
  ensure_llama_cache_env_locked();
  g_rustBrowserEngine = cogentlm_browser_engine_create();
  if (g_rustBrowserEngine == nullptr) {
    g_lastEngineError = "failed to create Rust browser engine";
    return -1;
  }
  const int init_status =
      cogentlm_browser_engine_load(g_rustBrowserEngine, model_path,
                                   runtime_config_json);
  if (init_status != 0) {
    g_lastEngineError = read_rust_engine_last_error_locked(g_rustBrowserEngine);
    if (g_lastEngineError.empty()) {
      g_lastEngineError = "Rust browser engine returned failure during load";
    }
    close_rust_engine_locked();
    return init_status;
  }
  g_isEngineInitialized = true;
  g_lastEngineError.clear();
  return 0;
}

} // namespace

extern "C" {

using CE_ReadAtCallback =
    int (*)(void *, std::uint64_t, std::uint8_t *, std::uintptr_t);
using CE_OpenShardCallback =
    int (*)(void *, const char *, std::uint16_t, std::uint16_t);
using CE_WriteShardCallback =
    int (*)(void *, const std::uint8_t *, std::uintptr_t);
using CE_CloseShardCallback = int (*)(void *);

int cogentlm_browser_cache_layout(std::uint64_t source_bytes,
                                  int source_bytes_known,
                                  std::uint64_t direct_load_max_bytes,
                                  std::uint64_t shard_max_bytes);
int cogentlm_gguf_plan_split_count(std::uint64_t source_bytes,
                                   std::uint64_t shard_max_bytes,
                                   void *user_data,
                                   CE_ReadAtCallback read_at);
int cogentlm_gguf_split_stream(std::uint64_t source_bytes,
                               const char *output_prefix,
                               std::uint64_t shard_max_bytes, void *user_data,
                               CE_ReadAtCallback read_at,
                               CE_OpenShardCallback open_shard,
                               CE_WriteShardCallback write_shard,
                               CE_CloseShardCallback close_shard);

EMSCRIPTEN_KEEPALIVE
int CE_RustBrowserEngineAbiVersion() {
  return static_cast<int>(cogentlm_browser_engine_abi_version());
}

EMSCRIPTEN_KEEPALIVE
std::uintptr_t CE_RustBrowserEngineCreate() {
  return reinterpret_cast<std::uintptr_t>(cogentlm_browser_engine_create());
}

EMSCRIPTEN_KEEPALIVE
int CE_RustBrowserEngineId(std::uintptr_t engine) {
  if (engine == 0) {
    return 0;
  }
  return static_cast<int>(
      cogentlm_browser_engine_id(reinterpret_cast<void *>(engine)));
}

EMSCRIPTEN_KEEPALIVE
int CE_RustBrowserEngineClose(std::uintptr_t engine) {
  if (engine == 0) {
    return kStatusInvalidArguments;
  }
  return cogentlm_browser_engine_close(reinterpret_cast<void *>(engine));
}

EMSCRIPTEN_KEEPALIVE
int CE_BrowserCacheLayout(double source_bytes, int source_bytes_known,
                          double direct_load_max_bytes,
                          double shard_max_bytes) {
  std::uint64_t source_bytes_u64 = 0;
  std::uint64_t direct_load_max_bytes_u64 = 0;
  std::uint64_t shard_max_bytes_u64 = 0;
  if (!read_size_arg(source_bytes, &source_bytes_u64) ||
      !read_size_arg(direct_load_max_bytes, &direct_load_max_bytes_u64) ||
      !read_size_arg(shard_max_bytes, &shard_max_bytes_u64)) {
    return kStatusInvalidArguments;
  }
  return cogentlm_browser_cache_layout(
      source_bytes_u64, source_bytes_known != 0, direct_load_max_bytes_u64,
      shard_max_bytes_u64);
}

EMSCRIPTEN_KEEPALIVE
int CE_GgufPlanSplitCount(double source_bytes, double shard_max_bytes,
                          void *user_data, CE_ReadAtCallback read_at) {
  std::uint64_t source_bytes_u64 = 0;
  std::uint64_t shard_max_bytes_u64 = 0;
  if (!read_size_arg(source_bytes, &source_bytes_u64) ||
      !read_size_arg(shard_max_bytes, &shard_max_bytes_u64) || !read_at) {
    return kStatusInvalidArguments;
  }
  return cogentlm_gguf_plan_split_count(source_bytes_u64,
                                        shard_max_bytes_u64, user_data,
                                        read_at);
}

EMSCRIPTEN_KEEPALIVE
int CE_GgufSplitStream(double source_bytes, const char *output_prefix,
                       double shard_max_bytes, void *user_data,
                       CE_ReadAtCallback read_at,
                       CE_OpenShardCallback open_shard,
                       CE_WriteShardCallback write_shard,
                       CE_CloseShardCallback close_shard) {
  std::uint64_t source_bytes_u64 = 0;
  std::uint64_t shard_max_bytes_u64 = 0;
  if (!output_prefix || !read_size_arg(source_bytes, &source_bytes_u64) ||
      !read_size_arg(shard_max_bytes, &shard_max_bytes_u64) || !read_at ||
      !open_shard || !write_shard || !close_shard) {
    return kStatusInvalidArguments;
  }
  return cogentlm_gguf_split_stream(source_bytes_u64, output_prefix,
                                    shard_max_bytes_u64, user_data, read_at,
                                    open_shard, write_shard, close_shard);
}

EMSCRIPTEN_KEEPALIVE
char *CE_DetectModelFromGgufBytes(const char *name, const std::uint8_t *bytes,
                                  double bytes_len) {
  std::uint64_t bytes_len_u64 = 0;
  if (!read_size_arg(bytes_len, &bytes_len_u64)) {
    return duplicate_heap_string(
        "{\"ok\":false,\"error\":{\"code\":\"INVALID_GGUF\",\"message\":\"invalid GGUF byte length\"}}");
  }
  const std::string value = take_rust_string(
      cogentlm_detect_model_from_gguf_bytes_json(
          name, bytes, static_cast<std::uintptr_t>(bytes_len_u64)));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
std::uintptr_t CE_Sha256Create() {
  return reinterpret_cast<std::uintptr_t>(cogentlm_sha256_create());
}

EMSCRIPTEN_KEEPALIVE
int CE_Sha256Update(std::uintptr_t hasher, const std::uint8_t *bytes,
                    double bytes_len) {
  std::uint64_t bytes_len_u64 = 0;
  if (!read_size_arg(bytes_len, &bytes_len_u64)) {
    return kStatusInvalidArguments;
  }
  return cogentlm_sha256_update(reinterpret_cast<void *>(hasher), bytes,
                                static_cast<std::uintptr_t>(bytes_len_u64));
}

EMSCRIPTEN_KEEPALIVE
char *CE_Sha256Finalize(std::uintptr_t hasher) {
  const std::string value = take_rust_string(
      cogentlm_sha256_finalize(reinterpret_cast<void *>(hasher)));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
int CE_Sha256Close(std::uintptr_t hasher) {
  return cogentlm_sha256_close(reinterpret_cast<void *>(hasher));
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceCreate(const char *config_json) {
  const std::string value =
      take_rust_string(cogentlm_model_service_create_json(config_json));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
int CE_ModelServiceClose(std::uintptr_t service) {
  return cogentlm_model_service_close(reinterpret_cast<void *>(service));
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceList(std::uintptr_t service) {
  const std::string value =
      take_rust_string(cogentlm_model_service_list_json(
          reinterpret_cast<void *>(service)));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceCurrent(std::uintptr_t service) {
  const std::string value =
      take_rust_string(cogentlm_model_service_current_json(
          reinterpret_cast<void *>(service)));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceManifest(std::uintptr_t service) {
  const std::string value =
      take_rust_string(cogentlm_model_service_manifest_json(
          reinterpret_cast<void *>(service)));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServicePrepareLoad(std::uintptr_t service,
                                 const char *source_json,
                                 const char *options_json) {
  const std::string value = take_rust_string(
      cogentlm_model_service_prepare_load_json(
          reinterpret_cast<void *>(service), source_json, options_json));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceCommitLoad(std::uintptr_t service,
                                const char *commit_json) {
  const std::string value =
      take_rust_string(cogentlm_model_service_commit_load_json(
          reinterpret_cast<void *>(service), commit_json));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceAbortLoad(std::uintptr_t service,
                               const char *error_json) {
  const std::string value =
      take_rust_string(cogentlm_model_service_abort_load_json(
          reinterpret_cast<void *>(service), error_json));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceRemove(std::uintptr_t service, const char *model_id) {
  const std::string value =
      take_rust_string(cogentlm_model_service_remove_json(
          reinterpret_cast<void *>(service), model_id));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceUnload(std::uintptr_t service) {
  const std::string value =
      take_rust_string(cogentlm_model_service_unload_json(
          reinterpret_cast<void *>(service)));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceSnapshot(std::uintptr_t service) {
  const std::string value =
      take_rust_string(cogentlm_model_service_snapshot_json(
          reinterpret_cast<void *>(service)));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceDrainEvents(std::uintptr_t service) {
  const std::string value =
      take_rust_string(cogentlm_model_service_drain_events_json(
          reinterpret_cast<void *>(service)));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceRecordEvent(std::uintptr_t service,
                                 const char *event_type,
                                 const char *patch_json) {
  const std::string value = take_rust_string(
      cogentlm_model_service_record_event_json(
          reinterpret_cast<void *>(service), event_type, patch_json));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
int CE_Init(const char *model_path, const char *runtime_config_json) {
  return init_engine_locked(model_path, runtime_config_json);
}

EMSCRIPTEN_KEEPALIVE
int CE_GetLastEngineErrorSize() {
  return static_cast<int>(g_lastEngineError.size());
}

EMSCRIPTEN_KEEPALIVE
int CE_CopyLastEngineError(char *buffer, std::int32_t capacity) {
  return copy_string_with_nul(g_lastEngineError, buffer, capacity);
}

EMSCRIPTEN_KEEPALIVE
void CE_Close() {
  close_rust_engine_locked();
}

EMSCRIPTEN_KEEPALIVE
char *CE_GetBackendObservabilityJson() {
  return duplicate_heap_string(backend_observability_json_locked().c_str());
}

EMSCRIPTEN_KEEPALIVE
const char *CE_GetMediaMarker() {
  if (!g_isEngineInitialized) {
    return nullptr;
  }
  g_mediaMarkerCache =
      take_rust_string(cogentlm_browser_engine_media_marker(g_rustBrowserEngine));
  return g_mediaMarkerCache.c_str();
}

EMSCRIPTEN_KEEPALIVE
const char *CE_GetChatTemplate() {
  if (!g_isEngineInitialized) {
    return nullptr;
  }
  g_chatTemplateCache =
      take_rust_string(cogentlm_browser_engine_chat_template(g_rustBrowserEngine));
  return g_chatTemplateCache.c_str();
}

EMSCRIPTEN_KEEPALIVE
char *CE_GetBosText() {
  if (!g_isEngineInitialized) {
    return duplicate_heap_string("");
  }
  const std::string value =
      take_rust_string(cogentlm_browser_engine_bos_text(g_rustBrowserEngine));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_GetEosText() {
  if (!g_isEngineInitialized) {
    return duplicate_heap_string("");
  }
  const std::string value =
      take_rust_string(cogentlm_browser_engine_eos_text(g_rustBrowserEngine));
  return duplicate_heap_string(value.c_str());
}

char *CE_PairingValidate(const char *classified_json,
                         const char *explicit_projector_id) {
  const std::string value = take_rust_string(
      cogentlm_pairing_validate_json(classified_json, explicit_projector_id));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
char *CE_ProbeChatBoundaryInfo() {
  if (!g_isEngineInitialized) {
    return duplicate_heap_string("");
  }
  const std::string value = take_rust_string(
      cogentlm_browser_engine_probe_chat_boundary_info(g_rustBrowserEngine));
  return duplicate_heap_string(value.c_str());
}

EMSCRIPTEN_KEEPALIVE
CE_RequestId CE_StartTextRequestWithTokenEmissionMode(
    const char *context_key, const char *prompt, int n_tokens,
    int token_emission_mode, const char *grammar) {
  if (!g_isEngineInitialized || prompt == nullptr ||
      !is_valid_prediction_tokens(n_tokens)) {
    return 0;
  }
  return cogentlm_browser_engine_start_text_request(
      g_rustBrowserEngine, context_key, prompt, n_tokens, token_emission_mode,
      grammar);
}

EMSCRIPTEN_KEEPALIVE
CE_RequestId CE_StartMediaRequestWithTokenEmissionMode(
    const char *context_key, const char *prompt, int n_tokens, int n_images,
    const uint8_t *images_flat_buffer, const int32_t *image_sizes,
    int token_emission_mode, const char *grammar) {
  if (!g_isEngineInitialized || prompt == nullptr ||
      !is_valid_prediction_tokens(n_tokens)) {
    return 0;
  }
  return cogentlm_browser_engine_start_media_request(
      g_rustBrowserEngine, context_key, prompt, n_tokens, n_images,
      images_flat_buffer, image_sizes, token_emission_mode, grammar);
}

EMSCRIPTEN_KEEPALIVE
CE_RequestId CE_StartChatRequestWithTokenEmissionMode(
    const char *context_key, const char *messages_json, int n_tokens,
    int n_images, const uint8_t *images_flat_buffer, const int32_t *image_sizes,
    int token_emission_mode, const char *grammar) {
  if (!g_isEngineInitialized || messages_json == nullptr ||
      !is_valid_prediction_tokens(n_tokens)) {
    return 0;
  }
  return cogentlm_browser_engine_start_chat_request(
      g_rustBrowserEngine, context_key, messages_json, n_tokens, n_images,
      images_flat_buffer, image_sizes, token_emission_mode, grammar);
}

EMSCRIPTEN_KEEPALIVE
int CE_CancelRequest(CE_RequestId request_id) {
  if (!g_isEngineInitialized || request_id == 0) {
    return 0;
  }
  return cogentlm_browser_engine_cancel_request(g_rustBrowserEngine,
                                                request_id);
}

EMSCRIPTEN_KEEPALIVE
int CE_GetRuntimeObservability(CE_RuntimeObservabilityMetrics *out_metrics) {
  if (!g_isEngineInitialized) {
    return -1;
  }
  return cogentlm_browser_engine_runtime_observability(g_rustBrowserEngine,
                                                       out_metrics);
}

EMSCRIPTEN_KEEPALIVE
int CE_RunSchedulerLoop(int32_t max_ticks, int32_t max_completed_responses,
                        int32_t max_emitted_tokens, int32_t max_duration_us,
                        int32_t streaming_active,
                        CE_SchedulerLoopResult *out_result) {
  return cogentlm_browser_engine_run_scheduler_loop(
      g_rustBrowserEngine, max_ticks, max_completed_responses,
      max_emitted_tokens, max_duration_us, streaming_active, out_result);
}

EMSCRIPTEN_KEEPALIVE
int CE_GetCompletedRequestStatus(CE_RequestId request_id) {
  if (!g_isEngineInitialized || request_id == 0) {
    return 4;
  }
  return cogentlm_browser_engine_completed_request_status(g_rustBrowserEngine,
                                                          request_id);
}

EMSCRIPTEN_KEEPALIVE
const uint8_t *CE_GetStreamingBufferPointer() {
  return cogentlm_browser_engine_streaming_buffer_pointer(g_rustBrowserEngine);
}

EMSCRIPTEN_KEEPALIVE
int32_t *CE_GetStreamingBufferUsedAddress() {
  return cogentlm_browser_engine_streaming_buffer_used_address(
      g_rustBrowserEngine);
}

EMSCRIPTEN_KEEPALIVE
int32_t *CE_GetStreamingBufferDropCountAddress() {
  return cogentlm_browser_engine_streaming_buffer_drop_count_address(
      g_rustBrowserEngine);
}

EMSCRIPTEN_KEEPALIVE
int CE_GetCompletedRequestOutputSize(CE_RequestId request_id) {
  return cogentlm_browser_engine_completed_request_output_size(
      g_rustBrowserEngine, request_id);
}

EMSCRIPTEN_KEEPALIVE
int CE_CopyCompletedRequestOutput(CE_RequestId request_id, char *buffer,
                                  int32_t capacity) {
  return cogentlm_browser_engine_copy_completed_request_output(
      g_rustBrowserEngine, request_id, reinterpret_cast<std::uint8_t *>(buffer),
      static_cast<std::uintptr_t>(capacity));
}

EMSCRIPTEN_KEEPALIVE
int CE_GetCompletedRequestErrorSize(CE_RequestId request_id) {
  return cogentlm_browser_engine_completed_request_error_size(
      g_rustBrowserEngine, request_id);
}

EMSCRIPTEN_KEEPALIVE
int CE_CopyCompletedRequestError(CE_RequestId request_id, char *buffer,
                                 int32_t capacity) {
  return cogentlm_browser_engine_copy_completed_request_error(
      g_rustBrowserEngine, request_id, reinterpret_cast<std::uint8_t *>(buffer),
      static_cast<std::uintptr_t>(capacity));
}

EMSCRIPTEN_KEEPALIVE
int CE_GetCompletedRequestRuntimeObservability(
    CE_RequestId request_id, CE_RuntimeObservabilityMetrics *out_metrics) {
  return cogentlm_browser_engine_completed_runtime_observability(
      g_rustBrowserEngine, request_id, out_metrics);
}

EMSCRIPTEN_KEEPALIVE
int CE_ConsumeCompletedRequest(CE_RequestId request_id) {
  return cogentlm_browser_engine_consume_completed_request(g_rustBrowserEngine,
                                                           request_id);
}

EMSCRIPTEN_KEEPALIVE
void CE_FreeString(char *str) {
  if (str) {
    std::free(str);
  }
}

// Called by the Rust browser engine between scheduler bursts. Lets the host
// (worker thread) snapshot the native streaming buffer into the SAB ring so
// the main thread can render tokens mid-flight without a full ccall round
// trip per token. Kept synchronous on purpose; the worker is dedicated to
// inference and the JS hook only does a fast HEAP->SAB copy.
EMSCRIPTEN_KEEPALIVE
void ce_native_yield() {
  // clang-format off
  EM_ASM({
    var fn = Module && Module._ce_yield_drain;
    if (typeof fn === 'function') {
      fn();
    }
  });
  // clang-format on
}

} // extern "C"
