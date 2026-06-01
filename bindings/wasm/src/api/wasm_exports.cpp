#include <emscripten/emscripten.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <sys/stat.h>

#include <nlohmann/json.hpp>

#include "ffi_types.h"

#include "../bridge_shim_c_api.h"
#include "cogent_shim.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"
#include "llama.h"

namespace {

constexpr int kStatusInvalidArguments = -2;
constexpr double kMaxExactInteger = 9007199254740991.0;

bool is_valid_prediction_tokens(int token_count) {
  return token_count > 0;
}

bool g_isEngineInitialized = false;
std::string g_lastEngineError;
std::once_flag g_backendInitOnce;

char *copy_heap_string(const std::string &value) {
  const std::size_t length = value.size();
  char *out = static_cast<char *>(std::malloc(length + 1));
  if (!out) {
    return nullptr;
  }
  std::memcpy(out, value.data(), length);
  out[length] = '\0';
  return out;
}

char *copy_json_error(const char *code, const char *message) {
  nlohmann::ordered_json response = {
      {"ok", false},
      {"error",
       {
           {"code", code},
           {"message", message},
       }},
  };
  return copy_heap_string(response.dump());
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

bool read_pointer_size_arg(double value, std::uintptr_t *out) {
  std::uint64_t parsed = 0;
  if (!read_size_arg(value, &parsed) ||
      parsed > std::numeric_limits<std::uintptr_t>::max()) {
    return false;
  }
  *out = static_cast<std::uintptr_t>(parsed);
  return true;
}

bool read_nonnegative_count(std::int32_t value, std::uintptr_t *out) {
  if (value < 0) {
    return false;
  }
  *out = static_cast<std::uintptr_t>(value);
  return true;
}

void *g_rustBrowserEngine = nullptr;
std::string g_mediaMarkerCache;
std::string g_chatTemplateCache;

std::string take_heap_string(char *value);
std::string read_rust_engine_last_error_locked(const void *engine);
void close_rust_engine_locked();
void ensure_backend_runtime_locked();
void ensure_llama_cache_env_locked();

std::string take_heap_string(char *value) {
  if (value == nullptr) {
    return "";
  }
  std::string out(value);
  std::free(value);
  return out;
}

void close_rust_engine_locked() {
  if (g_rustBrowserEngine != nullptr) {
    cogentlm_browser_engine_close(g_rustBrowserEngine);
    g_rustBrowserEngine = nullptr;
  }
  g_isEngineInitialized = false;
}

void ensure_backend_runtime_locked() {
  std::call_once(g_backendInitOnce, []() {
    llama_backend_init();
    cogent_backend_load_all();
  });
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
  ensure_backend_runtime_locked();

  std::string raw =
      take_heap_string(cogent_backend_observability_json(true));
  nlohmann::ordered_json value =
      nlohmann::ordered_json::parse(raw.empty() ? "{}" : raw, nullptr, false);
  if (value.is_discarded() || !value.is_object()) {
    value = nlohmann::ordered_json::object();
  }
  nlohmann::ordered_json compiled = nlohmann::ordered_json::object();
  if (auto it = value.find("compiled"); it != value.end() && it->is_object()) {
    compiled = *it;
  }

  value["profilingEnabled"] = false;
  if (!value.contains("compiled")) {
    value["compiled"] = compiled;
  }
  value["webgpuCompiled"] = compiled.value("webgpu", false);
  ggml_backend_reg_t webgpu_reg = ggml_backend_reg_by_name(GGML_WEBGPU_NAME);
  value["webgpuRegistered"] = webgpu_reg != nullptr;
  value["webgpuDeviceCount"] =
      webgpu_reg != nullptr ? ggml_backend_reg_dev_count(webgpu_reg) : 0;
  if (!value.contains("gpuOffloadSupported")) {
    value["gpuOffloadSupported"] = llama_supports_gpu_offload();
  }
  value["engineInitialized"] = g_isEngineInitialized;
  if (!value.contains("availableBackends")) {
    value["availableBackends"] = nlohmann::ordered_json::array();
  }
  if (!value.contains("devices")) {
    value["devices"] = nlohmann::ordered_json::array();
  }
  return value.dump();
}

int init_engine_locked(const char *model_path,
                       const char *runtime_config_json) {
  g_lastEngineError.clear();
  if (model_path == nullptr || runtime_config_json == nullptr) {
    g_lastEngineError = "engine init received a null string";
    return kStatusInvalidArguments;
  }

  close_rust_engine_locked();
  ensure_backend_runtime_locked();
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
  return cogentlm_wasm_browser_cache_layout(
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
  return cogentlm_wasm_gguf_plan_split_count(source_bytes_u64,
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
  return cogentlm_wasm_gguf_split_stream(source_bytes_u64, output_prefix,
                                         shard_max_bytes_u64, user_data, read_at,
                                         open_shard, write_shard, close_shard);
}

EMSCRIPTEN_KEEPALIVE
char *CE_DetectModelFromGgufBytes(const char *name, const std::uint8_t *bytes,
                                  double bytes_len) {
  std::uintptr_t bytes_len_usize = 0;
  if (name == nullptr || !read_pointer_size_arg(bytes_len, &bytes_len_usize) ||
      (!bytes && bytes_len_usize > 0)) {
    return copy_json_error("INVALID_GGUF", "invalid GGUF byte length");
  }
  const std::string value = take_heap_string(
      cogentlm_wasm_detect_model_from_gguf_bytes_json(
          name, bytes, bytes_len_usize));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
std::uintptr_t CE_Sha256Create() {
  return reinterpret_cast<std::uintptr_t>(cogentlm_wasm_sha256_create());
}

EMSCRIPTEN_KEEPALIVE
int CE_Sha256Update(std::uintptr_t hasher, const std::uint8_t *bytes,
                    double bytes_len) {
  std::uintptr_t bytes_len_usize = 0;
  if (hasher == 0 || !read_pointer_size_arg(bytes_len, &bytes_len_usize) ||
      (bytes == nullptr && bytes_len_usize > 0)) {
    return kStatusInvalidArguments;
  }
  return cogentlm_wasm_sha256_update(
      reinterpret_cast<void *>(hasher),
      bytes,
      bytes_len_usize);
}

EMSCRIPTEN_KEEPALIVE
char *CE_Sha256Finalize(std::uintptr_t hasher) {
  if (hasher == 0) {
    return nullptr;
  }
  const std::string value = take_heap_string(
      cogentlm_wasm_sha256_finalize(reinterpret_cast<void *>(hasher)));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
int CE_Sha256Close(std::uintptr_t hasher) {
  if (hasher == 0) {
    return kStatusInvalidArguments;
  }
  return cogentlm_wasm_sha256_close(reinterpret_cast<void *>(hasher));
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceCreate(const char *config_json) {
  if (config_json == nullptr) {
    return copy_json_error("INVALID_MODEL_SOURCE",
                           "service config JSON is missing");
  }
  const std::string value =
      take_heap_string(cogentlm_wasm_model_service_create_json(config_json));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
int CE_ModelServiceClose(std::uintptr_t service) {
  return cogentlm_wasm_model_service_close(service);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceList(std::uintptr_t service) {
  const std::string value =
      take_heap_string(cogentlm_wasm_model_service_list_json(service));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceCurrent(std::uintptr_t service) {
  const std::string value =
      take_heap_string(cogentlm_wasm_model_service_current_json(service));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceManifest(std::uintptr_t service) {
  const std::string value =
      take_heap_string(cogentlm_wasm_model_service_manifest_json(service));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServicePrepareLoad(std::uintptr_t service,
                                 const char *source_json,
                                 const char *options_json) {
  if (source_json == nullptr || options_json == nullptr) {
    return copy_json_error("INVALID_MODEL_SOURCE",
                           "load source or options JSON is missing");
  }
  const std::string value = take_heap_string(
      cogentlm_wasm_model_service_prepare_load_json(
          service, source_json, options_json));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceCommitLoad(std::uintptr_t service,
                                const char *commit_json) {
  if (commit_json == nullptr) {
    return copy_json_error("INVALID_MODEL_SOURCE",
                           "load commit JSON is missing");
  }
  const std::string value =
      take_heap_string(cogentlm_wasm_model_service_commit_load_json(
          service, commit_json));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceAbortLoad(std::uintptr_t service,
                               const char *error_json) {
  const std::string value =
      take_heap_string(cogentlm_wasm_model_service_abort_load_json(
          service, error_json));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceRemove(std::uintptr_t service, const char *model_id) {
  if (model_id == nullptr) {
    return copy_json_error("INVALID_MODEL_SOURCE", "model id is missing");
  }
  const std::string value =
      take_heap_string(cogentlm_wasm_model_service_remove_json(
          service, model_id));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceUnload(std::uintptr_t service) {
  const std::string value =
      take_heap_string(cogentlm_wasm_model_service_unload_json(service));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceSnapshot(std::uintptr_t service) {
  const std::string value =
      take_heap_string(cogentlm_wasm_model_service_snapshot_json(service));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceDrainEvents(std::uintptr_t service) {
  const std::string value =
      take_heap_string(cogentlm_wasm_model_service_drain_events_json(service));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ModelServiceRecordEvent(std::uintptr_t service,
                                 const char *event_type,
                                 const char *patch_json) {
  if (event_type == nullptr || patch_json == nullptr) {
    return copy_json_error("INVALID_MODEL_SOURCE",
                           "event type or patch JSON is missing");
  }
  const std::string value = take_heap_string(
      cogentlm_wasm_model_service_record_event_json(
          service, event_type, patch_json));
  return copy_heap_string(value);
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
  return copy_heap_string(backend_observability_json_locked());
}

EMSCRIPTEN_KEEPALIVE
const char *CE_GetMediaMarker() {
  if (!g_isEngineInitialized) {
    return nullptr;
  }
  g_mediaMarkerCache =
      take_heap_string(cogentlm_browser_engine_media_marker(g_rustBrowserEngine));
  return g_mediaMarkerCache.c_str();
}

EMSCRIPTEN_KEEPALIVE
const char *CE_GetChatTemplate() {
  if (!g_isEngineInitialized) {
    return nullptr;
  }
  g_chatTemplateCache =
      take_heap_string(cogentlm_browser_engine_chat_template(g_rustBrowserEngine));
  return g_chatTemplateCache.c_str();
}

EMSCRIPTEN_KEEPALIVE
char *CE_GetBosText() {
  if (!g_isEngineInitialized) {
    return copy_heap_string("");
  }
  const std::string value =
      take_heap_string(cogentlm_browser_engine_bos_text(g_rustBrowserEngine));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_GetEosText() {
  if (!g_isEngineInitialized) {
    return copy_heap_string("");
  }
  const std::string value =
      take_heap_string(cogentlm_browser_engine_eos_text(g_rustBrowserEngine));
  return copy_heap_string(value);
}

char *CE_PairingValidate(const char *classified_json,
                         const char *explicit_projector_id) {
  if (classified_json == nullptr) {
    return copy_json_error("INVALID_MODEL_SOURCE",
                           "classified asset JSON is missing");
  }
  const std::string value = take_heap_string(
      cogentlm_wasm_pairing_validate_json(
          classified_json,
          explicit_projector_id));
  return copy_heap_string(value);
}

EMSCRIPTEN_KEEPALIVE
char *CE_ProbeChatBoundaryInfo() {
  if (!g_isEngineInitialized) {
    return copy_heap_string("");
  }
  const std::string value = take_heap_string(
      cogentlm_browser_engine_probe_chat_boundary_info(g_rustBrowserEngine));
  return copy_heap_string(value);
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
      !is_valid_prediction_tokens(n_tokens) || n_images < 0 ||
      (n_images > 0 && (images_flat_buffer == nullptr || image_sizes == nullptr))) {
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
      !is_valid_prediction_tokens(n_tokens) || n_images < 0 ||
      (n_images > 0 && (images_flat_buffer == nullptr || image_sizes == nullptr))) {
    return 0;
  }
  return cogentlm_browser_engine_start_chat_request(
      g_rustBrowserEngine, context_key, messages_json, n_tokens, n_images,
      images_flat_buffer, image_sizes, token_emission_mode, grammar);
}

EMSCRIPTEN_KEEPALIVE
CE_RequestId CE_StartEmbeddingRequest(const char *context_key,
                                      const char *input, int normalize) {
  if (!g_isEngineInitialized || input == nullptr) {
    return 0;
  }
  return cogentlm_browser_engine_start_embedding_request(
      g_rustBrowserEngine, context_key, input, normalize);
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
  if (!g_isEngineInitialized || out_metrics == nullptr) {
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
  if (!g_isEngineInitialized || out_result == nullptr) {
    return -1;
  }
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
int CE_GetCompletedRequestOutputKind(CE_RequestId request_id) {
  return cogentlm_browser_engine_completed_request_output_kind(
      g_rustBrowserEngine, request_id);
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
  std::uintptr_t buffer_len = 0;
  if (buffer == nullptr || !read_nonnegative_count(capacity, &buffer_len)) {
    return kStatusInvalidArguments;
  }
  return cogentlm_browser_engine_copy_completed_request_output(
      g_rustBrowserEngine, request_id, reinterpret_cast<std::uint8_t *>(buffer),
      buffer_len);
}

EMSCRIPTEN_KEEPALIVE
int CE_GetCompletedRequestEmbeddingLength(CE_RequestId request_id) {
  return cogentlm_browser_engine_completed_request_embedding_length(
      g_rustBrowserEngine, request_id);
}

EMSCRIPTEN_KEEPALIVE
int CE_CopyCompletedRequestEmbedding(CE_RequestId request_id, float *buffer,
                                     int32_t value_count) {
  std::uintptr_t values_len = 0;
  if (buffer == nullptr || !read_nonnegative_count(value_count, &values_len)) {
    return kStatusInvalidArguments;
  }
  return cogentlm_browser_engine_copy_completed_request_embedding(
      g_rustBrowserEngine, request_id, buffer,
      values_len);
}

EMSCRIPTEN_KEEPALIVE
int CE_GetCompletedRequestEmbeddingPooling(CE_RequestId request_id) {
  return cogentlm_browser_engine_completed_request_embedding_pooling(
      g_rustBrowserEngine, request_id);
}

EMSCRIPTEN_KEEPALIVE
int CE_GetCompletedRequestEmbeddingNormalized(CE_RequestId request_id) {
  return cogentlm_browser_engine_completed_request_embedding_normalized(
      g_rustBrowserEngine, request_id);
}

EMSCRIPTEN_KEEPALIVE
int CE_GetCompletedRequestErrorSize(CE_RequestId request_id) {
  return cogentlm_browser_engine_completed_request_error_size(
      g_rustBrowserEngine, request_id);
}

EMSCRIPTEN_KEEPALIVE
int CE_CopyCompletedRequestError(CE_RequestId request_id, char *buffer,
                                 int32_t capacity) {
  std::uintptr_t buffer_len = 0;
  if (buffer == nullptr || !read_nonnegative_count(capacity, &buffer_len)) {
    return kStatusInvalidArguments;
  }
  return cogentlm_browser_engine_copy_completed_request_error(
      g_rustBrowserEngine, request_id, reinterpret_cast<std::uint8_t *>(buffer),
      buffer_len);
}

EMSCRIPTEN_KEEPALIVE
int CE_GetCompletedRequestRuntimeObservability(
    CE_RequestId request_id, CE_RuntimeObservabilityMetrics *out_metrics) {
  if (!g_isEngineInitialized || out_metrics == nullptr) {
    return -1;
  }
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
