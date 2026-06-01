#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "browser_engine_api.h"
#include "gguf_callbacks.h"
#include "cogentlm-wasm/src/bridge.rs.h"

namespace {

using BrowserEngineBox = rust::Box<cogentlm::wasm::BrowserEngine>;
using Sha256HasherBox = rust::Box<cogentlm::wasm::BrowserSha256Hasher>;

constexpr int kStatusOk = 0;
constexpr int kStatusFailure = -1;
constexpr int kStatusInvalidArguments = -2;
constexpr int kCompletedRequestStatusUnknown = 4;

const std::uint8_t kEmptyByte = 0;
const std::int32_t kEmptyInt = 0;

BrowserEngineBox & browser_engine_box(void * engine) {
  return *static_cast<BrowserEngineBox *>(engine);
}

const BrowserEngineBox & browser_engine_box(const void * engine) {
  return *static_cast<const BrowserEngineBox *>(engine);
}

Sha256HasherBox & sha256_box(void * hasher) {
  return *static_cast<Sha256HasherBox *>(hasher);
}

char * copy_to_heap_string(const rust::String & value) {
  const std::string copy = static_cast<std::string>(value);
  char * result = static_cast<char *>(std::malloc(copy.size() + 1));
  if (result == nullptr) {
    return nullptr;
  }
  std::memcpy(result, copy.c_str(), copy.size() + 1);
  return result;
}

char * copy_to_heap_string(const char * value) {
  const std::string copy(value == nullptr ? "" : value);
  char * result = static_cast<char *>(std::malloc(copy.size() + 1));
  if (result == nullptr) {
    return nullptr;
  }
  std::memcpy(result, copy.c_str(), copy.size() + 1);
  return result;
}

char * copy_lifecycle_argument_error(const char * message) {
  std::string value =
      "{\"ok\":false,\"error\":{\"code\":\"INVALID_MODEL_SOURCE\",\"message\":\"";
  value += message;
  value += "\"}}";
  return copy_to_heap_string(value.c_str());
}

char * copy_gguf_argument_error(const char * message) {
  std::string value =
      "{\"ok\":false,\"error\":{\"code\":\"INVALID_GGUF\",\"message\":\"";
  value += message;
  value += "\"}}";
  return copy_to_heap_string(value.c_str());
}

char * copy_pairing_argument_error(const char * message) {
  std::string value =
      "{\"ok\":false,\"error\":{\"code\":\"INVALID_MODEL_SOURCE\",\"message\":\"";
  value += message;
  value += "\"}}";
  return copy_to_heap_string(value.c_str());
}

rust::Str rust_str_or_empty(const char * value) {
  return rust::Str(value == nullptr ? "" : value);
}

bool image_payload_size(const std::int32_t * image_sizes, int image_count, std::size_t * out) {
  if (image_count < 0 || out == nullptr) {
    return false;
  }
  if (image_count == 0) {
    *out = 0;
    return true;
  }
  if (image_sizes == nullptr) {
    return false;
  }
  std::size_t total = 0;
  for (int i = 0; i < image_count; ++i) {
    if (image_sizes[i] <= 0) {
      return false;
    }
    const auto len = static_cast<std::size_t>(image_sizes[i]);
    if (len > std::numeric_limits<std::size_t>::max() - total) {
      return false;
    }
    total += len;
  }
  *out = total;
  return true;
}

rust::Slice<const std::uint8_t> const_u8_slice(
    const std::uint8_t * data,
    std::size_t len) {
  return rust::Slice<const std::uint8_t>(len == 0 ? &kEmptyByte : data, len);
}

rust::Slice<const std::int32_t> const_i32_slice(
    const std::int32_t * data,
    std::size_t len) {
  return rust::Slice<const std::int32_t>(len == 0 ? &kEmptyInt : data, len);
}

void copy_scheduler_loop_result(
    const cogentlm::wasm::BrowserSchedulerLoopResult & source,
    CE_SchedulerLoopResult * out) {
  out->ticks_executed = source.ticks_executed;
  out->progressed_ticks = source.progressed_ticks;
  out->completed_response_count = source.completed_response_count;
  out->emitted_token_count = source.emitted_token_count;
}

void copy_runtime_metrics(
    const cogentlm::wasm::BrowserRuntimeMetrics & source,
    CE_RuntimeObservabilityMetrics * out) {
  out->ttft_ms = source.ttft_ms;
  out->itl_avg_ms = source.itl_avg_ms;
  out->itl_p99_ms = source.itl_p99_ms;
  out->e2e_ms = source.e2e_ms;
  out->prefill_ms = source.prefill_ms;
  out->decode_ms = source.decode_ms;
  out->native_gpu_ms = source.native_gpu_ms;
  out->native_sync_ms = source.native_sync_ms;
  out->native_logic_ms = source.native_logic_ms;
  out->input_tokens = source.input_tokens;
  out->output_tokens = source.output_tokens;
  out->cache_hits = source.cache_hits;
  out->prefill_tokens = source.prefill_tokens;
}

} // namespace

extern "C" {

std::uint32_t cogentlm_browser_engine_abi_version() {
  return cogentlm::wasm::browser_engine_abi_version();
}

void * cogentlm_browser_engine_create() {
  return new BrowserEngineBox(cogentlm::wasm::browser_engine_create());
}

std::uint32_t cogentlm_browser_engine_id(void * engine) {
  if (engine == nullptr) {
    return 0;
  }
  return cogentlm::wasm::browser_engine_id(*browser_engine_box(engine));
}

int cogentlm_browser_engine_load(
    void * engine,
    const char * model_path,
    const char * runtime_config_json) {
  if (engine == nullptr || model_path == nullptr || runtime_config_json == nullptr) {
    return kStatusInvalidArguments;
  }
  return cogentlm::wasm::browser_engine_load(
      *browser_engine_box(engine),
      rust_str_or_empty(model_path),
      rust_str_or_empty(runtime_config_json));
}

int cogentlm_browser_engine_last_error_size(const void * engine) {
  if (engine == nullptr) {
    return 0;
  }
  return cogentlm::wasm::browser_engine_last_error_size(*browser_engine_box(engine));
}

int cogentlm_browser_engine_copy_last_error(
    const void * engine,
    std::uint8_t * buffer,
    std::uintptr_t buffer_len) {
  if (engine == nullptr || buffer == nullptr) {
    return kStatusInvalidArguments;
  }
  return cogentlm::wasm::browser_engine_copy_last_error(
      *browser_engine_box(engine),
      rust::Slice<std::uint8_t>(buffer, buffer_len));
}

int cogentlm_browser_engine_close(void * engine) {
  if (engine == nullptr) {
    return kStatusInvalidArguments;
  }
  delete static_cast<BrowserEngineBox *>(engine);
  return kStatusOk;
}

CE_RequestId cogentlm_browser_engine_start_text_request(
    void * engine,
    const char * context_key,
    const char * prompt,
    int max_tokens,
    int token_emission_mode,
    const char * grammar) {
  if (engine == nullptr || prompt == nullptr || max_tokens <= 0) {
    return 0;
  }
  return cogentlm::wasm::browser_engine_start_text_request(
      *browser_engine_box(engine),
      rust_str_or_empty(context_key),
      rust_str_or_empty(prompt),
      max_tokens,
      token_emission_mode,
      rust_str_or_empty(grammar));
}

CE_RequestId cogentlm_browser_engine_start_media_request(
    void * engine,
    const char * context_key,
    const char * prompt,
    int max_tokens,
    int image_count,
    const std::uint8_t * images_flat_buffer,
    const std::int32_t * image_sizes,
    int token_emission_mode,
    const char * grammar) {
  std::size_t image_bytes = 0;
  if (engine == nullptr || prompt == nullptr || max_tokens <= 0 ||
      !image_payload_size(image_sizes, image_count, &image_bytes) ||
      (image_bytes > 0 && images_flat_buffer == nullptr)) {
    return 0;
  }
  return cogentlm::wasm::browser_engine_start_media_request(
      *browser_engine_box(engine),
      rust_str_or_empty(context_key),
      rust_str_or_empty(prompt),
      max_tokens,
      const_u8_slice(images_flat_buffer, image_bytes),
      const_i32_slice(image_sizes, static_cast<std::size_t>(image_count)),
      token_emission_mode,
      rust_str_or_empty(grammar));
}

CE_RequestId cogentlm_browser_engine_start_chat_request(
    void * engine,
    const char * context_key,
    const char * messages_json,
    int max_tokens,
    int image_count,
    const std::uint8_t * images_flat_buffer,
    const std::int32_t * image_sizes,
    int token_emission_mode,
    const char * grammar) {
  std::size_t image_bytes = 0;
  if (engine == nullptr || messages_json == nullptr || max_tokens <= 0 ||
      !image_payload_size(image_sizes, image_count, &image_bytes) ||
      (image_bytes > 0 && images_flat_buffer == nullptr)) {
    return 0;
  }
  return cogentlm::wasm::browser_engine_start_chat_request(
      *browser_engine_box(engine),
      rust_str_or_empty(context_key),
      rust_str_or_empty(messages_json),
      max_tokens,
      const_u8_slice(images_flat_buffer, image_bytes),
      const_i32_slice(image_sizes, static_cast<std::size_t>(image_count)),
      token_emission_mode,
      rust_str_or_empty(grammar));
}

CE_RequestId cogentlm_browser_engine_start_embedding_request(
    void * engine,
    const char * context_key,
    const char * input,
    int normalize) {
  if (engine == nullptr || input == nullptr) {
    return 0;
  }
  return cogentlm::wasm::browser_engine_start_embedding_request(
      *browser_engine_box(engine),
      rust_str_or_empty(context_key),
      rust_str_or_empty(input),
      normalize);
}

int cogentlm_browser_engine_cancel_request(void * engine, CE_RequestId request_id) {
  if (engine == nullptr || request_id == 0) {
    return 0;
  }
  return cogentlm::wasm::browser_engine_cancel_request(
      *browser_engine_box(engine),
      request_id);
}

int cogentlm_browser_engine_run_scheduler_loop(
    void * engine,
    int max_ticks,
    int max_completed_responses,
    int max_emitted_tokens,
    int max_duration_us,
    int streaming_active,
    CE_SchedulerLoopResult * out_result) {
  if (engine == nullptr || out_result == nullptr) {
    return kStatusInvalidArguments;
  }
  cogentlm::wasm::BrowserSchedulerLoopResult result{};
  const int status = cogentlm::wasm::browser_engine_run_scheduler_loop(
      *browser_engine_box(engine),
      max_ticks,
      max_completed_responses,
      max_emitted_tokens,
      max_duration_us,
      streaming_active != 0,
      result);
  copy_scheduler_loop_result(result, out_result);
  return status;
}

int cogentlm_browser_engine_completed_request_status(
    const void * engine,
    CE_RequestId request_id) {
  if (engine == nullptr || request_id == 0) {
    return kCompletedRequestStatusUnknown;
  }
  return cogentlm::wasm::browser_engine_completed_request_status(
      *browser_engine_box(engine),
      request_id);
}

int cogentlm_browser_engine_completed_request_output_kind(
    const void * engine,
    CE_RequestId request_id) {
  if (engine == nullptr || request_id == 0) {
    return kStatusFailure;
  }
  return cogentlm::wasm::browser_engine_completed_request_output_kind(
      *browser_engine_box(engine),
      request_id);
}

int cogentlm_browser_engine_completed_request_output_size(
    const void * engine,
    CE_RequestId request_id) {
  if (engine == nullptr || request_id == 0) {
    return kStatusFailure;
  }
  return cogentlm::wasm::browser_engine_completed_request_output_size(
      *browser_engine_box(engine),
      request_id);
}

int cogentlm_browser_engine_copy_completed_request_output(
    const void * engine,
    CE_RequestId request_id,
    std::uint8_t * buffer,
    std::uintptr_t buffer_len) {
  if (engine == nullptr || request_id == 0 || buffer == nullptr) {
    return kStatusInvalidArguments;
  }
  return cogentlm::wasm::browser_engine_copy_completed_request_output(
      *browser_engine_box(engine),
      request_id,
      rust::Slice<std::uint8_t>(buffer, buffer_len));
}

int cogentlm_browser_engine_completed_request_embedding_length(
    const void * engine,
    CE_RequestId request_id) {
  if (engine == nullptr || request_id == 0) {
    return kStatusFailure;
  }
  return cogentlm::wasm::browser_engine_completed_request_embedding_length(
      *browser_engine_box(engine),
      request_id);
}

int cogentlm_browser_engine_copy_completed_request_embedding(
    const void * engine,
    CE_RequestId request_id,
    float * buffer,
    std::uintptr_t value_count) {
  if (engine == nullptr || request_id == 0 || buffer == nullptr) {
    return kStatusInvalidArguments;
  }
  return cogentlm::wasm::browser_engine_copy_completed_request_embedding(
      *browser_engine_box(engine),
      request_id,
      rust::Slice<float>(buffer, value_count));
}

int cogentlm_browser_engine_completed_request_embedding_pooling(
    const void * engine,
    CE_RequestId request_id) {
  if (engine == nullptr || request_id == 0) {
    return kStatusFailure;
  }
  return cogentlm::wasm::browser_engine_completed_request_embedding_pooling(
      *browser_engine_box(engine),
      request_id);
}

int cogentlm_browser_engine_completed_request_embedding_normalized(
    const void * engine,
    CE_RequestId request_id) {
  if (engine == nullptr || request_id == 0) {
    return kStatusFailure;
  }
  return cogentlm::wasm::browser_engine_completed_request_embedding_normalized(
      *browser_engine_box(engine),
      request_id);
}

int cogentlm_browser_engine_completed_request_error_size(
    const void * engine,
    CE_RequestId request_id) {
  if (engine == nullptr || request_id == 0) {
    return kStatusFailure;
  }
  return cogentlm::wasm::browser_engine_completed_request_error_size(
      *browser_engine_box(engine),
      request_id);
}

int cogentlm_browser_engine_copy_completed_request_error(
    const void * engine,
    CE_RequestId request_id,
    std::uint8_t * buffer,
    std::uintptr_t buffer_len) {
  if (engine == nullptr || request_id == 0 || buffer == nullptr) {
    return kStatusInvalidArguments;
  }
  return cogentlm::wasm::browser_engine_copy_completed_request_error(
      *browser_engine_box(engine),
      request_id,
      rust::Slice<std::uint8_t>(buffer, buffer_len));
}

int cogentlm_browser_engine_consume_completed_request(
    void * engine,
    CE_RequestId request_id) {
  if (engine == nullptr || request_id == 0) {
    return 0;
  }
  return cogentlm::wasm::browser_engine_consume_completed_request(
      *browser_engine_box(engine),
      request_id);
}

int cogentlm_browser_engine_runtime_observability(
    const void * engine,
    CE_RuntimeObservabilityMetrics * out_metrics) {
  if (engine == nullptr || out_metrics == nullptr) {
    return kStatusInvalidArguments;
  }
  cogentlm::wasm::BrowserRuntimeMetrics metrics{};
  const int status = cogentlm::wasm::browser_engine_runtime_observability(
      *browser_engine_box(engine),
      metrics);
  copy_runtime_metrics(metrics, out_metrics);
  return status;
}

int cogentlm_browser_engine_completed_runtime_observability(
    const void * engine,
    CE_RequestId request_id,
    CE_RuntimeObservabilityMetrics * out_metrics) {
  if (engine == nullptr || request_id == 0 || out_metrics == nullptr) {
    return kStatusInvalidArguments;
  }
  cogentlm::wasm::BrowserRuntimeMetrics metrics{};
  const int status = cogentlm::wasm::browser_engine_completed_runtime_observability(
      *browser_engine_box(engine),
      request_id,
      metrics);
  copy_runtime_metrics(metrics, out_metrics);
  return status;
}

std::uint8_t * cogentlm_browser_engine_streaming_buffer_pointer(void * engine) {
  if (engine == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<std::uint8_t *>(
      cogentlm::wasm::browser_engine_streaming_buffer_pointer(*browser_engine_box(engine)));
}

std::int32_t * cogentlm_browser_engine_streaming_buffer_used_address(void * engine) {
  if (engine == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<std::int32_t *>(
      cogentlm::wasm::browser_engine_streaming_buffer_used_address(*browser_engine_box(engine)));
}

std::int32_t * cogentlm_browser_engine_streaming_buffer_drop_count_address(void * engine) {
  if (engine == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<std::int32_t *>(
      cogentlm::wasm::browser_engine_streaming_buffer_drop_count_address(
          *browser_engine_box(engine)));
}

char * cogentlm_browser_engine_media_marker(const void * engine) {
  if (engine == nullptr) {
    return copy_to_heap_string("");
  }
  return copy_to_heap_string(
      cogentlm::wasm::browser_engine_media_marker(*browser_engine_box(engine)));
}

char * cogentlm_browser_engine_chat_template(const void * engine) {
  if (engine == nullptr) {
    return copy_to_heap_string("");
  }
  return copy_to_heap_string(
      cogentlm::wasm::browser_engine_chat_template(*browser_engine_box(engine)));
}

char * cogentlm_browser_engine_bos_text(const void * engine) {
  if (engine == nullptr) {
    return copy_to_heap_string("");
  }
  return copy_to_heap_string(
      cogentlm::wasm::browser_engine_bos_text(*browser_engine_box(engine)));
}

char * cogentlm_browser_engine_eos_text(const void * engine) {
  if (engine == nullptr) {
    return copy_to_heap_string("");
  }
  return copy_to_heap_string(
      cogentlm::wasm::browser_engine_eos_text(*browser_engine_box(engine)));
}

char * cogentlm_browser_engine_probe_chat_boundary_info(const void * engine) {
  if (engine == nullptr) {
    return copy_to_heap_string("");
  }
  return copy_to_heap_string(
      cogentlm::wasm::browser_engine_probe_chat_boundary_info(*browser_engine_box(engine)));
}

int cogentlm_wasm_browser_cache_layout(
    std::uint64_t source_bytes,
    bool source_bytes_known,
    std::uint64_t direct_load_max_bytes,
    std::uint64_t shard_max_bytes) {
  return cogentlm::wasm::browser_cache_layout(
      source_bytes,
      source_bytes_known,
      direct_load_max_bytes,
      shard_max_bytes);
}

int cogentlm_wasm_gguf_plan_split_count(
    std::uint64_t source_bytes,
    std::uint64_t shard_max_bytes,
    void * user_data,
    cogentlm::wasm::GgufReadAtCallback read_at) {
  if (read_at == nullptr) {
    return kStatusInvalidArguments;
  }
  cogentlm::wasm::GgufReadAt source(user_data, read_at);
  return cogentlm::wasm::gguf_plan_split_count(source_bytes, shard_max_bytes, source);
}

int cogentlm_wasm_gguf_split_stream(
    std::uint64_t source_bytes,
    const char * output_prefix,
    std::uint64_t shard_max_bytes,
    void * user_data,
    cogentlm::wasm::GgufReadAtCallback read_at,
    cogentlm::wasm::GgufOpenShardCallback open_shard,
    cogentlm::wasm::GgufWriteShardCallback write_shard,
    cogentlm::wasm::GgufCloseShardCallback close_shard) {
  if (output_prefix == nullptr || read_at == nullptr || open_shard == nullptr ||
      write_shard == nullptr || close_shard == nullptr) {
    return kStatusInvalidArguments;
  }
  cogentlm::wasm::GgufReadAt source(user_data, read_at);
  cogentlm::wasm::GgufShardSink sink(user_data, open_shard, write_shard, close_shard);
  return cogentlm::wasm::gguf_split_stream(
      source_bytes,
      rust_str_or_empty(output_prefix),
      shard_max_bytes,
      source,
      sink);
}

char * cogentlm_wasm_detect_model_from_gguf_bytes_json(
    const char * name,
    const std::uint8_t * bytes,
    std::uintptr_t bytes_len) {
  if (bytes_len > 0 && bytes == nullptr) {
    return copy_gguf_argument_error("invalid GGUF byte length");
  }
  return copy_to_heap_string(cogentlm::wasm::detect_model_from_gguf_bytes_json(
      rust_str_or_empty(name),
      const_u8_slice(bytes, bytes_len)));
}

void * cogentlm_wasm_sha256_create() {
  return new Sha256HasherBox(cogentlm::wasm::sha256_create());
}

int cogentlm_wasm_sha256_update(
    void * hasher,
    const std::uint8_t * bytes,
    std::uintptr_t bytes_len) {
  if (hasher == nullptr || (bytes_len > 0 && bytes == nullptr)) {
    return kStatusInvalidArguments;
  }
  cogentlm::wasm::sha256_update(*sha256_box(hasher), const_u8_slice(bytes, bytes_len));
  return kStatusOk;
}

char * cogentlm_wasm_sha256_finalize(void * hasher) {
  if (hasher == nullptr) {
    return nullptr;
  }
  auto * box = static_cast<Sha256HasherBox *>(hasher);
  const rust::String digest = cogentlm::wasm::sha256_finalize(std::move(*box));
  delete box;
  return copy_to_heap_string(digest);
}

int cogentlm_wasm_sha256_close(void * hasher) {
  if (hasher == nullptr) {
    return 0;
  }
  delete static_cast<Sha256HasherBox *>(hasher);
  return 1;
}

#if defined(__EMSCRIPTEN__)

char * cogentlm_wasm_pairing_validate_json(
    const char * classified_json,
    const char * explicit_projector_id) {
  if (classified_json == nullptr) {
    return copy_pairing_argument_error("classified asset JSON is missing");
  }
  return copy_to_heap_string(cogentlm::wasm::pairing_validate_json(
      rust_str_or_empty(classified_json),
      rust_str_or_empty(explicit_projector_id)));
}

char * cogentlm_wasm_model_service_create_json(const char * config_json) {
  if (config_json == nullptr) {
    return copy_lifecycle_argument_error("service config JSON is missing");
  }
  return copy_to_heap_string(
      cogentlm::wasm::model_service_create_json(rust_str_or_empty(config_json)));
}

int cogentlm_wasm_model_service_close(std::uintptr_t service) {
  return cogentlm::wasm::model_service_close(service);
}

char * cogentlm_wasm_model_service_list_json(std::uintptr_t service) {
  return copy_to_heap_string(cogentlm::wasm::model_service_list_json(service));
}

char * cogentlm_wasm_model_service_current_json(std::uintptr_t service) {
  return copy_to_heap_string(cogentlm::wasm::model_service_current_json(service));
}

char * cogentlm_wasm_model_service_manifest_json(std::uintptr_t service) {
  return copy_to_heap_string(cogentlm::wasm::model_service_manifest_json(service));
}

char * cogentlm_wasm_model_service_prepare_load_json(
    std::uintptr_t service,
    const char * source_json,
    const char * options_json) {
  if (source_json == nullptr || options_json == nullptr) {
    return copy_lifecycle_argument_error("load source or options JSON is missing");
  }
  return copy_to_heap_string(cogentlm::wasm::model_service_prepare_load_json(
      service,
      rust_str_or_empty(source_json),
      rust_str_or_empty(options_json)));
}

char * cogentlm_wasm_model_service_commit_load_json(
    std::uintptr_t service,
    const char * commit_json) {
  if (commit_json == nullptr) {
    return copy_lifecycle_argument_error("load commit JSON is missing");
  }
  return copy_to_heap_string(cogentlm::wasm::model_service_commit_load_json(
      service,
      rust_str_or_empty(commit_json)));
}

char * cogentlm_wasm_model_service_abort_load_json(
    std::uintptr_t service,
    const char * error_json) {
  return copy_to_heap_string(cogentlm::wasm::model_service_abort_load_json(
      service,
      rust_str_or_empty(error_json)));
}

char * cogentlm_wasm_model_service_remove_json(
    std::uintptr_t service,
    const char * model_id) {
  if (model_id == nullptr) {
    return copy_lifecycle_argument_error("model id is missing");
  }
  return copy_to_heap_string(cogentlm::wasm::model_service_remove_json(
      service,
      rust_str_or_empty(model_id)));
}

char * cogentlm_wasm_model_service_unload_json(std::uintptr_t service) {
  return copy_to_heap_string(cogentlm::wasm::model_service_unload_json(service));
}

char * cogentlm_wasm_model_service_snapshot_json(std::uintptr_t service) {
  return copy_to_heap_string(cogentlm::wasm::model_service_snapshot_json(service));
}

char * cogentlm_wasm_model_service_drain_events_json(std::uintptr_t service) {
  return copy_to_heap_string(cogentlm::wasm::model_service_drain_events_json(service));
}

char * cogentlm_wasm_model_service_record_event_json(
    std::uintptr_t service,
    const char * event_type,
    const char * patch_json) {
  if (event_type == nullptr || patch_json == nullptr) {
    return copy_lifecycle_argument_error("event type or patch JSON is missing");
  }
  return copy_to_heap_string(cogentlm::wasm::model_service_record_event_json(
      service,
      rust_str_or_empty(event_type),
      rust_str_or_empty(patch_json)));
}

#endif

}
