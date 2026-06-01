#pragma once

#include <cstdint>

#include "ffi_types.h"

extern "C" {

std::uint32_t cogentlm_browser_engine_abi_version();
void * cogentlm_browser_engine_create();
std::uint32_t cogentlm_browser_engine_id(void * engine);
int cogentlm_browser_engine_load(
    void * engine,
    const char * model_path,
    const char * runtime_config_json);
int cogentlm_browser_engine_last_error_size(const void * engine);
int cogentlm_browser_engine_copy_last_error(
    const void * engine,
    std::uint8_t * buffer,
    std::uintptr_t buffer_len);
int cogentlm_browser_engine_close(void * engine);
CE_RequestId cogentlm_browser_engine_start_text_request(
    void * engine,
    const char * context_key,
    const char * prompt,
    int max_tokens,
    int token_emission_mode,
    const char * grammar);
CE_RequestId cogentlm_browser_engine_start_media_request(
    void * engine,
    const char * context_key,
    const char * prompt,
    int max_tokens,
    int image_count,
    const std::uint8_t * images_flat_buffer,
    const std::int32_t * image_sizes,
    int token_emission_mode,
    const char * grammar);
CE_RequestId cogentlm_browser_engine_start_chat_request(
    void * engine,
    const char * context_key,
    const char * messages_json,
    int max_tokens,
    int image_count,
    const std::uint8_t * images_flat_buffer,
    const std::int32_t * image_sizes,
    int token_emission_mode,
    const char * grammar);
CE_RequestId cogentlm_browser_engine_start_embedding_request(
    void * engine,
    const char * context_key,
    const char * input,
    int normalize);
int cogentlm_browser_engine_cancel_request(void * engine, CE_RequestId request_id);
int cogentlm_browser_engine_run_scheduler_loop(
    void * engine,
    int max_ticks,
    int max_completed_responses,
    int max_emitted_tokens,
    int max_duration_us,
    int streaming_active,
    CE_SchedulerLoopResult * out_result);
int cogentlm_browser_engine_completed_request_status(
    const void * engine,
    CE_RequestId request_id);
int cogentlm_browser_engine_completed_request_output_kind(
    const void * engine,
    CE_RequestId request_id);
int cogentlm_browser_engine_completed_request_output_size(
    const void * engine,
    CE_RequestId request_id);
int cogentlm_browser_engine_copy_completed_request_output(
    const void * engine,
    CE_RequestId request_id,
    std::uint8_t * buffer,
    std::uintptr_t buffer_len);
int cogentlm_browser_engine_completed_request_embedding_length(
    const void * engine,
    CE_RequestId request_id);
int cogentlm_browser_engine_copy_completed_request_embedding(
    const void * engine,
    CE_RequestId request_id,
    float * buffer,
    std::uintptr_t value_count);
int cogentlm_browser_engine_completed_request_embedding_pooling(
    const void * engine,
    CE_RequestId request_id);
int cogentlm_browser_engine_completed_request_embedding_normalized(
    const void * engine,
    CE_RequestId request_id);
int cogentlm_browser_engine_completed_request_error_size(
    const void * engine,
    CE_RequestId request_id);
int cogentlm_browser_engine_copy_completed_request_error(
    const void * engine,
    CE_RequestId request_id,
    std::uint8_t * buffer,
    std::uintptr_t buffer_len);
int cogentlm_browser_engine_consume_completed_request(void * engine, CE_RequestId request_id);
int cogentlm_browser_engine_runtime_observability(
    const void * engine,
    CE_RuntimeObservabilityMetrics * out_metrics);
int cogentlm_browser_engine_completed_runtime_observability(
    const void * engine,
    CE_RequestId request_id,
    CE_RuntimeObservabilityMetrics * out_metrics);
std::uint8_t * cogentlm_browser_engine_streaming_buffer_pointer(void * engine);
std::int32_t * cogentlm_browser_engine_streaming_buffer_used_address(void * engine);
std::int32_t * cogentlm_browser_engine_streaming_buffer_drop_count_address(void * engine);
char * cogentlm_browser_engine_media_marker(const void * engine);
char * cogentlm_browser_engine_chat_template(const void * engine);
char * cogentlm_browser_engine_bos_text(const void * engine);
char * cogentlm_browser_engine_eos_text(const void * engine);
char * cogentlm_browser_engine_probe_chat_boundary_info(const void * engine);
int cogentlm_wasm_browser_cache_layout(
    std::uint64_t source_bytes,
    bool source_bytes_known,
    std::uint64_t direct_load_max_bytes,
    std::uint64_t shard_max_bytes);
int cogentlm_wasm_gguf_plan_split_count(
    std::uint64_t source_bytes,
    std::uint64_t shard_max_bytes,
    void * user_data,
    CE_ReadAtCallback read_at);
int cogentlm_wasm_gguf_split_stream(
    std::uint64_t source_bytes,
    const char * output_prefix,
    std::uint64_t shard_max_bytes,
    void * user_data,
    CE_ReadAtCallback read_at,
    CE_OpenShardCallback open_shard,
    CE_WriteShardCallback write_shard,
    CE_CloseShardCallback close_shard);
char * cogentlm_wasm_detect_model_from_gguf_bytes_json(
    const char * name,
    const std::uint8_t * bytes,
    std::uintptr_t bytes_len);
char * cogentlm_wasm_pairing_validate_json(
    const char * classified_json,
    const char * explicit_projector_id);
void * cogentlm_wasm_sha256_create();
int cogentlm_wasm_sha256_update(
    void * hasher,
    const std::uint8_t * bytes,
    std::uintptr_t bytes_len);
char * cogentlm_wasm_sha256_finalize(void * hasher);
int cogentlm_wasm_sha256_close(void * hasher);
char * cogentlm_wasm_model_service_create_json(const char * config_json);
int cogentlm_wasm_model_service_close(std::uintptr_t service);
char * cogentlm_wasm_model_service_list_json(std::uintptr_t service);
char * cogentlm_wasm_model_service_current_json(std::uintptr_t service);
char * cogentlm_wasm_model_service_manifest_json(std::uintptr_t service);
char * cogentlm_wasm_model_service_prepare_load_json(
    std::uintptr_t service,
    const char * source_json,
    const char * options_json);
char * cogentlm_wasm_model_service_commit_load_json(
    std::uintptr_t service,
    const char * commit_json);
char * cogentlm_wasm_model_service_abort_load_json(
    std::uintptr_t service,
    const char * error_json);
char * cogentlm_wasm_model_service_remove_json(
    std::uintptr_t service,
    const char * model_id);
char * cogentlm_wasm_model_service_unload_json(std::uintptr_t service);
char * cogentlm_wasm_model_service_snapshot_json(std::uintptr_t service);
char * cogentlm_wasm_model_service_drain_events_json(std::uintptr_t service);
char * cogentlm_wasm_model_service_record_event_json(
    std::uintptr_t service,
    const char * event_type,
    const char * patch_json);

}
