/////////////////////////////////////////////////////////////////////////////////////////////////
//
// ffi_types.h
//
// - Minimal FFI surface for inference-only callbacks.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <stdint.h>

// JS/Wasm interop calls these functions through `ccall`, so request ids must
// stay in a JS-safe scalar ABI. Do not widen this to uint64_t without also
// changing the exported calling convention.
typedef uint32_t CE_RequestId;
typedef int32_t (*CE_TokenCallback)(const char *token_piece,
                                    int32_t token_length);

typedef enum CE_TokenEmissionMode {
  CE_TOKEN_EMISSION_NONE = 0,
  CE_TOKEN_EMISSION_RUNTIME_EVENTS = 1,
  CE_TOKEN_EMISSION_DIRECT_CALLBACK = 2,
} CE_TokenEmissionMode;

typedef struct CE_InitConfig {
  int32_t n_ctx;
  int32_t n_batch;
  int32_t n_ubatch;
  int32_t n_seq_max;
  int32_t n_threads;
  int32_t n_threads_batch;
  int32_t gpu_layers;
  int32_t flash_attention;
  int32_t kv_unified;
  int32_t max_cached_sessions;
  int32_t retained_prefix_tokens;
  int32_t prefill_chunk_size;
  int32_t prefix_cache_interval_tokens;
  int32_t max_prefix_cache_entries;
  int32_t scheduler_policy;
  int32_t decode_token_reserve;
  int32_t adaptive_prefill_chunking;
  int32_t enable_runtime_observability;
  int32_t enable_backend_profiling;
  const char *mmproj_path;
  int32_t multimodal_use_gpu;
  int32_t image_min_tokens;
  int32_t image_max_tokens;
  int32_t sampling_repeat_last_n;
  float sampling_repeat_penalty;
  float sampling_frequency_penalty;
  float sampling_presence_penalty;
  int32_t sampling_top_k;
  float sampling_top_p;
  float sampling_min_p;
  float sampling_temperature;
  int32_t sampling_seed;
} CE_InitConfig;

typedef struct CE_RuntimeObservabilityMetrics {
  double total_ms;
  double prompt_eval_ms;
  double decode_eval_ms;
  double sample_ms;
  double queue_delay_ms;
  double ttft_ms;
  double mean_itl_ms;
  double tail_itl_ms;
  double e2e_ms;

  int32_t input_token_count;
  int32_t prompt_eval_tokens;
  int32_t decode_eval_count;
  int32_t sample_count;
  int32_t output_token_count;
  int32_t first_sampled_token_id;
  int32_t batch_participation_count;

} CE_RuntimeObservabilityMetrics;

typedef struct CE_SchedulerBurstResult {
  int32_t ticks_executed;
  int32_t progressed_ticks;
  int32_t completed_response_count;
  int32_t emitted_token_count;
} CE_SchedulerBurstResult;

typedef struct CE_RuntimeEvent {
  CE_RequestId request_id;
  int32_t kind;
  int32_t status;
  int32_t text_offset;
  int32_t text_length;
} CE_RuntimeEvent;

typedef struct CE_RuntimeEventDrainResult {
  int32_t event_count;
  int32_t text_bytes;
} CE_RuntimeEventDrainResult;

#ifdef __cplusplus
static_assert(sizeof(CE_RequestId) == 4,
              "CE_RequestId must stay 32-bit for JS/Wasm FFI calls.");
static_assert(sizeof(CE_RuntimeObservabilityMetrics) == 104,
              "CE_RuntimeObservabilityMetrics layout changed. Update the TS FFI reader.");
static_assert(sizeof(CE_SchedulerBurstResult) == 16,
              "CE_SchedulerBurstResult layout changed. Update the TS FFI reader.");
static_assert(sizeof(CE_RuntimeEvent) == 20,
              "CE_RuntimeEvent layout changed. Update the TS FFI reader.");
static_assert(sizeof(CE_RuntimeEventDrainResult) == 8,
              "CE_RuntimeEventDrainResult layout changed. Update the TS FFI reader.");
#endif
