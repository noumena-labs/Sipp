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

typedef enum CE_TokenEmissionMode {
  CE_TOKEN_EMISSION_NONE = 0,
  // Native appends to the streaming buffer; JS drains via the SAB ring on
  // each ce_native_yield.  See request_queue.h for the wire format.
  CE_TOKEN_EMISSION_STREAMING_BUFFER = 1,
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
  // Latency (User Experience)
  double ttft_ms;
  double itl_avg_ms;
  double itl_p99_ms;
  double e2e_ms;

  // Phases (Compute)
  double prefill_ms;
  double decode_ms;

  // Native (Hardware & Engine).  native_gpu_ms is raw decode+sync wall time;
  // in WebGPU+wasm that window includes any event-loop wait inside
  // llama_synchronize for the GPU-completion microtask.
  double native_gpu_ms;
  double native_sync_ms;
  double native_logic_ms;

  // Counts
  int32_t input_tokens;
  int32_t output_tokens;
  int32_t cache_hits;
  int32_t prefill_tokens;
} CE_RuntimeObservabilityMetrics;

typedef struct CE_SchedulerLoopResult {
  int32_t ticks_executed;
  int32_t progressed_ticks;
  int32_t completed_response_count;
  int32_t emitted_token_count;
} CE_SchedulerLoopResult;

#ifdef __cplusplus
static_assert(sizeof(CE_RequestId) == 4,
              "CE_RequestId must stay 32-bit for JS/Wasm FFI calls.");
static_assert(sizeof(CE_RuntimeObservabilityMetrics) == 88,
              "CE_RuntimeObservabilityMetrics layout changed. Update the TS FFI reader.");
static_assert(sizeof(CE_SchedulerLoopResult) == 16,
              "CE_SchedulerLoopResult layout changed. Update the TS FFI reader.");
#endif
