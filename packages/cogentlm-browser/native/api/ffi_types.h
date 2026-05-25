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
  // Rust appends to the streaming buffer; JS drains it through the browser
  // scheduler pump.
  CE_TOKEN_EMISSION_STREAMING_BUFFER = 1,
} CE_TokenEmissionMode;

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
