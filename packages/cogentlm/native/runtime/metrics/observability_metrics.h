/////////////////////////////////////////////////////////////////////////////////////////////////
//
// observability_metrics.h
//
// - Runtime observability DTOs.
// - These metrics are optional and should stay outside the core inference contract.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

namespace noumena::cogentengine {

struct RuntimeObservabilityMetrics {
  // Latency (User Experience)
  double ttft_ms = 0.0;
  double itl_avg_ms = 0.0;
  double itl_p99_ms = 0.0;
  double e2e_ms = 0.0;

  // Phases (Compute)
  double prefill_ms = 0.0;
  double decode_ms = 0.0;

  // Native (Hardware & Engine)
  // Raw wall-clock `gpu_end - gpu_start` around llama_decode + llama_synchronize.
  // In WebGPU+wasm, llama_synchronize is an event-loop dependency, so this
  // number includes any browser-side wait for the GPU completion microtask
  // to be picked up.  Do NOT pre-subtract FFI/yield costs here — those are
  // reported separately so the consumer can see the decomposition.
  double native_gpu_ms = 0.0;
  double native_sync_ms = 0.0;
  double native_logic_ms = 0.0;

  // Wall-clock from the previous tick's `gpu_end` to this tick's `gpu_start`.
  // This is the worker-thread JS work window between successive syncs:
  // ce_native_yield, the streaming drain hook, scheduler-pump bookkeeping,
  // drainRuntimeEvents, and any postMessage processing.  Contributes directly
  // to event-loop contention seen by the next llama_synchronize.
  double inter_decode_js_ms = 0.0;
  // Subset of inter_decode_js_ms spent suspended inside ce_native_yield()
  // (the JSPI await + the _ce_yield_drain hook).  The remainder is JS pump
  // work outside the wasm-suspend window.
  double yield_wait_ms = 0.0;

  // Counts
  int32_t input_tokens = 0;
  int32_t output_tokens = 0;
  int32_t cache_hits = 0;
};

} // namespace noumena::cogentengine
