export interface RequestObservabilityMetrics {
  /** Time to first token (milliseconds from enqueue to first emission). */
  ttftMs: number;
  /** Average inter-token latency (milliseconds between output tokens). */
  itlAvgMs: number;
  /** Tail inter-token latency (P99/max observed between output tokens). */
  itlP99Ms: number;
  /** End-to-end latency (milliseconds from enqueue to terminal state). */
  e2eMs: number;

  /** Time spent in the prefill (prompt evaluation) phase. */
  prefillMs: number;
  /** Time spent in the decoding (token generation) phase. */
  decodeMs: number;

  /**
   * Raw wall-clock window around llama_decode + llama_synchronize.  In
   * WebGPU+wasm this includes any event-loop wait inside llama_synchronize
   * (the GPU-completion microtask waiting behind queued JS work), so it is
   * NOT a pure GPU-only number — pair it with `interDecodeJsMs` to attribute
   * how much of the gap to native vs. JS-side contention.
   */
  nativeGpuMs: number;
  /** Cumulative time spent in backend synchronization (llama_synchronize). */
  nativeSyncMs: number;
  /** Internal engine logic overhead (scheduling, batching, bookkeeping). */
  nativeLogicMs: number;
  /**
   * Cumulative wall-clock between successive `gpu_end → gpu_start` boundaries.
   * Captures all worker-thread JS work between decodes: ce_native_yield, the
   * streaming-buffer drain hook, scheduler-pump bookkeeping, drainRuntimeEvents,
   * and postMessage processing.  Idle gaps >500ms are excluded (treated as
   * request boundaries).
   */
  interDecodeJsMs: number;
  /**
   * Subset of `interDecodeJsMs` spent suspended inside `ce_native_yield()`
   * (the JSPI await plus the `_ce_yield_drain` hook).  The remainder is JS
   * pump work that ran with wasm fully returned to JS (drainRuntimeEvents,
   * settle logic, postMessage dispatch).
   */
  yieldWaitMs: number;

  /** Total number of tokens processed in the prompt. */
  inputTokens: number;
  /** Total number of tokens generated in the response. */
  outputTokens: number;
  /** Number of tokens reused from KV cache (LCP/Prefix hits). */
  cacheHits: number;
}

export interface RuntimeAggregateObservabilityMetrics extends RequestObservabilityMetrics {}
