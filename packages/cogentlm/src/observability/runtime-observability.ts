export interface RequestObservabilityMetrics {
  /**
   * Time to first token: enqueue → first sampled token.  Sampled at the
   * moment llama_sampler_sample produces the first token, not at the time
   * the JS consumer sees it (the JS-side delivery is unbounded and consumer-
   * controlled).
   */
  ttftMs: number;
  /** Average inter-token latency (ms between consecutive emitted tokens). */
  itlAvgMs: number;
  /**
   * Worst-observed inter-token latency.  Field is named `p99` for backwards
   * compatibility; the implementation is `max(itl)` since the per-request
   * sample count is small enough (typically ≤ outputTokens) that max is a
   * faithful tail-latency proxy.
   */
  itlP99Ms: number;
  /** End-to-end latency: enqueue → completion. */
  e2eMs: number;

  /** Wall-clock summed over ticks where this request had a prefill contribution. */
  prefillMs: number;
  /** Wall-clock summed over ticks where this request had a decode contribution. */
  decodeMs: number;

  /**
   * Raw wall-clock window around llama_decode + llama_synchronize.  In
   * WebGPU+wasm this includes any event-loop wait inside llama_synchronize
   * (the GPU-completion microtask waiting behind queued JS work).
   */
  nativeGpuMs: number;
  /** Cumulative time spent in backend synchronization (llama_synchronize). */
  nativeSyncMs: number;
  /** Internal engine logic overhead (scheduling, batching, bookkeeping). */
  nativeLogicMs: number;

  /** Total number of tokens processed in the prompt. */
  inputTokens: number;
  /** Total number of tokens generated in the response. */
  outputTokens: number;
  /** Number of tokens reused from KV cache (LCP / prefix hits). */
  cacheHits: number;
}

export interface RuntimeAggregateObservabilityMetrics extends RequestObservabilityMetrics {}
