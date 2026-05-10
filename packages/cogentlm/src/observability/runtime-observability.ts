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

  /** Cumulative wall time spent in GPU compute (llama_decode). */
  nativeGpuMs: number;
  /** Cumulative time spent in backend synchronization (llama_synchronize). */
  nativeSyncMs: number;
  /** Internal engine logic overhead (scheduling, batching, bookkeeping). */
  nativeLogicMs: number;

  /** Total number of tokens processed in the prompt. */
  inputTokens: number;
  /** Total number of tokens generated in the response. */
  outputTokens: number;
  /** Number of tokens reused from KV cache (LCP/Prefix hits). */
  cacheHits: number;
}

export interface RuntimeAggregateObservabilityMetrics extends RequestObservabilityMetrics {}
