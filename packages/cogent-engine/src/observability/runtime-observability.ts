interface ObservabilityMetricsBase {
  totalMs: number;
  promptEvalMs: number;
  decodeEvalMs: number;
  sampleMs: number;
  queueDelayMs: number;
  ttftMs: number;
  meanItlMs: number;
  tailItlMs: number;
  e2elMs: number;
  inputTokenCount: number;
  promptEvalTokens: number;
  decodeEvalCount: number;
  sampleCount: number;
  outputTokenCount: number;
  batchParticipationCount: number;
  decodeFirstTickCount: number;
  chunkedPrefillTickCount: number;
  mixedWorkloadTickCount: number;
  lcpReuseTokens: number;
  prefixCacheRestoreTokens: number;
  prefixCacheHitCount: number;
  prefixCacheStoreCount: number;
}

export interface RequestObservabilityMetrics extends ObservabilityMetricsBase {}

export interface RuntimeAggregateObservabilityMetrics extends ObservabilityMetricsBase {}

export type RuntimeObservabilityMetrics = RuntimeAggregateObservabilityMetrics;
