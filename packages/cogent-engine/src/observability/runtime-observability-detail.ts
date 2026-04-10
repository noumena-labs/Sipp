import type {
  RequestObservabilityMetrics,
  RuntimeAggregateObservabilityMetrics,
} from './runtime-observability.js';

interface DetailedObservabilityMetricsBase
  extends RequestObservabilityMetrics {
  promptEvalMs: number;
  decodeEvalMs: number;
  sampleMs: number;
  queueDelayMs: number;
  meanItlMs: number;
  tailItlMs: number;
  e2elMs: number;
  promptEvalTokens: number;
  decodeEvalCount: number;
  sampleCount: number;
  batchParticipationCount: number;
  decodeFirstTickCount: number;
  chunkedPrefillTickCount: number;
  mixedWorkloadTickCount: number;
  lcpReuseTokens: number;
  prefixCacheRestoreTokens: number;
  prefixCacheHitCount: number;
  prefixCacheStoreCount: number;
}

export interface DetailedRequestObservabilityMetrics
  extends DetailedObservabilityMetricsBase {}

export interface DetailedRuntimeAggregateObservabilityMetrics
  extends DetailedObservabilityMetricsBase,
    RuntimeAggregateObservabilityMetrics {}

export type DetailedRuntimeObservabilityMetrics =
  DetailedRuntimeAggregateObservabilityMetrics;

export function deriveTokensPerSecond(metrics: {
  outputTokenCount: number;
  decodeEvalMs: number;
  totalMs: number;
}): number | null {
  const effectiveMs =
    metrics.decodeEvalMs > 0 ? metrics.decodeEvalMs : metrics.totalMs;
  if (effectiveMs <= 0 || metrics.outputTokenCount <= 0) {
    return null;
  }
  return (metrics.outputTokenCount * 1000) / effectiveMs;
}

export function withDerivedObservabilityMetrics<T extends {
  totalMs: number;
  ttftMs: number;
  inputTokenCount: number;
  outputTokenCount: number;
  decodeEvalMs: number;
}>(metrics: T): T & { tokensPerSecond: number | null } {
  return {
    ...metrics,
    tokensPerSecond: deriveTokensPerSecond(metrics),
  };
}
