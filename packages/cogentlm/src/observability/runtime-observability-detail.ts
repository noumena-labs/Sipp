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
  nativeSchedulerTickMs: number;
  nativeSchedulerAdmitMs: number;
  nativeSchedulerFinalizeMs: number;
  nativeSchedulerCommitMs: number;
  nativePolicyPrepareMs: number;
  nativePolicyPlanMs: number;
  nativeBatchBuildMs: number;
  nativeLlamaDecodeWallMs: number;
  nativeSynchronizeMs: number;
  nativeKvUpdateMs: number;
  nativeSamplerWallMs: number;
  nativeTokenEmitMs: number;
  nativePrefixCacheMs: number;
  nativeObservabilityMs: number;
  promptEvalTokens: number;
  decodeEvalCount: number;
  sampleCount: number;
  batchParticipationCount: number;
  firstSampledTokenId: number;
  decodeFirstTickCount: number;
  chunkedPrefillTickCount: number;
  mixedWorkloadTickCount: number;
  lcpReuseTokens: number;
  prefixCacheRestoreTokens: number;
  prefixCacheHitCount: number;
  prefixCacheStoreCount: number;
  nativePolicyTickCount: number;
  nativeSchedulerTickCount: number;
}

export interface DetailedRequestObservabilityMetrics
  extends DetailedObservabilityMetricsBase {}

export interface DetailedRuntimeAggregateObservabilityMetrics
  extends DetailedObservabilityMetricsBase,
    RuntimeAggregateObservabilityMetrics {}

export type DetailedRuntimeObservabilityMetrics =
  DetailedRuntimeAggregateObservabilityMetrics;

export function deriveTokensPerSecond(metrics: {
  meanItlMs: number;
}): number | null {
  if (metrics.meanItlMs <= 0) {
    return null;
  }
  return 1000 / metrics.meanItlMs;
}

export function withDerivedObservabilityMetrics<T extends {
  totalMs: number;
  ttftMs: number;
  inputTokenCount: number;
  outputTokenCount: number;
  decodeEvalMs: number;
  meanItlMs: number;
}>(metrics: T): T & { tokensPerSecond: number | null } {
  return {
    ...metrics,
    tokensPerSecond: deriveTokensPerSecond(metrics),
  };
}
