import type {
  RequestObservabilityMetrics,
  RuntimeAggregateObservabilityMetrics,
} from './runtime-observability.js';

export interface DetailedRequestObservabilityMetrics extends RequestObservabilityMetrics {}

export interface DetailedRuntimeAggregateObservabilityMetrics
  extends RequestObservabilityMetrics,
    RuntimeAggregateObservabilityMetrics {}

export type DetailedRuntimeObservabilityMetrics = DetailedRuntimeAggregateObservabilityMetrics;

export function deriveTokensPerSecond(metrics: {
  decodeMs: number;
  outputTokens: number;
}): number | null {
  if (metrics.decodeMs <= 0 || metrics.outputTokens <= 0) {
    return null;
  }
  return (metrics.outputTokens / metrics.decodeMs) * 1000;
}

export function withDerivedObservabilityMetrics<T extends RequestObservabilityMetrics>(
  metrics: T
): T & { tokensPerSecond: number | null } {
  return {
    ...metrics,
    tokensPerSecond: deriveTokensPerSecond(metrics),
  };
}
