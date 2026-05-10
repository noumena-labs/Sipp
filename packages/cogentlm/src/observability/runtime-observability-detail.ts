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
  itlAvgMs: number;
}): number | null {
  if (metrics.itlAvgMs <= 0) {
    return null;
  }
  return 1000 / metrics.itlAvgMs;
}

export function withDerivedObservabilityMetrics<T extends RequestObservabilityMetrics>(
  metrics: T
): T & { tokensPerSecond: number | null } {
  return {
    ...metrics,
    tokensPerSecond: deriveTokensPerSecond(metrics),
  };
}
