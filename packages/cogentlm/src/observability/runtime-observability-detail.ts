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

export function derivePrefillTokensPerSecond(metrics: {
  prefillMs: number;
  prefillTokens: number;
}): number | null {
  // A prefill of exactly 1 token is valid in cache-hit scenarios where the
  // engine re-decodes the last cached token to generate logits.
  if (metrics.prefillMs < 0.1 || metrics.prefillTokens < 1) {
    return null;
  }
  return (metrics.prefillTokens / metrics.prefillMs) * 1000;
}

export function withDerivedObservabilityMetrics<T extends RequestObservabilityMetrics>(
  metrics: T
): T & { tokensPerSecond: number | null; prefillTokensPerSecond: number | null } {
  return {
    ...metrics,
    tokensPerSecond: deriveTokensPerSecond(metrics),
    prefillTokensPerSecond: derivePrefillTokensPerSecond(metrics),
  };
}
