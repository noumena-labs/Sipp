interface ObservabilityMetricsBase {
  totalMs: number;
  ttftMs: number;
  tokensPerSecond: number | null;
  inputTokenCount: number;
  outputTokenCount: number;
}

export interface RequestObservabilityMetrics extends ObservabilityMetricsBase {}

export interface RuntimeAggregateObservabilityMetrics extends ObservabilityMetricsBase {}

export type RuntimeObservabilityMetrics = RuntimeAggregateObservabilityMetrics;
