import type { NativeRuntimeConfig, RequestObservabilityMetrics } from '@noumena-labs/cogentlm';

export interface EnvironmentInfo {
  browserLabel: string;
  language: string;
  hardwareConcurrency: number | null;
  deviceMemory: number | null;
  crossOriginIsolated: boolean;
  hasNavigatorGpu: boolean;
  adapterAvailable: boolean;
  adapterLabel: string;
  adapterVendor: string | null;
  adapterArchitecture: string | null;
  adapterDescription: string | null;
  adapterError: string | null;
}

export interface MetricSummary {
  meanMs: number;
  medianMs: number;
  p99Ms: number;
  minMs: number;
  maxMs: number;
}

export type RequestObservability = RequestObservabilityMetrics;

export type BenchmarkOperation = 'chat' | 'query' | 'embed';

export interface BenchmarkRun {
  label: string;
  operation: BenchmarkOperation;
  outputKind: 'text' | 'embedding';
  wallMs: number;
  ttftMs: number | null;
  itlAvgMs: number | null;
  itlP99Ms: number | null;
  tps: number | null;
  inputTokens: number | null;
  outputTokens: number;
  prefillTokens: number | null;
  prefillTps: number | null;
  outputLength: number;
  outputPreview: string;
  embeddingDimensions: number | null;
  embeddingPooling: string | null;
  embeddingNormalized: boolean | null;
  observability: RequestObservability | null;
}

export interface GroupSummary {
  serving: {
    successfulRequests: number;
    benchmarkDurationMs: number;
    totalInputTokens: number;
    totalGeneratedTokens: number;
    totalPrefillTokens: number;
    requestThroughputRps: number | null;
    outputTokenThroughputTps: number | null;
    totalTokenThroughputTps: number | null;
  };
  runtime: {
    ttftMs: MetricSummary | null;
    itlAvgMs: MetricSummary | null;
    itlP99Ms: MetricSummary | null;
    tps: MetricSummary | null;
    prefillTps: MetricSummary | null;
    avgInputTokens: number | null;
    avgOutputTokens: number | null;
    avgPrefillTokens: number | null;
    avgPrefillMs: number | null;
    avgDecodeMs: number | null;
    avgNativeGpuMs: number | null;
    avgNativeSyncMs: number | null;
    avgNativeLogicMs: number | null;
    avgCacheHits: number | null;
  };
}

export interface GroupResult {
  id: string;
  label: string;
  warmupRuns: number;
  measuredRuns: number;
  benchmarkDurationMs: number;
  runs: BenchmarkRun[];
  summary: GroupSummary;
}

export interface ScenarioDefinition {
  id: string;
  label: string;
  prompt: string;
  promptBucket: string;
  promptChars: number;
  promptWords: number;
  outputTokenLimit: number;
  outputBucket: string;
}

export interface ScenarioResult {
  definition: ScenarioDefinition;
  runtime: { loadRuntimeMs: number };
  coldPrompt: GroupResult;
  hotFreshContext: GroupResult;
  hotReuseContext: GroupResult;
}

export interface MemorySnapshot {
  label: string;
  capturedAt: string;
  source: string;
  usedJsHeapBytes: number | null;
  totalJsHeapBytes: number | null;
  jsHeapLimitBytes: number | null;
  userAgentBytes: number | null;
  error: string | null;
}

export interface ImageInput {
  enabled: boolean;
  source: 'url' | 'base64';
  url: string;
  base64: string;
  mimeType: string;
  fileName: string;
  projectorUrl: string;
  projectorFileName: string;
}

export interface ConfigOptions {
  prompt: string;
  tokenCount: number;
  warmupRuns: number;
  measuredRuns: number;
  workerTransport: {
    preset: 'default' | 'low-buffer' | 'no-buffer' | 'custom';
    bufferedTokenLimit: number;
    flushIntervalMs: number;
  };
  initConfig: NativeRuntimeConfig & {
    debugCompareMultimodalEmbeddings: boolean;
  };
  imageInput: ImageInput;
}

export interface MixedLoadDefinition {
  id: string;
  label: string;
  background: ScenarioDefinition & { promptMode: 'chat' | 'query'; contextBucket: string; concurrency: number };
  foreground: ScenarioDefinition & { promptMode: 'chat' | 'query'; contextBucket: string; concurrency: number };
  concurrency: number;
}

export interface MixedLoadResult {
  definition: MixedLoadDefinition;
  unsupported?: boolean;
  reason?: string;
  runtime: { loadRuntimeMs: number | null };
  foreground?: GroupResult;
  background?: GroupResult;
}

export interface BenchmarkLogEntry {
  scenarioId: string;
  scenarioLabel: string;
  groupId: string;
  groupLabel: string;
  runLabel: string;
  operation: BenchmarkOperation;
  outputKind: BenchmarkRun['outputKind'];
  wallMs: number;
  outputTokens: number;
  embeddingDimensions: number | null;
  observability: RequestObservability | null;
}

export interface BenchmarkTraceReport {
  runCount: number;
  logs: BenchmarkLogEntry[];
  analysis: {
    ttftMs: MetricSummary | null;
    itlAvgMs: MetricSummary | null;
    itlP99Ms: MetricSummary | null;
    tps: MetricSummary | null;
  };
}
