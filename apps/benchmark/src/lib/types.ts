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

export interface RequestObservability {
  inputTokenCount: number;
  outputTokenCount: number;
  promptEvalTokens: number;
  promptEvalMs: number;
  decodeEvalMs: number;
  sampleMs: number;
  queueDelayMs: number;
  tailItlMs: number;
  batchParticipationCount: number;
  decodeFirstTickCount: number;
  chunkedPrefillTickCount: number;
  mixedWorkloadTickCount: number;
  lcpReuseTokens: number;
  prefixCacheRestoreTokens: number;
  prefixCacheHitCount: number;
  prefixCacheStoreCount: number;
  ttftMs?: number;
}

export interface BenchmarkRun {
  label: string;
  wallMs: number;
  ttftMs: number | null;
  tpotMs: number | null;
  itlMsValues: number[];
  inputTokenCount: number | null;
  outputTokenCount: number;
  outputLength: number;
  outputPreview: string;
  requestObservability: RequestObservability | null;
}

export interface GroupSummary {
  serving: {
    successfulRequests: number;
    benchmarkDurationMs: number;
    totalInputTokens: number;
    totalGeneratedTokens: number;
    requestThroughputRps: number | null;
    outputTokenThroughputTps: number | null;
    totalTokenThroughputTps: number | null;
    ttftMs: MetricSummary | null;
    tpotMs: MetricSummary | null;
    itlMs: MetricSummary | null;
    e2elMs: MetricSummary;
  };
  runtime: {
    avgLogicalInputTokenCount: number | null;
    avgPromptEvalTokens: number | null;
    avgPromptEvalMs: number | null;
    avgDecodeEvalMs: number | null;
    avgSampleMs: number | null;
    avgOutputTokenCount: number | null;
    avgQueueDelayMs: number | null;
    avgTailItlMs: number | null;
    avgBatchParticipationCount: number | null;
    avgDecodeFirstTickCount: number | null;
    avgChunkedPrefillTickCount: number | null;
    avgMixedWorkloadTickCount: number | null;
    avgLcpReuseTokens: number | null;
    avgPrefixCacheRestoreTokens: number | null;
    avgPrefixCacheHitCount: number | null;
    avgPrefixCacheStoreCount: number | null;
    promptTokensPerSecond: number | null;
    decodeTokensPerSecond: number | null;
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
  runtime: { initEngineMs: number };
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

export interface ConfigOptions {
  prompt: string;
  tokenCount: number;
  warmupRuns: number;
  measuredRuns: number;
  initConfig: {
    prefillChunkSize: number;
    schedulerPolicy: string;
    decodeTokenReserve: number;
  };
}

export interface MixedLoadDefinition {
  id: string;
  label: string;
  background: ScenarioDefinition & { promptFormat: string; contextBucket: string; concurrency: number };
  foreground: ScenarioDefinition & { promptFormat: string; contextBucket: string; concurrency: number };
  concurrency: number;
}

export interface MixedLoadResult {
  definition: MixedLoadDefinition;
  unsupported?: boolean;
  reason?: string;
  runtime: { initEngineMs: number | null };
  foreground?: GroupResult;
  background?: GroupResult;
}
