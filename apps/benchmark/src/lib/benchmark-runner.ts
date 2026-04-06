import type {
  BenchmarkRun,
  GroupResult,
  GroupSummary,
  RuntimeObservability,
  ScenarioDefinition,
  ScenarioResult,
  MemorySnapshot
} from './types';
import { round, measureAsync } from './utils';
import { CogentEngine } from 'cogent-engine';

function summarize(values: number[]) {
  const sorted = [...values].sort((left, right) => left - right);
  const total = sorted.reduce((acc, value) => acc + value, 0);
  const percentileIndex = Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.99) - 1);

  return {
    meanMs: round(total / sorted.length),
    medianMs: round(sorted[Math.floor(sorted.length / 2)]),
    p99Ms: round(sorted[percentileIndex]),
    minMs: round(sorted[0]),
    maxMs: round(sorted[sorted.length - 1]),
  };
}

function summarizeOptional(values: number[]) {
  const filtered = values.filter((value) => value != null && Number.isFinite(value));
  return filtered.length === 0 ? null : summarize(filtered);
}

function averageRuntimeObservabilityMetric(
  observabilityRuns: (RuntimeObservability | null)[],
  metric: (m: RuntimeObservability) => number | undefined | null
) {
  const values = observabilityRuns
    .filter((metrics): metrics is RuntimeObservability => metrics !== null)
    .map(metric)
    .filter((value): value is number => value != null && Number.isFinite(value) && value >= 0);

  if (values.length === 0) return null;
  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
}

function summarizePromptThroughput(observabilityRuns: (RuntimeObservability | null)[]) {
  const values = observabilityRuns
    .filter((metrics): metrics is RuntimeObservability => metrics !== null)
    .map((metrics) => {
      if (metrics.promptEvalMs <= 0 || metrics.promptEvalTokens <= 0) return 0;
      return (metrics.promptEvalTokens * 1000) / metrics.promptEvalMs;
    })
    .filter((value) => value > 0);

  if (values.length === 0) return null;
  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
}

function summarizeDecodeThroughput(observabilityRuns: (RuntimeObservability | null)[]) {
  const values = observabilityRuns
    .filter((metrics): metrics is RuntimeObservability => metrics !== null)
    .map((metrics) => {
      if (metrics.decodeEvalMs <= 0 || metrics.outputTokenCount <= 0) return 0;
      return (metrics.outputTokenCount * 1000) / metrics.decodeEvalMs;
    })
    .filter((value) => value > 0);

  if (values.length === 0) return null;
  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
}

function summarizeRunGroup(runs: BenchmarkRun[], benchmarkDurationMs: number): GroupSummary {
  const observabilityRuns = runs.map((run) => run.runtimeObservability);
  const totalInputTokens = runs.reduce((acc, run) => acc + (run.inputTokenCount ?? 0), 0);
  const totalGeneratedTokens = runs.reduce((acc, run) => acc + (run.outputTokenCount ?? 0), 0);
  const allItls = runs.flatMap((run) => run.itlMsValues);
  const benchmarkDurationSeconds = benchmarkDurationMs > 0 ? benchmarkDurationMs / 1000 : 0;

  return {
    serving: {
      successfulRequests: runs.length,
      benchmarkDurationMs,
      totalInputTokens,
      totalGeneratedTokens,
      requestThroughputRps: benchmarkDurationSeconds > 0 ? round(runs.length / benchmarkDurationSeconds) : null,
      outputTokenThroughputTps: benchmarkDurationSeconds > 0 ? round(totalGeneratedTokens / benchmarkDurationSeconds) : null,
      totalTokenThroughputTps: benchmarkDurationSeconds > 0 ? round((totalInputTokens + totalGeneratedTokens) / benchmarkDurationSeconds) : null,
      ttftMs: summarizeOptional(runs.map((run) => run.ttftMs!).filter(Boolean)),
      tpotMs: summarizeOptional(runs.map((run) => run.tpotMs!).filter(Boolean)),
      itlMs: summarizeOptional(allItls),
      e2elMs: summarize(runs.map((run) => run.wallMs)),
    },
    runtime: {
      avgLogicalInputTokenCount: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.inputTokenCount),
      avgPromptEvalTokens: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.promptEvalTokens),
      avgPromptEvalMs: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.promptEvalMs),
      avgDecodeEvalMs: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.decodeEvalMs),
      avgSampleMs: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.sampleMs),
      avgOutputTokenCount: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.outputTokenCount),
      avgQueueDelayMs: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.queueDelayMs),
      avgTailItlMs: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.tailItlMs),
      avgSchedulerTickCount: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.schedulerTickCount),
      avgBatchParticipationCount: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.batchParticipationCount),
      avgDecodeFirstTickCount: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.decodeFirstTickCount),
      avgChunkedPrefillTickCount: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.chunkedPrefillTickCount),
      avgMixedWorkloadTickCount: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.mixedWorkloadTickCount),
      avgLcpReuseTokens: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.lcpReuseTokens),
      avgPrefixCacheRestoreTokens: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.prefixCacheRestoreTokens),
      avgPrefixCacheHitCount: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.prefixCacheHitCount),
      avgPrefixCacheStoreCount: averageRuntimeObservabilityMetric(observabilityRuns, (m) => m.prefixCacheStoreCount),
      promptTokensPerSecond: summarizePromptThroughput(observabilityRuns),
      decodeTokensPerSecond: summarizeDecodeThroughput(observabilityRuns),
    },
  };
}

export function createGroupResult(id: string, label: string, warmupRuns: number, measuredRuns: number, group: { benchmarkDurationMs: number, runs: BenchmarkRun[], summary: GroupSummary }): GroupResult {
  return {
    id,
    label,
    warmupRuns,
    measuredRuns,
    benchmarkDurationMs: group.benchmarkDurationMs,
    runs: group.runs,
    summary: group.summary,
  };
}

export async function runPromptGroup(
  targetEngine: CogentEngine,
  groupLabel: string,
  prompt: string,
  tokenCount: number,
  warmupRuns: number,
  measuredRuns: number,
  contextKeyFactory: (index: number) => string,
  setStatus: (s: string) => void
): Promise<{ benchmarkDurationMs: number, runs: BenchmarkRun[], summary: GroupSummary }> {
  for (let i = 0; i < warmupRuns; i++) {
    setStatus(`${groupLabel}: warmup ${i + 1}/${warmupRuns}`);
    await targetEngine.submitPrompt(contextKeyFactory(i), prompt, { nTokens: tokenCount });
  }

  const runs: BenchmarkRun[] = [];
  const benchmarkStart = performance.now();
  
  for (let i = 0; i < measuredRuns; i++) {
    setStatus(`${groupLabel}: run ${i + 1}/${measuredRuns}`);
    const start = performance.now();
    let ttftMs: number | null = null;
    const tokenEventTimes: number[] = [];
    
    const output = await targetEngine.submitPrompt(contextKeyFactory(i + warmupRuns), prompt, {
      nTokens: tokenCount,
      onToken: () => {
        const elapsedMs = round(performance.now() - start);
        tokenEventTimes.push(elapsedMs);
        if (ttftMs == null) ttftMs = elapsedMs;
      },
    });

    const wallMs = round(performance.now() - start);
    // @ts-ignore getRuntimeObservability exist but types may vary
    const runtimeObservability = (targetEngine.getRuntimeObservability?.() as RuntimeObservability) || null;
    const outputTokenCount = runtimeObservability?.outputTokenCount ?? tokenEventTimes.length;
    const itlMsValues: number[] = [];
    for (let j = 1; j < tokenEventTimes.length; j++) {
      itlMsValues.push(round(tokenEventTimes[j] - tokenEventTimes[j - 1]));
    }
    const tpotMs = ttftMs != null && outputTokenCount > 1 ? round((wallMs - ttftMs) / (outputTokenCount - 1)) : null;

    runs.push({
      label: `${groupLabel}-${i + 1}`,
      wallMs,
      ttftMs,
      tpotMs,
      itlMsValues,
      inputTokenCount: runtimeObservability?.inputTokenCount ?? null,
      outputTokenCount,
      outputLength: output.length,
      outputPreview: output.slice(0, 160).replace(/\s+/g, ' ').trim(),
      runtimeObservability,
    });
  }

  const benchmarkDurationMs = round(performance.now() - benchmarkStart);
  return {
    benchmarkDurationMs,
    runs,
    summary: summarizeRunGroup(runs, benchmarkDurationMs),
  };
}

export async function runScenarioBenchmark(
  targetEngine: CogentEngine,
  scenario: ScenarioDefinition,
  modelPath: string,
  warmupRuns: number,
  measuredRuns: number,
  initConfig: any,
  setStatus: (s: string) => void,
  skipInitEngine?: boolean
): Promise<ScenarioResult> {
  let initEngineMs = 0;
  if (!skipInitEngine) {
    setStatus(`${scenario.label}: initializing engine...`);
    const { ms } = await measureAsync(() => targetEngine.initEngine(modelPath, initConfig));
    initEngineMs = ms;
  }

  const coldPrompt = await runPromptGroup(
    targetEngine,
    `${scenario.label}: cold prompt`,
    scenario.prompt,
    scenario.outputTokenLimit,
    0,
    1,
    () => `${scenario.id}-cold`,
    setStatus
  );

  const hotFreshContext = await runPromptGroup(
    targetEngine,
    `${scenario.label}: hot fresh context`,
    scenario.prompt,
    scenario.outputTokenLimit,
    warmupRuns,
    measuredRuns,
    (index) => `${scenario.id}-fresh-${index}`,
    setStatus
  );

  const hotReuseContext = await runPromptGroup(
    targetEngine,
    `${scenario.label}: hot reused context`,
    scenario.prompt,
    scenario.outputTokenLimit,
    warmupRuns,
    measuredRuns,
    () => `${scenario.id}-reuse`,
    setStatus
  );

  return {
    definition: scenario,
    runtime: { initEngineMs },
    coldPrompt: createGroupResult('coldPrompt', 'Cold Prompt', 0, 1, coldPrompt),
    hotFreshContext: createGroupResult('hotFreshContext', 'Hot Prompt: Fresh Context', warmupRuns, measuredRuns, hotFreshContext),
    hotReuseContext: createGroupResult('hotReuseContext', 'Hot Prompt: Reused Context', warmupRuns, measuredRuns, hotReuseContext),
  };
}

export async function captureBrowserMemorySnapshot(label: string, includeDetailed?: boolean): Promise<MemorySnapshot> {
  const snapshot: MemorySnapshot = {
    label,
    capturedAt: new Date().toISOString(),
    source: 'unavailable',
    usedJsHeapBytes: null,
    totalJsHeapBytes: null,
    jsHeapLimitBytes: null,
    userAgentBytes: null,
    error: null,
  };

  // @ts-ignore performance.memory is standard in chrome only
  if (typeof performance !== 'undefined' && performance.memory) {
    snapshot.source = 'performance.memory';
    // @ts-ignore
    snapshot.usedJsHeapBytes = performance.memory.usedJSHeapSize ?? null;
    // @ts-ignore
    snapshot.totalJsHeapBytes = performance.memory.totalJSHeapSize ?? null;
    // @ts-ignore
    snapshot.jsHeapLimitBytes = performance.memory.jsHeapSizeLimit ?? null;
  }

  // @ts-ignore
  if (includeDetailed && typeof performance !== 'undefined' && typeof performance.measureUserAgentSpecificMemory === 'function') {
    try {
      // @ts-ignore
      const uaMemory = await performance.measureUserAgentSpecificMemory();
      snapshot.userAgentBytes = uaMemory.bytes ?? null;
      snapshot.source = snapshot.source === 'performance.memory' 
        ? 'performance.memory + measureUserAgentSpecificMemory' 
        : 'measureUserAgentSpecificMemory';
    } catch (error: any) {
      snapshot.error = typeof error === 'object' && error?.message ? error.message : String(error);
    }
  }

  return snapshot;
}

export function supportsQueuedRequestApi(targetEngine: any): boolean {
  return (
    targetEngine != null &&
    typeof targetEngine.queuePrompt === 'function' &&
    typeof targetEngine.runQueuedRequest === 'function'
  );
}

export async function runQueuedMixedLoadPair(
  targetEngine: any,
  definition: import('./types').MixedLoadDefinition,
  runIndex: number
) {
  const backgroundContextKey = `${definition.background.id}-mixed-${runIndex}`;
  const foregroundContextKey = `${definition.foreground.id}-mixed-${runIndex}`;

  const backgroundStart = performance.now();
  let backgroundTtftMs: number | null = null;
  const backgroundTokenEventTimes: number[] = [];
  const backgroundRequestId = await targetEngine.queuePrompt(
    backgroundContextKey,
    definition.background.prompt,
    {
      nTokens: definition.background.outputTokenLimit,
      promptFormat: definition.background.promptFormat,
      onToken: () => {
        const elapsedMs = round(performance.now() - backgroundStart);
        backgroundTokenEventTimes.push(elapsedMs);
        if (backgroundTtftMs == null) backgroundTtftMs = elapsedMs;
      },
    }
  );

  const foregroundStart = performance.now();
  let foregroundTtftMs: number | null = null;
  const foregroundTokenEventTimes: number[] = [];
  const foregroundRequestId = await targetEngine.queuePrompt(
    foregroundContextKey,
    definition.foreground.prompt,
    {
      nTokens: definition.foreground.outputTokenLimit,
      promptFormat: definition.foreground.promptFormat,
      onToken: () => {
        const elapsedMs = round(performance.now() - foregroundStart);
        foregroundTokenEventTimes.push(elapsedMs);
        if (foregroundTtftMs == null) foregroundTtftMs = elapsedMs;
      },
    }
  );

  const foregroundResponse = await targetEngine.runQueuedRequest(foregroundRequestId);
  const foregroundWallMs = round(performance.now() - foregroundStart);
  const backgroundResponse = await targetEngine.runQueuedRequest(backgroundRequestId);
  const backgroundWallMs = round(performance.now() - backgroundStart);

  const toRun = (label: string, contextKey: string, wallMs: number, ttftMs: number | null, tokenEventTimes: number[], response: any) => {
    const perf = response.perf ?? null;
    const outputTokenCount = perf?.outputTokenCount ?? tokenEventTimes.length;
    const itlMsValues: number[] = [];
    for (let i = 1; i < tokenEventTimes.length; i++) {
      itlMsValues.push(round(tokenEventTimes[i] - tokenEventTimes[i - 1]));
    }
    const effectiveTtftMs = ttftMs ?? perf?.ttftMs ?? null;
    const tpotMs = effectiveTtftMs != null && outputTokenCount > 1 ? round((wallMs - effectiveTtftMs) / (outputTokenCount - 1)) : null;

    return {
      label,
      contextKey,
      wallMs,
      ttftMs: effectiveTtftMs,
      tpotMs,
      itlMsValues,
      inputTokenCount: perf?.inputTokenCount ?? null,
      outputTokenCount,
      outputLength: response.outputText.length,
      outputPreview: response.outputText.slice(0, 160).replace(/\s+/g, ' ').trim(),
      runtimeObservability: perf,
    };
  };

  return {
    backgroundRun: toRun(
      `${definition.id}-background-${runIndex + 1}`,
      backgroundContextKey,
      backgroundWallMs,
      backgroundTtftMs,
      backgroundTokenEventTimes,
      backgroundResponse
    ),
    foregroundRun: toRun(
      `${definition.id}-foreground-${runIndex + 1}`,
      foregroundContextKey,
      foregroundWallMs,
      foregroundTtftMs,
      foregroundTokenEventTimes,
      foregroundResponse
    ),
  };
}

export async function runMixedLoadBenchmark(
  targetEngine: any,
  definition: import('./types').MixedLoadDefinition,
  modelPath: string,
  warmupRuns: number,
  measuredRuns: number,
  initConfig: any,
  setStatus: (s: string) => void,
  skipInitEngine?: boolean
): Promise<import('./types').MixedLoadResult> {
  let initEngineMs = 0;
  if (!skipInitEngine) {
    setStatus(`${definition.label}: initializing engine...`);
    const { ms } = await measureAsync(() => targetEngine.initEngine(modelPath, initConfig));
    initEngineMs = ms;
  }

  for (let i = 0; i < warmupRuns; i++) {
    setStatus(`${definition.label}: warmup ${i + 1}/${warmupRuns}`);
    await runQueuedMixedLoadPair(targetEngine, definition, i);
  }

  const foregroundRuns: import('./types').BenchmarkRun[] = [];
  const backgroundRuns: import('./types').BenchmarkRun[] = [];
  const benchmarkStart = performance.now();
  
  for (let i = 0; i < measuredRuns; i++) {
    setStatus(`${definition.label}: run ${i + 1}/${measuredRuns}`);
    const pair = await runQueuedMixedLoadPair(targetEngine, definition, i + warmupRuns);
    backgroundRuns.push(pair.backgroundRun);
    foregroundRuns.push(pair.foregroundRun);
  }

  const benchmarkDurationMs = round(performance.now() - benchmarkStart);
  return {
    definition,
    runtime: { initEngineMs },
    foreground: createGroupResult('hotFreshContext', `${definition.foreground.label} Under Mixed Load`, warmupRuns, measuredRuns, {
      benchmarkDurationMs,
      runs: foregroundRuns,
      summary: summarizeRunGroup(foregroundRuns, benchmarkDurationMs),
    }),
    background: createGroupResult('hotFreshContext', `${definition.background.label} Under Mixed Load`, warmupRuns, measuredRuns, {
      benchmarkDurationMs,
      runs: backgroundRuns,
      summary: summarizeRunGroup(backgroundRuns, benchmarkDurationMs),
    }),
  };
}
