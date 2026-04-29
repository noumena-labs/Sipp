import { CogentEngine, type ModelLoadOptions, type ModelSource } from '@noumena-labs/cogent-engine';
import type {
  BenchmarkRun,
  GroupResult,
  GroupSummary,
  MemorySnapshot,
  RequestObservability,
  ScenarioDefinition,
  ScenarioResult,
} from './types';
import { measureAsync, round } from './utils';

type BenchmarkRuntimeOptions = NonNullable<ModelLoadOptions['runtime']>;

interface ObservedQueryRun {
  output: string;
  wallMs: number;
  ttftMs: number | null;
  tokenTimes: number[];
  observability: RequestObservability | null;
}

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
  const filtered = values.filter((value) => Number.isFinite(value));
  return filtered.length === 0 ? null : summarize(filtered);
}

function averageOptional(values: Array<number | null | undefined>): number | null {
  const filtered = values.filter((value): value is number => value != null && Number.isFinite(value));
  if (filtered.length === 0) return null;
  return round(filtered.reduce((sum, value) => sum + value, 0) / filtered.length);
}

function cloneRuntimeObservation(
  observation: RequestObservability | null | undefined
): RequestObservability | null {
  if (observation == null) {
    return null;
  }
  return {
    ...observation,
    execution: { ...observation.execution },
  };
}

function observeSessionCompletion(
  targetEngine: CogentEngine,
  session: string
): {
  promise: Promise<RequestObservability | null>;
  dispose: () => void;
} {
  let unsubscribe: (() => void) | null = null;

  const promise = new Promise<RequestObservability | null>((resolve) => {
    unsubscribe = targetEngine.observability.subscribe((event) => {
      const query = event.snapshot.query;
      if (query?.session !== session) {
        return;
      }
      if (event.type !== 'query-complete' && event.type !== 'error') {
        return;
      }
      const done = unsubscribe;
      unsubscribe = null;
      done?.();
      resolve(cloneRuntimeObservation(event.snapshot.runtime));
    });
  });

  return {
    promise,
    dispose: () => {
      const done = unsubscribe;
      unsubscribe = null;
      done?.();
    },
  };
}

async function runObservedQuery(
  targetEngine: CogentEngine,
  prompt: string,
  options: {
    session: string;
    maxTokens: number;
  }
): Promise<ObservedQueryRun> {
  const start = performance.now();
  let ttftMs: number | null = null;
  const tokenTimes: number[] = [];
  const sessionObserver = observeSessionCompletion(targetEngine, options.session);

  try {
    const [output, observability] = await Promise.all([
      targetEngine.query(prompt, {
        maxTokens: options.maxTokens,
        session: options.session,
        onToken: () => {
          const elapsed = round(performance.now() - start);
          tokenTimes.push(elapsed);
          ttftMs ??= elapsed;
        },
      }),
      sessionObserver.promise,
    ]);

    return {
      output,
      wallMs: round(performance.now() - start),
      ttftMs,
      tokenTimes,
      observability,
    };
  } finally {
    sessionObserver.dispose();
  }
}

function summarizeRunGroup(runs: BenchmarkRun[], benchmarkDurationMs: number): GroupSummary {
  const observations = runs
    .map((run) => run.requestObservability)
    .filter((value): value is RequestObservability => value != null);
  const totalInputTokens = runs.reduce(
    (acc, run) => acc + (run.requestObservability?.inputTokenCount ?? 0),
    0
  );
  const totalGeneratedTokens = runs.reduce((acc, run) => acc + run.outputTokenCount, 0);
  const benchmarkDurationSeconds = benchmarkDurationMs > 0 ? benchmarkDurationMs / 1000 : 0;
  return {
    serving: {
      successfulRequests: runs.length,
      benchmarkDurationMs,
      totalInputTokens,
      totalGeneratedTokens,
      requestThroughputRps:
        benchmarkDurationSeconds > 0 ? round(runs.length / benchmarkDurationSeconds) : null,
      // End-to-end output throughput uses the full benchmark wall time.
      outputTokenThroughputTps:
        benchmarkDurationSeconds > 0 ? round(totalGeneratedTokens / benchmarkDurationSeconds) : null,
      // Total throughput counts both prompt and generated tokens over wall time.
      totalTokenThroughputTps:
        benchmarkDurationSeconds > 0
          ? round((totalInputTokens + totalGeneratedTokens) / benchmarkDurationSeconds)
          : null,
      appObservedTtftMs: summarizeOptional(
        runs.map((run) => run.appObservedTtftMs).filter((value): value is number => value != null)
      ),
      appObservedTpotMs: summarizeOptional(
        runs.map((run) => run.appObservedTpotMs).filter((value): value is number => value != null)
      ),
      appObservedItlMs: summarizeOptional(runs.flatMap((run) => run.appObservedItlMsValues)),
      e2elMs: summarize(runs.map((run) => run.wallMs)),
    },
    runtime: {
      nativeTtftMs: summarizeOptional(observations.map((item) => item.ttftMs)),
      nativeMeanItlMs: summarizeOptional(
        observations.map((item) => item.meanItlMs).filter((value): value is number => value != null)
      ),
      nativeTailItlMs: summarizeOptional(
        observations.map((item) => item.tailItlMs).filter((value): value is number => value != null)
      ),
      nativeDecodeTokensPerSecond: summarizeOptional(
        observations.map((item) => item.tokensPerSecond).filter((value): value is number => value != null)
      ),
      avgLogicalInputTokenCount: averageOptional(observations.map((item) => item.inputTokenCount)),
      avgPromptEvalTokens: averageOptional(observations.map((item) => item.promptEvalTokens)),
      avgPromptEvalMs: averageOptional(observations.map((item) => item.promptEvalMs)),
      avgDecodeEvalMs: averageOptional(observations.map((item) => item.decodeEvalMs)),
      avgSampleMs: averageOptional(observations.map((item) => item.sampleMs)),
      avgOutputTokenCount: averageOptional(observations.map((item) => item.outputTokenCount)),
      avgQueueDelayMs: averageOptional(observations.map((item) => item.queueDelayMs)),
      avgTailItlMs: averageOptional(observations.map((item) => item.tailItlMs)),
      avgBatchParticipationCount: averageOptional(
        observations.map((item) => item.batchParticipationCount)
      ),
      avgDecodeFirstTickCount: averageOptional(observations.map((item) => item.decodeFirstTickCount)),
      avgChunkedPrefillTickCount: averageOptional(
        observations.map((item) => item.chunkedPrefillTickCount)
      ),
      avgMixedWorkloadTickCount: averageOptional(observations.map((item) => item.mixedWorkloadTickCount)),
      avgLcpReuseTokens: averageOptional(observations.map((item) => item.lcpReuseTokens)),
      avgPrefixCacheRestoreTokens: averageOptional(
        observations.map((item) => item.prefixCacheRestoreTokens)
      ),
      avgPrefixCacheHitCount: averageOptional(observations.map((item) => item.prefixCacheHitCount)),
      avgPrefixCacheStoreCount: averageOptional(observations.map((item) => item.prefixCacheStoreCount)),
      promptTokensPerSecond: averageOptional(
        observations.map((item) =>
          item.promptEvalTokens != null && item.promptEvalMs != null && item.promptEvalMs > 0
            ? (item.promptEvalTokens / item.promptEvalMs) * 1000
            : null
        )
      ),
      decodeTokensPerSecond: averageOptional(observations.map((item) => item.tokensPerSecond)),
    },
  };
}

function createRun(
  label: string,
  wallMs: number,
  ttftMs: number | null,
  tokenTimes: number[],
  output: string,
  observability: RequestObservability | null = null
): BenchmarkRun {
  const appObservedItlMsValues: number[] = [];
  for (let i = 1; i < tokenTimes.length; i++) {
    appObservedItlMsValues.push(round(tokenTimes[i] - tokenTimes[i - 1]));
  }
  const appObservedTpotMs =
    ttftMs != null && tokenTimes.length > 1 ? round((wallMs - ttftMs) / (tokenTimes.length - 1)) : null;
  return {
    label,
    wallMs,
    appObservedTtftMs: ttftMs,
    appObservedTpotMs,
    appObservedItlMsValues,
    nativeTtftMs: observability?.ttftMs ?? null,
    nativeMeanItlMs: observability?.meanItlMs ?? null,
    nativeTailItlMs: observability?.tailItlMs ?? null,
    nativeDecodeTokensPerSecond: observability?.tokensPerSecond ?? null,
    inputTokenCount: observability?.inputTokenCount ?? null,
    outputTokenCount: observability?.outputTokenCount ?? tokenTimes.length,
    outputLength: output.length,
    outputPreview: output.slice(0, 160).replace(/\s+/g, ' ').trim(),
    requestObservability: observability,
  };
}

export function createGroupResult(
  id: string,
  label: string,
  warmupRuns: number,
  measuredRuns: number,
  group: { benchmarkDurationMs: number; runs: BenchmarkRun[]; summary: GroupSummary }
): GroupResult {
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
  sessionFactory: (index: number) => string,
  setStatus: (s: string) => void
): Promise<{ benchmarkDurationMs: number; runs: BenchmarkRun[]; summary: GroupSummary }> {
  for (let i = 0; i < warmupRuns; i++) {
    setStatus(`${groupLabel}: warmup ${i + 1}/${warmupRuns}`);
    await targetEngine.query(prompt, {
      maxTokens: tokenCount,
      session: sessionFactory(i),
    });
  }

  const runs: BenchmarkRun[] = [];
  const benchmarkStart = performance.now();
  for (let i = 0; i < measuredRuns; i++) {
    setStatus(`${groupLabel}: run ${i + 1}/${measuredRuns}`);
    const run = await runObservedQuery(targetEngine, prompt, {
      maxTokens: tokenCount,
      session: sessionFactory(i + warmupRuns),
    });
    runs.push(
      createRun(
        `${groupLabel}-${i + 1}`,
        run.wallMs,
        run.ttftMs,
        run.tokenTimes,
        run.output,
        run.observability
      )
    );
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
  modelSource: ModelSource,
  warmupRuns: number,
  measuredRuns: number,
  runtime: BenchmarkRuntimeOptions,
  setStatus: (s: string) => void,
  alreadyLoaded?: boolean
): Promise<ScenarioResult> {
  let loadRuntimeMs = 0;
  if (!alreadyLoaded) {
    setStatus(`${scenario.label}: loading model...`);
    const measured = await measureAsync(() =>
      targetEngine.models.load(modelSource, { runtime, observability: 'profile' })
    );
    loadRuntimeMs = measured.ms;
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
    runtime: { loadRuntimeMs },
    coldPrompt: createGroupResult('coldPrompt', 'Cold Prompt', 0, 1, coldPrompt),
    hotFreshContext: createGroupResult('hotFreshContext', 'Hot Prompt: Fresh Context', warmupRuns, measuredRuns, hotFreshContext),
    hotReuseContext: createGroupResult('hotReuseContext', 'Hot Prompt: Reused Context', warmupRuns, measuredRuns, hotReuseContext),
  };
}

export async function captureBrowserMemorySnapshot(
  label: string,
  includeDetailed?: boolean
): Promise<MemorySnapshot> {
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

  if (typeof performance !== 'undefined' && 'memory' in performance) {
    const memory = (performance as Performance & {
      memory?: { usedJSHeapSize?: number; totalJSHeapSize?: number; jsHeapSizeLimit?: number };
    }).memory;
    snapshot.source = 'performance.memory';
    snapshot.usedJsHeapBytes = memory?.usedJSHeapSize ?? null;
    snapshot.totalJsHeapBytes = memory?.totalJSHeapSize ?? null;
    snapshot.jsHeapLimitBytes = memory?.jsHeapSizeLimit ?? null;
  }

  const detailedMemory = performance as Performance & {
    measureUserAgentSpecificMemory?: () => Promise<{ bytes?: number }>;
  };
  if (includeDetailed && typeof detailedMemory.measureUserAgentSpecificMemory === 'function') {
    try {
      const uaMemory = await detailedMemory.measureUserAgentSpecificMemory();
      snapshot.userAgentBytes = uaMemory.bytes ?? null;
      snapshot.source =
        snapshot.source === 'performance.memory'
          ? 'performance.memory + measureUserAgentSpecificMemory'
          : 'measureUserAgentSpecificMemory';
    } catch (error) {
      snapshot.error = error instanceof Error ? error.message : String(error);
    }
  }

  return snapshot;
}

export function supportsConcurrentQueryApi(targetEngine: CogentEngine | null): boolean {
  return targetEngine != null;
}

export async function runMixedLoadBenchmark(
  targetEngine: CogentEngine,
  definition: import('./types').MixedLoadDefinition,
  modelSource: ModelSource,
  warmupRuns: number,
  measuredRuns: number,
  runtime: BenchmarkRuntimeOptions,
  setStatus: (s: string) => void,
  alreadyLoaded?: boolean
): Promise<import('./types').MixedLoadResult> {
  let loadRuntimeMs = 0;
  if (!alreadyLoaded) {
    const measured = await measureAsync(() =>
      targetEngine.models.load(modelSource, { runtime, observability: 'profile' })
    );
    loadRuntimeMs = measured.ms;
  }

  for (let i = 0; i < warmupRuns; i++) {
    setStatus(`${definition.label}: warmup ${i + 1}/${warmupRuns}`);
    await Promise.all([
      runObservedQuery(targetEngine, definition.background.prompt, {
        maxTokens: definition.background.outputTokenLimit,
        session: `${definition.background.id}-warmup-${i}`,
      }),
      runObservedQuery(targetEngine, definition.foreground.prompt, {
        maxTokens: definition.foreground.outputTokenLimit,
        session: `${definition.foreground.id}-warmup-${i}`,
      }),
    ]);
  }

  const foregroundRuns: BenchmarkRun[] = [];
  const backgroundRuns: BenchmarkRun[] = [];
  const benchmarkStart = performance.now();
  for (let i = 0; i < measuredRuns; i++) {
    setStatus(`${definition.label}: run ${i + 1}/${measuredRuns}`);
    const [backgroundRun, foregroundRun] = await Promise.all([
      runObservedQuery(targetEngine, definition.background.prompt, {
        maxTokens: definition.background.outputTokenLimit,
        session: `${definition.background.id}-mixed-${i}`,
      }),
      runObservedQuery(targetEngine, definition.foreground.prompt, {
        maxTokens: definition.foreground.outputTokenLimit,
        session: `${definition.foreground.id}-mixed-${i}`,
      }),
    ]);
    backgroundRuns.push(
      createRun(
        `${definition.id}-background-${i + 1}`,
        backgroundRun.wallMs,
        backgroundRun.ttftMs,
        backgroundRun.tokenTimes,
        backgroundRun.output,
        backgroundRun.observability
      )
    );
    foregroundRuns.push(
      createRun(
        `${definition.id}-foreground-${i + 1}`,
        foregroundRun.wallMs,
        foregroundRun.ttftMs,
        foregroundRun.tokenTimes,
        foregroundRun.output,
        foregroundRun.observability
      )
    );
  }

  const benchmarkDurationMs = round(performance.now() - benchmarkStart);
  return {
    definition,
    runtime: { loadRuntimeMs },
    foreground: createGroupResult('foreground', `${definition.foreground.label} Under Mixed Load`, warmupRuns, measuredRuns, {
      benchmarkDurationMs,
      runs: foregroundRuns,
      summary: summarizeRunGroup(foregroundRuns, benchmarkDurationMs),
    }),
    background: createGroupResult('background', `${definition.background.label} Under Mixed Load`, warmupRuns, measuredRuns, {
      benchmarkDurationMs,
      runs: backgroundRuns,
      summary: summarizeRunGroup(backgroundRuns, benchmarkDurationMs),
    }),
  };
}

export type { RequestObservability };
