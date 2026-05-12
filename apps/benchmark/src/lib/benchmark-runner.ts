import { CogentEngine, type ModelLoadOptions, type ModelSource } from '@noumena-labs/cogentlm';
import type {
  BenchmarkRun,
  BenchmarkTraceReport,
  GroupResult,
  GroupSummary,
  MemorySnapshot,
  MixedLoadResult,
  RequestObservability,
  ScenarioDefinition,
  ScenarioResult,
} from './types';
import { measureAsync, round } from './utils';

type BenchmarkRuntimeOptions = NonNullable<ModelLoadOptions['runtime']>;

export interface ObservedQueryRun {
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
  return { ...observation };
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

export async function runObservedQuery(
  targetEngine: CogentEngine,
  prompt: string,
  options: {
    session: string;
    maxTokens: number;
    onToken?: (tokens: string[]) => void;
    media?: Uint8Array[];
    /**
     * When false, the chat call is made WITHOUT an `onToken` callback so the
     * engine runs in NONE emission mode (no per-token FFI/SAB activity).
     * TTFT and per-token timing then come from native runtime_observability
     * instead of JS-side wall-clock instrumentation.  Default true.
     */
    streamTokens?: boolean;
  }
): Promise<ObservedQueryRun> {
  const start = performance.now();
  let ttftMs: number | null = null;
  const tokenTimes: number[] = [];
  const sessionObserver = observeSessionCompletion(targetEngine, options.session);
  const streamTokens = options.streamTokens ?? true;

  try {
    const messages = options.media == null
      ? [{ role: 'user', content: prompt }]
      : { messages: [{ role: 'user', content: prompt }], media: options.media };
    // Pass `onToken` only when streaming is enabled.  Omitting it triggers
    // TOKEN_EMISSION_NONE on the engine side, which is the NONE-mode path.
    const chatOptions = streamTokens
      ? {
          maxTokens: options.maxTokens,
          session: options.session,
          onToken: (tokens: string[]) => {
            const elapsed = round(performance.now() - start);
            for (const token of tokens) {
              tokenTimes.push(elapsed);
              ttftMs ??= elapsed;
            }
            options.onToken?.(tokens);
          },
        }
      : {
          maxTokens: options.maxTokens,
          session: options.session,
        };
    const [output, observability] = await Promise.all([
      targetEngine.chat(messages, chatOptions),
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
    .map((run) => run.observability)
    .filter((value): value is RequestObservability => value != null);
  
  const totalInputTokens = runs.reduce(
    (acc, run) => acc + (run.observability?.inputTokens ?? 0),
    0
  );
  const totalGeneratedTokens = runs.reduce((acc, run) => acc + run.outputTokens, 0);
  const benchmarkDurationSeconds = benchmarkDurationMs > 0 ? benchmarkDurationMs / 1000 : 0;
  
  // Native decode TPS: output_tokens / decode_ms.
  // Includes GPU synchronization overhead to accurately reflect pure 
  // hardware-native inference performance, consistent with industry standards.
  const tpsValues = observations
    .map((item) =>
      item.decodeMs > 0 && item.outputTokens > 0
        ? (item.outputTokens * 1000) / item.decodeMs
        : 0
    )
    .filter((v) => v > 0);

  return {
    serving: {
      successfulRequests: runs.length,
      benchmarkDurationMs,
      totalInputTokens,
      totalGeneratedTokens,
      requestThroughputRps:
        benchmarkDurationSeconds > 0 ? round(runs.length / benchmarkDurationSeconds) : null,
      outputTokenThroughputTps:
        benchmarkDurationSeconds > 0 ? round(totalGeneratedTokens / benchmarkDurationSeconds) : null,
      totalTokenThroughputTps:
        benchmarkDurationSeconds > 0
          ? round((totalInputTokens + totalGeneratedTokens) / benchmarkDurationSeconds)
          : null,
    },
    runtime: {
      ttftMs: summarizeOptional(observations.map((item) => item.ttftMs)),
      itlAvgMs: summarizeOptional(observations.map((item) => item.itlAvgMs)),
      itlP99Ms: summarizeOptional(observations.map((item) => item.itlP99Ms)),
      tps: summarizeOptional(tpsValues),
      avgInputTokens: averageOptional(observations.map((item) => item.inputTokens)),
      avgOutputTokens: averageOptional(observations.map((item) => item.outputTokens)),
      avgPrefillMs: averageOptional(observations.map((item) => item.prefillMs)),
      avgDecodeMs: averageOptional(observations.map((item) => item.decodeMs)),
      avgNativeGpuMs: averageOptional(observations.map((item) => item.nativeGpuMs)),
      avgNativeSyncMs: averageOptional(observations.map((item) => item.nativeSyncMs)),
      avgNativeLogicMs: averageOptional(observations.map((item) => item.nativeLogicMs)),
      avgCacheHits: averageOptional(observations.map((item) => item.cacheHits)),
    },
  };
}

function createRun(
  label: string,
  wallMs: number,
  ttftMs: number | null,
  tokenTimes: number[],
  output: string,
  observability: RequestObservability | null
): BenchmarkRun {
  return {
    label,
    wallMs,
    ttftMs: observability?.ttftMs ?? null,
    itlAvgMs: observability?.itlAvgMs ?? null,
    itlP99Ms: observability?.itlP99Ms ?? null,
    tps:
      (observability?.decodeMs ?? 0) > 0 && (observability?.outputTokens ?? 0) > 0
        ? (observability!.outputTokens * 1000) / observability!.decodeMs
        : null,
    inputTokens: observability?.inputTokens ?? null,
    outputTokens: observability?.outputTokens ?? tokenTimes.length,
    outputLength: output.length,
    outputPreview: output.slice(0, 160).replace(/\s+/g, ' ').trim(),
    observability,
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
  setStatus: (s: string) => void,
  streamTokens: boolean = true
): Promise<{ benchmarkDurationMs: number; runs: BenchmarkRun[]; summary: GroupSummary }> {
  for (let i = 0; i < warmupRuns; i++) {
    setStatus(`${groupLabel}: warmup ${i + 1}/${warmupRuns}`);
    await targetEngine.chat([{ role: 'user', content: prompt }], {
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
      streamTokens,
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
  alreadyLoaded?: boolean,
  streamTokens: boolean = true
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
    setStatus,
    streamTokens
  );
  const hotFreshContext = await runPromptGroup(
    targetEngine,
    `${scenario.label}: hot fresh context`,
    scenario.prompt,
    scenario.outputTokenLimit,
    warmupRuns,
    measuredRuns,
    (index) => `${scenario.id}-fresh-${index}`,
    setStatus,
    streamTokens
  );
  const hotReuseContext = await runPromptGroup(
    targetEngine,
    `${scenario.label}: hot reused context`,
    scenario.prompt,
    scenario.outputTokenLimit,
    warmupRuns,
    measuredRuns,
    () => `${scenario.id}-reuse`,
    setStatus,
    streamTokens
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
  alreadyLoaded?: boolean,
  streamTokens: boolean = true
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
        streamTokens,
      }),
      runObservedQuery(targetEngine, definition.foreground.prompt, {
        maxTokens: definition.foreground.outputTokenLimit,
        session: `${definition.foreground.id}-warmup-${i}`,
        streamTokens,
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
        streamTokens,
      }),
      runObservedQuery(targetEngine, definition.foreground.prompt, {
        maxTokens: definition.foreground.outputTokenLimit,
        session: `${definition.foreground.id}-mixed-${i}`,
        streamTokens,
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

function collectGroupLogs(
  scenarioId: string,
  scenarioLabel: string,
  group: GroupResult
): BenchmarkTraceReport['logs'] {
  return group.runs.map((run) => ({
    scenarioId,
    scenarioLabel,
    groupId: group.id,
    groupLabel: group.label,
    runLabel: run.label,
    wallMs: run.wallMs,
    outputTokens: run.outputTokens,
    observability: run.observability,
  }));
}

export function buildBenchmarkTraceReport(
  scenarios: ScenarioResult[],
  mixedLoad: MixedLoadResult | null
): BenchmarkTraceReport {
  const logs: BenchmarkTraceReport['logs'] = [];
  for (const scenario of scenarios) {
    logs.push(
      ...collectGroupLogs(
        scenario.definition.id,
        scenario.definition.label,
        scenario.coldPrompt
      ),
      ...collectGroupLogs(
        scenario.definition.id,
        scenario.definition.label,
        scenario.hotFreshContext
      ),
      ...collectGroupLogs(
        scenario.definition.id,
        scenario.definition.label,
        scenario.hotReuseContext
      )
    );
  }
  if (mixedLoad?.foreground != null) {
    logs.push(
      ...collectGroupLogs(
        mixedLoad.definition.id,
        mixedLoad.definition.label,
        mixedLoad.foreground
      )
    );
  }
  if (mixedLoad?.background != null) {
    logs.push(
      ...collectGroupLogs(
        mixedLoad.definition.id,
        mixedLoad.definition.label,
        mixedLoad.background
      )
    );
  }

  const observations = logs
    .map((log) => log.observability)
    .filter((value): value is RequestObservability => value != null);

  // Native decode TPS: output_tokens / sum-of-llama_decode-wall-times.
  // Excludes JS yield, GPU sync, and streaming delivery — reflects pure
  // inference throughput, the number the engine reports about itself.
  const tpsValues = observations
    .map((item) =>
      item.decodeMs > 0 && item.outputTokens > 0
        ? (item.outputTokens * 1000) / item.decodeMs
        : 0
    )
    .filter((v) => v > 0);

  return {
    runCount: logs.length,
    logs,
    analysis: {
      ttftMs: summarizeOptional(observations.map((item) => item.ttftMs)),
      itlAvgMs: summarizeOptional(observations.map((item) => item.itlAvgMs)),
      itlP99Ms: summarizeOptional(observations.map((item) => item.itlP99Ms)),
      tps: summarizeOptional(tpsValues),
    },
  };
}

export type { RequestObservability };
