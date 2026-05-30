import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import {
  CogentClient,
  type BrowserRuntimeSmokeResult,
  type ModelInfo,
  type ModelSource,
  type ObservabilitySnapshot,
  type TokenBatch,
} from '@noumena-labs/cogentlm-browser';
import { MetricCard } from './components/MetricCard';
import {
  buildBenchmarkScenarios,
  buildMixedLoadDefinition,
  DEFAULT_BENCHMARK_PROMPTS,
  describeRuntimeBackend,
  ENCODER_DECODER_BENCHMARK_PROMPTS,
  summarizeMemorySnapshots,
  type BenchmarkPromptSet,
} from './lib/helpers';
import {
  buildBenchmarkTraceReport,
  captureBrowserMemorySnapshot,
  runMixedLoadBenchmark,
  runObservedRequest,
  runScenarioBenchmark,
  supportsConcurrentQueryApi,
  type BenchmarkTokenObserver,
} from './lib/benchmark-runner';
import {
  formatBytes,
  formatMs,
  fileToBase64,
  round,
  validateImageFile,
} from './lib/utils';
import {
  formatSize,
  getDefaultVariant,
  getModelById,
  getVariantPrimaryUrl,
  isVisionModel,
  MODEL_REGISTRY,
} from './lib/model-registry';
import type {
  BenchmarkOperation,
  BenchmarkTraceReport,
  GroupResult,
  MemorySnapshot,
  MetricSummary,
  MixedLoadResult,
  RequestObservability,
  ScenarioResult,
} from './lib/types';

declare global {
  interface Window {
    __cogentBench?: {
      getEnvironment(): Promise<Record<string, unknown>>;
      getRuntimeObservability(): ObservabilitySnapshot | null;
      getBackendObservability(): unknown;
      getBrowserRuntimeSmoke(): {
        result: BrowserRuntimeSmokeResult | null;
        error: string | null;
      };
      runBrowserRuntimeSmoke(): Promise<BrowserRuntimeSmokeResult>;
      getLastReport(): BenchmarkReport | null;
    };
  }
}

interface BenchmarkReport {
  schema: 'cogent.benchmark.browser.v9';
  generatedAt: string;
  model: ModelInfo | null;
  source: { label: string; bytes: number | null };
  settings: {
    operation: BenchmarkOperation;
    prompt: string;
    tokenCount: number;
    warmupRuns: number;
    measuredRuns: number;
    runtime: ReturnType<typeof getDefaultRuntimeOptions>;
  };
  observability: ObservabilitySnapshot | null;
  scenarios: ScenarioResult[];
  mixedLoad: MixedLoadResult | null;
  memory: {
    snapshots: MemorySnapshot[];
    summary: ReturnType<typeof summarizeMemorySnapshots>;
  };
  trace: BenchmarkTraceReport;
}

function getDefaultRuntimeOptions() {
  return {
    placement: {
      gpu_layers: 'all' as const,
    },
    context: {
      n_parallel: 1,
    },
    cache: {
      mode: 'live_slot_prefix' as const,
      max_session_entries: 8,
    },
  };
}

async function inspectBrowserEnvironment(): Promise<Record<string, unknown>> {
  const gpu = navigator.gpu;
  if (gpu == null) {
    return {
      hasNavigatorGpu: false,
      adapterAvailable: false,
      crossOriginIsolated: window.crossOriginIsolated,
    };
  }

  const adapter = await gpu.requestAdapter();
  const adapterWithInfo = adapter as (GPUAdapter & {
    requestAdapterInfo?: () => Promise<{
      vendor?: string;
      architecture?: string;
      device?: string;
      description?: string;
    }>;
  }) | null;
  const info = adapterWithInfo == null
    ? null
    : await adapterWithInfo.requestAdapterInfo?.().catch(() => null);
  return {
    hasNavigatorGpu: true,
    adapterAvailable: adapter != null,
    crossOriginIsolated: window.crossOriginIsolated,
    adapterInfo: info == null ? null : {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    },
  };
}

function sourceLabel(source: ModelSource): string {
  if (typeof source === 'string') return source;
  if (source instanceof File) return source.name;
  if (Array.isArray(source)) {
    const first = source[0];
    return typeof first === 'string' ? first : first?.name ?? 'model shards';
  }
  if ('model' in source) {
    return sourceLabel(source.model);
  }
  return 'model.gguf';
}

function isModelSourceObject(
  source: ModelSource
): source is Extract<ModelSource, { model: ModelSource }> {
  return (
    typeof source === 'object' &&
    source != null &&
    !(source instanceof File) &&
    !Array.isArray(source) &&
    'model' in source
  );
}

function sourceKey(source: ModelSource): string {
  if (typeof source === 'string') return `string:${source}`;
  if (source instanceof File) {
    return `file:${source.name}:${source.size}:${source.lastModified}`;
  }
  if (Array.isArray(source)) {
    return `array:[${source.map((item) => sourceKey(item)).join('|')}]`;
  }
  if (!isModelSourceObject(source)) {
    return 'unknown:model-source';
  }
  return `pair:model=${sourceKey(source.model)};projector=${source.projector == null ? 'none' : sourceKey(source.projector)
    }`;
}

function withProjector(source: ModelSource, projector?: string | File): ModelSource {
  if (projector == null) return source;
  if (
    typeof source === 'object' &&
    source != null &&
    !(source instanceof File) &&
    !Array.isArray(source)
  ) {
    return { ...source, projector };
  }
  return { model: source, projector };
}

async function fetchImageBytes(source: string): Promise<Uint8Array[]> {
  const response = await fetch(source);
  return [new Uint8Array(await response.arrayBuffer())];
}

function formatSummary(summary: MetricSummary | null, unit: string = 'ms'): string {
  if (summary == null) return 'n/a';
  return `${round(summary.meanMs)}${unit} avg / ${round(summary.p99Ms)}${unit} p99`;
}

function formatTps(value: number | null | undefined): string {
  return value == null ? 'n/a' : `${round(value)} tok/s`;
}

function downloadJson(filename: string, value: unknown): void {
  const blob = new Blob([JSON.stringify(value, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

const DEFAULT_QUERY_PROMPT = 'Describe how to benchmark browser-hosted inference.';
const ENCODER_DECODER_QUERY_PROMPT = 'translate English to German: The house is wonderful.';
const DEFAULT_TOKEN_COUNT = 64;
const ENCODER_DECODER_TOKEN_COUNT = 32;

function defaultOperationForModel(model: ModelInfo): BenchmarkOperation {
  const capabilities = model.capabilities;
  if (capabilities == null) {
    return model.chatTemplate == null ? 'query' : 'chat';
  }
  if (capabilities.supportsEmbeddings && !capabilities.supportsTextGeneration) {
    return 'embed';
  }
  if (capabilities.modelClass === 'encoder_decoder') {
    return 'query';
  }
  return capabilities.hasChatTemplate ? 'chat' : 'query';
}

function modelSupportsOperation(model: ModelInfo, operation: BenchmarkOperation): boolean {
  const capabilities = model.capabilities;
  if (capabilities == null) {
    return true;
  }
  if (operation === 'embed') {
    return capabilities.supportsEmbeddings;
  }
  if (operation === 'chat') {
    return capabilities.supportsTextGeneration && capabilities.hasChatTemplate;
  }
  return capabilities.supportsTextGeneration;
}

function yesNo(value: boolean | undefined): string {
  return value == null ? 'unknown' : value ? 'yes' : 'no';
}

function isEncoderDecoderModel(model: ModelInfo | null): boolean {
  return model?.capabilities?.modelClass === 'encoder_decoder';
}

function benchmarkPromptSetForModel(model: ModelInfo | null): BenchmarkPromptSet {
  return isEncoderDecoderModel(model)
    ? ENCODER_DECODER_BENCHMARK_PROMPTS
    : DEFAULT_BENCHMARK_PROMPTS;
}

function defaultPromptForModel(model: ModelInfo | null): string {
  return isEncoderDecoderModel(model) ? ENCODER_DECODER_QUERY_PROMPT : DEFAULT_QUERY_PROMPT;
}

function effectivePromptForModel(model: ModelInfo | null, currentPrompt: string): string {
  if (currentPrompt === DEFAULT_QUERY_PROMPT || currentPrompt === ENCODER_DECODER_QUERY_PROMPT) {
    return defaultPromptForModel(model);
  }
  return currentPrompt;
}

function defaultTokenCountForModel(model: ModelInfo | null): number {
  return isEncoderDecoderModel(model) ? ENCODER_DECODER_TOKEN_COUNT : DEFAULT_TOKEN_COUNT;
}

function effectiveTokenCountForModel(model: ModelInfo | null, currentTokenCount: number): number {
  if (
    currentTokenCount === DEFAULT_TOKEN_COUNT ||
    currentTokenCount === ENCODER_DECODER_TOKEN_COUNT
  ) {
    return defaultTokenCountForModel(model);
  }
  return currentTokenCount;
}

export default function App() {
  const [client, setClient] = useState<CogentClient | null>(null);
  const [status, setStatus] = useState('booting');
  const [isBusy, setIsBusy] = useState(false);
  const [modelType, setModelType] = useState<'registry' | 'url' | 'file'>('registry');
  const [selectedRegistryId, setSelectedRegistryId] = useState(MODEL_REGISTRY[0].id);
  const selectedModel = getModelById(selectedRegistryId) ?? MODEL_REGISTRY[0];
  const selectedVariant = getDefaultVariant(selectedModel);
  const [modelUrl, setModelUrl] = useState(getVariantPrimaryUrl(selectedVariant));
  const [projectorUrl, setProjectorUrl] = useState('');
  const [operation, setOperation] = useState<BenchmarkOperation>('chat');
  const [prompt, setPrompt] = useState(DEFAULT_QUERY_PROMPT);
  const [tokenCount, setTokenCount] = useState(DEFAULT_TOKEN_COUNT);
  const [warmupRuns, setWarmupRuns] = useState(1);
  const [measuredRuns, setMeasuredRuns] = useState(3);
  // Three transport modes:
  //   off    — client submits TOKEN_EMISSION_NONE; nothing crosses to main.
  //   tokens — client submits StreamingBuffer; main drains token batches
  //            without DOM work.  This isolates callback/token-plane cost from
  //            rendering cost.
  //   render — client submits StreamingBuffer; main drains SAB and writes
  //            textContent as token batches arrive. Pays the UI/rendering tax.
  type StreamMode = 'off' | 'tokens' | 'render';
  const [streamMode, setStreamMode] = useState<StreamMode>('render');
  const streamTokens = streamMode !== 'off';
  const [imageSource, setImageSource] = useState('');
  const [imageEnabled, setImageEnabled] = useState(false);
  const [currentModel, setCurrentModel] = useState<ModelInfo | null>(null);
  const [installedModels, setInstalledModels] = useState<ModelInfo[]>([]);
  const [observability, setObservability] = useState<ObservabilitySnapshot | null>(null);
  const [response, setResponse] = useState('');
  const [lastRun, setLastRun] = useState<{
    operation: BenchmarkOperation;
    outputKind: 'text' | 'embedding';
    wallMs: number;
    ttftMs: number | null;
    tokens: number;
    tps: number | null;
    prefillTokens: number | null;
    prefillTps: number | null;
    embeddingDimensions: number | null;
    embeddingPooling: string | null;
    embeddingNormalized: boolean | null;
    observability: RequestObservability | null;
  } | null>(null);
  const [lastLoadMs, setLastLoadMs] = useState(0);
  const [sourceInfo, setSourceInfo] = useState<{ label: string; bytes: number } | null>(null);
  const [scenarioResults, setScenarioResults] = useState<ScenarioResult[]>([]);
  const [mixedLoadResult, setMixedLoadResult] = useState<MixedLoadResult | null>(null);
  const [memorySnapshots, setMemorySnapshots] = useState<MemorySnapshot[]>([]);
  const [benchmarkReport, setBenchmarkReport] = useState<BenchmarkReport | null>(null);
  const [browserSmoke, setBrowserSmoke] = useState<BrowserRuntimeSmokeResult | null>(null);
  const [browserSmokeError, setBrowserSmokeError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const projectorFileInputRef = useRef<HTMLInputElement>(null);
  const loadedSourceKeyRef = useRef<string | null>(null);
  const responseElementRef = useRef<HTMLDivElement>(null);

  const createResponseRenderer = (maxStreams = 1, flushMode: 'frame' | 'immediate' = 'frame') => {
    let frame = 0;
    const order: string[] = [];
    const textByLabel = new Map<string, string>();

    const flush = () => {
      frame = 0;
      if (responseElementRef.current == null) {
        return;
      }
      responseElementRef.current.textContent = order
        .map((label) => {
          const text = textByLabel.get(label) ?? '';
          return maxStreams === 1 ? text : `${label}\n${text}`;
        })
        .join('\n\n');
    };

    const schedule = () => {
      if (flushMode === 'immediate') {
        flush();
        return;
      }
      if (frame === 0) {
        frame = window.requestAnimationFrame(flush);
      }
    };

    return {
      reset() {
        if (frame !== 0) {
          window.cancelAnimationFrame(frame);
          frame = 0;
        }
        order.length = 0;
        textByLabel.clear();
        if (responseElementRef.current != null) {
          responseElementRef.current.textContent = '';
        }
      },
      start(label: string) {
        if (!textByLabel.has(label)) {
          order.push(label);
        }
        while (order.length > maxStreams) {
          const dropped = order.shift();
          if (dropped != null) {
            textByLabel.delete(dropped);
          }
        }
        textByLabel.set(label, '');
        schedule();
      },
      append(label: string, batch: Pick<TokenBatch, 'text'>) {
        textByLabel.set(label, `${textByLabel.get(label) ?? ''}${batch.text}`);
        schedule();
      },
      finish() {
        if (frame !== 0) {
          window.cancelAnimationFrame(frame);
          flush();
        }
      },
    };
  };

  const runBrowserRuntimeSmoke = async (): Promise<BrowserRuntimeSmokeResult> => {
    setBrowserSmokeError(null);
    const result = await CogentClient.browserRuntimeSmoke();
    setBrowserSmoke(result);
    return result;
  };

  useEffect(() => {
    let disposed = false;
    let created: CogentClient | null = null;
    let unsubscribe: (() => void) | null = null;

    void (async () => {
      try {
        const nextClient = new CogentClient();
        if (disposed) {
          await nextClient.close();
          return;
        }
        created = nextClient;
        let pendingSnapshot: any = null;
        const updateInterval = setInterval(() => {
          if (pendingSnapshot) {
            setObservability(pendingSnapshot);
            setCurrentModel(pendingSnapshot.model);
            pendingSnapshot = null;
          }
        }, 250); // Steady 4 FPS for metrics to keep main thread clear

        unsubscribe = nextClient.observability.subscribe((event) => {
          pendingSnapshot = event.snapshot;
        });

        // Cleanup interval on dispose
        const originalUnsubscribe = unsubscribe;
        unsubscribe = () => {
          clearInterval(updateInterval);
          originalUnsubscribe();
        };
        setClient(nextClient);
        setCurrentModel(nextClient.models.current());
        setObservability(nextClient.observability.current());
        setInstalledModels(await nextClient.models.list());
        setStatus('idle');
        void CogentClient.browserRuntimeSmoke()
          .then((result) => {
            if (!disposed) {
              setBrowserSmoke(result);
              setBrowserSmokeError(null);
            }
          })
          .catch((error) => {
            if (!disposed) {
              setBrowserSmoke(null);
              setBrowserSmokeError(error instanceof Error ? error.message : String(error));
            }
          });
      } catch (error) {
        if (disposed) {
          return;
        }
        setStatus(error instanceof Error ? error.message : String(error));
      }
    })();

    return () => {
      disposed = true;
      unsubscribe?.();
      if (created != null) {
        void created.close();
      }
    };
  }, []);

  useEffect(() => {
    window.__cogentBench = {
      getEnvironment: inspectBrowserEnvironment,
      getRuntimeObservability: () => observability,
      getBackendObservability: () => browserSmoke?.backend ?? observability?.profile ?? null,
      getBrowserRuntimeSmoke: () => ({
        result: browserSmoke,
        error: browserSmokeError,
      }),
      runBrowserRuntimeSmoke,
      getLastReport: () => benchmarkReport,
    };

    return () => {
      if (window.__cogentBench?.runBrowserRuntimeSmoke === runBrowserRuntimeSmoke) {
        delete window.__cogentBench;
      }
    };
  }, [benchmarkReport, observability, browserSmoke, browserSmokeError]);

  const projectorOverride = (): string | File | undefined => {
    const file = projectorFileInputRef.current?.files?.[0];
    if (file != null) return file;
    const url = projectorUrl.trim();
    return url.length > 0 ? url : undefined;
  };

  const modelSource = (): ModelSource | null => {
    const projector = projectorOverride();
    if (modelType === 'registry') return withProjector(selectedVariant.source, projector);
    if (modelType === 'url') {
      return modelUrl.trim().length > 0 ? withProjector(modelUrl.trim(), projector) : null;
    }
    const files = Array.from(fileInputRef.current?.files ?? []);
    if (files.length === 0) return null;
    return withProjector(files.length === 1 ? files[0] : files, projector);
  };

  const refreshModels = async (targetClient: CogentClient) => {
    setCurrentModel(targetClient.models.current());
    setObservability(targetClient.observability.current());
    setInstalledModels(await targetClient.models.list());
  };

  const loadModelSelection = async (
    targetClient: CogentClient,
    source: ModelSource
  ): Promise<ModelInfo> => {
    const start = performance.now();
    const info = await targetClient.models.load(source, {
      observability: 'profile',
      runtime: getDefaultRuntimeOptions(),
      onProgress: (progress) => {
        if (progress.phase === 'download') {
          setStatus(`Downloading model ${Math.floor(progress.percent ?? 0)}%`);
        } else if (progress.phase === 'load') {
          setStatus('Loading into memory');
        }
      },
    });
    setLastLoadMs(round(performance.now() - start));
    setSourceInfo({ label: sourceLabel(source), bytes: info.bytes });
    loadedSourceKeyRef.current = sourceKey(source);
    if (!modelSupportsOperation(info, operation)) {
      setOperation(defaultOperationForModel(info));
    }
    const modelPrompt = effectivePromptForModel(info, prompt);
    if (modelPrompt !== prompt) {
      setPrompt(modelPrompt);
    }
    const modelTokenCount = effectiveTokenCountForModel(info, tokenCount);
    if (modelTokenCount !== tokenCount) {
      setTokenCount(modelTokenCount);
    }
    await refreshModels(targetClient);
    return info;
  };

  const loadSelectedModel = async () => {
    if (client == null) return;
    const source = modelSource();
    if (source == null) {
      setStatus('Select a model source first.');
      return;
    }
    setIsBusy(true);
    try {
      const info = await loadModelSelection(client, source);
      setStatus(info.status === 'ready' ? `loaded ${info.name}` : `${info.name}: ${info.status}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsBusy(false);
    }
  };

  const runQuery = async () => {
    if (client == null) return;
    setIsBusy(true);
    setResponse('');
    setLastRun(null);
    try {
      const source = modelSource();
      if (source == null) {
        setStatus('Select a model source first.');
        return;
      }
      const nextSourceKey = sourceKey(source);
      const loadedModel = client.models.current();
      let requestOperation = operation;
      let requestPrompt = prompt;
      let requestTokenCount = tokenCount;
      if (
        loadedModel == null ||
        loadedModel.status !== 'ready' ||
        (loadedSourceKeyRef.current != null && loadedSourceKeyRef.current !== nextSourceKey)
      ) {
        const info = await loadModelSelection(client, source);
        if (!modelSupportsOperation(info, requestOperation)) {
          requestOperation = defaultOperationForModel(info);
        }
        requestPrompt = effectivePromptForModel(info, requestPrompt);
        requestTokenCount = effectiveTokenCountForModel(info, requestTokenCount);
        setStatus(info.status === 'ready' ? `loaded ${info.name}` : `${info.name}: ${info.status}`);
      } else if (!modelSupportsOperation(loadedModel, requestOperation)) {
        requestOperation = defaultOperationForModel(loadedModel);
        setOperation(requestOperation);
        requestPrompt = effectivePromptForModel(loadedModel, requestPrompt);
        requestTokenCount = effectiveTokenCountForModel(loadedModel, requestTokenCount);
      } else {
        requestPrompt = effectivePromptForModel(loadedModel, requestPrompt);
        requestTokenCount = effectiveTokenCountForModel(loadedModel, requestTokenCount);
      }
      if (requestPrompt !== prompt) {
        setPrompt(requestPrompt);
      }
      if (requestTokenCount !== tokenCount) {
        setTokenCount(requestTokenCount);
      }

      const image =
        requestOperation !== 'embed' && imageEnabled && imageSource.trim().length > 0
          ? await fetchImageBytes(imageSource.trim())
          : undefined;
      const effectiveStreamTokens = requestOperation !== 'embed' && streamTokens;
      const queryRenderer = effectiveStreamTokens && streamMode === 'render' ? createResponseRenderer(1, 'frame') : null;
      queryRenderer?.reset();
      queryRenderer?.start('response');
      const onTokenBatch =
        requestOperation === 'embed'
          ? undefined
          : streamMode === 'render'
          ? (batch: TokenBatch) => {
            queryRenderer?.append('response', batch);
          }
          : streamMode === 'tokens'
            ? (_batch: TokenBatch) => {
              /* Token stream drained with no DOM work. */
            }
            : undefined;
      try {
        const run = await runObservedRequest(client, requestPrompt, {
          operation: requestOperation,
          maxTokens: requestTokenCount,
          session: `query-${Date.now()}`,
          media: image,
          streamTokens: effectiveStreamTokens,
          tokenFlush: streamMode === 'render' ? 'token' : 'batch',
          onTokenBatch,
        });
        setResponse(run.output); // Sync React state at the end
        setObservability(client.observability.current());
        setLastRun({
          operation: run.operation,
          outputKind: run.outputKind,
          wallMs: run.wallMs,
          ttftMs: run.observability?.ttftMs ?? run.ttftMs,
          tokens:
            run.outputKind === 'embedding'
              ? run.embeddingDimensions ?? 0
              : run.observability?.outputTokens ?? run.tokenTimes.length,
          tps:
            (run.observability?.decodeMs ?? 0) > 0 &&
              (run.observability?.outputTokens ?? 0) > 0
              ? (run.observability!.outputTokens * 1000) /
              run.observability!.decodeMs
              : null,
          prefillTps:
            (run.observability?.prefillMs ?? 0) >= 0.1 &&
              (run.observability?.prefillTokens ?? 0) >= 1
              ? (run.observability!.prefillTokens * 1000) /
              run.observability!.prefillMs
              : null,
          prefillTokens: run.observability?.prefillTokens ?? null,
          embeddingDimensions: run.embeddingDimensions,
          embeddingPooling: run.embeddingPooling,
          embeddingNormalized: run.embeddingNormalized,
          observability: run.observability,
        });
      } finally {
        queryRenderer?.finish();
      }
      setStatus('idle');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsBusy(false);
    }
  };

  const runBenchmark = async () => {
    if (client == null) return;
    const source = modelSource();
    if (source == null) {
      setStatus('Select a model source first.');
      return;
    }

    setIsBusy(true);
    setScenarioResults([]);
    setMixedLoadResult(null);
    setMemorySnapshots([]);
    setBenchmarkReport(null);
    let benchmarkOperation = operation;
    const loadedModel = client.models.current();
    let benchmarkPrompt = prompt;
    let benchmarkTokenCount = tokenCount;
    if (loadedModel != null && !modelSupportsOperation(loadedModel, benchmarkOperation)) {
      benchmarkOperation = defaultOperationForModel(loadedModel);
      setOperation(benchmarkOperation);
    }
    let effectiveStreamTokens = false;
    let benchmarkRenderer: ReturnType<typeof createResponseRenderer> | null = null;
    let benchmarkTokenObserver: BenchmarkTokenObserver | undefined;

    try {
      const info = await loadModelSelection(client, source);
      if (!modelSupportsOperation(info, benchmarkOperation)) {
        benchmarkOperation = defaultOperationForModel(info);
      }
      benchmarkPrompt = effectivePromptForModel(info, benchmarkPrompt);
      benchmarkTokenCount = effectiveTokenCountForModel(info, benchmarkTokenCount);
      if (benchmarkPrompt !== prompt) {
        setPrompt(benchmarkPrompt);
      }
      if (benchmarkTokenCount !== tokenCount) {
        setTokenCount(benchmarkTokenCount);
      }
      const promptSet = benchmarkPromptSetForModel(info);
      effectiveStreamTokens = benchmarkOperation !== 'embed' && streamTokens;
      benchmarkRenderer = effectiveStreamTokens && streamMode === 'render'
        ? createResponseRenderer(2, 'frame')
        : null;
      const benchmarkTokenFlush = streamMode === 'render' ? 'token' : 'batch';
      benchmarkTokenObserver =
        effectiveStreamTokens && streamMode === 'render'
          ? {
            onRunStart: (label) => {
              benchmarkRenderer?.start(label);
            },
            onTokenBatch: (label, batch) => {
              benchmarkRenderer?.append(label, batch);
            },
          }
          : undefined;
      benchmarkRenderer?.reset();
      const snapshots: MemorySnapshot[] = [];
      snapshots.push(await captureBrowserMemorySnapshot('before-benchmark', true));

      const scenarios = buildBenchmarkScenarios(benchmarkPrompt, benchmarkTokenCount, promptSet);
      const results: ScenarioResult[] = [];
      for (const scenario of scenarios) {
        results.push(
          await runScenarioBenchmark(
            client,
            benchmarkOperation,
            scenario,
            source,
            warmupRuns,
            measuredRuns,
            getDefaultRuntimeOptions(),
            setStatus,
            true,
            effectiveStreamTokens,
            benchmarkTokenFlush,
            benchmarkTokenObserver
          )
        );
      }

      snapshots.push(await captureBrowserMemorySnapshot('after-scenarios', true));

      let mixed: MixedLoadResult | null = null;
      if (benchmarkOperation !== 'embed' && supportsConcurrentQueryApi(client)) {
        mixed = await runMixedLoadBenchmark(
          client,
          benchmarkOperation,
          buildMixedLoadDefinition(benchmarkOperation, promptSet),
          source,
          warmupRuns,
          measuredRuns,
          getDefaultRuntimeOptions(),
          setStatus,
          true,
          effectiveStreamTokens,
          benchmarkTokenFlush,
          benchmarkTokenObserver
        );
        snapshots.push(await captureBrowserMemorySnapshot('after-mixed-load', true));
      }
      benchmarkRenderer?.finish();
      const trace = buildBenchmarkTraceReport(results, mixed);

      const report: BenchmarkReport = {
        schema: 'cogent.benchmark.browser.v9',
        generatedAt: new Date().toISOString(),
        model: info,
        source: {
          label: sourceLabel(source),
          bytes: info.bytes,
        },
        settings: {
          operation: benchmarkOperation,
          prompt: benchmarkPrompt,
          tokenCount: benchmarkTokenCount,
          warmupRuns,
          measuredRuns,
          runtime: getDefaultRuntimeOptions(),
        },
        observability: client.observability.current(),
        scenarios: results,
        mixedLoad: mixed,
        memory: {
          snapshots,
          summary: summarizeMemorySnapshots(snapshots),
        },
        trace,
      };

      setScenarioResults(results);
      setMixedLoadResult(mixed);
      setMemorySnapshots(snapshots);
      setBenchmarkReport(report);
      setStatus('benchmark complete');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      benchmarkRenderer?.finish();
      setIsBusy(false);
    }
  };

  const uploadImage = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file == null) return;
    const validation = validateImageFile(file);
    if (!validation.valid) {
      setStatus(validation.error ?? 'Invalid image.');
      return;
    }
    setImageSource(await fileToBase64(file));
    setImageEnabled(true);
  };

  const renderGroup = (title: string, group: GroupResult) => (
    <div className="result-card" key={`${group.id}-${title}`}>
      <h3>{title}</h3>
      <div className="metric-group-title">Latency (User Experience)</div>
      <div className="metric-grid">
        <MetricCard label="Operation" value={group.runs[0]?.operation ?? operation} />
        <MetricCard label="TTFT" value={formatSummary(group.summary.runtime.ttftMs)} />
        <MetricCard label="ITL Avg" value={formatSummary(group.summary.runtime.itlAvgMs)} />
        <MetricCard label="ITL P99" value={formatSummary(group.summary.runtime.itlP99Ms)} />
        <MetricCard label="Prefill TPS" value={formatSummary(group.summary.runtime.prefillTps, 'tok/s')} />
        <MetricCard label="Decode TPS" value={formatSummary(group.summary.runtime.tps, 'tok/s')} />
      </div>

      <div className="metric-group-title">Compute Profile</div>
      <div className="metric-grid">
        <MetricCard
          label="Prefill"
          value={group.summary.runtime.avgPrefillMs != null ? formatMs(group.summary.runtime.avgPrefillMs) : 'n/a'}
        />
        <MetricCard
          label="Decode"
          value={group.summary.runtime.avgDecodeMs != null ? formatMs(group.summary.runtime.avgDecodeMs) : 'n/a'}
        />
        <MetricCard
          label="Input Tokens"
          value={group.summary.runtime.avgInputTokens ?? 'n/a'}
        />
        <MetricCard
          label="Prefill Tokens"
          value={group.summary.runtime.avgPrefillTokens ?? 'n/a'}
        />
        <MetricCard
          label="Output Tokens"
          value={group.summary.runtime.avgOutputTokens ?? 'n/a'}
        />
        {group.runs[0]?.outputKind === 'embedding' ? (
          <>
            <MetricCard
              label="Dimensions"
              value={group.runs[0].embeddingDimensions ?? 'n/a'}
            />
            <MetricCard
              label="Pooling"
              value={group.runs[0].embeddingPooling ?? 'n/a'}
            />
          </>
        ) : null}
      </div>

      <div className="metric-group-title">Native Pipeline</div>
      <div className="metric-grid">
        <MetricCard
          label="GPU Wall"
          value={group.summary.runtime.avgNativeGpuMs != null ? formatMs(group.summary.runtime.avgNativeGpuMs) : 'n/a'}
        />
        <MetricCard
          label="Sync/Wait"
          value={group.summary.runtime.avgNativeSyncMs != null ? formatMs(group.summary.runtime.avgNativeSyncMs) : 'n/a'}
        />
        <MetricCard
          label="CPU Logic"
          value={group.summary.runtime.avgNativeLogicMs != null ? formatMs(group.summary.runtime.avgNativeLogicMs) : 'n/a'}
        />
        <MetricCard
          label="Cache Hits"
          value={group.summary.runtime.avgCacheHits ?? 'n/a'}
        />
      </div>

      {group.runs[0]?.outputPreview ? (
        <p className="result-preview">{group.runs[0].outputPreview}</p>
      ) : null}
    </div>
  );

  return (
    <div className="shell">
      <header className="hero">
        <div className="eyebrow">Browser Benchmark</div>
        <h1>CogentClient Minimal API</h1>
        <p>
          Load with <code>client.models.load()</code>, inspect with{' '}
          <code>client.models.current()</code>, run with <code>client.chat()</code>,{' '}
          <code>client.query()</code>, or <code>client.embed()</code>, consume{' '}
          <code>.response</code> and <code>.tokens</code>, and benchmark with
          the same minimal surface.
        </p>
      </header>

      <div className="layout">
        <div className="column">
          <section className="section">
            <div className="section-header">
              <h2>Model</h2>
            </div>
            <div className="field-grid">
              <div className="row">
                <label>Source Type</label>
                <div className="toggle-group">
                  <button
                    type="button"
                    className={`toggle-item ${modelType === 'registry' ? 'active' : ''}`}
                    onClick={() => setModelType('registry')}
                  >
                    Library
                  </button>
                  <button
                    type="button"
                    className={`toggle-item ${modelType === 'url' ? 'active' : ''}`}
                    onClick={() => setModelType('url')}
                  >
                    URL
                  </button>
                  <button
                    type="button"
                    className={`toggle-item ${modelType === 'file' ? 'active' : ''}`}
                    onClick={() => setModelType('file')}
                  >
                    Local File
                  </button>
                </div>
              </div>
              <div className="row">
                {modelType === 'registry' ? (
                  <select
                    value={selectedRegistryId}
                    onChange={(event) => {
                      const entry = getModelById(event.target.value);
                      if (entry == null) return;
                      setSelectedRegistryId(entry.id);
                      setModelUrl(getVariantPrimaryUrl(getDefaultVariant(entry)));
                      setImageEnabled(isVisionModel(entry));
                    }}
                    style={{ flex: 1 }}
                  >
                    {MODEL_REGISTRY.map((model) => {
                      const variant = getDefaultVariant(model);
                      return (
                        <option key={model.id} value={model.id}>
                          {model.name} ({model.parameterCount}) -{' '}
                          {formatSize(variant.sizeBytes + (variant.projectorSizeBytes ?? 0))}
                        </option>
                      );
                    })}
                  </select>
                ) : modelType === 'url' ? (
                  <input
                    value={modelUrl}
                    onChange={(event) => setModelUrl(event.target.value)}
                    placeholder="https://.../model.gguf"
                  />
                ) : (
                  <input type="file" accept=".gguf" ref={fileInputRef} multiple />
                )}
              </div>
              <div className="row">
                <label>Optional Projector</label>
                <input
                  value={projectorUrl}
                  onChange={(event) => setProjectorUrl(event.target.value)}
                  placeholder="https://.../mmproj.gguf"
                />
                <input type="file" accept=".gguf" ref={projectorFileInputRef} />
              </div>
              <div className="button-row">
                <button
                  type="button"
                  onClick={loadSelectedModel}
                  disabled={isBusy || client == null}
                >
                  Load Model
                </button>
              </div>
            </div>
          </section>

          <section className="section">
            <div className="section-header">
              <h2>Request</h2>
            </div>
            <div className="field-grid">
              <div className="row">
                <label>Operation</label>
                <div className="toggle-group">
                  <button
                    type="button"
                    className={`toggle-item ${operation === 'chat' ? 'active' : ''}`}
                    onClick={() => setOperation('chat')}
                  >
                    Chat
                  </button>
                  <button
                    type="button"
                    className={`toggle-item ${operation === 'query' ? 'active' : ''}`}
                    onClick={() => setOperation('query')}
                  >
                    Query
                  </button>
                  <button
                    type="button"
                    className={`toggle-item ${operation === 'embed' ? 'active' : ''}`}
                    onClick={() => setOperation('embed')}
                  >
                    Embed
                  </button>
                </div>
              </div>
              <div className="row">
                <label>Prompt</label>
                <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
              </div>
              <div className="row">
                <label>Max Tokens</label>
                <input
                  type="number"
                  value={tokenCount}
                  disabled={operation === 'embed'}
                  onChange={(event) =>
                    setTokenCount(Number.parseInt(event.target.value, 10) || 0)
                  }
                />
              </div>
              {operation !== 'embed' ? (
                <div className="row">
                  <label>
                    <input
                      type="checkbox"
                      checked={imageEnabled}
                      onChange={(event) => setImageEnabled(event.target.checked)}
                    />{' '}
                    Attach Image
                  </label>
                </div>
              ) : null}
              {operation !== 'embed' && imageEnabled ? (
                <div className="row">
                  <label>Image URL or File</label>
                  <div className="field-grid">
                    <input
                      value={imageSource.startsWith('data:') ? '' : imageSource}
                      onChange={(event) => setImageSource(event.target.value)}
                      placeholder="https://.../image.jpg"
                    />
                    <input
                      type="file"
                      accept="image/jpeg,image/png,image/webp,image/gif"
                      onChange={uploadImage}
                      style={{ padding: '8px' }}
                    />
                  </div>
                </div>
              ) : null}
              <div className="button-row">
                <button
                  type="button"
                  onClick={runQuery}
                  disabled={isBusy || client == null}
                >
                  Run Request
                </button>
              </div>
            </div>
          </section>

          <section className="section">
            <div className="section-header">
              <h2>Benchmark</h2>
            </div>
            <div className="field-grid">
              <div className="field-grid field-grid-compact">
                <div className="row">
                  <label>Warmup Runs</label>
                  <input
                    type="number"
                    value={warmupRuns}
                    onChange={(event) =>
                      setWarmupRuns(Number.parseInt(event.target.value, 10) || 0)
                    }
                  />
                </div>
                <div className="row">
                  <label>Measured Runs</label>
                  <input
                    type="number"
                    value={measuredRuns}
                    onChange={(event) =>
                      setMeasuredRuns(Number.parseInt(event.target.value, 10) || 0)
                    }
                  />
                </div>
                <div className="row">
                  <label>Observability</label>
                  <input value={observability?.mode ?? 'off'} readOnly />
                </div>
                <div className="row">
                  <label title="Off = native NONE (no emission). Tokens = client exposes batched tokens for JS-side draining with no DOM work. Render = client flushes token-sized batches and writes textContent as they arrive.">
                    Stream Tokens
                  </label>
                  <div className="toggle-group">
                    <button
                      type="button"
                      className={`toggle-item ${streamMode === 'off' ? 'active' : ''}`}
                      onClick={() => setStreamMode('off')}
                      title="Off — NONE (native baseline)"
                    >
                      Off
                    </button>
                    <button
                      type="button"
                      className={`toggle-item ${streamMode === 'tokens' ? 'active' : ''}`}
                      onClick={() => setStreamMode('tokens')}
                      title="On — drain token batches without DOM rendering"
                    >
                      Tokens
                    </button>
                    <button
                      type="button"
                      className={`toggle-item ${streamMode === 'render' ? 'active' : ''}`}
                      onClick={() => setStreamMode('render')}
                      title="On — rendered (with DOM)"
                    >
                      Render
                    </button>
                  </div>
                </div>
                <div className="row">
                  <label>Active Transport</label>
                  <input
                    value={observability?.runtime?.execution.tokenPath ?? 'pending'}
                    readOnly
                  />
                </div>
              </div>
              <div className="button-row">
                <button
                  type="button"
                  onClick={runBenchmark}
                  disabled={isBusy || client == null}
                >
                  Run Benchmark
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() =>
                    benchmarkReport == null
                      ? undefined
                      : downloadJson('cogent-benchmark.json', benchmarkReport)
                  }
                  disabled={benchmarkReport == null}
                >
                  Export JSON
                </button>
              </div>
            </div>
          </section>

          <p className="status">Status: {status}</p>
        </div>

        <div className="column">
          <section className="section">
            <div className="section-header">
              <h2>State</h2>
            </div>
            <div className="metric-grid">
              <MetricCard
                label="Current Model"
                value={currentModel?.name ?? 'none'}
                tone={currentModel?.loaded ? 'ok' : 'warn'}
              />
              <MetricCard label="Status" value={currentModel?.status ?? 'none'} />
              <MetricCard label="Model Class" value={currentModel?.capabilities?.modelClass ?? 'unknown'} />
              <MetricCard
                label="Text"
                value={yesNo(currentModel?.capabilities?.supportsTextGeneration)}
              />
              <MetricCard
                label="Embeddings"
                value={yesNo(currentModel?.capabilities?.supportsEmbeddings)}
              />
              <MetricCard
                label="Pooling"
                value={currentModel?.capabilities?.embedding?.pooling ?? 'n/a'}
              />
              <MetricCard label="Installed" value={installedModels.length} />
              <MetricCard label="Load Time" value={formatMs(lastLoadMs)} />
              <MetricCard label="Model Bytes" value={formatBytes(sourceInfo?.bytes ?? null)} />
              <MetricCard label="Source" value={sourceInfo?.label ?? 'none'} />
              <MetricCard label="Obs Mode" value={observability?.mode ?? 'off'} />
              <MetricCard label="Obs State" value={observability?.state ?? 'idle'} />
              <MetricCard
                label="Current Decode TPS"
                value={
                  lastRun?.tps != null
                    ? formatTps(lastRun.tps)
                    : observability?.runtime?.tokensPerSecond == null
                      ? 'n/a'
                      : round(observability.runtime.tokensPerSecond)
                }
              />
              <MetricCard
                label="Current Prefill TPS"
                value={
                  lastRun?.prefillTps != null
                    ? formatTps(lastRun.prefillTps)
                    : observability?.runtime?.prefillTokensPerSecond == null
                      ? 'n/a'
                      : round(observability.runtime.prefillTokensPerSecond)
                }
              />
              <MetricCard
                label="Backend"
                value={describeRuntimeBackend(observability?.profile)}
              />
              <MetricCard
                label="Rust Engine"
                value={
                  browserSmokeError ??
                  (browserSmoke == null
                    ? 'pending'
                    : browserSmoke.rustEngine.available
                      ? `abi ${browserSmoke.rustEngine.abiVersion}`
                      : browserSmoke.rustEngine.error ?? 'unavailable')
                }
                tone={browserSmoke?.rustEngine.available ? 'ok' : browserSmokeError ? 'warn' : undefined}
              />
              <MetricCard
                label="Rust GGUF Ingest"
                value={
                  browserSmokeError ??
                  (browserSmoke == null
                    ? 'pending'
                    : browserSmoke.ggufIngest.available
                      ? `ready (${browserSmoke.ggufIngest.plannedShardCount} shards)`
                      : browserSmoke.ggufIngest.error ?? 'unavailable')
                }
                tone={browserSmoke?.ggufIngest.available ? 'ok' : browserSmokeError ? 'warn' : undefined}
              />
              <MetricCard
                label="WebGPU Smoke"
                value={
                  browserSmokeError ??
                  (browserSmoke == null
                    ? 'pending'
                    : browserSmoke.webgpuReady
                      ? `ready (${browserSmoke.backend?.webgpuDeviceCount ?? 0})`
                      : browserSmoke.backend?.webgpuCompiled
                        ? 'compiled, unavailable'
                        : 'not compiled')
                }
                tone={browserSmoke?.webgpuReady ? 'ok' : browserSmokeError ? 'warn' : undefined}
              />
            </div>
          </section>

          <section className="section">
            <div className="section-header">
              <h2>Response</h2>
            </div>
            <div className="metric-grid">
              {lastRun == null ? (
                <MetricCard label="Last Run" value="No query yet" />
              ) : (
                <>
                  <MetricCard label="Operation" value={lastRun.operation} />
                  <MetricCard label="Latency" value={formatMs(lastRun.wallMs)} />
                  <MetricCard
                    label="TTFT"
                    value={lastRun.ttftMs == null ? 'n/a' : formatMs(lastRun.ttftMs)}
                  />
                  {lastRun.outputKind === 'embedding' ? (
                    <>
                      <MetricCard label="Dimensions" value={lastRun.embeddingDimensions ?? 'n/a'} />
                      <MetricCard label="Pooling" value={lastRun.embeddingPooling ?? 'n/a'} />
                      <MetricCard
                        label="Normalized"
                        value={lastRun.embeddingNormalized == null ? 'n/a' : yesNo(lastRun.embeddingNormalized)}
                      />
                    </>
                  ) : (
                    <>
                      <MetricCard label="Tokens" value={lastRun.tokens || 'n/a'} />
                      <MetricCard label="TPS" value={lastRun.tps == null ? 'n/a' : formatTps(lastRun.tps)} />
                    </>
                  )}
                  <MetricCard label="Prefill TPS" value={lastRun.prefillTps == null ? 'n/a' : formatTps(lastRun.prefillTps)} />
                  {lastRun.observability && (
                    <>
                      <div className="metric-group-title" style={{ gridColumn: '1 / -1', marginTop: '1rem' }}>Compute Phases</div>
                      <MetricCard label="Prefill" value={formatMs(lastRun.observability.prefillMs)} />
                      <MetricCard label="Decode" value={formatMs(lastRun.observability.decodeMs)} />
                      <MetricCard label="Input" value={lastRun.observability.inputTokens} />
                      <MetricCard label="Prefill Tokens" value={lastRun.observability.prefillTokens} />
                      <MetricCard label="Cache Hits" value={lastRun.observability.cacheHits} />

                      <div className="metric-group-title" style={{ gridColumn: '1 / -1', marginTop: '1rem' }}>Hardware Efficiency</div>
                      <MetricCard label="Native GPU" value={formatMs(lastRun.observability.nativeGpuMs)} />
                      <MetricCard label="Native Sync" value={formatMs(lastRun.observability.nativeSyncMs)} />
                      <MetricCard label="Engine Logic" value={formatMs(lastRun.observability.nativeLogicMs)} />
                    </>
                  )}
                </>
              )}
            </div>
            <div
              className={`response ${isBusy ? 'generating' : ''}`}
              style={{ marginTop: '16px' }}
              ref={responseElementRef}
            >
              {response || (isBusy ? 'Running request...' : 'Ready for request.')}
            </div>
          </section>

          <section className="section">
            <div className="section-header">
              <h2>Benchmark Results</h2>
            </div>
            {scenarioResults.length === 0 && mixedLoadResult == null ? (
              <p className="hint">
                Run the benchmark to capture SISO, SILO, LISO, LILO, mixed-load, and memory
                snapshots with the minimal client API.
              </p>
            ) : (
              <div className="benchmark-results">
                {scenarioResults.map((scenario) => (
                  <div className="result-stack" key={scenario.definition.id}>
                    <div className="result-card">
                      <h3>{scenario.definition.label}</h3>
                      <div className="metric-grid">
                        <MetricCard label="Scenario" value={scenario.definition.id.toUpperCase()} />
                        <MetricCard
                          label="Load Runtime"
                          value={formatMs(scenario.runtime.loadRuntimeMs)}
                        />
                        <MetricCard
                          label="Output Limit"
                          value={scenario.definition.outputTokenLimit}
                        />
                      </div>
                    </div>
                    {renderGroup('Cold Prompt', scenario.coldPrompt)}
                    {renderGroup('Hot Fresh Context', scenario.hotFreshContext)}
                    {renderGroup('Hot Reused Context', scenario.hotReuseContext)}
                  </div>
                ))}

                {mixedLoadResult != null ? (
                  mixedLoadResult.unsupported ? (
                    <div className="result-card">
                      <h3>{mixedLoadResult.definition.label}</h3>
                      <p className="result-detail">{mixedLoadResult.reason ?? 'Unsupported.'}</p>
                    </div>
                  ) : (
                    <div className="result-stack">
                      <div className="result-card">
                        <h3>{mixedLoadResult.definition.label}</h3>
                        <div className="metric-grid">
                          <MetricCard
                            label="Concurrency"
                            value={mixedLoadResult.definition.concurrency}
                          />
                          <MetricCard
                            label="Foreground"
                            value={mixedLoadResult.definition.foreground.label}
                          />
                          <MetricCard
                            label="Background"
                            value={mixedLoadResult.definition.background.label}
                          />
                        </div>
                      </div>
                      {mixedLoadResult.foreground
                        ? renderGroup('Foreground Under Load', mixedLoadResult.foreground)
                        : null}
                      {mixedLoadResult.background
                        ? renderGroup('Background Under Load', mixedLoadResult.background)
                        : null}
                    </div>
                  )
                ) : null}

                {memorySnapshots.length > 0 && benchmarkReport != null ? (
                  <div className="result-card">
                    <h3>Memory Snapshots</h3>
                    <div className="metric-grid">
                      <MetricCard
                        label="Snapshots"
                        value={benchmarkReport.memory.summary.snapshotCount}
                      />
                      <MetricCard
                        label="JS Heap Peak"
                        value={formatBytes(benchmarkReport.memory.summary.maxUsedJsHeapBytes)}
                      />
                      <MetricCard
                        label="UA Memory Peak"
                        value={formatBytes(benchmarkReport.memory.summary.maxUserAgentBytes)}
                      />
                    </div>
                  </div>
                ) : null}

                {benchmarkReport != null ? (
                  <div className="result-card">
                    <h3>Timing Trace</h3>
                    <div className="metric-grid">
                      <MetricCard label="Raw Runs" value={benchmarkReport.trace.runCount} />
                      <MetricCard
                        label="Trace TTFT"
                        value={formatSummary(benchmarkReport.trace.analysis.ttftMs)}
                      />
                      <MetricCard
                        label="Trace Mean ITL"
                        value={formatSummary(benchmarkReport.trace.analysis.itlAvgMs)}
                      />
                      <MetricCard
                        label="Trace Tail ITL"
                        value={formatSummary(benchmarkReport.trace.analysis.itlP99Ms)}
                      />
                      <MetricCard
                        label="Trace Decode TPS"
                        value={formatSummary(
                          benchmarkReport.trace.analysis.tps,
                          'tok/s'
                        )}
                      />
                    </div>
                  </div>
                ) : null}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
