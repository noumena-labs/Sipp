import { useEffect, useRef, useState, type ChangeEvent, type ReactNode } from 'react';
import {
  CogentClient,
  type BrowserRuntimeSmokeResult,
  type ModelInfo,
  type ModelSource,
  type ObservabilitySnapshot,
  type TokenBatch,
} from '@noumena-labs/cogentlm';
import {
  Activity,
  Cpu,
  Database,
  Download,
  FileJson,
  Gauge,
  Play,
  Settings2,
  TerminalSquare,
} from 'lucide-react';
import { MetricCard } from './components/MetricCard';
import { Panel } from './components/Panel';
import { SegmentedControl } from './components/SegmentedControl';
import { StatusBadge } from './components/StatusBadge';
import {
  buildBenchmarkScenarios,
  buildBenchmarkBackendProfile,
  buildMixedLoadDefinition,
  DEFAULT_BENCHMARK_PROMPTS,
  describeRuntimeBackend,
  ENCODER_DECODER_BENCHMARK_PROMPTS,
  runtimeOptionsForMixedLoad,
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
  getEmbeddingModels,
  getDefaultVariant,
  getModelById,
  getVariantPrimaryUrl,
  getVisionModels,
  isEmbeddingModel,
  isVisionModel,
  MODEL_REGISTRY,
  type ModelCapability,
  type ModelRegistryEntry,
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
    __cogentPlayground?: {
      getEnvironment(): Promise<Record<string, unknown>>;
      getRuntimeObservability(): ObservabilitySnapshot | null;
      getBackendObservability(): unknown;
      getRuntimeSmoke(): {
        result: BrowserRuntimeSmokeResult | null;
        error: string | null;
      };
      runRuntimeSmoke(): Promise<BrowserRuntimeSmokeResult>;
      getLastReport(): BenchmarkReport | null;
    };
  }
}

interface BenchmarkReport {
  schema: 'cogent.playground.browser.v1';
  generatedAt: string;
  model: ModelInfo | null;
  source: { label: string; bytes: number | null };
  settings: {
    operation: BenchmarkOperation;
    prompt: string;
    tokenCount: number;
    warmupRuns: number;
    measuredRuns: number;
    emitTokens: boolean;
    runtime: ReturnType<typeof getDefaultRuntimeOptions>;
  };
  environment: Awaited<ReturnType<typeof inspectBrowserEnvironment>>;
  backend: ReturnType<typeof buildBenchmarkBackendProfile>;
  observability: ObservabilitySnapshot | null;
  scenarios: ScenarioResult[];
  mixedLoad: MixedLoadResult | null;
  memory: {
    snapshots: MemorySnapshot[];
    summary: ReturnType<typeof summarizeMemorySnapshots>;
  };
  trace: BenchmarkTraceReport;
}

interface ImageSelectionMeta {
  readonly name: string;
  readonly size: number;
  readonly type: string;
}

const AUTO_SPLIT_SMOKE_BYTES = 256 * 1024 * 1024;

function getDefaultRuntimeOptions() {
  return {
    placement: {
      gpu_layers: 'all' as const,
    },
    context: {
      n_parallel: 1,
    },
    cache: {
      mode: 'live_slot_and_snapshot' as const,
    },
  };
}

function getClientOptions() {
  const params = new URLSearchParams(window.location.search);
  if (params.get('forceAutoSplit') !== '1') {
    return {};
  }
  return {
    browserCache: {
      directLoadMaxBytes: AUTO_SPLIT_SMOKE_BYTES,
      shardMaxBytes: AUTO_SPLIT_SMOKE_BYTES,
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
    adapterLabel: info?.device ?? info?.description ?? null,
    adapterVendor: info?.vendor ?? null,
    adapterArchitecture: info?.architecture ?? null,
    adapterDescription: info?.description ?? null,
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
  return `${round(summary.mean)}${unit} avg / ${round(summary.p99)}${unit} p99`;
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

function logBenchmarkReport(report: BenchmarkReport): void {
  console.groupCollapsed('[CogentLM playground] benchmark suite complete');
  console.log('backend profile', report.backend);
  console.log('runtime config', report.settings.runtime);
  console.log('summary', report.trace.analysis);
  console.table(report.trace.rows);
  console.groupEnd();
}

const DEFAULT_QUERY_PROMPT = 'Describe how to benchmark browser-hosted inference.';
const ENCODER_DECODER_QUERY_PROMPT = 'translate English to German: The house is wonderful.';
const DEFAULT_VISION_PROMPT =
  'Describe the image in detail, including visible objects, text, context, and uncertainty.';
const DEFAULT_EMBED_PROMPT = 'search_query: browser-hosted inference playground observability';
const DEFAULT_TOKEN_COUNT = 64;
const ENCODER_DECODER_TOKEN_COUNT = 32;
type ModelSourceType = 'registry' | 'url' | 'file';
type PlaygroundView =
  | 'requests'
  | 'vision'
  | 'embeddings'
  | 'benchmarks'
  | 'observability'
  | 'reports';
type TextOperation = Extract<BenchmarkOperation, 'chat' | 'query'>;
type EmbeddingUseCase = 'semanticSearch' | 'ragDocument' | 'classification' | 'clustering';

const CAPABILITY_LABELS: Record<ModelCapability, string> = {
  embedding: 'Embed',
  text: 'Text',
  vision: 'Vision',
};

const EMBEDDING_USE_CASES: readonly {
  readonly description: string;
  readonly label: string;
  readonly prefix: string;
  readonly prompt: string;
  readonly value: EmbeddingUseCase;
}[] = [
  {
    description: 'Turn a user search into a vector for similarity search.',
    label: 'Semantic Search',
    prefix: 'search_query:',
    prompt: DEFAULT_EMBED_PROMPT,
    value: 'semanticSearch',
  },
  {
    description: 'Encode a passage before storing it in a retrieval index.',
    label: 'RAG Document',
    prefix: 'search_document:',
    prompt:
      'search_document: The CogentLM playground runs local models and captures latency, cache, memory, and backend metrics.',
    value: 'ragDocument',
  },
  {
    description: 'Compare short labels or examples by vector distance.',
    label: 'Classification',
    prefix: 'plain text',
    prompt:
      'Classify this support ticket by comparing it with known examples: vision setup fails because the projector is missing.',
    value: 'classification',
  },
  {
    description: 'Group related text snippets without generating tokens.',
    label: 'Clustering',
    prefix: 'plain text',
    prompt: 'Cluster feedback about browser inference latency, memory pressure, and model setup.',
    value: 'clustering',
  },
];

function capabilityLabel(capability: ModelCapability): string {
  return CAPABILITY_LABELS[capability];
}

function totalVariantSize(model: ModelRegistryEntry): number {
  const variant = getDefaultVariant(model);
  return variant.sizeBytes + (variant.projectorSizeBytes ?? 0);
}

function formatModelOption(model: ModelRegistryEntry): string {
  const variant = getDefaultVariant(model);
  return [
    `[${capabilityLabel(model.capability)}] ${model.name}`,
    `(${model.parameterCount}, ${variant.quant})`,
    `- ${formatSize(totalVariantSize(model))}`,
  ].join(' ');
}

function isManagedPrompt(value: string): boolean {
  return [
    DEFAULT_QUERY_PROMPT,
    ENCODER_DECODER_QUERY_PROMPT,
    DEFAULT_VISION_PROMPT,
    ...EMBEDDING_USE_CASES.map((useCase) => useCase.prompt),
  ].includes(value);
}

function registryModelSupportsOperation(
  model: ModelRegistryEntry,
  operation: BenchmarkOperation
): boolean {
  if (operation === 'embed') {
    return isEmbeddingModel(model);
  }
  return model.capability !== 'embedding';
}

function defaultOperationForRegistryEntry(model: ModelRegistryEntry): BenchmarkOperation {
  if (isEmbeddingModel(model)) {
    return 'embed';
  }
  if (model.modelClass === 'encoder_decoder') {
    return 'query';
  }
  return 'chat';
}

function defaultPromptForRegistryEntry(model: ModelRegistryEntry): string {
  if (isVisionModel(model)) {
    return DEFAULT_VISION_PROMPT;
  }
  if (model.modelClass === 'encoder_decoder') {
    return ENCODER_DECODER_QUERY_PROMPT;
  }
  return DEFAULT_QUERY_PROMPT;
}

function defaultTokenCountForRegistryEntry(model: ModelRegistryEntry): number {
  return model.modelClass === 'encoder_decoder'
    ? ENCODER_DECODER_TOKEN_COUNT
    : DEFAULT_TOKEN_COUNT;
}

const MODEL_SOURCE_OPTIONS: readonly { readonly label: string; readonly value: ModelSourceType }[] = [
  { label: 'Library', value: 'registry' },
  { label: 'URL', value: 'url' },
  { label: 'Local File', value: 'file' },
];

const TEXT_OPERATION_OPTIONS: readonly { readonly label: string; readonly value: TextOperation }[] = [
  { label: 'Chat', value: 'chat' },
  { label: 'Query', value: 'query' },
];

const OPERATION_OPTIONS: readonly { readonly label: string; readonly value: BenchmarkOperation }[] = [
  ...TEXT_OPERATION_OPTIONS,
  { label: 'Embed', value: 'embed' },
];

const VIEW_OPTIONS: readonly {
  readonly icon: typeof Activity;
  readonly label: string;
  readonly title: string;
  readonly value: PlaygroundView;
}[] = [
  { icon: TerminalSquare, label: 'Requests', title: 'Chat and query requests', value: 'requests' },
  { icon: Cpu, label: 'Vision', title: 'Image and prompt requests', value: 'vision' },
  { icon: Database, label: 'Embeddings', title: 'Vector embedding requests', value: 'embeddings' },
  { icon: Gauge, label: 'Benchmarks', title: 'Benchmark suite and traces', value: 'benchmarks' },
  { icon: Activity, label: 'Observability', title: 'Runtime and request observability', value: 'observability' },
  { icon: FileJson, label: 'Reports', title: 'Benchmark report output', value: 'reports' },
];

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
  const [activeView, setActiveView] = useState<PlaygroundView>('requests');
  const [modelType, setModelType] = useState<ModelSourceType>('registry');
  const [selectedRegistryId, setSelectedRegistryId] = useState(MODEL_REGISTRY[0].id);
  const selectedModel = getModelById(selectedRegistryId) ?? MODEL_REGISTRY[0];
  const selectedVariant = getDefaultVariant(selectedModel);
  const [modelUrl, setModelUrl] = useState(getVariantPrimaryUrl(selectedVariant));
  const [projectorUrl, setProjectorUrl] = useState('');
  const [operation, setOperation] = useState<BenchmarkOperation>('chat');
  const [prompt, setPrompt] = useState(DEFAULT_QUERY_PROMPT);
  const [embeddingUseCase, setEmbeddingUseCase] =
    useState<EmbeddingUseCase>('semanticSearch');
  const [tokenCount, setTokenCount] = useState(DEFAULT_TOKEN_COUNT);
  const [warmupRuns, setWarmupRuns] = useState(1);
  const [measuredRuns, setMeasuredRuns] = useState(3);
  const [emitTokens, setEmitTokens] = useState(true);
  const [imageSource, setImageSource] = useState('');
  const [imageMeta, setImageMeta] = useState<ImageSelectionMeta | null>(null);
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
    decodeTps: number | null;
    e2eTps: number | null;
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
        const nextClient = new CogentClient(getClientOptions());
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
        setCurrentModel(nextClient.currentLocal());
        setObservability(nextClient.observability.current());
        setInstalledModels(await nextClient.listLocal());
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
    window.__cogentPlayground = {
      getEnvironment: inspectBrowserEnvironment,
      getRuntimeObservability: () => observability,
      getBackendObservability: () => browserSmoke?.backend ?? observability?.profile ?? null,
      getRuntimeSmoke: () => ({
        result: browserSmoke,
        error: browserSmokeError,
      }),
      runRuntimeSmoke: runBrowserRuntimeSmoke,
      getLastReport: () => benchmarkReport,
    };

    return () => {
      if (window.__cogentPlayground?.runRuntimeSmoke === runBrowserRuntimeSmoke) {
        delete window.__cogentPlayground;
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
    if (modelType === 'registry') return selectedVariant.source;
    if (modelType === 'url') {
      const projector = projectorOverride();
      return modelUrl.trim().length > 0 ? withProjector(modelUrl.trim(), projector) : null;
    }
    const files = Array.from(fileInputRef.current?.files ?? []);
    if (files.length === 0) return null;
    return files.length === 1 ? files[0] : files;
  };

  const applyEmbeddingUseCase = (value: EmbeddingUseCase): void => {
    const nextUseCase =
      EMBEDDING_USE_CASES.find((useCase) => useCase.value === value) ?? EMBEDDING_USE_CASES[0];
    setEmbeddingUseCase(nextUseCase.value);
    setPrompt(nextUseCase.prompt);
  };

  const configureRegistryEntry = (entry: ModelRegistryEntry): void => {
    const variant = getDefaultVariant(entry);
    setSelectedRegistryId(entry.id);
    setModelUrl(getVariantPrimaryUrl(variant));
    setProjectorUrl('');

    if (isVisionModel(entry)) {
      setActiveView('vision');
      setOperation(defaultOperationForRegistryEntry(entry));
      setImageEnabled(true);
      setPrompt(defaultPromptForRegistryEntry(entry));
      setTokenCount(defaultTokenCountForRegistryEntry(entry));
      return;
    }

    if (isEmbeddingModel(entry)) {
      setActiveView('embeddings');
      setOperation('embed');
      setImageEnabled(false);
      applyEmbeddingUseCase('semanticSearch');
      return;
    }

    setActiveView('requests');
    setOperation(defaultOperationForRegistryEntry(entry));
    setImageEnabled(false);
    if (isManagedPrompt(prompt)) {
      setPrompt(defaultPromptForRegistryEntry(entry));
    }
    setTokenCount(defaultTokenCountForRegistryEntry(entry));
  };

  const refreshModels = async (targetClient: CogentClient) => {
    setCurrentModel(targetClient.currentLocal());
    setObservability(targetClient.observability.current());
    setInstalledModels(await targetClient.listLocal());
  };

  const loadLocalSelection = async (
    targetClient: CogentClient,
    source: ModelSource
  ): Promise<ModelInfo> => {
    const start = performance.now();
    await targetClient.add('playground-local', {
      kind: 'local',
      source,
      options: {
        observability: 'profile',
        runtime: getDefaultRuntimeOptions(),
        onProgress: (progress) => {
          if (progress.phase === 'download') {
            setStatus(`Downloading model ${Math.floor(progress.percent ?? 0)}%`);
          } else if (progress.phase === 'store') {
            setStatus(`Storing model ${Math.floor(progress.percent ?? 0)}%`);
          } else if (progress.phase === 'split') {
            setStatus(`Preparing model shards ${Math.floor(progress.percent ?? 0)}%`);
          } else if (progress.phase === 'load') {
            setStatus('Loading into memory');
          }
        },
      },
    });
    const info = targetClient.currentLocal();
    if (info == null) {
      throw new Error('Local model did not become active.');
    }
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
      const info = await loadLocalSelection(client, source);
      setStatus(info.status === 'ready' ? `loaded ${info.name}` : `${info.name}: ${info.status}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsBusy(false);
    }
  };

  const runQuery = async (
    requestedOperation: BenchmarkOperation = operation,
    targetView: PlaygroundView = 'requests',
    includeImage = imageEnabled
  ) => {
    if (client == null) return;
    setActiveView(targetView);
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
      const loadedModel = client.currentLocal();
      let requestOperation = requestedOperation;
      let requestPrompt = prompt;
      let requestTokenCount = tokenCount;
      if (
        loadedModel == null ||
        loadedModel.status !== 'ready' ||
        (loadedSourceKeyRef.current != null && loadedSourceKeyRef.current !== nextSourceKey)
      ) {
        const info = await loadLocalSelection(client, source);
        if (!modelSupportsOperation(info, requestOperation)) {
          setStatus(`${info.name} does not support ${requestOperation}.`);
          return;
        }
        requestPrompt = effectivePromptForModel(info, requestPrompt);
        requestTokenCount = effectiveTokenCountForModel(info, requestTokenCount);
        setStatus(info.status === 'ready' ? `loaded ${info.name}` : `${info.name}: ${info.status}`);
      } else if (!modelSupportsOperation(loadedModel, requestOperation)) {
        setStatus(`${loadedModel.name} does not support ${requestOperation}.`);
        return;
      } else {
        requestPrompt = effectivePromptForModel(loadedModel, requestPrompt);
        requestTokenCount = effectiveTokenCountForModel(loadedModel, requestTokenCount);
      }
      if (requestPrompt !== prompt) {
        setPrompt(requestPrompt);
      }
      if (requestOperation !== operation) {
        setOperation(requestOperation);
      }
      if (requestTokenCount !== tokenCount) {
        setTokenCount(requestTokenCount);
      }

      const image =
        requestOperation !== 'embed' && includeImage && imageSource.trim().length > 0
          ? await fetchImageBytes(imageSource.trim())
          : undefined;
      const requestEmitTokens = requestOperation !== 'embed' && emitTokens;
      const queryRenderer = requestEmitTokens ? createResponseRenderer(1, 'frame') : null;
      queryRenderer?.reset();
      queryRenderer?.start('response');
      const onTokenBatch =
        !requestEmitTokens
          ? undefined
          : (batch: TokenBatch) => {
              queryRenderer?.append('response', batch);
            };
      try {
        const run = await runObservedRequest(client, requestPrompt, {
          operation: requestOperation,
          maxTokens: requestTokenCount,
          session: `query-${Date.now()}`,
          media: image,
          emitTokens: requestEmitTokens,
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
          decodeTps: run.observability?.decodeTokensPerSecond ?? null,
          e2eTps: run.observability?.e2eTokensPerSecond ?? null,
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

    setActiveView('benchmarks');
    setIsBusy(true);
    setScenarioResults([]);
    setMixedLoadResult(null);
    setMemorySnapshots([]);
    setBenchmarkReport(null);
    let benchmarkOperation = operation;
    const loadedModel = client.currentLocal();
    let benchmarkPrompt = prompt;
    let benchmarkTokenCount = tokenCount;
    if (loadedModel != null && !modelSupportsOperation(loadedModel, benchmarkOperation)) {
      benchmarkOperation = defaultOperationForModel(loadedModel);
      setOperation(benchmarkOperation);
    }
    let benchmarkEmitTokens = false;
    let benchmarkRenderer: ReturnType<typeof createResponseRenderer> | null = null;
    let benchmarkTokenObserver: BenchmarkTokenObserver | undefined;

    try {
      const info = await loadLocalSelection(client, source);
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
      benchmarkEmitTokens = benchmarkOperation !== 'embed' && emitTokens;
      benchmarkRenderer = benchmarkEmitTokens ? createResponseRenderer(2, 'frame') : null;
      benchmarkTokenObserver =
        benchmarkEmitTokens
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

      const defaultRuntime = getDefaultRuntimeOptions();
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
            defaultRuntime,
            setStatus,
            true,
            benchmarkEmitTokens,
            benchmarkTokenObserver
          )
        );
      }

      snapshots.push(await captureBrowserMemorySnapshot('after-scenarios', true));

      let mixed: MixedLoadResult | null = null;
      if (
        benchmarkOperation !== 'embed' &&
        info.capabilities?.modelClass !== 'encoder_decoder' &&
        supportsConcurrentQueryApi(client)
      ) {
        const mixedDefinition = buildMixedLoadDefinition(benchmarkOperation, promptSet);
        mixed = await runMixedLoadBenchmark(
          client,
          benchmarkOperation,
          mixedDefinition,
          source,
          warmupRuns,
          measuredRuns,
          runtimeOptionsForMixedLoad(defaultRuntime, mixedDefinition.concurrency),
          setStatus,
          false,
          benchmarkEmitTokens,
          benchmarkTokenObserver
        );
        snapshots.push(await captureBrowserMemorySnapshot('after-mixed-load', true));
      }
      benchmarkRenderer?.finish();
      const trace = buildBenchmarkTraceReport(results, mixed);
      const environment = await inspectBrowserEnvironment();
      const observabilitySnapshot = client.observability.current();
      const backend = buildBenchmarkBackendProfile(
        environment,
        browserSmoke?.backend ?? observabilitySnapshot.profile ?? null,
        defaultRuntime.placement.gpu_layers
      );

      const report: BenchmarkReport = {
        schema: 'cogent.playground.browser.v1',
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
          emitTokens: benchmarkEmitTokens,
          runtime: defaultRuntime,
        },
        environment,
        backend,
        observability: observabilitySnapshot,
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
      logBenchmarkReport(report);
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
    setImageMeta({
      name: file.name,
      size: file.size,
      type: file.type || 'image',
    });
    setImageEnabled(true);
  };

  const renderDetailTable = (
    rows: readonly { readonly label: string; readonly value: ReactNode }[]
  ) => (
    <div className="detail-table">
      {rows.map((row) => (
        <div className="detail-row" key={row.label}>
          <span>{row.label}</span>
          <strong>{row.value}</strong>
        </div>
      ))}
    </div>
  );

  const renderRunMetrics = () => (
    <div className="metric-grid">
      {lastRun == null ? (
        <MetricCard label="Last Run" value="No request yet" />
      ) : (
        <>
          <MetricCard label="Operation" value={lastRun.operation} />
          <MetricCard label="Latency" value={formatMs(lastRun.wallMs)} />
          <MetricCard label="TTFT" value={lastRun.ttftMs == null ? 'n/a' : formatMs(lastRun.ttftMs)} />
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
              <MetricCard label="Decode TPS" value={lastRun.decodeTps == null ? 'n/a' : formatTps(lastRun.decodeTps)} />
              <MetricCard label="E2E TPS" value={lastRun.e2eTps == null ? 'n/a' : formatTps(lastRun.e2eTps)} />
            </>
          )}
          <MetricCard label="Prefill TPS" value={lastRun.prefillTps == null ? 'n/a' : formatTps(lastRun.prefillTps)} />
        </>
      )}
    </div>
  );

  const renderLiveOutput = (emptyLabel: string) => (
    <div className={`response-console ${isBusy ? 'generating' : ''}`} ref={responseElementRef}>
      {response || (isBusy ? 'Running...' : emptyLabel)}
    </div>
  );

  const selectedEmbeddingUseCase =
    EMBEDDING_USE_CASES.find((useCase) => useCase.value === embeddingUseCase) ??
    EMBEDDING_USE_CASES[0];
  const selectedSourceForState = modelSource();
  const selectedSourceKey =
    selectedSourceForState == null ? null : sourceKey(selectedSourceForState);
  const isLoadedSelectedSource =
    currentModel?.loaded === true &&
    selectedSourceKey != null &&
    loadedSourceKeyRef.current === selectedSourceKey;

  const selectedRegistryOperationReason = (
    requestedOperation: BenchmarkOperation
  ): string | null => {
    if (modelType !== 'registry') {
      return null;
    }
    if (registryModelSupportsOperation(selectedModel, requestedOperation)) {
      return null;
    }
    return `${selectedModel.name} is a ${capabilityLabel(selectedModel.capability)} model.`;
  };

  const loadedModelOperationReason = (
    requestedOperation: BenchmarkOperation
  ): string | null => {
    if (!isLoadedSelectedSource || currentModel == null) {
      return null;
    }
    if (modelSupportsOperation(currentModel, requestedOperation)) {
      return null;
    }
    return `${currentModel.name} does not support ${requestedOperation}.`;
  };

  const operationDisabledReason = (
    requestedOperation: BenchmarkOperation
  ): string | null =>
    selectedRegistryOperationReason(requestedOperation) ??
    loadedModelOperationReason(requestedOperation);

  const textOperation: TextOperation = operation === 'query' ? 'query' : 'chat';
  const textRequestDisabledReason = operationDisabledReason(textOperation);
  const visionDisabledReason =
    modelType === 'registry' && !isVisionModel(selectedModel)
      ? `${selectedModel.name} is a ${capabilityLabel(selectedModel.capability)} model. Select a Vision model to enable image inputs.`
      : isLoadedSelectedSource && currentModel != null && currentModel.modality !== 'vision'
        ? `${currentModel.name} is loaded as a ${currentModel.modality} model. Load a Vision model to enable image inputs.`
        : null;
  const embeddingDisabledReason =
    modelType === 'registry' && !isEmbeddingModel(selectedModel)
      ? `${selectedModel.name} is a ${capabilityLabel(selectedModel.capability)} model. Select an Embed model to generate vectors.`
      : isLoadedSelectedSource &&
          currentModel?.capabilities != null &&
          !currentModel.capabilities.supportsEmbeddings
        ? `${currentModel.name} does not support embeddings. Load an Embed model to generate vectors.`
        : null;
  const urlProjectorConfigured =
    projectorUrl.trim().length > 0 ||
    (projectorFileInputRef.current?.files?.length ?? 0) > 0;
  const projectorDetail =
    modelType === 'registry'
      ? isVisionModel(selectedModel)
        ? 'registry/default'
        : 'n/a'
      : modelType === 'url'
        ? urlProjectorConfigured
          ? 'url/file override'
          : 'not provided'
        : 'local source files';
  const sourceDetailRows =
    modelType === 'registry'
      ? [
          { label: 'Capability', value: capabilityLabel(selectedModel.capability) },
          { label: 'Variant', value: getDefaultVariant(selectedModel).quant },
          { label: 'Download', value: formatSize(totalVariantSize(selectedModel)) },
          { label: 'Projector', value: projectorDetail },
        ]
      : modelType === 'url'
        ? [
            { label: 'Capability', value: currentModel?.capabilities == null ? 'unknown until load' : 'loaded' },
            { label: 'Model URL', value: modelUrl.trim().length > 0 ? 'provided' : 'missing' },
            { label: 'Projector', value: projectorDetail },
          ]
        : [
            { label: 'Capability', value: currentModel?.capabilities == null ? 'unknown until load' : 'loaded' },
            { label: 'Source', value: 'local GGUF file' },
            { label: 'Projector', value: projectorDetail },
          ];
  const imagePreviewSource =
    imageEnabled && imageSource.trim().length > 0 ? imageSource.trim() : null;
  const imageSelectionKind =
    imagePreviewSource == null ? 'none' : imageMeta == null ? 'url' : 'file';
  const imageSelectionName =
    imageMeta?.name ??
    (imagePreviewSource == null
      ? 'none'
      : imagePreviewSource.startsWith('data:')
        ? 'inline image'
        : imagePreviewSource);
  const benchmarkDisabledReason = operationDisabledReason(operation);
  const tokenEmissionDisabledReason =
    operation === 'embed' ? 'Embedding requests return vectors and do not emit tokens.' : null;
  const visionSurfaceClass =
    visionDisabledReason == null ? 'form-stack' : 'form-stack disabled-surface';
  const embeddingSurfaceClass =
    embeddingDisabledReason == null ? 'form-stack' : 'form-stack disabled-surface';
  const operationOptionsForCurrentModel = OPERATION_OPTIONS.map((option) => {
    const disabledReason = operationDisabledReason(option.value);
    return {
      ...option,
      disabled: disabledReason != null,
      title: disabledReason ?? option.label,
    };
  });
  const textOperationOptionsForCurrentModel = TEXT_OPERATION_OPTIONS.map((option) => {
    const disabledReason = operationDisabledReason(option.value);
    return {
      ...option,
      disabled: disabledReason != null,
      title: disabledReason ?? option.label,
    };
  });
  const embeddingUseCaseOptions = EMBEDDING_USE_CASES.map((useCase) => ({
    disabled: embeddingDisabledReason != null,
    label: useCase.label,
    title: embeddingDisabledReason ?? useCase.description,
    value: useCase.value,
  }));

  const runVisionRequest = async () => {
    if (visionDisabledReason != null) {
      setStatus(visionDisabledReason);
      return;
    }
    if (imageSource.trim().length === 0) {
      setStatus('Add an image for the vision request.');
      return;
    }
    setImageEnabled(true);
    await runQuery(textOperation, 'vision', true);
  };

  const runEmbeddingRequest = async () => {
    if (embeddingDisabledReason != null) {
      setStatus(embeddingDisabledReason);
      return;
    }
    await runQuery('embed', 'embeddings', false);
  };

  const renderRequestObservability = () => {
    const metrics = lastRun?.observability;
    if (metrics == null) {
      return null;
    }
    return (
      <div className="detail-grid">
        {renderDetailTable([
          { label: 'Prefill', value: formatMs(metrics.prefillMs) },
          { label: 'Decode', value: formatMs(metrics.decodeMs) },
          { label: 'E2E', value: formatMs(metrics.e2eMs) },
          { label: 'ITL Avg', value: formatMs(metrics.itlAvgMs) },
          { label: 'ITL P99', value: formatMs(metrics.itlP99Ms) },
        ])}
        {renderDetailTable([
          { label: 'Input Tokens', value: metrics.inputTokens },
          { label: 'Output Tokens', value: metrics.outputTokens },
          { label: 'Prefill Tokens', value: metrics.prefillTokens },
          { label: 'Cache Mode', value: metrics.cacheMode },
          { label: 'Cache Source', value: metrics.cacheSource },
          { label: 'Cache Hits', value: metrics.cacheHits },
        ])}
        {renderDetailTable([
          { label: 'Native GPU', value: formatMs(metrics.nativeGpuMs) },
          { label: 'Native Sync', value: formatMs(metrics.nativeSyncMs) },
          { label: 'Engine Logic', value: formatMs(metrics.nativeLogicMs) },
          { label: 'Execution', value: metrics.execution.mode },
          { label: 'Worker', value: yesNo(metrics.execution.workerBacked) },
          { label: 'Token Path', value: metrics.execution.tokenPath ?? 'none' },
          { label: 'JS Drain', value: metrics.jsTokenDrainMs == null ? 'n/a' : formatMs(metrics.jsTokenDrainMs) },
        ])}
      </div>
    );
  };

  const renderFeatureOutputPanel = (title: string, emptyLabel: string) => (
    <Panel title={title}>
      {renderRunMetrics()}
      {renderRequestObservability()}
      {renderLiveOutput(emptyLabel)}
    </Panel>
  );

  const renderGroup = (title: string, group: GroupResult) => (
    <div className="result-panel" key={`${group.id}-${title}`}>
      <div className="result-panel-header">
        <h4>{title}</h4>
        <StatusBadge
          tone={
            group.cacheReuse.expected
              ? group.cacheReuse.invalidRunLabels.length === 0
                ? 'ok'
                : 'warn'
              : 'neutral'
          }
        >
          {group.cacheReuse.expected
            ? group.cacheReuse.invalidRunLabels.length === 0
              ? `cache ${group.cacheReuse.expectedSource}`
              : `cache invalid ${group.cacheReuse.invalidRunLabels.length}`
            : 'cache n/a'}
        </StatusBadge>
      </div>
      <div className="data-table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Latency</th>
              <th>Compute</th>
              <th>Native</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Primary</td>
              <td>TTFT {formatSummary(group.summary.runtime.ttftMs)}</td>
              <td>
                Prefill {group.summary.runtime.avgPrefillMs == null ? 'n/a' : formatMs(group.summary.runtime.avgPrefillMs)}
              </td>
              <td>
                GPU {group.summary.runtime.avgNativeGpuMs == null ? 'n/a' : formatMs(group.summary.runtime.avgNativeGpuMs)}
              </td>
            </tr>
            <tr>
              <td>Throughput</td>
              <td>E2E {formatSummary(group.summary.runtime.e2eTps, 'tok/s')}</td>
              <td>Decode {formatSummary(group.summary.runtime.decodeTps, 'tok/s')}</td>
              <td>
                Sync {group.summary.runtime.avgNativeSyncMs == null ? 'n/a' : formatMs(group.summary.runtime.avgNativeSyncMs)}
              </td>
            </tr>
            <tr>
              <td>Tokens</td>
              <td>ITL P99 {formatSummary(group.summary.runtime.itlP99Ms)}</td>
              <td>
                In {group.summary.runtime.avgInputTokens ?? 'n/a'} / Out {group.summary.runtime.avgOutputTokens ?? 'n/a'}
              </td>
              <td>
                CPU {group.summary.runtime.avgNativeLogicMs == null ? 'n/a' : formatMs(group.summary.runtime.avgNativeLogicMs)}
              </td>
            </tr>
            <tr>
              <td>Cache</td>
              <td>ITL Avg {formatSummary(group.summary.runtime.itlAvgMs)}</td>
              <td>Prefill TPS {formatSummary(group.summary.runtime.prefillTps, 'tok/s')}</td>
              <td>Hits {group.summary.runtime.avgCacheHits ?? 'n/a'}</td>
            </tr>
          </tbody>
        </table>
      </div>
      {group.runs[0]?.outputPreview ? (
        <p className="result-preview">{group.runs[0].outputPreview}</p>
      ) : null}
    </div>
  );

  const runtimeStatusTone = isBusy ? 'info' : status === 'idle' ? 'ok' : status === 'booting' ? 'warn' : 'neutral';
  const engineStatus = browserSmokeError ?? (
    browserSmoke == null
      ? 'pending'
      : browserSmoke.rustEngine.available
        ? `abi ${browserSmoke.rustEngine.abiVersion}`
        : browserSmoke.rustEngine.error ?? 'unavailable'
  );
  const ingestStatus = browserSmokeError ?? (
    browserSmoke == null
      ? 'pending'
      : browserSmoke.ggufIngest.available
        ? `ready (${browserSmoke.ggufIngest.plannedShardCount} shards)`
        : browserSmoke.ggufIngest.error ?? 'unavailable'
  );
  const webgpuStatus = browserSmokeError ?? (
    browserSmoke == null
      ? 'pending'
      : browserSmoke.webgpuReady
        ? `ready (${browserSmoke.backend?.webgpuDeviceCount ?? 0})`
        : browserSmoke.backend?.webgpuCompiled
          ? 'compiled, unavailable'
          : 'not compiled'
  );

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <div className="brand-block">
          <div className="brand-mark">
            <Cpu size={20} aria-hidden="true" />
          </div>
          <div>
            <h1>CogentLM Playground</h1>
            <p>Local inference console</p>
          </div>
        </div>
        <div className="topbar-status">
          <StatusBadge tone={runtimeStatusTone}>{status}</StatusBadge>
          <StatusBadge tone="info">{capabilityLabel(selectedModel.capability)}</StatusBadge>
          <StatusBadge tone={currentModel?.loaded ? 'ok' : 'warn'}>
            {currentModel?.name ?? 'no model'}
          </StatusBadge>
        </div>
      </header>

      <main className="app-layout">
        <aside className="control-rail">
          <Panel title="Model Source">
            <div className="form-stack">
              <div className="field">
                <label>Source Type</label>
                <SegmentedControl
                  ariaLabel="Model source type"
                  onChange={setModelType}
                  options={MODEL_SOURCE_OPTIONS}
                  value={modelType}
                />
              </div>
              <div className="field">
                <label>Model</label>
                {modelType === 'registry' ? (
                  <select
                    value={selectedRegistryId}
                    onChange={(event) => {
                      const entry = getModelById(event.target.value);
                      if (entry == null) return;
                      configureRegistryEntry(entry);
                    }}
                  >
                    {MODEL_REGISTRY.map((model) => (
                      <option key={model.id} value={model.id}>
                        {formatModelOption(model)}
                      </option>
                    ))}
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
              {renderDetailTable(sourceDetailRows)}
              {modelType === 'url' ? (
                <div className="field">
                  <label>Projector URL (optional)</label>
                  <input
                    value={projectorUrl}
                    onChange={(event) => setProjectorUrl(event.target.value)}
                    placeholder="https://.../mmproj.gguf"
                  />
                  <input type="file" accept=".gguf" ref={projectorFileInputRef} />
                </div>
              ) : null}
              <button
                className="button button-primary"
                disabled={isBusy || client == null}
                onClick={loadSelectedModel}
                type="button"
              >
                <Database size={16} aria-hidden="true" />
                Load Model
              </button>
            </div>
          </Panel>

          <Panel title="Runtime Snapshot">
            <div className="rail-heading">
              <Settings2 size={16} aria-hidden="true" />
              <span>Current model and transport</span>
            </div>
            <div className="form-stack">
              {renderDetailTable([
                { label: 'Observability', value: observability?.mode ?? 'off' },
                { label: 'State', value: observability?.state ?? 'idle' },
                { label: 'Transport', value: observability?.runtime?.execution.tokenPath ?? 'pending' },
                { label: 'Installed', value: installedModels.length },
                { label: 'Load Time', value: formatMs(lastLoadMs) },
                { label: 'Selected', value: capabilityLabel(selectedModel.capability) },
              ])}
              <div className="button-row">
                <button
                  className="button button-secondary"
                  onClick={() => setActiveView('observability')}
                  type="button"
                >
                  <Activity size={16} aria-hidden="true" />
                  Observability
                </button>
                <button
                  className="button button-secondary"
                  disabled={benchmarkReport == null}
                  onClick={() =>
                    benchmarkReport == null
                      ? undefined
                      : downloadJson('cogent-playground-report.json', benchmarkReport)
                  }
                  type="button"
                >
                  <Download size={16} aria-hidden="true" />
                  Export Report
                </button>
              </div>
            </div>
          </Panel>
        </aside>

        <div className="workbench">
          <nav className="workspace-tabs" aria-label="Playground views">
            {VIEW_OPTIONS.map((view) => {
              const Icon = view.icon;
              return (
                <button
                  className={`workspace-tab ${activeView === view.value ? 'active' : ''}`}
                  aria-current={activeView === view.value ? 'page' : undefined}
                  key={view.value}
                  onClick={() => setActiveView(view.value)}
                  title={view.title}
                  type="button"
                >
                  <Icon size={16} aria-hidden="true" />
                  {view.label}
                </button>
              );
            })}
          </nav>

          {activeView === 'requests' ? (
            <div className="workspace-view">
              <Panel
                title="Text Requests"
                actions={
                  <button
                    className="button button-primary"
                    disabled={isBusy || client == null || textRequestDisabledReason != null}
                    onClick={() => void runQuery(textOperation, 'requests', false)}
                    title={textRequestDisabledReason ?? undefined}
                    type="button"
                  >
                    <Play size={16} aria-hidden="true" />
                    Run {textOperation === 'chat' ? 'Chat' : 'Query'}
                  </button>
                }
              >
                <div className="form-stack">
                  <div className="form-grid request-grid">
                    <div className="field">
                      <label>Request Type</label>
                      <SegmentedControl
                        ariaLabel="Text request type"
                        onChange={setOperation}
                        options={textOperationOptionsForCurrentModel}
                        value={textOperation}
                      />
                    </div>
                    <div className="field">
                      <label>Max Tokens</label>
                      <input
                        disabled={textRequestDisabledReason != null}
                        type="number"
                        value={tokenCount}
                        onChange={(event) =>
                          setTokenCount(Number.parseInt(event.target.value, 10) || 0)
                        }
                      />
                    </div>
                  </div>
                  <div className="field">
                    <label>Prompt</label>
                    <textarea
                      disabled={textRequestDisabledReason != null}
                      value={prompt}
                      onChange={(event) => setPrompt(event.target.value)}
                    />
                  </div>
                  <div className="field">
                    <label>Token Emission</label>
                    <SegmentedControl
                      ariaLabel="Text token emission"
                      disabled={textRequestDisabledReason != null}
                      onChange={(value) => setEmitTokens(value === 'on')}
                      options={[
                        { label: 'Off', value: 'off', title: textRequestDisabledReason ?? 'Off' },
                        { label: 'On', value: 'on', title: textRequestDisabledReason ?? 'On' },
                      ]}
                      value={emitTokens ? 'on' : 'off'}
                    />
                  </div>
                </div>
              </Panel>

              {renderFeatureOutputPanel('Response', 'Ready for a text request.')}
            </div>
          ) : null}

          {activeView === 'vision' ? (
            <div className="workspace-view">
              <Panel
                title="Vision Request"
                actions={
                  <button
                    className="button button-primary"
                    disabled={isBusy || client == null || visionDisabledReason != null}
                    onClick={() => void runVisionRequest()}
                    title={visionDisabledReason ?? undefined}
                    type="button"
                  >
                    <Play size={16} aria-hidden="true" />
                    Run Vision
                  </button>
                }
              >
                <div className="form-stack">
                  {modelType === 'registry' ? (
                    <div className="field">
                      <label>Vision Model</label>
                      <select
                        value={isVisionModel(selectedModel) ? selectedRegistryId : ''}
                        onChange={(event) => {
                          const entry = getModelById(event.target.value);
                          if (entry == null) return;
                          configureRegistryEntry(entry);
                        }}
                      >
                        <option value="" disabled>
                          Select a vision model
                        </option>
                        {getVisionModels().map((model) => (
                          <option key={model.id} value={model.id}>
                            {formatModelOption(model)}
                          </option>
                        ))}
                      </select>
                    </div>
                  ) : null}
                  <div
                    aria-disabled={visionDisabledReason != null}
                    className={visionSurfaceClass}
                    title={visionDisabledReason ?? undefined}
                  >
                    <div className="form-grid request-grid">
                      <div className="field">
                        <label>Vision Mode</label>
                        <SegmentedControl
                          ariaLabel="Vision request type"
                          disabled={visionDisabledReason != null}
                          onChange={setOperation}
                          options={textOperationOptionsForCurrentModel}
                          value={textOperation}
                        />
                      </div>
                      <div className="field">
                        <label>Max Tokens</label>
                        <input
                          disabled={visionDisabledReason != null}
                          type="number"
                          value={tokenCount}
                          onChange={(event) =>
                            setTokenCount(Number.parseInt(event.target.value, 10) || 0)
                          }
                        />
                      </div>
                    </div>
                    <div className="field">
                      <label>Prompt</label>
                      <textarea
                        disabled={visionDisabledReason != null}
                        value={prompt}
                        onChange={(event) => setPrompt(event.target.value)}
                      />
                    </div>
                    <label className="checkbox-row">
                      <input
                        checked={imageEnabled}
                        disabled={visionDisabledReason != null}
                        type="checkbox"
                        onChange={(event) => setImageEnabled(event.target.checked)}
                      />
                      <span>Attach Image</span>
                    </label>
                    {imageEnabled ? (
                      <div className="field">
                        <label>Image URL or File</label>
                        <div className="form-grid">
                          <input
                            disabled={visionDisabledReason != null}
                            value={imageSource.startsWith('data:') ? '' : imageSource}
                            onChange={(event) => {
                              setImageSource(event.target.value);
                              setImageMeta(null);
                            }}
                            placeholder="https://.../image.jpg"
                          />
                          <input
                            accept="image/jpeg,image/png,image/webp,image/gif"
                            disabled={visionDisabledReason != null}
                            type="file"
                            onChange={uploadImage}
                          />
                        </div>
                      </div>
                    ) : null}
                    <div className="image-preview-panel">
                      <div className="image-preview-frame">
                        {imagePreviewSource == null ? (
                          <div className="image-preview-empty">No image selected</div>
                        ) : (
                          <img alt="Selected vision input" src={imagePreviewSource} />
                        )}
                      </div>
                      {renderDetailTable([
                        { label: 'Image Source', value: imageSelectionKind },
                        { label: 'Image Name', value: imageSelectionName },
                        {
                          label: 'Image Size',
                          value: imageMeta == null ? 'unknown' : formatBytes(imageMeta.size),
                        },
                        { label: 'Image Type', value: imageMeta?.type ?? 'unknown' },
                      ])}
                    </div>
                    {renderDetailTable([
                      { label: 'Selected Capability', value: capabilityLabel(selectedModel.capability) },
                      {
                        label: 'Projector',
                        value: projectorDetail,
                      },
                      { label: 'Loaded Media Marker', value: currentModel?.mediaMarker ?? 'not loaded' },
                    ])}
                  </div>
                </div>
              </Panel>

              {renderFeatureOutputPanel('Vision Response', 'Ready for a vision request.')}
            </div>
          ) : null}

          {activeView === 'embeddings' ? (
            <div className="workspace-view">
              <Panel
                title="Embedding Request"
                actions={
                  <button
                    className="button button-primary"
                    disabled={isBusy || client == null || embeddingDisabledReason != null}
                    onClick={() => void runEmbeddingRequest()}
                    title={embeddingDisabledReason ?? undefined}
                    type="button"
                  >
                    <Play size={16} aria-hidden="true" />
                    Run Embed
                  </button>
                }
              >
                <div className="form-stack">
                  {modelType === 'registry' ? (
                    <div className="field">
                      <label>Embedding Model</label>
                      <select
                        value={isEmbeddingModel(selectedModel) ? selectedRegistryId : ''}
                        onChange={(event) => {
                          const entry = getModelById(event.target.value);
                          if (entry == null) return;
                          configureRegistryEntry(entry);
                        }}
                      >
                        <option value="" disabled>
                          Select an embedding model
                        </option>
                        {getEmbeddingModels().map((model) => (
                          <option key={model.id} value={model.id}>
                            {formatModelOption(model)}
                          </option>
                        ))}
                      </select>
                    </div>
                  ) : null}
                  <div
                    aria-disabled={embeddingDisabledReason != null}
                    className={embeddingSurfaceClass}
                    title={embeddingDisabledReason ?? undefined}
                  >
                    <div className="field">
                      <label>Use Case</label>
                      <SegmentedControl
                        ariaLabel="Embedding use case"
                        disabled={embeddingDisabledReason != null}
                        onChange={applyEmbeddingUseCase}
                        options={embeddingUseCaseOptions}
                        value={embeddingUseCase}
                      />
                    </div>
                    <div className="field">
                      <label>Input Text</label>
                      <textarea
                        disabled={embeddingDisabledReason != null}
                        value={prompt}
                        onChange={(event) => setPrompt(event.target.value)}
                      />
                    </div>
                    <div className="detail-grid">
                      {renderDetailTable([
                        { label: 'Use Case', value: selectedEmbeddingUseCase.label },
                        { label: 'Why', value: selectedEmbeddingUseCase.description },
                        { label: 'Input Prefix', value: selectedEmbeddingUseCase.prefix },
                      ])}
                      {renderDetailTable([
                        {
                          label: 'Model Embeddings',
                          value: yesNo(currentModel?.capabilities?.supportsEmbeddings),
                        },
                        {
                          label: 'Dimensions',
                          value: currentModel?.capabilities?.embedding?.dimensions ?? 'unknown',
                        },
                        {
                          label: 'Pooling',
                          value: currentModel?.capabilities?.embedding?.pooling ?? 'unknown',
                        },
                        {
                          label: 'Vector Preview',
                          value:
                            lastRun?.outputKind === 'embedding'
                              ? `${lastRun.embeddingDimensions ?? 'unknown'} dimensions`
                              : 'run embed to preview',
                        },
                      ])}
                    </div>
                  </div>
                </div>
              </Panel>

              {renderFeatureOutputPanel('Embedding Result', 'Ready for an embedding request.')}
            </div>
          ) : null}

          {activeView === 'benchmarks' ? (
            <div className="workspace-view">
              <Panel
                title="Benchmark Suite"
                actions={
                  <button
                    className="button button-primary"
                    disabled={isBusy || client == null || benchmarkDisabledReason != null}
                    onClick={runBenchmark}
                    title={benchmarkDisabledReason ?? undefined}
                    type="button"
                  >
                    <Gauge size={16} aria-hidden="true" />
                    Run Benchmark Suite
                  </button>
                }
              >
                <div className="form-stack benchmark-controls">
                  <div className="form-grid">
                    <div className="field">
                      <label>Operation</label>
                      <SegmentedControl
                        ariaLabel="Benchmark operation"
                        onChange={setOperation}
                        options={operationOptionsForCurrentModel}
                        value={operation}
                      />
                    </div>
                    <div className="field">
                      <label>Token Emission</label>
                      <SegmentedControl
                        ariaLabel="Benchmark token emission"
                        disabled={tokenEmissionDisabledReason != null}
                        onChange={(value) => setEmitTokens(value === 'on')}
                        options={[
                          { label: 'Off', value: 'off', title: tokenEmissionDisabledReason ?? 'Off' },
                          { label: 'On', value: 'on', title: tokenEmissionDisabledReason ?? 'On' },
                        ]}
                        value={emitTokens ? 'on' : 'off'}
                      />
                    </div>
                  </div>
                  <div className="form-grid">
                    <div className="field">
                      <label>Warmup Runs</label>
                      <input
                        type="number"
                        value={warmupRuns}
                        onChange={(event) => setWarmupRuns(Number.parseInt(event.target.value, 10) || 0)}
                      />
                    </div>
                    <div className="field">
                      <label>Measured Runs</label>
                      <input
                        type="number"
                        value={measuredRuns}
                        onChange={(event) => setMeasuredRuns(Number.parseInt(event.target.value, 10) || 0)}
                      />
                    </div>
                  </div>
                  {renderDetailTable([
                    { label: 'Scenarios', value: 'SISO, SILO, LISO, LILO' },
                    { label: 'Cache Checks', value: operation === 'embed' ? 'n/a for embed' : 'repeated prompt reuse' },
                    { label: 'Mixed Load', value: operation === 'embed' ? 'unsupported for embed' : 'foreground/background' },
                    { label: 'Report', value: benchmarkReport == null ? 'not generated' : benchmarkReport.generatedAt },
                  ])}
                </div>
                {scenarioResults.length === 0 && mixedLoadResult == null ? (
                  <div className="empty-state">No benchmark results yet.</div>
                ) : (
                  <div className="benchmark-results">
                    {scenarioResults.map((scenario) => (
                      <div className="scenario-block" key={scenario.definition.id}>
                        <div className="scenario-header">
                          <div>
                            <h3>{scenario.definition.label}</h3>
                            <p>{scenario.definition.id.toUpperCase()}</p>
                          </div>
                          <div className="scenario-meta">
                            <StatusBadge tone="neutral">
                              load {formatMs(scenario.runtime.loadRuntimeMs)}
                            </StatusBadge>
                            <StatusBadge tone="neutral">
                              limit {scenario.definition.outputTokenLimit}
                            </StatusBadge>
                          </div>
                        </div>
                        {renderGroup('Cold Prompt', scenario.coldPrompt)}
                        {renderGroup('Hot Fresh Context', scenario.hotFreshContext)}
                        {renderGroup('Repeated Prompt', scenario.repeatedPrompt)}
                      </div>
                    ))}

                    {mixedLoadResult == null ? null : mixedLoadResult.unsupported ? (
                      <div className="result-panel">
                        <div className="result-panel-header">
                          <h4>{mixedLoadResult.definition.label}</h4>
                          <StatusBadge tone="warn">unsupported</StatusBadge>
                        </div>
                        <p className="result-preview">{mixedLoadResult.reason ?? 'Unsupported.'}</p>
                      </div>
                    ) : (
                      <div className="scenario-block">
                        <div className="scenario-header">
                          <div>
                            <h3>{mixedLoadResult.definition.label}</h3>
                            <p>Concurrency {mixedLoadResult.definition.concurrency}</p>
                          </div>
                          <div className="scenario-meta">
                            <StatusBadge tone="neutral">
                              foreground {mixedLoadResult.definition.foreground.label}
                            </StatusBadge>
                            <StatusBadge tone="neutral">
                              background {mixedLoadResult.definition.background.label}
                            </StatusBadge>
                          </div>
                        </div>
                        {mixedLoadResult.foreground == null
                          ? null
                          : renderGroup('Foreground Under Load', mixedLoadResult.foreground)}
                        {mixedLoadResult.background == null
                          ? null
                          : renderGroup('Background Under Load', mixedLoadResult.background)}
                      </div>
                    )}
                  </div>
                )}
                {renderLiveOutput('Benchmark output will stream here.')}
              </Panel>
            </div>
          ) : null}

          {activeView === 'observability' ? (
            <div className="workspace-view two-column">
              <Panel title="Runtime State">
                <div className="metric-grid">
                  <MetricCard
                    label="Current Model"
                    value={currentModel?.name ?? 'none'}
                    tone={currentModel?.loaded ? 'ok' : 'warn'}
                  />
                  <MetricCard label="Status" value={currentModel?.status ?? 'none'} />
                  <MetricCard label="Selected Capability" value={capabilityLabel(selectedModel.capability)} />
                  <MetricCard label="Loaded Modality" value={currentModel?.modality ?? 'none'} />
                  <MetricCard label="Model Class" value={currentModel?.capabilities?.modelClass ?? 'unknown'} />
                  <MetricCard label="Model Bytes" value={formatBytes(sourceInfo?.bytes ?? null)} />
                  <MetricCard label="Source" value={sourceInfo?.label ?? 'none'} />
                  <MetricCard label="Backend" value={describeRuntimeBackend(observability?.profile)} />
                  <MetricCard label="Memory Snapshots" value={memorySnapshots.length} />
                </div>
                {renderDetailTable([
                  { label: 'Text', value: yesNo(currentModel?.capabilities?.supportsTextGeneration) },
                  { label: 'Vision', value: yesNo(currentModel?.modality === 'vision') },
                  { label: 'Embeddings', value: yesNo(currentModel?.capabilities?.supportsEmbeddings) },
                  { label: 'Pooling', value: currentModel?.capabilities?.embedding?.pooling ?? 'n/a' },
                  { label: 'Observability State', value: observability?.state ?? 'idle' },
                ])}
              </Panel>

              <Panel title="Engine Health">
                <div className="metric-grid">
                  <MetricCard
                    label="Rust Engine"
                    value={engineStatus}
                    tone={browserSmoke?.rustEngine.available ? 'ok' : browserSmokeError ? 'warn' : undefined}
                  />
                  <MetricCard
                    label="Rust GGUF Ingest"
                    value={ingestStatus}
                    tone={browserSmoke?.ggufIngest.available ? 'ok' : browserSmokeError ? 'warn' : undefined}
                  />
                  <MetricCard
                    label="WebGPU Smoke"
                    value={webgpuStatus}
                    tone={browserSmoke?.webgpuReady ? 'ok' : browserSmokeError ? 'warn' : undefined}
                  />
                </div>
                {renderDetailTable([
                  {
                    label: 'Current Decode TPS',
                    value:
                      lastRun?.decodeTps != null
                        ? formatTps(lastRun.decodeTps)
                        : observability?.runtime?.decodeTokensPerSecond == null
                          ? 'n/a'
                          : round(observability.runtime.decodeTokensPerSecond),
                  },
                  {
                    label: 'Current E2E TPS',
                    value:
                      lastRun?.e2eTps != null
                        ? formatTps(lastRun.e2eTps)
                        : observability?.runtime?.e2eTokensPerSecond == null
                          ? 'n/a'
                          : round(observability.runtime.e2eTokensPerSecond),
                  },
                  {
                    label: 'Current Prefill TPS',
                    value:
                      lastRun?.prefillTps != null
                        ? formatTps(lastRun.prefillTps)
                        : observability?.runtime?.prefillTokensPerSecond == null
                          ? 'n/a'
                          : round(observability.runtime.prefillTokensPerSecond),
                  },
                ])}
              </Panel>

              <Panel title="Request Observability">
                {lastRun == null ? (
                  <div className="empty-state">No request metrics yet.</div>
                ) : (
                  <>
                    {renderRunMetrics()}
                    {renderRequestObservability()}
                  </>
                )}
              </Panel>

              <Panel title="Backend Profile">
                <div className="metric-grid">
                  <MetricCard
                    label="Profiling"
                    value={yesNo(observability?.profile?.profilingEnabled)}
                    tone={observability?.profile?.profilingEnabled ? 'ok' : undefined}
                  />
                  <MetricCard
                    label="WebGPU"
                    value={describeRuntimeBackend(observability?.profile)}
                    tone={observability?.profile?.webgpuRegistered ? 'ok' : 'warn'}
                  />
                  <MetricCard
                    label="Devices"
                    value={observability?.profile?.devices.length ?? 0}
                  />
                </div>
                {renderDetailTable([
                  {
                    label: 'Available Backends',
                    value:
                      observability?.profile?.availableBackends
                        ?.map((backend) => `${backend.name}:${backend.deviceCount}`)
                        .join(', ') || 'none',
                  },
                  {
                    label: 'Runtime Devices',
                    value:
                      observability?.profile?.devices
                        ?.map((device) => device.description || device.name || device.type)
                        .join(' | ') || 'none',
                  },
                  { label: 'Memory Snapshots', value: memorySnapshots.length },
                  {
                    label: 'Last JS Heap Peak',
                    value: formatBytes(benchmarkReport?.memory.summary.maxUsedJsHeapBytes ?? null),
                  },
                  {
                    label: 'Last UA Memory Peak',
                    value: formatBytes(benchmarkReport?.memory.summary.maxUserAgentBytes ?? null),
                  },
                ])}
              </Panel>

              <Panel className="wide-panel" title="Raw Observability Snapshot">
                <pre className="json-preview">
                  {JSON.stringify(
                    {
                      observability,
                      runtimeSmoke: browserSmoke,
                      lastRun: lastRun?.observability ?? null,
                    },
                    null,
                    2
                  )}
                </pre>
              </Panel>
            </div>
          ) : null}

          {activeView === 'reports' ? (
            <div className="workspace-view">
              <Panel
                title="Reports"
                actions={
                  <button
                    className="button button-primary"
                    disabled={benchmarkReport == null}
                    onClick={() =>
                      benchmarkReport == null
                        ? undefined
                        : downloadJson('cogent-playground-report.json', benchmarkReport)
                    }
                    type="button"
                  >
                    <Download size={16} aria-hidden="true" />
                    Export JSON
                  </button>
                }
              >
                {benchmarkReport == null ? (
                  <div className="empty-state">No report generated.</div>
                ) : (
                  <>
                    <div className="metric-grid">
                      <MetricCard label="Schema" value={benchmarkReport.schema} />
                      <MetricCard label="Raw Runs" value={benchmarkReport.trace.runCount} />
                      <MetricCard label="Snapshots" value={benchmarkReport.memory.summary.snapshotCount} />
                      <MetricCard
                        label="JS Heap Peak"
                        value={formatBytes(benchmarkReport.memory.summary.maxUsedJsHeapBytes)}
                      />
                      <MetricCard
                        label="UA Memory Peak"
                        value={formatBytes(benchmarkReport.memory.summary.maxUserAgentBytes)}
                      />
                      <MetricCard label="Backend" value={benchmarkReport.backend.inferredExecutionBackend} />
                    </div>
                    {renderDetailTable([
                      { label: 'Generated', value: benchmarkReport.generatedAt },
                      { label: 'Model', value: benchmarkReport.model?.name ?? 'none' },
                      { label: 'Source', value: benchmarkReport.source.label },
                      { label: 'Operation', value: benchmarkReport.settings.operation },
                      { label: 'Prompt Tokens', value: benchmarkReport.settings.tokenCount },
                      { label: 'Runtime', value: benchmarkReport.backend.runtimeBackendStatus },
                    ])}
                    <div className="data-table-wrap">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Trace</th>
                            <th>Mean</th>
                            <th>P99</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td>TTFT</td>
                            <td>{benchmarkReport.trace.analysis.ttftMs?.mean ?? 'n/a'}</td>
                            <td>{benchmarkReport.trace.analysis.ttftMs?.p99 ?? 'n/a'}</td>
                          </tr>
                          <tr>
                            <td>ITL</td>
                            <td>{benchmarkReport.trace.analysis.itlAvgMs?.mean ?? 'n/a'}</td>
                            <td>{benchmarkReport.trace.analysis.itlP99Ms?.p99 ?? 'n/a'}</td>
                          </tr>
                          <tr>
                            <td>Decode TPS</td>
                            <td>{benchmarkReport.trace.analysis.decodeTps?.mean ?? 'n/a'}</td>
                            <td>{benchmarkReport.trace.analysis.decodeTps?.p99 ?? 'n/a'}</td>
                          </tr>
                          <tr>
                            <td>E2E TPS</td>
                            <td>{benchmarkReport.trace.analysis.e2eTps?.mean ?? 'n/a'}</td>
                            <td>{benchmarkReport.trace.analysis.e2eTps?.p99 ?? 'n/a'}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <pre className="json-preview">
                      {JSON.stringify(benchmarkReport, null, 2)}
                    </pre>
                  </>
                )}
              </Panel>
            </div>
          ) : null}
        </div>
      </main>
    </div>
  );
}
