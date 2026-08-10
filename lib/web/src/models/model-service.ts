import type { EngineRuntime } from '../runtime/engine-runtime.js';
import {
  buildBoundaryMarkers,
  sliceUndeliveredSuffix,
  TokenBoundaryTextSanitizer,
} from '../engine/chat-boundary-sanitizer.js';
import type {
  GenerateRequestHandle,
  GenerateResponse,
  NativeRuntimeConfig,
  PromptOptions,
  SamplingRuntimeOverride,
  TransportObservability,
} from '../engine/inference-types.js';
import { hasSamplingRuntimeOverrideFields } from '../engine/inference-types.js';
import {
  createAbortError,
  createLinkedAbortController,
  isAbortError,
} from '../utils/abort.js';
import { AsyncSerialQueue } from '../utils/async-queue.js';
import { attachCleanupFailures, releaseAll, releaseAllAsync } from '../utils/cleanup.js';
import { AssetStore } from './asset-store.js';
import { RemoteAcquisitionHost } from './remote-acquisition-host.js';
import { ModelRegistryStore } from './model-registry-store.js';
import type {
  RustLifecycleBridge,
  RustLifecycleInstallSource,
  RustLifecycleInstallValue,
  RustRemoteCommandValue,
} from '../wasm/wasm-bridge.js';
import { queryErrorFromLifecycleError } from '../wasm/wasm-bridge.js';
import {
  QueryError,
  type AssetRecord,
  type AudioResult,
  type BrowserBackendPreference,
  type CatalogModelInfo,
  type CatalogObservabilityEvent,
  type ChatInput,
  type ClassifiedAsset,
  type ClassifiedAssetFile,
  type EmbedOptions,
  type EmbeddingResult,
  type EngineEvent,
  type EngineState,
  type ModelEntry,
  type ModelInfo,
  type ModelAddOptions,
  type ModelAddSource,
  type ModelLoadOptions,
  type ObservabilityEvent,
  type ObservabilitySnapshot,
  type QueryObservation,
  type QueryInput,
  type QueryOptions,
  type GenerationResult,
  type InternalTextRequestOptions,
  type ListenOptions,
  type TokenBatch,
  type RegistryManifest,
  type RuntimeBundleDescriptor,
  type RuntimeBundleFile,
  type RuntimeSessionDescriptor,
  type RuntimeSessionSnapshot,
  type SpeakOptions,
  type WebGpuAdapterInfo,
} from './types.js';
import {
  audioResultFromGenerateResponse,
  embeddingResultFromGenerateResponse,
  generationResultFromGenerateResponse,
  generationResultFromText,
  ObservabilityController,
  observabilityEventToStateEvent,
  observabilitySnapshotToEngineState,
  toBackendProfileObservation,
  toRuntimeObservation,
} from './observability-controller.js';
import type {
  RuntimeBackendConstraint,
  WasmThreadingMode,
} from '../engine/runtime-assets.js';

interface InstalledAsset {
  record: AssetRecord;
  file: File;
}

interface AssetClassifier {
  classify(assetId: string, file: File, signal?: AbortSignal): Promise<ClassifiedAssetFile>;
}

interface RuntimeRequestOptions {
  contextKey?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  sampling?: SamplingRuntimeOverride;
  stop?: readonly string[];
  signal?: AbortSignal;
  emitTokens?: boolean;
  tokenBatchSink?: (batch: TokenBatch) => void;
  grammar?: string;
  onRequestStarted?: (requestId: number) => void;
}

const DEFAULT_TRANSCRIPTION_MAX_TOKENS = 512;
const MAX_LOCAL_TOKEN_COUNT = 0x7fffffff;
const MAX_SPEECH_DURATION_MS = 0xffffffff;
const OPFS_LOCK_RETRY_DELAYS_MS = [25, 50, 100, 200, 400] as const;

function validateSpeechLanguage(operation: string, language: string | undefined): void {
  if (language == null) {
    return;
  }
  if (language.trim().length === 0 || language.trim() !== language) {
    throw new QueryError(
      'QUERY_FAILED',
      `${operation} language must not be empty or contain surrounding whitespace.`
    );
  }
}

type NavigatorWithGpu = Navigator & {
  gpu?: {
    requestAdapter(): Promise<NavigatorGpuAdapter | null>;
  };
};
type NavigatorGpuAdapter = {
  readonly features?: {
    has(feature: string): boolean;
  };
  readonly info?: Partial<WebGpuAdapterInfo> | null;
  requestAdapterInfo?: () => Promise<Partial<WebGpuAdapterInfo> | null>;
};

interface ResolvedBrowserBackend {
  backend: Exclude<BrowserBackendPreference, 'auto'>;
  webgpuAdapter: WebGpuAdapterInfo | null;
}

function isFile(value: unknown): value is File {
  return typeof File !== 'undefined' && value instanceof File;
}

async function resolveBrowserBackend(
  backend: BrowserBackendPreference | undefined,
  constraint: RuntimeBackendConstraint | null
): Promise<ResolvedBrowserBackend> {
  const requestedBackend = backend === 'auto' ? undefined : backend;
  if (constraint === 'cpu-only') {
    if (requestedBackend === 'webgpu') {
      throw new QueryError(
        'UNSUPPORTED_OPERATION',
        'The active browser runtime is CPU-only because this browser did not pass the JSPI ' +
          'suspend/resume probe. Remove backend: "webgpu", or supply WebGPU-capable ' +
          'moduleUrl and wasmUrl assets.'
      );
    }
    return { backend: 'cpu', webgpuAdapter: null };
  }
  if (requestedBackend === 'cpu') {
    return { backend: requestedBackend, webgpuAdapter: null };
  }
  const gpu = (globalThis.navigator as NavigatorWithGpu | undefined)?.gpu;
  const adapter = gpu == null ? null : await gpu.requestAdapter();
  if (requestedBackend === 'webgpu') {
    return { backend: requestedBackend, webgpuAdapter: await readWebGpuAdapterInfo(adapter) };
  }
  if (adapter?.features?.has('shader-f16') !== true) {
    return { backend: 'cpu', webgpuAdapter: null };
  }
  return { backend: 'webgpu', webgpuAdapter: await readWebGpuAdapterInfo(adapter) };
}

/**
 * Read the adapter identity in the scope that runs the engine. The wasm
 * backend requests its adapter from this same scope, so this mirrors the GPU
 * inference actually executes on (decisive on hybrid-GPU machines, where the
 * browser, not the app, picks the physical GPU).
 */
async function readWebGpuAdapterInfo(
  adapter: NavigatorGpuAdapter | null
): Promise<WebGpuAdapterInfo | null> {
  if (adapter == null) {
    return null;
  }
  const info = adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? null;
  if (info == null) {
    return null;
  }
  return {
    vendor: info.vendor ?? '',
    architecture: info.architecture ?? '',
    device: info.device ?? '',
    description: info.description ?? '',
  };
}

function hostSleep(delayMs: number): Promise<void> {
  return new Promise<void>((resolve) => setTimeout(resolve, delayMs));
}

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

const textEncoder = new TextEncoder();

function tokenBatchFromText(
  requestId: string,
  streamId: number,
  sequenceStart: number,
  text: string
): TokenBatch {
  const byteCount = utf8ByteLength(text);
  return {
    requestId,
    streamId,
    sequenceStart,
    text,
    frameCount: 1,
    byteCount,
    stats: {
      framesSent: sequenceStart + 1,
      bytesSent: byteCount,
      batchesSent: sequenceStart + 1,
    },
  };
}

function utf8ByteLength(text: string): number {
  return textEncoder.encode(text).byteLength;
}

function requestHandleLabel(request: GenerateRequestHandle): string {
  return `${request.generation}:${request.requestId}`;
}

function normalizeLocalSourceFileName(file: File): string {
  const trimmed = (file.name || 'model.gguf').trim();
  const defaultValue = trimmed.length > 0 ? trimmed : 'model.gguf';
  return defaultValue.replace(/[\\/:*?"<>|]+/g, '-');
}

function browserDefaultThreadCount(): number {
  const hardwareConcurrency = globalThis.navigator?.hardwareConcurrency;
  const cores =
    typeof hardwareConcurrency === 'number' && Number.isFinite(hardwareConcurrency)
      ? Math.trunc(hardwareConcurrency)
      : 4;
  return Math.max(1, Math.min(4, cores));
}

function applyBrowserRuntimeDefaults(
  runtime: NativeRuntimeConfig | undefined,
  wasmThreading: WasmThreadingMode
): NativeRuntimeConfig {
  const threadCount = wasmThreading === 'pthread' ? browserDefaultThreadCount() : 1;
  return {
    ...runtime,
    context: {
      ...runtime?.context,
      n_threads: runtime?.context?.n_threads ?? threadCount,
      n_threads_batch: runtime?.context?.n_threads_batch ?? threadCount,
      warmup: runtime?.context?.warmup ?? false,
    },
  };
}

export class ModelService {
  private chatBoundaryMarkersPromise: Promise<readonly string[]> | null = null;
  private readonly lifecycleOperations = new AsyncSerialQueue();
  private readonly observability = new ObservabilityController();
  private readonly engineEventListeners = new Set<(event: EngineEvent) => void>();
  private rustLifecyclePromise: Promise<RustLifecycleBridge> | null = null;

  constructor(
    private readonly runtime: EngineRuntime,
    private readonly registry = new ModelRegistryStore(),
    private readonly assetStore = new AssetStore(),
    assetClassifier?: AssetClassifier,
    /** @internal Test seam; production always waits on real time. */
    private readonly sleep: (delayMs: number) => Promise<void> = hostSleep
  ) {
    this.assetClassifier = assetClassifier ?? {
      classify: async (assetId, file, signal) => {
        const detection = await runtime.detectModelFromGgufFile(file, signal);
        return {
          assetId,
          file,
          inspection: detection.inspection,
          name: detection.modelName,
        };
      },
    };
    this.observability.subscribe((event) => {
      this.emitEngineEvent(observabilityEventToStateEvent(event));
    });
  }

  private readonly assetClassifier: AssetClassifier;

  public current(): ModelInfo | null {
    const session = this.runtime.currentRuntimeSession();
    return session == null ? null : modelInfoFromCatalog(session.model, session);
  }

  private requireOperation(
    operation: keyof RuntimeSessionSnapshot['capabilities']['operations']
  ): RuntimeSessionSnapshot {
    const session = this.runtime.currentRuntimeSession();
    if (session == null) {
      throw new QueryError('MODEL_NOT_READY', 'No model is loaded. Call client.add(...) first.');
    }
    if (!session.capabilities.operations[operation]) {
      throw new QueryError(
        'UNSUPPORTED_OPERATION',
        `Loaded model "${session.model.id}" does not support ${operation}.`
      );
    }
    return session;
  }

  public async list(): Promise<ModelInfo[]> {
    const manifest = await this.registry.read();
    const rust = await this.getRustLifecycle(manifest);
    const session = this.runtime.currentRuntimeSession();
    return (await rust.list()).map((model) => modelInfoFromCatalog(model, session));
  }

  public currentObservability(): ObservabilitySnapshot {
    return this.observability.current();
  }

  public subscribeObservability(listener: (event: ObservabilityEvent) => void): () => void {
    return this.observability.subscribe(listener);
  }

  public state(): EngineState {
    return observabilitySnapshotToEngineState(this.observability.current());
  }

  public subscribeEvents(listener: (event: EngineEvent) => void): () => void {
    this.engineEventListeners.add(listener);
    return () => {
      this.engineEventListeners.delete(listener);
    };
  }

  public async add(
    source: ModelAddSource,
    options: ModelAddOptions = {}
  ): Promise<ModelInfo> {
    return this.lifecycleOperations.run(async () => {
      if (options.signal?.aborted) {
        throw new DOMException('Model install aborted.', 'AbortError');
      }
      return await this.addWithRustLifecycle(source, options);
    });
  }

  public async load(modelId: string, options: ModelLoadOptions = {}): Promise<ModelInfo> {
    return this.lifecycleOperations.run(async () => {
      if (options.signal?.aborted) {
        throw new DOMException('Model load aborted.', 'AbortError');
      }
      return await this.loadWithRustLifecycle(modelId, options);
    });
  }

  public async remove(id: string): Promise<void> {
    await this.lifecycleOperations.run(async () => {
      const manifest = await this.registry.read();
      const rust = await this.getRustLifecycle(manifest);
      const activeModelId = this.runtime.currentRuntimeSession()?.model.id ?? null;
      const removed = await rust.remove(id, activeModelId);
      await this.replaceManifest(removed.manifest);
      for (const asset of removed.orphanedAssets) {
        await this.assetStore.delete(asset);
      }
      this.ingestRustEvents(removed.events);
    });
  }

  public async runQuery(
    input: QueryInput,
    options: InternalTextRequestOptions
  ): Promise<GenerationResult> {
    this.requireOperation('query');
    let prompt = typeof input === 'string' ? input : input.prompt;
    const media = typeof input === 'string' ? undefined : input.media;
    if (media != null && media.length > 0) {
      const marker = this.runtime.readMediaMarker();
      if (marker == null) {
        throw new QueryError('MODEL_NOT_READY', 'The loaded model does not accept media input.');
      }
      if (!prompt.includes(marker)) {
        prompt = `${Array.from({ length: media.length }, () => marker).join('\n')}\n${prompt}`;
      }
    }
    const response = await this.runRuntimeRequest(
      options,
      media,
      (contextKey, promptOptions) => this.runtime.enqueueQuery(contextKey, prompt, promptOptions),
      'Model query'
    );
    return generationResultFromGenerateResponse(response, {
      maxTokens: options.maxTokens,
    });
  }

  public async runEmbedding(
    input: string,
    options: EmbedOptions
  ): Promise<EmbeddingResult> {
    this.requireOperation('embed');

    const response = await this.runRuntimeRequest(
      {
        contextKey: options.contextKey,
        signal: options.signal,
      },
      undefined,
      (contextKey) =>
        this.runtime.enqueueEmbedding(contextKey, input, {
          normalize: options.normalize ?? true,
          signal: options.signal,
        }),
      'Model embedding'
    );
    return embeddingResultFromGenerateResponse(response);
  }

  public async runListen(
    audio: Uint8Array,
    options: ListenOptions
  ): Promise<GenerationResult> {
    this.requireOperation('listen');
    if (audio.byteLength === 0) {
      throw new QueryError('QUERY_FAILED', 'Listen audio must not be empty.');
    }
    validateSpeechLanguage('Listen', options.language);
    if (
      options.maxTokens != null &&
      (!Number.isInteger(options.maxTokens) ||
        options.maxTokens <= 0 ||
        options.maxTokens > MAX_LOCAL_TOKEN_COUNT)
    ) {
      throw new QueryError(
        'QUERY_FAILED',
        `Listen maxTokens must be an integer between 1 and ${MAX_LOCAL_TOKEN_COUNT}.`
      );
    }
    const maxTokens = options.maxTokens ?? DEFAULT_TRANSCRIPTION_MAX_TOKENS;
    const response = await this.runRuntimeRequest(
      {
        maxTokens,
        signal: options.signal,
      },
      undefined,
      (_contextKey, promptOptions) =>
        this.runtime.enqueueListen(
          audio,
          options.language ?? '',
          promptOptions
        ),
      'Model listen'
    );
    return generationResultFromGenerateResponse(response, {
      maxTokens,
    });
  }

  public async runSpeak(text: string, options: SpeakOptions): Promise<AudioResult> {
    this.requireOperation('speak');
    if (text.trim().length === 0 || text.trim() !== text) {
      throw new QueryError(
        'QUERY_FAILED',
        'Speak text must not be empty or contain surrounding whitespace.'
      );
    }
    validateSpeechLanguage('Speak', options.language);
    if (options.speakerAudio != null && options.speakerAudio.byteLength === 0) {
      throw new QueryError('QUERY_FAILED', 'Speak speakerAudio must not be empty.');
    }
    if (
      options.maxDurationMs != null &&
      (!Number.isInteger(options.maxDurationMs) ||
        options.maxDurationMs <= 0 ||
        options.maxDurationMs > MAX_SPEECH_DURATION_MS)
    ) {
      throw new QueryError(
        'QUERY_FAILED',
        `Speak maxDurationMs must be an integer between 1 and ${MAX_SPEECH_DURATION_MS}.`
      );
    }
    const response = await this.runRuntimeRequest(
      { signal: options.signal },
      undefined,
      (_contextKey, promptOptions) =>
        this.runtime.enqueueSpeak(
          text,
          options.language ?? '',
          options.speakerAudio ?? new Uint8Array(),
          options.maxDurationMs,
          promptOptions
        ),
      'Model speak'
    );
    return audioResultFromGenerateResponse(response);
  }

  private async runRuntimeRequest(
    options: RuntimeRequestOptions,
    media: Uint8Array[] | undefined,
    enqueue: (contextKey: string, promptOptions: PromptOptions) => Promise<GenerateRequestHandle>,
    operationLabel = 'Model query'
  ): Promise<GenerateResponse> {
    const deliverTokenBatch = (batch: TokenBatch): void => {
      if (batch.text.length === 0) {
        return;
      }
      options.tokenBatchSink?.(batch);
    };
    const promptOptions: PromptOptions = {
      nTokens: options.maxTokens,
      signal: options.signal,
      emitTokens: options.emitTokens === true || options.tokenBatchSink != null,
      tokenBatchSink: options.tokenBatchSink == null ? undefined : deliverTokenBatch,
      media,
      stop: options.stop,
      sampling: samplingRuntimeOverride(options),
      grammar: options.grammar,
      onRequestStarted: options.onRequestStarted,
    };
    const contextKey = options.contextKey ?? 'default';
    const emitsTokens = promptOptions.emitTokens === true;
    const transportStart = this.runtime.getTransportObservability();
    const requestTransport = (): TransportObservability =>
      this.requestTransportObservability(transportStart, emitsTokens);
    const start = nowMs();
    this.observability.emit('query-start', {
      state: 'querying',
      query: {
        contextKey,
        status: 'running',
        wallMs: null,
        ttftMs: null,
        outputTokens: null,
      },
    });
    let request: GenerateRequestHandle | null = null;
    let failureRecorded = false;
    try {
      request = await enqueue(contextKey, promptOptions);
      const requestLabel = requestHandleLabel(request);
      this.emitEngineEvent({
        type: 'request-started',
        requestId: requestLabel,
        streamId: request.requestId,
      });
      const response = await this.runtime.awaitQuery(request, { signal: options.signal });
      const terminalError = response.cancelled
        ? new DOMException(response.errorMessage ?? 'Queued request cancelled.', 'AbortError')
        : response.failed
          ? new Error(response.errorMessage ?? 'Queued prompt failed.')
          : null;
      if (terminalError != null) {
        this.recordQueryFailure(
          contextKey,
          start,
          terminalError,
          response,
          requestTransport()
        );
        this.emitEngineEvent({
          type: 'request-failed',
          requestId: requestLabel,
          error: terminalError.message,
        });
        failureRecorded = true;
        throw terminalError;
      }
      this.recordQuerySuccess(
        contextKey,
        start,
        response,
        requestTransport()
      );
      this.emitEngineEvent({
        type: 'request-completed',
        requestId: requestLabel,
      });
      return response;
    } catch (error) {
      if (!failureRecorded) {
        this.recordQueryFailure(
          contextKey,
          start,
          error,
          undefined,
          requestTransport()
        );
      }
      if (error instanceof QueryError) {
        throw error;
      }
      const wrapped = new QueryError(
        'QUERY_FAILED',
        error instanceof Error && error.message.trim().length > 0
          ? `${operationLabel} failed: ${error.message}`
          : `${operationLabel} failed.`,
        { cause: error }
      );
      if (!failureRecorded && request != null) {
        this.emitEngineEvent({
          type: 'request-failed',
          requestId: requestHandleLabel(request),
          error: wrapped.message,
        });
      }
      throw wrapped;
    }
  }

  public async runChat(
    input: ChatInput,
    options: InternalTextRequestOptions
  ): Promise<GenerationResult> {
    const current = this.requireOperation('chat');
    const messages = isChatInputObject(input) ? input.messages : input;
    const media = isChatInputObject(input) ? input.media : undefined;
    if (media != null && media.length > 0 && this.runtime.readMediaMarker() == null) {
      throw new QueryError('MODEL_NOT_READY', 'The loaded model does not accept media input.');
    }
    const boundaryMarkers = await this.getChatBoundaryMarkers();
    const outputSanitizer = new TokenBoundaryTextSanitizer(boundaryMarkers);
    const linkedAbort = createLinkedAbortController(options.signal);
    let deliveredOutputText = '';
    let assistantText = '';
    let stoppedAtBoundary = false;

    let safeSequence = 0;
    let lastBatch: TokenBatch | null = null;
    const shouldDeliverTokens = options.tokenBatchSink != null;
    const consumeOutputTokens = (batch: TokenBatch): void => {
      lastBatch = batch;
      const text = batch.text;
      if (text.length === 0 || outputSanitizer.reachedBoundary) {
        return;
      }
      deliveredOutputText += text;
      const result = outputSanitizer.consume(text);
      if (result.safeText.length > 0) {
        assistantText += result.safeText;
        options.tokenBatchSink?.(
          tokenBatchFromText(batch.requestId, batch.streamId, safeSequence++, result.safeText)
        );
      }
      if (result.hitBoundary) {
        stoppedAtBoundary = true;
        linkedAbort.controller.abort();
      }
    };

    const flushOutputText = (): void => {
      const safeText = outputSanitizer.flush();
      if (safeText.length > 0) {
        assistantText += safeText;
        const source = lastBatch ?? tokenBatchFromText('0', 0, safeSequence, safeText);
        options.tokenBatchSink?.(
          tokenBatchFromText(source.requestId, source.streamId, safeSequence++, safeText)
        );
      }
    };

    try {
      const rawResult = await this.runRuntimeRequest(
        {
          ...options,
          signal: linkedAbort.signal,
          ...(shouldDeliverTokens ? { tokenBatchSink: consumeOutputTokens } : {}),
        },
        media == null ? undefined : [...media],
        (contextKey, promptOptions) => this.runtime.enqueueChat(contextKey, messages, promptOptions),
        'Model chat'
      );
      const rawText = rawResult.outputText;
      if (rawText == null) {
        throw new Error('Runtime completed chat() without text output.');
      }
      const unseenOutputSuffix = shouldDeliverTokens
        ? sliceUndeliveredSuffix(deliveredOutputText, rawText)
        : rawText;
      if (!outputSanitizer.reachedBoundary && unseenOutputSuffix.length > 0) {
        const source = lastBatch ?? tokenBatchFromText(
          String(rawResult.requestId),
          rawResult.requestId,
          safeSequence,
          unseenOutputSuffix
        );
        consumeOutputTokens(
          tokenBatchFromText(source.requestId, source.streamId, safeSequence, unseenOutputSuffix)
        );
      }
      flushOutputText();
      return generationResultFromGenerateResponse(rawResult, {
        text: assistantText.trim(),
        maxTokens: options.maxTokens,
      });
    } catch (error) {
      if (stoppedAtBoundary && options.signal?.aborted !== true) {
        flushOutputText();
        return generationResultFromText({
          id: -1,
          text: assistantText.trim(),
          finishReason: 'stop',
          metrics: null,
        });
      }
      throw error;
    } finally {
      linkedAbort.dispose();
    }
  }

  public async close(): Promise<void> {
    try {
      await releaseAllAsync('Failed to close the browser model service.', [
        {
          label: 'close Rust lifecycle service',
          release: () => this.closeRustLifecycle(),
        },
        {
          label: 'close Wasm engine runtime',
          release: () => this.runtime.close(),
        },
      ]);
    } finally {
      this.observability.markClosed();
    }
  }

  private async addWithRustLifecycle(
    source: ModelAddSource,
    options: ModelAddOptions
  ): Promise<ModelInfo> {
    const addOptions: ModelAddOptions = {
      ...options,
      onProgress: (progress) => {
        options.onProgress?.(progress);
        this.emitEngineEvent({
          type: 'load-progress',
          loadedBytes: progress.loadedBytes,
          totalBytes: progress.totalBytes,
          assetName: progress.assetName,
        });
      },
    };
    const manifest = await this.registry.read();
    const rustPromise = this.getRustLifecycle(manifest);
    const remoteHost = source.kind === 'remote'
      ? new RemoteAcquisitionHost(
        this.assetStore,
        this.runtime,
        manifest,
        async (assetId, file, signal) => {
          const result = await this.assetClassifier.classify(assetId, file, signal);
          return {
            assetId: result.assetId,
            name: result.name,
            inspection: result.inspection,
          };
        },
        addOptions
      )
      : null;
    let rust: RustLifecycleBridge | null = null;
    let remoteAcquisitionId: string | null = null;
    let remoteManifestCommitted = false;
    try {
      rust = await rustPromise;
      const installed = source.kind === 'remote'
        ? await this.acquireRemote(
          rust,
          remoteHost as RemoteAcquisitionHost,
          source,
          (acquisitionId) => {
            remoteAcquisitionId = acquisitionId;
          }
        )
        : await rust.install(await this.buildRustInstallSource(source, manifest, addOptions));
      remoteAcquisitionId = null;
      await this.replaceManifest(installed.manifest);
      if (remoteHost != null) {
        remoteManifestCommitted = true;
        await remoteHost.commitJournal();
      }
      this.ingestRustEvents(installed.events);
      return modelInfoFromCatalog(installed.model, this.runtime.currentRuntimeSession());
    } catch (error) {
      const lifecycle = rust;
      const journalHost = remoteHost;
      const acquisitionId: string | null = remoteAcquisitionId;
      const committed = remoteManifestCommitted;
      try {
        await releaseAllAsync('Failed to clean up an unsuccessful model install.', [
          ...(acquisitionId != null && lifecycle != null && journalHost != null
            ? [{
                label: 'cancel remote acquisition',
                release: () =>
                  this.cancelRemoteAcquisition(lifecycle, journalHost, acquisitionId),
              }]
            : []),
          ...(journalHost != null && !committed
            ? [{
                label: 'discard uncommitted acquisition journal',
                release: () => journalHost.cleanupUncommittedJournal(manifest),
              }]
            : []),
        ]);
      } catch (cleanupError) {
        throw attachCleanupFailures(error, cleanupError);
      }
      throw error;
    }
  }

  private async loadWithRustLifecycle(
    modelId: string,
    options: ModelLoadOptions
  ): Promise<ModelInfo> {
    const loadOptions: ModelLoadOptions = {
      ...options,
      onProgress: (progress) => {
        options.onProgress?.(progress);
        this.emitEngineEvent({
          type: 'load-progress',
          loadedBytes: progress.loadedBytes,
          totalBytes: progress.totalBytes,
          assetName: progress.assetName,
        });
      },
    };
    const observabilityMode = options.observability ?? 'off';
    const manifest = await this.registry.read();
    const rustPromise = this.getRustLifecycle(manifest);
    const wasmThreading = this.runtime.getWasmThreadingMode();
    const backendPromise = resolveBrowserBackend(
      options.backend,
      this.runtime.backendConstraint
    );
    const [resolvedRust, resolvedBackend] = await Promise.all([rustPromise, backendPromise]);
    const runtimeConfig = applyBrowserRuntimeDefaults(options.runtime, wasmThreading);
    const rustOptions = {
      backend: resolvedBackend.backend,
      runtime: runtimeConfig,
      observability: observabilityMode,
    } as const;
    const prepared = await resolvedRust.prepareLoad({ modelId }, rustOptions);
    this.ingestRustEvents(prepared.events);

    if (prepared.model.status !== 'ready') {
      throw new QueryError(
        'MODEL_NOT_READY',
        `Model "${prepared.model.id}" is not ready to load.`
      );
    }

    const entry = prepared.manifest.models[prepared.model.id];
    if (entry == null) {
      throw new QueryError('STORAGE_CORRUPT', `Rust lifecycle omitted model "${prepared.model.id}".`);
    }

    const targetSession: RuntimeSessionDescriptor = {
      model: prepared.model,
      runtimeFingerprint: prepared.runtimeFingerprint,
    };
    const descriptor = await this.openBundleForEntry(
      entry,
      prepared.manifest,
      options.signal
    );
    loadOptions.onProgress?.({
      phase: 'load',
      loadedBytes: 0,
      totalBytes: null,
      percent: null,
      assetName: entry.name,
    });
    const activation = await this.runtime.activateRuntime(descriptor, {
      session: targetSession,
      config: prepared.runtimeConfig,
      signal: options.signal,
      commit: async (report) => {
        const runtime = toRuntimeObservation(
          report.runtimeObservability,
          this.runtime.getTransportObservability()
        );
        const profile = toBackendProfileObservation(
          report.backendObservability,
          resolvedBackend.webgpuAdapter
        );
        const committed = await resolvedRust.commitLoad({
          loadId: prepared.loadId,
          modelId: prepared.model.id,
          runtimeFingerprint: prepared.runtimeFingerprint,
          runtime,
          profile,
        });
        await this.replaceManifest(committed.manifest);
        return committed;
      },
    });
    const committed = activation.committed;
    loadOptions.onProgress?.({
      phase: 'load',
      loadedBytes: 1,
      totalBytes: 1,
      percent: 100,
      assetName: committed.model.name,
    });
    this.ingestRustEvents(committed.events);
    return modelInfoFromCatalog(committed.model, activation.session);
  }

  private async acquireRemote(
    rust: RustLifecycleBridge,
    host: RemoteAcquisitionHost,
    source: Extract<ModelAddSource, { kind: 'remote' }>,
    setAcquisitionId: (acquisitionId: string | null) => void
  ): Promise<RustLifecycleInstallValue> {
    let response = await rust.remoteAcquisition({
      command: 'begin',
      urls: source.urls,
    });
    while (response.kind === 'action') {
      setAcquisitionId(response.action.acquisitionId);
      const result = await host.execute(response.action);
      response = await rust.remoteAcquisition({
        command: 'advance',
        event: result.event,
        ...(result.assets == null ? {} : { assets: result.assets }),
        ...(result.classified == null ? {} : { classified: result.classified }),
      });
    }
    setAcquisitionId(null);
    if (response.kind === 'cancelled') {
      throw new QueryError('REMOTE_LOAD_FAILED', 'Remote acquisition was cancelled.');
    }
    if (response.kind === 'failed') {
      throw queryErrorFromLifecycleError(response.error, 'Remote acquisition failed.');
    }
    return response.installed;
  }

  private async cancelRemoteAcquisition(
    rust: RustLifecycleBridge,
    host: RemoteAcquisitionHost,
    acquisitionId: string
  ): Promise<void> {
    let response: RustRemoteCommandValue = await rust.remoteAcquisition({
      command: 'cancel',
      acquisitionId,
    });
    while (response.kind === 'action') {
      const result = await host.execute(response.action);
      response = await rust.remoteAcquisition({
        command: 'advance',
        event: result.event,
      });
    }
  }

  private async buildRustInstallSource(
    source: Extract<ModelAddSource, { kind: 'local' }>,
    manifest: RegistryManifest,
    options: ModelAddOptions
  ): Promise<RustLifecycleInstallSource> {
    const installed = await this.installLocalSource(source, manifest, options);
    const classified = await this.classifyAssets(installed, options.signal);
    return {
      assets: installed.map((asset) => asset.record),
      classified: classified.map((file) => ({
        assetId: file.assetId,
        name: file.name,
        inspection: file.inspection,
      })),
    };
  }

  private async getRustLifecycle(
    manifest: RegistryManifest
  ): Promise<RustLifecycleBridge> {
    if (this.rustLifecyclePromise == null) {
      this.rustLifecyclePromise = this.runtime.createRustLifecycleBridge(manifest);
    }
    return await this.rustLifecyclePromise;
  }

  private async replaceManifest(manifest: RegistryManifest): Promise<void> {
    await this.registry.write((draft) => {
      draft.version = manifest.version;
      draft.projectorIndexRevision = manifest.projectorIndexRevision;
      draft.assets = JSON.parse(JSON.stringify(manifest.assets)) as RegistryManifest['assets'];
      draft.models = JSON.parse(JSON.stringify(manifest.models)) as RegistryManifest['models'];
    });
  }

  private ingestRustEvents(events: readonly CatalogObservabilityEvent[]): void {
    for (const event of events) {
      const runtimeEvent = this.runtimeEvent(event);
      this.observability.ingest(runtimeEvent);
      this.emitEngineEvent(observabilityEventToStateEvent(runtimeEvent));
    }
  }

  private runtimeEvent(event: CatalogObservabilityEvent): ObservabilityEvent {
    const model = event.snapshot.model;
    const session = this.runtime.currentRuntimeSession();
    return {
      ...event,
      snapshot: {
        ...event.snapshot,
        model: model == null ? null : modelInfoFromCatalog(model, session),
      },
    };
  }

  private emitEngineEvent(event: EngineEvent): void {
    for (const listener of this.engineEventListeners) {
      listener(event);
    }
  }

  private async closeRustLifecycle(): Promise<void> {
    if (this.rustLifecyclePromise == null) {
      return;
    }
    const rust = await this.rustLifecyclePromise;
    await rust.close();
    this.rustLifecyclePromise = null;
  }

  private recordQuerySuccess(
    contextKey: string,
    start: number,
    response: GenerateResponse,
    transport: TransportObservability
  ): void {
    const metrics = response.observability ?? null;
    const runtime = toRuntimeObservation(
      metrics ?? this.runtime.getRuntimeObservability(),
      transport
    );
    this.observability.emit('query-complete', {
      state: 'ready',
      query: this.toQueryObservation(contextKey, 'success', start, response),
      ...(runtime == null ? {} : { runtime }),
    });
  }

  private recordQueryFailure(
    contextKey: string,
    start: number,
    error: unknown,
    response?: GenerateResponse,
    transport: TransportObservability = this.runtime.getTransportObservability()
  ): void {
    const metrics = response?.observability ?? null;
    const runtime = toRuntimeObservation(
      metrics ?? this.runtime.getRuntimeObservability(),
      transport
    );
    this.observability.emit('error', {
      state: 'error',
      query: {
        ...this.toQueryObservation(
          contextKey,
          isAbortError(error) || response?.cancelled === true ? 'cancelled' : 'failed',
          start,
          response
        ),
        errorCode: error instanceof QueryError ? error.code : undefined,
        errorMessage: error instanceof Error ? error.message : String(error),
      },
      ...(runtime == null ? {} : { runtime }),
    });
  }

  private requestTransportObservability(
    start: TransportObservability,
    emitsTokens: boolean
  ): TransportObservability {
    const current = this.runtime.getTransportObservability();
    const transport: TransportObservability = {
      ...current,
      wasmRunLoopCalls: current.wasmRunLoopCalls - start.wasmRunLoopCalls,
      wasmRunLoopMs: current.wasmRunLoopMs - start.wasmRunLoopMs,
      activeTokenEmission: emitsTokens,
      activeTokenTransport: emitsTokens ? 'token-stream' : 'none',
    };
    if (!emitsTokens) {
      delete transport.tokenDrainCalls;
      delete transport.tokenDrainMs;
      return transport;
    }
    // The runtime accumulates ring-drain cost across every request it serves,
    // so report the delta observed while this request was in flight. Concurrent
    // requests each attribute the whole overlapping window to themselves.
    transport.tokenDrainCalls =
      (current.tokenDrainCalls ?? 0) - (start.tokenDrainCalls ?? 0);
    transport.tokenDrainMs = (current.tokenDrainMs ?? 0) - (start.tokenDrainMs ?? 0);
    return transport;
  }

  private toQueryObservation(
    contextKey: string,
    status: QueryObservation['status'],
    start: number,
    response?: GenerateResponse
  ): QueryObservation {
    const metrics = response?.observability ?? null;
    return {
      contextKey,
      status,
      wallMs: Math.max(0, nowMs() - start),
      ttftMs: metrics?.ttftMs ?? null,
      outputTokens: metrics?.outputTokens ?? null,
    };
  }

  private async installLocalSource(
    source: Extract<ModelAddSource, { kind: 'local' }>,
    manifest: RegistryManifest,
    options: ModelAddOptions
  ): Promise<InstalledAsset[]> {
    if (source.files.length === 0 || !source.files.every(isFile)) {
      throw new QueryError(
        'INVALID_MODEL_SOURCE',
        'Local model source requires at least one File.'
      );
    }
    return source.files.length === 1
      ? await this.installLocalModelAssets(source.files[0], manifest, options)
      : await Promise.all(
        source.files.map((file) => this.installLocalAsset(file, 'shard', manifest, options))
      );
  }

  private async installLocalModelAssets(
    file: File,
    manifest: RegistryManifest,
    options: ModelAddOptions
  ): Promise<InstalledAsset[]> {
    const existingSplit = this.findLocalSplitAssets(manifest, file);
    if (existingSplit != null) {
      const assets: InstalledAsset[] = [];
      for (const record of existingSplit) {
        assets.push({
          record,
          file: await this.assetStore.getFile(record),
        });
      }
      return assets;
    }

    const records = await this.assetStore.installLocalGguf(
      file,
      this.runtime,
      manifest,
      options.signal,
      options.onProgress
    );
    return await this.installedAssetsFromRecords(records, manifest);
  }

  private async installedAssetsFromRecords(
    records: readonly AssetRecord[],
    manifest: RegistryManifest
  ): Promise<InstalledAsset[]> {
    const assets: InstalledAsset[] = [];
    for (const record of records) {
      const existing = manifest.assets[record.id];
      if (existing != null && existing.storagePath !== record.storagePath) {
        await this.assetStore.delete(record);
      }
      const effective = existing ?? record;
      assets.push({
        record: effective,
        file: await this.assetStore.getFile(effective),
      });
    }
    return assets;
  }

  private async installLocalAsset(
    file: File,
    kind: AssetRecord['kind'],
    manifest: RegistryManifest,
    options: ModelAddOptions
  ): Promise<InstalledAsset> {
    const record = await this.assetStore.installFile({
      kind,
      file,
      signal: options.signal,
      onProgress: options.onProgress,
    });
    const existing = manifest.assets[record.id];
    if (existing != null && existing.storagePath !== record.storagePath) {
      await this.assetStore.delete(record);
    }
    return {
      record: existing ?? record,
      file: await this.assetStore.getFile(existing ?? record),
    };
  }

  private findLocalSplitAssets(
    manifest: RegistryManifest,
    file: File
  ): AssetRecord[] | null {
    const sourceFileName = normalizeLocalSourceFileName(file);
    return this.findCompleteSplitAssets(
      Object.values(manifest.assets).filter(
        (asset) =>
          asset.kind === 'shard' &&
          asset.sourceUrl == null &&
          asset.sourceFileName === sourceFileName &&
          asset.sourceFileLastModified === file.lastModified &&
          asset.sourceBytes === file.size &&
          Number.isInteger(asset.sourcePartIndex) &&
          Number.isInteger(asset.sourcePartCount)
      )
    );
  }

  private findCompleteSplitAssets(candidates: AssetRecord[]): AssetRecord[] | null {
    candidates.sort((left, right) => (left.sourcePartIndex ?? 0) - (right.sourcePartIndex ?? 0));
    if (candidates.length === 0) {
      return null;
    }

    const first = candidates[0];
    const count = first?.sourcePartCount;
    if (typeof count !== 'number' || !Number.isInteger(count) || count <= 0 || candidates.length !== count) {
      return null;
    }
    for (let index = 0; index < candidates.length; index += 1) {
      const candidate = candidates[index];
      if (candidate.sourcePartCount !== count || candidate.sourcePartIndex !== index) {
        return null;
      }
    }
    return candidates;
  }

  private async classifyAssets(
    assets: InstalledAsset[],
    signal?: AbortSignal
  ): Promise<ClassifiedAssetFile[]> {
    return Promise.all(
      assets.map(async (asset) => {
        if (asset.record.inspection != null) {
          return {
            assetId: asset.record.id,
            file: asset.file,
            inspection: asset.record.inspection,
            name: asset.file.name,
          };
        }
        return await this.assetClassifier.classify(asset.record.id, asset.file, signal);
      })
    );
  }

  private async openBundleForEntry(
    entry: ModelEntry,
    manifest: RegistryManifest,
    signal?: AbortSignal
  ): Promise<RuntimeBundleDescriptor> {
    for (let attempt = 0; ; attempt += 1) {
      try {
        return await this.openBundleOnce(entry, manifest);
      } catch (error) {
        const delayMs = OPFS_LOCK_RETRY_DELAYS_MS[attempt];
        if (delayMs == null || !isOpfsExclusiveLockError(error)) {
          throw error;
        }
        if (signal?.aborted) {
          throw createAbortError('Model load aborted while waiting for an OPFS lock.');
        }
        await this.sleep(delayMs);
        if (signal?.aborted) {
          throw createAbortError('Model load aborted while waiting for an OPFS lock.');
        }
      }
    }
  }

  private async openBundleOnce(
    entry: ModelEntry,
    manifest: RegistryManifest
  ): Promise<RuntimeBundleDescriptor> {
    const modelFiles: RuntimeBundleFile[] = [];
    let projector: RuntimeBundleFile | undefined;
    try {
      for (const assetId of entry.modelAssetIds) {
        modelFiles.push(await this.openEntryAsset(entry, manifest, assetId, 'asset'));
      }

      if (entry.projectorAssetId != null) {
        projector = await this.openEntryAsset(
          entry,
          manifest,
          entry.projectorAssetId,
          'projector'
        );
      }

      return { modelFiles, projector };
    } catch (error) {
      const openedProjector = projector;
      try {
        releaseAll('Failed to close a partially opened runtime model bundle.', [
          ...modelFiles.map((file) => ({
            label: `close model handle "${file.name}"`,
            release: () => file.handle.close(),
          })),
          ...(openedProjector == null
            ? []
            : [{
              label: `close projector handle "${openedProjector.name}"`,
              release: () => openedProjector.handle.close(),
            }]),
        ]);
      } catch (cleanupError) {
        throw attachCleanupFailures(error, cleanupError);
      }
      throw error;
    }
  }

  /**
   * Opens one of an entry's assets, marking the model broken when the asset is
   * missing from the manifest or the store reports it unusable.
   */
  private async openEntryAsset(
    entry: ModelEntry,
    manifest: RegistryManifest,
    assetId: string,
    role: 'asset' | 'projector'
  ): Promise<RuntimeBundleFile> {
    const asset = manifest.assets[assetId];
    if (asset == null) {
      await this.markBroken(entry.id);
      throw new QueryError(
        'MODEL_BROKEN',
        `Model "${entry.id}" references a missing ${role}.`
      );
    }
    try {
      return await this.assetStore.openSyncHandle(asset);
    } catch (error) {
      if (error instanceof QueryError && error.code === 'MODEL_BROKEN') {
        await this.markBroken(entry.id);
      }
      throw error;
    }
  }

  private async markBroken(id: string): Promise<void> {
    await this.registry.write((draft) => {
      const entry = draft.models[id];
      if (entry != null) {
        entry.status = 'broken';
        entry.updatedAt = new Date().toISOString();
      }
    });
  }

  /**
   * Probes chat-template boundaries once per runtime session. A Worker hosts
   * exactly one session, so there is nothing to invalidate against; a failed
   * probe clears the cache so the next chat retries it.
   */
  private getChatBoundaryMarkers(): Promise<readonly string[]> {
    this.chatBoundaryMarkersPromise ??= this.runtime
      .probeChatTemplateBoundaryInfo()
      .then(buildBoundaryMarkers)
      .catch((error) => {
        this.chatBoundaryMarkersPromise = null;
        throw error;
      });
    return this.chatBoundaryMarkersPromise;
  }
}

function isOpfsExclusiveLockError(error: unknown): boolean {
  if (typeof DOMException !== 'function' || !(error instanceof DOMException)) {
    return false;
  }
  const withCleanup = error as DOMException & { readonly cleanupFailures?: unknown };
  return withCleanup.cleanupFailures == null && error.name === 'NoModificationAllowedError';
}

function samplingRuntimeOverride(
  options: RuntimeRequestOptions
): PromptOptions['sampling'] {
  const overrideConfig: NonNullable<PromptOptions['sampling']> = {
    ...(options.sampling ?? {}),
  };
  mergeSamplingOverrideField(overrideConfig, 'temperature', options.temperature);
  mergeSamplingOverrideField(overrideConfig, 'top_p', options.topP);
  return hasSamplingRuntimeOverrideFields(overrideConfig) ? overrideConfig : undefined;
}

function mergeSamplingOverrideField(
  overrideConfig: SamplingRuntimeOverride,
  field: keyof Pick<SamplingRuntimeOverride, 'temperature' | 'top_p'>,
  value: number | undefined
): void {
  if (value == null) {
    return;
  }
  if (overrideConfig[field] != null && overrideConfig[field] !== value) {
    throw new QueryError(
      'QUERY_FAILED',
      `${field} conflicts with sampling.${field}`
    );
  }
  overrideConfig[field] = value;
}

function isChatInputObject(input: ChatInput): input is Extract<ChatInput, { messages: unknown }> {
  return !Array.isArray(input);
}

function modelInfoFromCatalog(
  model: CatalogModelInfo,
  session: RuntimeSessionSnapshot | null
): ModelInfo {
  if (
    session == null ||
    session.model.id !== model.id ||
    session.model.assetFingerprint !== model.assetFingerprint
  ) {
    return {
      ...model,
      loaded: false,
      chatTemplate: null,
      bosText: '',
      eosText: '',
      mediaMarker: null,
      capabilities: null,
    };
  }
  return {
    ...model,
    loaded: true,
    chatTemplate: session.chatTemplate,
    bosText: session.bosText,
    eosText: session.eosText,
    mediaMarker: session.mediaMarker,
    capabilities: session.capabilities,
  };
}
