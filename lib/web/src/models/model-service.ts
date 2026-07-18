import type { EngineRuntime } from '../runtime/engine-runtime.js';
import {
  buildBoundaryMarkers,
  sliceUndeliveredSuffix,
  TokenBoundaryTextSanitizer,
} from '../engine/chat-boundary-sanitizer.js';
import type {
  GenerateRequestId,
  GenerateResponse,
  NativeRuntimeConfig,
  PromptOptions,
  SamplingRuntimeOverride,
  TransportObservability,
} from '../engine/inference-types.js';
import { hasSamplingRuntimeOverrideFields } from '../engine/inference-types.js';
import { createLinkedAbortController, isAbortError } from '../utils/abort.js';
import { AssetStore } from './asset-store.js';
import { RemoteAcquisitionHost } from './remote-acquisition-host.js';
import { ModelRegistryStore } from './model-registry-store.js';
import type {
  RustLifecycleBridge,
  RustLifecycleInstallSource,
  RustLifecycleInstallValue,
  RustLifecyclePrepareLoadValue,
  RustRemoteCommandValue,
} from '../wasm/wasm-bridge.js';
import { queryErrorFromLifecycleError } from '../wasm/wasm-bridge.js';
import {
  QueryError,
  type AssetRecord,
  type BrowserBackendPreference,
  type ChatInput,
  type ClassifiedAsset,
  type ClassifiedAssetFile,
  type EmbedOptions,
  type EmbeddingResult,
  type EngineEvent,
  type EngineState,
  type InternalBundleDescriptor,
  type LoadedModelState,
  type ModelBundleFileProjectorDescriptor,
  type ModelBundleShard,
  type ModelEntry,
  type ModelDetectionResult,
  type ModelInfo,
  type ModelInstallOptions,
  type ModelInstallSource,
  type ModelLifecycleService,
  type ModelLoadOptions,
  type ObservabilityEvent,
  type ObservabilitySnapshot,
  type QueryObservation,
  type QueryInput,
  type QueryOptions,
  type GenerationResult,
  type InternalTextRequestOptions,
  type TokenBatch,
  type RegistryManifest,
  type WebGpuAdapterInfo,
} from './types.js';
import {
  embeddingResultFromGenerateResponse,
  generationResultFromGenerateResponse,
  generationResultFromText,
  ObservabilityController,
  observabilityEventToStateEvent,
  observabilitySnapshotToEngineState,
  toBackendProfileObservation,
  toRuntimeObservation,
} from './observability-controller.js';
import type { RuntimeBackendOverride, WasmThreadingMode } from '../engine/runtime-assets.js';

interface InstalledAsset {
  record: AssetRecord;
  file: File;
}

interface SourceInstallResult {
  assets: InstalledAsset[];
  explicitProjectorAssetId: string | null;
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
  defaultBackendOverride: RuntimeBackendOverride | null
): Promise<ResolvedBrowserBackend> {
  const requestedBackend = backend === 'auto' ? undefined : backend;
  if (requestedBackend === 'cpu') {
    return { backend: requestedBackend, webgpuAdapter: null };
  }
  if (requestedBackend == null && defaultBackendOverride === 'cpu') {
    return { backend: 'cpu', webgpuAdapter: null };
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
      drainMs: 0,
      drainCalls: 0,
    },
  };
}

function utf8ByteLength(text: string): number {
  return textEncoder.encode(text).byteLength;
}

function entryAssetFingerprint(entry: Pick<ModelEntry, 'modelAssetIds' | 'projectorAssetId'>): string {
  return JSON.stringify({
    modelAssetIds: [...entry.modelAssetIds].sort((left, right) => left.localeCompare(right)),
    projectorAssetId: entry.projectorAssetId ?? null,
  });
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

export class ModelService implements ModelLifecycleService {
  private currentLoaded: LoadedModelState | null = null;
  private chatBoundaryMarkersPromise: Promise<readonly string[]> | null = null;
  private chatBoundaryMarkersKey: string | null = null;
  private operationChain: Promise<void> = Promise.resolve();
  private transitioning = false;
  private readonly observability = new ObservabilityController();
  private readonly engineEventListeners = new Set<(event: EngineEvent) => void>();
  private rustLifecyclePromise: Promise<RustLifecycleBridge> | null = null;

  constructor(
    private readonly runtime: EngineRuntime,
    private readonly registry = new ModelRegistryStore(),
    private readonly assetStore = new AssetStore(),
    assetClassifier?: AssetClassifier
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
    const current = this.currentLoaded;
    if (current == null) {
      return null;
    }
    return this.currentSnapshot ?? null;
  }

  private currentSnapshot: ModelInfo | null = null;

  public async list(): Promise<ModelInfo[]> {
    const manifest = await this.registry.read();
    const rust = await this.getRustLifecycle(manifest);
    return rust.list();
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

  public async install(
    source: ModelInstallSource,
    options: ModelInstallOptions = {}
  ): Promise<ModelInfo> {
    return this.withLifecycleLock(async () => {
      if (options.signal?.aborted) {
        throw new DOMException('Model install aborted.', 'AbortError');
      }
      return await this.installWithRustLifecycle(source, options);
    });
  }

  public async load(modelId: string, options: ModelLoadOptions = {}): Promise<ModelInfo> {
    return this.withLifecycleLock(async () => {
      if (options.signal?.aborted) {
        throw new DOMException('Model load aborted.', 'AbortError');
      }
      return await this.loadWithRustLifecycle(modelId, options);
    });
  }

  public async remove(id: string): Promise<void> {
    await this.withLifecycleLock(async () => {
      const manifest = await this.registry.read();
      const rust = await this.getRustLifecycle(manifest);
      const removed = rust.remove(id);
      await this.replaceManifest(removed.manifest);
      for (const asset of removed.orphanedAssets) {
        await this.assetStore.delete(asset);
      }
      this.ingestRustEvents(removed.events);
    });
  }

  public async unload(): Promise<void> {
    await this.withLifecycleLock(async () => {
      const rust = await this.getRustLifecycle(await this.registry.read());
      if (this.currentLoaded != null) {
        this.runtime.close();
        this.currentLoaded = null;
        this.currentSnapshot = null;
      }
      const snapshot = rust.unload();
      this.ingestRustEvents(rust.drainEvents());
      this.observability.ingest({ type: 'load-complete', snapshot });
      this.emitEngineEvent({ type: 'state', state: this.state() });
    });
  }

  public async runQuery(
    input: QueryInput,
    options: InternalTextRequestOptions
  ): Promise<GenerationResult> {
    if (this.transitioning) {
      throw new QueryError('MODEL_NOT_READY', 'A model lifecycle transition is in progress.');
    }
    if (this.currentLoaded == null) {
      throw new QueryError('MODEL_NOT_READY', 'No model is loaded. Call client.add(...) first.');
    }
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
    if (this.transitioning) {
      throw new QueryError('MODEL_NOT_READY', 'A model lifecycle transition is in progress.');
    }
    if (this.currentLoaded == null) {
      throw new QueryError('MODEL_NOT_READY', 'No model is loaded. Call client.add(...) first.');
    }

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

  private async runRuntimeRequest(
    options: RuntimeRequestOptions,
    media: Uint8Array[] | undefined,
    enqueue: (contextKey: string, promptOptions: PromptOptions) => Promise<GenerateRequestId>,
    operationLabel = 'Model query'
  ): Promise<GenerateResponse> {
    let tokenDrainMs = 0;
    let tokenDrainCalls = 0;
    const deliverTokenBatch = (batch: TokenBatch): void => {
      if (batch.text.length === 0) {
        return;
      }
      tokenDrainMs = batch.stats.drainMs;
      tokenDrainCalls = batch.stats.drainCalls;
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
    let requestId = 0;
    let failureRecorded = false;
    try {
      requestId = await enqueue(contextKey, promptOptions);
      this.emitEngineEvent({ type: 'request-started', requestId: String(requestId), streamId: requestId });
      const response = await this.runtime.awaitQuery(requestId, { signal: options.signal });
      if (response.cancelled) {
        const error = new DOMException(response.errorMessage ?? 'Queued request cancelled.', 'AbortError');
        this.recordQueryFailure(
          contextKey,
          start,
          error,
          response,
          this.requestTransportObservability(emitsTokens, tokenDrainMs, tokenDrainCalls)
        );
        this.emitEngineEvent({
          type: 'request-failed',
          requestId: String(requestId),
          error: error.message,
        });
        failureRecorded = true;
        throw error;
      }
      if (response.failed) {
        const error = new Error(response.errorMessage ?? 'Queued prompt failed.');
        this.recordQueryFailure(
          contextKey,
          start,
          error,
          response,
          this.requestTransportObservability(emitsTokens, tokenDrainMs, tokenDrainCalls)
        );
        this.emitEngineEvent({
          type: 'request-failed',
          requestId: String(requestId),
          error: error.message,
        });
        failureRecorded = true;
        throw error;
      }
      this.recordQuerySuccess(
        contextKey,
        start,
        response,
        this.requestTransportObservability(emitsTokens, tokenDrainMs, tokenDrainCalls)
      );
      this.emitEngineEvent({
        type: 'request-completed',
        requestId: String(requestId),
      });
      return response;
    } catch (error) {
      if (!failureRecorded) {
        this.recordQueryFailure(
          contextKey,
          start,
          error,
          undefined,
          this.requestTransportObservability(emitsTokens, tokenDrainMs, tokenDrainCalls)
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
      if (!failureRecorded && requestId !== 0) {
        this.emitEngineEvent({
          type: 'request-failed',
          requestId: String(requestId),
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
    if (this.transitioning) {
      throw new QueryError('MODEL_NOT_READY', 'A model lifecycle transition is in progress.');
    }
    if (this.currentLoaded == null) {
      throw new QueryError('MODEL_NOT_READY', 'No model is loaded. Call client.add(...) first.');
    }

    const current = this.currentLoaded;
    const messages = isChatInputObject(input) ? input.messages : input;
    const media = isChatInputObject(input) ? input.media : undefined;
    if (media != null && media.length > 0 && this.runtime.readMediaMarker() == null) {
      throw new QueryError('MODEL_NOT_READY', 'The loaded model does not accept media input.');
    }
    const boundaryMarkers = await this.getChatBoundaryMarkers(current);
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

  public close(): void {
    this.runtime.close();
    this.currentLoaded = null;
    this.currentSnapshot = null;
    void this.closeRustLifecycle();
    this.observability.markClosed();
  }

  private async installWithRustLifecycle(
    source: ModelInstallSource,
    options: ModelInstallOptions
  ): Promise<ModelInfo> {
    const installOptions: ModelInstallOptions = {
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
        installOptions
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
        : rust.install(await this.buildRustInstallSource(source, manifest, installOptions));
      remoteAcquisitionId = null;
      await this.replaceManifest(installed.manifest);
      if (remoteHost != null) {
        remoteManifestCommitted = true;
        await remoteHost.commitJournal();
      }
      this.ingestRustEvents(installed.events);
      return installed.model;
    } catch (error) {
      if (remoteAcquisitionId != null && rust != null && remoteHost != null) {
        await this.cancelRemoteAcquisition(rust, remoteHost, remoteAcquisitionId);
      }
      if (remoteHost != null && !remoteManifestCommitted) {
        await remoteHost.cleanupUncommittedJournal(manifest);
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
      this.runtime.getDefaultBackendOverride()
    );
    let prepared: RustLifecyclePrepareLoadValue | null = null;
    let rust: RustLifecycleBridge | null = null;
    try {
      const [resolvedRust, resolvedBackend] = await Promise.all([rustPromise, backendPromise]);
      rust = resolvedRust;
      const runtimeConfig = applyBrowserRuntimeDefaults(
        options.runtime,
        wasmThreading
      );
      const rustOptions = {
        backend: resolvedBackend.backend,
        runtime: runtimeConfig,
        observability: observabilityMode,
      } as const;
      prepared = rust.prepareLoad({ modelId }, rustOptions);
      await this.replaceManifest(prepared.manifest);
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

      if (prepared.loadRequired) {
        const descriptor = await this.openBundleForEntry(entry, prepared.manifest);
        loadOptions.onProgress?.({
          phase: 'load',
          loadedBytes: 0,
          totalBytes: null,
          percent: null,
          assetName: entry.name,
        });
        const staged = await this.runtime.stageModelBundle(descriptor, {
          signal: options.signal,
        });
        await this.runtime.loadRuntimeModel(staged, prepared.runtimeConfig);
      }

      const runtime = toRuntimeObservation(
        this.runtime.getRuntimeObservability(),
        this.runtime.getTransportObservability()
      );
      // Backend identity (registry, devices, adapter) is cheap read-only data
      // and stays available in every observability mode; only the native
      // profiling instrumentation remains gated on 'profile'.
      const profile = toBackendProfileObservation(
        await this.runtime.getBackendObservability(),
        resolvedBackend.webgpuAdapter
      );
      const committed = rust.commitLoad({
        loadId: prepared.loadId,
        modelId: prepared.model.id,
        runtimeFingerprint: prepared.runtimeFingerprint,
        chatTemplate: this.runtime.getChatTemplate(),
        bosText: this.runtime.getBosText(),
        eosText: this.runtime.getEosText(),
        mediaMarker: this.runtime.readMediaMarker(),
        runtime,
        profile,
      });
      await this.replaceManifest(committed.manifest);
      const loadedEntry = committed.manifest.models[committed.model.id] ?? entry;
      this.currentLoaded = {
        id: committed.model.id,
        assetFingerprint: entryAssetFingerprint(loadedEntry),
        runtimeFingerprint: prepared.runtimeFingerprint,
      };
      this.currentSnapshot = committed.model;
      loadOptions.onProgress?.({
        phase: 'load',
        loadedBytes: 1,
        totalBytes: 1,
        percent: 100,
        assetName: committed.model.name,
      });
      this.ingestRustEvents(committed.events);
      return committed.model;
    } catch (error) {
      if (rust == null) {
        rust = await rustPromise.catch(() => null);
      }
      if (prepared != null && rust != null) {
        const snapshot = rust.abortLoad({
          message: error instanceof Error ? error.message : String(error),
        });
        this.observability.ingest({ type: 'error', snapshot });
        this.ingestRustEvents(rust.drainEvents());
      }
      throw error;
    }
  }

  private async acquireRemote(
    rust: RustLifecycleBridge,
    host: RemoteAcquisitionHost,
    source: Extract<ModelInstallSource, { kind: 'remote' }>,
    setAcquisitionId: (acquisitionId: string | null) => void
  ): Promise<RustLifecycleInstallValue> {
    let response = rust.remoteAcquisition({
      command: 'begin',
      modelUrls: source.modelUrls,
      ...(source.projectorUrl == null ? {} : { projectorUrl: source.projectorUrl }),
    });
    while (response.kind === 'action') {
      setAcquisitionId(response.action.acquisitionId);
      const result = await host.execute(response.action);
      response = rust.remoteAcquisition({
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
    let response: RustRemoteCommandValue = rust.remoteAcquisition({
      command: 'cancel',
      acquisitionId,
    });
    while (response.kind === 'action') {
      const result = await host.execute(response.action);
      response = rust.remoteAcquisition({
        command: 'advance',
        event: result.event,
      });
    }
  }

  private async buildRustInstallSource(
    source: Extract<ModelInstallSource, { kind: 'local' }>,
    manifest: RegistryManifest,
    options: ModelInstallOptions
  ): Promise<RustLifecycleInstallSource> {
    const installed = await this.installLocalSource(source, manifest, options);
    const classified = await this.classifyAssets(installed.assets, options.signal);
    const sourceProjectorAssetId = this.resolveSourceProjectorAssetId(
      classified,
      installed.explicitProjectorAssetId
    );
    return {
      assets: installed.assets.map((asset) => asset.record),
      classified: classified.map((file) => ({
        assetId: file.assetId,
        name: file.name,
        inspection: file.inspection,
      })),
      explicitProjectorAssetId: sourceProjectorAssetId,
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

  private ingestRustEvents(events: readonly ObservabilityEvent[]): void {
    for (const event of events) {
      this.observability.ingest(event);
      this.emitEngineEvent(observabilityEventToStateEvent(event));
    }
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
    rust.close();
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
    emitsTokens: boolean,
    tokenDrainMs = 0,
    tokenDrainCalls = 0
  ): TransportObservability {
    const current = this.runtime.getTransportObservability();
    const transport: TransportObservability = {
      ...current,
      activeTokenEmission: emitsTokens,
      activeTokenTransport: emitsTokens ? 'token-stream' : 'none',
    };
    if (!emitsTokens) {
      delete transport.tokenDrainCalls;
      delete transport.tokenDrainMs;
      return transport;
    }
    transport.tokenDrainCalls = tokenDrainCalls;
    transport.tokenDrainMs = tokenDrainMs;
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
    source: Extract<ModelInstallSource, { kind: 'local' }>,
    manifest: RegistryManifest,
    options: ModelInstallOptions
  ): Promise<SourceInstallResult> {
    if (source.modelFiles.length === 0 || !source.modelFiles.every(isFile)) {
      throw new QueryError(
        'INVALID_MODEL_SOURCE',
        'Local model source requires at least one File.'
      );
    }
    const base = source.modelFiles.length === 1
      ? await this.installLocalModelAssets(source.modelFiles[0], manifest, options)
      : await Promise.all(
        source.modelFiles.map((file) => this.installLocalAsset(file, 'shard', manifest, options))
      );
    const projector = source.projectorFile == null
      ? null
      : await this.installLocalAsset(source.projectorFile, 'projector', manifest, options);
    return {
      assets: [...base, ...(projector == null ? [] : [projector])],
      explicitProjectorAssetId: projector?.record.id ?? null,
    };
  }

  private async installLocalModelAssets(
    file: File,
    manifest: RegistryManifest,
    options: ModelInstallOptions
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
    options: ModelInstallOptions
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
        if (asset.record.inspection?.version === 1) {
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

  private resolveSourceProjectorAssetId(
    classified: readonly ClassifiedAssetFile[],
    explicitProjectorAssetId: string | null
  ): string | null {
    if (explicitProjectorAssetId != null) {
      return explicitProjectorAssetId;
    }
    const projectors = classified.filter((file) => file.inspection.role === 'projector');
    return projectors.length === 1 ? projectors[0].assetId : null;
  }

  private async getAssetFileForEntry(entry: ModelEntry, asset: AssetRecord): Promise<File> {
    try {
      return await this.assetStore.getFile(asset);
    } catch (error) {
      if (error instanceof QueryError && error.code === 'MODEL_BROKEN') {
        await this.markBroken(entry.id);
      }
      throw error;
    }
  }

  private async openBundleForEntry(
    entry: ModelEntry,
    manifest: RegistryManifest
  ): Promise<InternalBundleDescriptor> {
    const detection = this.detectionForEntry(entry, manifest);
    if (detection == null) {
      await this.markBroken(entry.id);
      throw new QueryError(
        'MODEL_BROKEN',
        `Installed model "${entry.id}" is missing detection metadata; reinstall the model.`
      );
    }

    const shards: ModelBundleShard[] = [];
    try {
      for (const assetId of entry.modelAssetIds) {
        const asset = manifest.assets[assetId];
        if (asset == null) {
          await this.markBroken(entry.id);
          throw new QueryError(
            'MODEL_BROKEN',
            `Installed model "${entry.id}" references a missing asset.`
          );
        }
        try {
          shards.push(await this.assetStore.openSyncHandle(asset));
        } catch (error) {
          if (error instanceof QueryError && error.code === 'MODEL_BROKEN') {
            await this.markBroken(entry.id);
          }
          throw error;
        }
      }

      let projector: ModelBundleFileProjectorDescriptor | undefined;
      if (entry.projectorAssetId != null) {
        const projectorAsset = manifest.assets[entry.projectorAssetId];
        if (projectorAsset == null) {
          await this.markBroken(entry.id);
          throw new QueryError(
            'MODEL_BROKEN',
            `Installed model "${entry.id}" references a missing projector.`
          );
        }
        try {
          projector = { file: await this.assetStore.getFile(projectorAsset) };
        } catch (error) {
          if (error instanceof QueryError && error.code === 'MODEL_BROKEN') {
            await this.markBroken(entry.id);
          }
          throw error;
        }
      }

      return { shards, projector, detection };
    } catch (error) {
      for (const shard of shards) {
        try {
          shard.handle.close();
        } catch {}
      }
      throw error;
    }
  }

  private detectionForEntry(
    entry: ModelEntry,
    manifest: RegistryManifest
  ): ModelDetectionResult | undefined {
    for (const assetId of entry.modelAssetIds) {
      const inspection = manifest.assets[assetId]?.inspection;
      if (inspection != null) {
        return {
          inspection,
          detectionMethod: inspection.role === 'unknown' ? 'none' : 'gguf-metadata',
          modelName: entry.name,
          modelType: null,
          modelArchitecture: inspection.architecture,
        };
      }
    }
    return undefined;
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

  private async withLifecycleLock<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.operationChain;
    let release!: () => void;
    this.operationChain = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    this.transitioning = true;
    try {
      return await operation();
    } finally {
      this.transitioning = false;
      release();
    }
  }

  private getChatBoundaryMarkers(current: LoadedModelState): Promise<readonly string[]> {
    const key = `${current.id}:${current.assetFingerprint}`;
    if (this.chatBoundaryMarkersPromise == null || this.chatBoundaryMarkersKey !== key) {
      this.chatBoundaryMarkersKey = key;
      this.chatBoundaryMarkersPromise = this.runtime.probeChatTemplateBoundaryInfo()
        .then(buildBoundaryMarkers)
        .catch((error) => {
          this.chatBoundaryMarkersPromise = null;
          this.chatBoundaryMarkersKey = null;
          throw error;
        });
    }
    return this.chatBoundaryMarkersPromise;
  }
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
