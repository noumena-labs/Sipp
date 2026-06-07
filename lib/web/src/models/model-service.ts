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
  TransportObservability,
} from '../engine/inference-types.js';
import { createLinkedAbortController, isAbortError } from '../utils/abort.js';
import { AssetStore, type RemoteAssetMetadata } from './asset-store.js';
import { ModelRegistryStore } from './model-registry-store.js';
import type {
  RustLifecycleBridge,
  RustLifecycleLoadSource,
  RustLifecyclePrepareLoadValue,
} from '../wasm/wasm-bridge.js';
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
  type ModelLifecycleService,
  type ModelLoadOptions,
  type ModelSource,
  type ObservabilityEvent,
  type ObservabilitySnapshot,
  type QueryObservation,
  type QueryInput,
  type QueryOptions,
  type GenerationResult,
  type InternalTextRequestOptions,
  type TokenBatch,
  type RegistryManifest,
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
import type { WasmThreadingMode } from '../engine/runtime-assets.js';

interface InstalledAsset {
  record: AssetRecord;
  file: File;
}

interface SourceInstallResult {
  assets: InstalledAsset[];
  source: 'remote' | 'local';
  explicitProjectorAssetId: string | null;
}

interface AssetClassifier {
  classify(assetId: string, file: File, signal?: AbortSignal): Promise<ClassifiedAssetFile>;
}

interface RuntimeRequestOptions {
  session?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stop?: readonly string[];
  signal?: AbortSignal;
  emitTokens?: boolean;
  tokenBatchSink?: (batch: TokenBatch) => void;
  grammar?: string;
  onRequestStarted?: (requestId: number) => void;
}

type BaseSource = string | File | readonly string[] | readonly File[];
type NavigatorWithGpu = Navigator & {
  gpu?: {
    requestAdapter(): Promise<NavigatorGpuAdapter | null>;
  };
};
type NavigatorGpuAdapter = {
  readonly features?: {
    has(feature: string): boolean;
  };
};

function isFile(value: unknown): value is File {
  return typeof File !== 'undefined' && value instanceof File;
}

function isStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

function isFileArray(value: unknown): value is readonly File[] {
  return Array.isArray(value) && value.every((item) => isFile(item));
}

function isSourceObject(source: ModelSource): source is Extract<ModelSource, { model: BaseSource }> {
  return typeof source === 'object' && source != null && !isFile(source) && !Array.isArray(source);
}

async function resolveBrowserBackend(
  backend: BrowserBackendPreference | undefined
): Promise<Exclude<BrowserBackendPreference, 'auto'>> {
  if (backend === 'cpu' || backend === 'webgpu') {
    return backend;
  }
  const gpu = (globalThis.navigator as NavigatorWithGpu | undefined)?.gpu;
  const adapter = gpu == null ? null : await gpu.requestAdapter();
  return adapter?.features?.has('shader-f16') === true ? 'webgpu' : 'cpu';
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
  private browserSplitCleanup: Promise<void> | null = null;
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

  public async load(source: ModelSource, options: ModelLoadOptions = {}): Promise<ModelInfo> {
    return this.withLifecycleLock(async () => {
      if (options.signal?.aborted) {
        throw new DOMException('Model load aborted.', 'AbortError');
      }
      return await this.loadWithRustLifecycle(source, options);
    });
  }

  public async remove(id: string): Promise<void> {
    await this.withLifecycleLock(async () => {
      const manifest = await this.registry.read();
      const rust = await this.getRustLifecycle(manifest);
      const wasCurrent = this.currentLoaded?.id === id;
      const removed = rust.remove(id);
      if (wasCurrent) {
        this.runtime.close();
        this.currentLoaded = null;
        this.currentSnapshot = null;
      }
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
      (session, promptOptions) => this.runtime.enqueueQuery(session, prompt, promptOptions),
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
        session: options.contextKey,
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
    enqueue: (session: string, promptOptions: PromptOptions) => Promise<GenerateRequestId>,
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
      sampling: requestSamplingPatch(options),
      grammar: options.grammar,
      onRequestStarted: options.onRequestStarted,
    };
    const session = options.session ?? 'default';
    const emitsTokens = promptOptions.emitTokens === true;
    const start = nowMs();
    this.observability.emit('query-start', {
      state: 'querying',
      query: {
        session,
        status: 'running',
        wallMs: null,
        ttftMs: null,
        outputTokens: null,
      },
    });
    let requestId = 0;
    let failureRecorded = false;
    try {
      requestId = await enqueue(session, promptOptions);
      this.emitEngineEvent({ type: 'request-started', requestId: String(requestId), streamId: requestId });
      const response = await this.runtime.awaitQuery(requestId, { signal: options.signal });
      if (response.cancelled) {
        const error = new DOMException(response.errorMessage ?? 'Queued request cancelled.', 'AbortError');
        this.recordQueryFailure(
          session,
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
          session,
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
        session,
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
          session,
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
        (session, promptOptions) => this.runtime.enqueueChat(session, messages, promptOptions),
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

  private async loadWithRustLifecycle(
    source: ModelSource,
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
    const backendPromise = resolveBrowserBackend(options.backend);
    let prepared: RustLifecyclePrepareLoadValue | null = null;
    let rust: RustLifecycleBridge | null = null;
    try {
      const rustSource = await this.buildRustLoadSource(source, manifest, loadOptions);
      const [resolvedRust, backend] = await Promise.all([rustPromise, backendPromise]);
      rust = resolvedRust;
      const runtimeConfig = applyBrowserRuntimeDefaults(
        options.runtime,
        this.runtime.getWasmThreadingMode()
      );
      prepared = rust.prepareLoad(rustSource, {
        backend,
        runtime: runtimeConfig,
        observability: observabilityMode,
      });
      await this.replaceManifest(prepared.manifest);
      this.ingestRustEvents(prepared.events);

      if (prepared.model.status === 'needs_projector') {
        this.currentLoaded = null;
        this.currentSnapshot = null;
        return prepared.model;
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
      const profile =
        observabilityMode === 'profile'
          ? toBackendProfileObservation(await this.runtime.getBackendObservability())
          : undefined;
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

  private async buildRustLoadSource(
    source: ModelSource,
    manifest: RegistryManifest,
    options: ModelLoadOptions
  ): Promise<RustLifecycleLoadSource> {
    const existing = this.resolveInstalledModel(manifest, source);
    const classifiedProjectors = await this.classifiedInstalledProjectors(manifest, options.signal);
    if (existing != null && !isSourceObject(source)) {
      return {
        kind: 'installed',
        id: existing.id,
        classifiedProjectors,
      };
    }

    const installed = await this.installSource(source, manifest, options);
    const classified = await this.classifyAssets(installed.assets, options.signal);
    const sourceProjectorAssetId = this.resolveSourceProjectorAssetId(
      classified,
      installed.explicitProjectorAssetId
    );
    return {
      kind: 'assets',
      assets: installed.assets.map((asset) => asset.record),
      classified: classified.map((file) => ({
        assetId: file.assetId,
        name: file.name,
        inspection: file.inspection,
      })),
      explicitProjectorAssetId: sourceProjectorAssetId,
      classifiedProjectors,
    };
  }

  private async classifiedInstalledProjectors(
    manifest: RegistryManifest,
    signal?: AbortSignal
  ): Promise<ClassifiedAsset[]> {
    const projectors: ClassifiedAsset[] = [];
    for (const asset of Object.values(manifest.assets)) {
      if (asset.kind !== 'projector' || asset.refCount <= 0) {
        continue;
      }
      if (asset.inspection != null) {
        projectors.push({
          assetId: asset.id,
          name: asset.name,
          inspection: asset.inspection,
        });
        continue;
      }
      try {
        const file = await this.assetStore.getFile(asset);
        const classified = await this.assetClassifier.classify(asset.id, file, signal);
        projectors.push({
          assetId: classified.assetId,
          name: classified.name,
          inspection: classified.inspection,
        });
      } catch (error) {
        if (error instanceof QueryError && error.code === 'MODEL_BROKEN') {
          continue;
        }
        throw error;
      }
    }
    return projectors;
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
    session: string,
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
      query: this.toQueryObservation(session, 'success', start, response),
      ...(runtime == null ? {} : { runtime }),
    });
  }

  private recordQueryFailure(
    session: string,
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
          session,
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
    session: string,
    status: QueryObservation['status'],
    start: number,
    response?: GenerateResponse
  ): QueryObservation {
    const metrics = response?.observability ?? null;
    return {
      session,
      status,
      wallMs: Math.max(0, nowMs() - start),
      ttftMs: metrics?.ttftMs ?? null,
      outputTokens: metrics?.outputTokens ?? null,
    };
  }

  private async installSource(
    source: ModelSource,
    manifest: RegistryManifest,
    options: ModelLoadOptions
  ): Promise<SourceInstallResult> {
    if (isSourceObject(source)) {
      const base = await this.installBaseSource(source.model, manifest, options, false);
      const projector =
        source.projector == null
          ? null
          : await this.installProjectorSource(source.projector, manifest, options);
      return {
        assets: [...base.assets, ...(projector?.assets ?? [])],
        source: base.source,
        explicitProjectorAssetId: projector?.assets[0]?.record.id ?? null,
      };
    }
    const base = await this.installBaseSource(source, manifest, options, false);
    return {
      ...base,
      explicitProjectorAssetId: null,
    };
  }

  private async installBaseSource(
    source: BaseSource,
    manifest: RegistryManifest,
    options: ModelLoadOptions,
    includeProjector: boolean
  ): Promise<SourceInstallResult> {
    if (typeof source === 'string') {
      const installed = manifest.models[source];
      if (installed != null) {
        return await this.assetsForEntry(installed, manifest, includeProjector);
      }
      return {
        assets: await this.installRemoteModelAssets(source, manifest, options),
        source: 'remote',
        explicitProjectorAssetId: null,
      };
    }
    if (isFile(source)) {
      return {
        assets: await this.installLocalModelAssets(source, manifest, options),
        source: 'local',
        explicitProjectorAssetId: null,
      };
    }
    if (isStringArray(source)) {
      if (source.length === 0) {
        throw new QueryError('INVALID_MODEL_SOURCE', 'Model URL array must not be empty.');
      }
      return {
        assets: await Promise.all(
          source.map((url) => this.installRemoteAsset(url, 'shard', manifest, options))
        ),
        source: 'remote',
        explicitProjectorAssetId: null,
      };
    }
    if (isFileArray(source)) {
      if (source.length === 0) {
        throw new QueryError('INVALID_MODEL_SOURCE', 'Model file array must not be empty.');
      }
      return {
        assets: await Promise.all(
          source.map((file) => this.installLocalAsset(file, 'shard', manifest, options))
        ),
        source: 'local',
        explicitProjectorAssetId: null,
      };
    }
    throw new QueryError('INVALID_MODEL_SOURCE', 'Unsupported model source.');
  }

  private async cleanupBrowserSplitArtifacts(manifest: RegistryManifest): Promise<void> {
    if (this.browserSplitCleanup == null) {
      this.browserSplitCleanup = this.assetStore.cleanupBrowserSplitArtifacts(manifest).catch((error) => {
        this.browserSplitCleanup = null;
        throw error;
      });
    }
    await this.browserSplitCleanup;
  }

  private async installProjectorSource(
    source: string | File,
    manifest: RegistryManifest,
    options: ModelLoadOptions
  ): Promise<SourceInstallResult> {
    if (typeof source === 'string') {
      return {
        assets: [await this.installRemoteAsset(source, 'projector', manifest, options)],
        source: 'remote',
        explicitProjectorAssetId: null,
      };
    }
    if (isFile(source)) {
      return {
        assets: [await this.installLocalAsset(source, 'projector', manifest, options)],
        source: 'local',
        explicitProjectorAssetId: null,
      };
    }
    throw new QueryError('INVALID_MODEL_SOURCE', 'Projector source must be a URL or File.');
  }

  private async installRemoteAsset(
    url: string,
    kind: AssetRecord['kind'],
    manifest: RegistryManifest,
    options: ModelLoadOptions
  ): Promise<InstalledAsset> {
    options.onProgress?.({
      phase: 'metadata',
      loadedBytes: 0,
      totalBytes: null,
      percent: null,
      assetName: url,
    });
    const metadata = await this.assetStore.resolveRemoteMetadata(url, options.signal);
    const existing = this.findRemoteAsset(manifest, metadata, kind);
    if (existing != null) {
      return {
        record: existing,
        file: await this.assetStore.getFile(existing),
      };
    }
    const record = await this.assetStore.downloadRemote(
      metadata,
      kind,
      options.signal,
      options.onProgress
    );
    return {
      record,
      file: await this.assetStore.getFile(record),
    };
  }

  private async installRemoteModelAssets(
    url: string,
    manifest: RegistryManifest,
    options: ModelLoadOptions
  ): Promise<InstalledAsset[]> {
    options.onProgress?.({
      phase: 'metadata',
      loadedBytes: 0,
      totalBytes: null,
      percent: null,
      assetName: url,
    });
    const metadata = await this.assetStore.resolveRemoteMetadata(url, options.signal);
    const existingSingle = this.findRemoteAsset(manifest, metadata, 'model');
    if (existingSingle != null) {
      return [
        {
          record: existingSingle,
          file: await this.assetStore.getFile(existingSingle),
        },
      ];
    }

    const existingSplit = this.findRemoteSplitAssets(manifest, metadata);
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

    if (this.assetStore.requiresBrowserSplit(metadata.bytes)) {
      await this.cleanupBrowserSplitArtifacts(manifest);
    }
    const records = await this.assetStore.downloadRemoteSplitGguf(
      metadata,
      this.runtime,
      options.signal,
      options.onProgress
    );
    return await this.installedAssetsFromRecords(records, manifest);
  }

  private async installLocalModelAssets(
    file: File,
    manifest: RegistryManifest,
    options: ModelLoadOptions
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

    if (this.assetStore.requiresBrowserSplit(file.size)) {
      await this.cleanupBrowserSplitArtifacts(manifest);
    }
    const records = await this.assetStore.installLocalSplitGguf(
      file,
      this.runtime,
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
    options: ModelLoadOptions
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

  private findRemoteAsset(
    manifest: RegistryManifest,
    metadata: RemoteAssetMetadata,
    kind: AssetRecord['kind']
  ): AssetRecord | null {
    return (
      Object.values(manifest.assets).find(
        (asset) =>
          asset.kind === kind &&
          asset.sourceUrl === metadata.canonicalUrl &&
          asset.sourceEtag === metadata.etag &&
          asset.sourceLastModified === metadata.lastModified &&
          asset.bytes === metadata.bytes
      ) ?? null
    );
  }

  private findRemoteSplitAssets(
    manifest: RegistryManifest,
    metadata: RemoteAssetMetadata
  ): AssetRecord[] | null {
    return this.findCompleteSplitAssets(
      Object.values(manifest.assets).filter(
        (asset) =>
          asset.kind === 'shard' &&
          asset.sourceUrl === metadata.canonicalUrl &&
          (asset.sourceEtag ?? '') === metadata.etag &&
          (asset.sourceLastModified ?? '') === metadata.lastModified &&
          asset.sourceBytes === metadata.bytes &&
          Number.isInteger(asset.sourcePartIndex) &&
          Number.isInteger(asset.sourcePartCount)
      )
    );
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

  private async assetsForEntry(
    entry: ModelEntry,
    manifest: RegistryManifest,
    includeProjector: boolean
  ): Promise<SourceInstallResult> {
    const assetIds = [...entry.modelAssetIds, includeProjector ? entry.projectorAssetId : undefined].filter(
      (assetId): assetId is string => typeof assetId === 'string'
    );
    const assets: InstalledAsset[] = [];
    for (const assetId of assetIds) {
      const record = manifest.assets[assetId];
      if (record == null) {
        await this.markBroken(entry.id);
        throw new QueryError('MODEL_BROKEN', `Installed model "${entry.id}" references a missing asset.`);
      }
      assets.push({
        record,
        file: await this.getAssetFileForEntry(entry, record),
      });
    }
    return {
      assets,
      source: assets.some((asset) => asset.record.sourceUrl != null) ? 'remote' : 'local',
      explicitProjectorAssetId: null,
    };
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

  private resolveInstalledModel(manifest: RegistryManifest, source: ModelSource): ModelEntry | null {
    if (typeof source !== 'string') {
      return null;
    }
    return manifest.models[source] ?? null;
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

function requestSamplingPatch(
  options: RuntimeRequestOptions
): PromptOptions['sampling'] {
  const patch: NonNullable<PromptOptions['sampling']> = {};
  if (options.temperature != null) {
    patch.temperature = options.temperature;
  }
  if (options.topP != null) {
    patch.top_p = options.topP;
  }
  return patch.temperature == null && patch.top_p == null ? undefined : patch;
}

function isChatInputObject(input: ChatInput): input is Extract<ChatInput, { messages: unknown }> {
  return !Array.isArray(input);
}
