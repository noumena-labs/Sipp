import type { EngineRuntime } from '../runtime/engine-runtime.js';
import { RuntimePairingValidationError } from '../runtime/engine-runtime.js';
import {
  buildBoundaryMarkers,
  sliceUnstreamedSuffix,
  StreamingBoundaryTextSanitizer,
} from '../core/chat-boundary-sanitizer.js';
import type {
  GenerateRequestId,
  GenerateResponse,
  NativeRuntimeConfig,
  PromptOptions,
} from '../core/inference-types.js';
import type {
  InternalBundleDescriptor,
  ModelBundleFileProjectorDescriptor,
  ModelDetectionResult,
} from '../bundle/model-bundle-types.js';
import { createLinkedAbortController, isAbortError } from '../utils/abort.js';
import { stableJson } from '../utils/stable-json.js';
import { AssetStore, type RemoteAssetMetadata } from './asset-store.js';
import { ModelRegistryStore } from './model-registry-store.js';
import { ModelAssetClassifier } from './model-asset-classifier.js';
import type { ClassifiedAsset, ClassifiedAssetFile, PairingPlan } from './pairing-types.js';
import type { RustLifecycleBridge } from '../wasm/lifecycle-bridge.js';
import type {
  RustLifecycleLoadSource,
  RustLifecyclePrepareLoadValue,
} from '../wasm/wasm-bridge.js';
import {
  QueryError,
  type AssetRecord,
  type ChatInput,
  type ChatOptions,
  type EngineEvent,
  type EngineState,
  type LoadedModelState,
  type ModelEntry,
  type ModelInfo,
  type ModelLifecycleService,
  type ModelLoadOptions,
  type ModelPairingReasonCode,
  type ModelRuntimeOptions,
  type ModelSource,
  type ObservabilityEvent,
  type ObservabilityMode,
  type ObservabilitySnapshot,
  type QueryObservation,
  type QueryInput,
  type QueryOptions,
  type RequestResult,
  type TokenBatch,
  type RegistryManifest,
} from './types.js';
import {
  EngineEventController,
  observabilityEventToStateEvent,
  observabilitySnapshotToEngineState,
  requestResultFromGenerateResponse,
} from './engine-protocol-adapter.js';
import {
  applyObservabilityMode,
  ObservabilityController,
  resolveObservabilityMode,
  toBackendProfileObservation,
  toRuntimeObservation,
} from './observability-controller.js';

interface InstalledAsset {
  record: AssetRecord;
  file: File;
}

interface SourceInstallResult {
  assets: InstalledAsset[];
  source: 'remote' | 'local';
  explicitProjectorAssetId: string | null;
}

type BaseSource = string | File | readonly string[] | readonly File[];

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

function toRuntimeConfig(
  options: ModelRuntimeOptions | undefined,
  mode: ObservabilityMode
): NativeRuntimeConfig {
  return applyObservabilityMode(options, mode);
}

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function tokenBatchFromText(
  requestId: string,
  streamId: number,
  sequenceStart: number,
  text: string
): TokenBatch {
  return {
    requestId,
    streamId,
    sequenceStart,
    text,
    frameCount: 1,
    byteCount: utf8ByteLength(text),
    stats: {
      framesSent: sequenceStart + 1,
      bytesSent: utf8ByteLength(text),
      framesDropped: 0,
      batchesSent: sequenceStart + 1,
    },
  };
}

function utf8ByteLength(text: string): number {
  return new TextEncoder().encode(text).byteLength;
}

function entryAssetFingerprint(entry: Pick<ModelEntry, 'modelAssetIds' | 'projectorAssetId'>): string {
  return stableJson({
    modelAssetIds: [...entry.modelAssetIds].sort((left, right) => left.localeCompare(right)),
    projectorAssetId: entry.projectorAssetId ?? null,
  });
}

const BROWSER_GGUF_SPLIT_DIRECT_LOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024;

function normalizeLocalSourceFileName(file: File): string {
  const trimmed = (file.name || 'model.gguf').trim();
  const defaultValue = trimmed.length > 0 ? trimmed : 'model.gguf';
  return defaultValue.replace(/[\\/:*?"<>|]+/g, '-');
}

export class ModelService implements ModelLifecycleService {
  private currentLoaded: LoadedModelState | null = null;
  private chatBoundaryMarkersPromise: Promise<readonly string[]> | null = null;
  private chatBoundaryMarkersKey: string | null = null;
  private operationChain: Promise<void> = Promise.resolve();
  private transitioning = false;
  private readonly observability = new ObservabilityController();
  private readonly engineEvents = new EngineEventController();
  private browserSplitCleanup: Promise<void> | null = null;
  private rustLifecyclePromise: Promise<RustLifecycleBridge> | null = null;
  private rustHashProviderPromise: Promise<void> | null = null;

  constructor(
    private readonly runtime: EngineRuntime,
    private readonly registry = new ModelRegistryStore(),
    private readonly assetStore = new AssetStore(),
    assetClassifier?: ModelAssetClassifier
  ) {
    this.assetClassifier = assetClassifier ?? new ModelAssetClassifier(runtime);
    this.observability.subscribe((event) => {
      this.engineEvents.emit(observabilityEventToStateEvent(event));
    });
  }

  private readonly assetClassifier: ModelAssetClassifier;

  public currentModel(): ModelInfo | null {
    const current = this.currentLoaded;
    if (current == null) {
      return null;
    }
    return this.currentSnapshot ?? null;
  }

  public current(): ModelInfo | null {
    return this.currentModel();
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

  public currentState(): EngineState {
    return observabilitySnapshotToEngineState(this.observability.current());
  }

  public state(): EngineState {
    return this.currentState();
  }

  public subscribeEvents(listener: (event: EngineEvent) => void): () => void {
    return this.engineEvents.subscribe(listener);
  }

  public async load(source: ModelSource, options: ModelLoadOptions = {}): Promise<ModelInfo> {
    return this.withLifecycleLock(async () => {
      if (options.signal?.aborted) {
        throw new DOMException('Model load aborted.', 'AbortError');
      }
      const rust = await this.getRustLifecycle(await this.registry.read());
      return await this.loadWithRustLifecycle(rust, source, options);
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
      this.engineEvents.emit({ type: 'state', state: this.currentState() });
    });
  }

  public async query(input: QueryInput, options: QueryOptions = {}): Promise<RequestResult> {
    return await this.queryResult(input, options);
  }

  public async queryResult(input: QueryInput, options: QueryOptions = {}): Promise<RequestResult> {
    const response = await this.runQuery(input, options);
    return requestResultFromGenerateResponse(response, {
      maxTokens: options.maxTokens,
    });
  }

  private async runQuery(input: QueryInput, options: QueryOptions = {}): Promise<GenerateResponse> {
    if (this.transitioning) {
      throw new QueryError('MODEL_NOT_READY', 'A model lifecycle transition is in progress.');
    }
    if (this.currentLoaded == null) {
      throw new QueryError('MODEL_NOT_READY', 'No model is loaded. Call engine.models.load(...) first.');
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
    return await this.runRuntimeRequest(options, media, (session, promptOptions) =>
      this.runtime.enqueueQuery(session, prompt, promptOptions)
    );
  }

  private async runRuntimeRequest(
    options: QueryOptions | ChatOptions,
    media: Uint8Array[] | undefined,
    enqueue: (session: string, promptOptions: PromptOptions) => Promise<GenerateRequestId>
  ): Promise<GenerateResponse> {
    let activeRequestId: number | null = null;
    let nextSequence = 0;
    const emitTokens = (batch: TokenBatch): void => {
      const requestId = activeRequestId ?? Number(batch.streamId);
      const text = batch.text;
      if (text.length === 0) {
        return;
      }
      options.onTokens?.({
        ...batch,
        requestId: String(requestId),
        streamId: requestId,
        sequenceStart: nextSequence,
      });
      nextSequence += Math.max(1, batch.frameCount);
    };
    const promptOptions: PromptOptions = {
      nTokens: options.maxTokens,
      signal: options.signal,
      onTokens: options.onTokens == null ? undefined : emitTokens,
      tokenFlush: options.onTokens == null ? undefined : options.tokenFlush ?? 'token',
      media,
      grammar: options.grammar,
      // Forward the internal streaming-claim hook if the caller (worker
      // entry) attached one. See PromptOptions.__internalRequestStarted
      // for why this exists. We use a property-key escape hatch so the
      // public ChatOptions / QueryOptions types don't have to advertise
      // this internal hook.
      __internalRequestStarted: (
        options as ChatOptions & {
          __internalRequestStarted?: (requestId: number) => void;
        }
      ).__internalRequestStarted,
    };
    const session = options.session ?? 'default';
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
    let failureRecorded = false;
    try {
      const requestId = await enqueue(session, promptOptions);
      activeRequestId = requestId;
      this.engineEvents.emit({ type: 'request-started', requestId: String(requestId), streamId: requestId });
      const response = await this.runtime.awaitQuery(requestId, { signal: options.signal });
      if (response.cancelled) {
        const error = new DOMException(response.errorMessage ?? 'Queued request cancelled.', 'AbortError');
        this.recordQueryFailure(session, start, error, response);
        this.engineEvents.emit({
          type: 'request-failed',
          requestId: String(requestId),
          error: error.message,
        });
        failureRecorded = true;
        throw error;
      }
      if (response.failed) {
        const error = new Error(response.errorMessage ?? 'Queued prompt failed.');
        this.recordQueryFailure(session, start, error, response);
        this.engineEvents.emit({
          type: 'request-failed',
          requestId: String(requestId),
          error: error.message,
        });
        failureRecorded = true;
        throw error;
      }
      this.recordQuerySuccess(session, start, response);
      this.engineEvents.emit({
        type: 'request-completed',
        result: requestResultFromGenerateResponse(response, {
          maxTokens: options.maxTokens,
        }),
      });
      return response;
    } catch (error) {
      if (!failureRecorded) {
        this.recordQueryFailure(session, start, error);
      }
      if (error instanceof QueryError) {
        throw error;
      }
      const wrapped = new QueryError(
        'QUERY_FAILED',
        error instanceof Error && error.message.trim().length > 0
          ? `Model query failed: ${error.message}`
          : 'Model query failed.',
        { cause: error }
      );
      if (!failureRecorded && activeRequestId != null) {
        this.engineEvents.emit({
          type: 'request-failed',
          requestId: String(activeRequestId),
          error: wrapped.message,
        });
      }
      throw wrapped;
    }
  }

  public async chat(input: ChatInput, options: ChatOptions = {}): Promise<RequestResult> {
    return await this.chatResult(input, options);
  }

  public async chatResult(input: ChatInput, options: ChatOptions = {}): Promise<RequestResult> {
    if (this.transitioning) {
      throw new QueryError('MODEL_NOT_READY', 'A model lifecycle transition is in progress.');
    }
    if (this.currentLoaded == null) {
      throw new QueryError('MODEL_NOT_READY', 'No model is loaded. Call engine.models.load(...) first.');
    }

    const current = this.currentLoaded;
    const messages = isChatInputObject(input) ? input.messages : input;
    const media = isChatInputObject(input) ? input.media : undefined;
    if (media != null && media.length > 0 && this.runtime.readMediaMarker() == null) {
      throw new QueryError('MODEL_NOT_READY', 'The loaded model does not accept media input.');
    }
    const boundaryMarkers = await this.getChatBoundaryMarkers(current);
    const outputSanitizer = new StreamingBoundaryTextSanitizer(boundaryMarkers);
    const linkedAbort = createLinkedAbortController(options.signal);
    let streamedOutputText = '';
    let assistantText = '';
    let stoppedAtBoundary = false;

    let safeSequence = 0;
    let lastBatch: TokenBatch | null = null;
    const shouldStreamTokens = options.onTokens != null;
    const consumeOutputTokens = (batch: TokenBatch): void => {
      lastBatch = batch;
      const text = batch.text;
      if (text.length === 0 || outputSanitizer.reachedBoundary) {
        return;
      }
      streamedOutputText += text;
      const result = outputSanitizer.consume(text);
      if (result.safeText.length > 0) {
        assistantText += result.safeText;
        options.onTokens?.(
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
        options.onTokens?.(
          tokenBatchFromText(source.requestId, source.streamId, safeSequence++, safeText)
        );
      }
    };

    try {
      const rawResult = await this.runRuntimeRequest(
        {
          ...options,
          signal: linkedAbort.signal,
          ...(shouldStreamTokens ? { onTokens: consumeOutputTokens } : {}),
        },
        media == null ? undefined : [...media],
        (session, promptOptions) => this.runtime.enqueueChat(session, messages, promptOptions)
      );
      const rawText = rawResult.outputText;
      const unseenOutputSuffix = shouldStreamTokens
        ? sliceUnstreamedSuffix(streamedOutputText, rawText)
        : rawText;
      if (!outputSanitizer.reachedBoundary && unseenOutputSuffix.length > 0) {
        const source = lastBatch ?? tokenBatchFromText(String(rawResult.requestId), 0, safeSequence, unseenOutputSuffix);
        consumeOutputTokens(
          tokenBatchFromText(source.requestId, source.streamId, safeSequence, unseenOutputSuffix)
        );
      }
      flushOutputText();
      return requestResultFromGenerateResponse(rawResult, {
        text: assistantText.trim(),
        maxTokens: options.maxTokens,
      });
    } catch (error) {
      if (stoppedAtBoundary && options.signal?.aborted !== true) {
        flushOutputText();
        return requestResultFromGenerateResponse(
          {
            requestId: -1,
            completed: true,
            failed: false,
            cancelled: false,
            outputText: assistantText.trim(),
            observability: null,
          },
          {
            text: assistantText.trim(),
            finishReason: 'stop',
          }
        );
      }
      throw error;
    } finally {
      linkedAbort.dispose();
    }
  }

  public getChatTemplate(): string | null {
    return this.runtime.getChatTemplate();
  }

  public getBosText(): string {
    return this.runtime.getBosText();
  }

  public getEosText(): string {
    return this.runtime.getEosText();
  }

  public getMediaMarker(): string | null {
    return this.runtime.readMediaMarker();
  }

  public close(): void {
    this.runtime.close();
    this.currentLoaded = null;
    this.currentSnapshot = null;
    void this.closeRustLifecycle();
    this.observability.markClosed();
  }

  private async loadWithRustLifecycle(
    rust: RustLifecycleBridge,
    source: ModelSource,
    options: ModelLoadOptions
  ): Promise<ModelInfo> {
    const loadOptions: ModelLoadOptions = {
      ...options,
      onProgress: (progress) => {
        options.onProgress?.(progress);
        this.engineEvents.emit({
          type: 'load-progress',
          loadedBytes: progress.loadedBytes,
          totalBytes: progress.totalBytes,
          assetName: progress.assetName,
        });
      },
    };
    const observabilityMode = resolveObservabilityMode(options.observability);
    let prepared: RustLifecyclePrepareLoadValue | null = null;
    try {
      const manifest = await this.registry.read();
      await this.cleanupBrowserSplitArtifacts(manifest);
      const rustSource = await this.buildRustLoadSource(source, manifest, loadOptions);
      prepared = rust.prepareLoad(rustSource, {
        runtime: options.runtime,
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
        const files = await this.filesForEntry(entry, prepared.manifest);
        const descriptor = this.buildDescriptor(
          files.modelFiles,
          files.projectorFile,
          entry,
          prepared.manifest
        );
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
      if (prepared != null) {
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
    await this.ensureRustHashProvider();
    if (this.rustLifecyclePromise == null) {
      this.rustLifecyclePromise = this.runtime.createRustLifecycleBridge(manifest);
    }
    return await this.rustLifecyclePromise;
  }

  private async ensureRustHashProvider(): Promise<void> {
    if (this.rustHashProviderPromise == null) {
      this.rustHashProviderPromise = this.runtime.createRustHashProvider().then((provider) => {
        this.assetStore.setHashProvider(provider);
      });
    }
    await this.rustHashProviderPromise;
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
      this.engineEvents.emit(observabilityEventToStateEvent(event));
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
    response: GenerateResponse
  ): void {
    const metrics = response.observability ?? null;
    const runtime = toRuntimeObservation(
      metrics ?? this.runtime.getRuntimeObservability(),
      this.runtime.getTransportObservability()
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
    response?: GenerateResponse
  ): void {
    const metrics = response?.observability ?? null;
    const runtime = toRuntimeObservation(
      metrics ?? this.runtime.getRuntimeObservability(),
      this.runtime.getTransportObservability()
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

    if (metadata.bytes <= BROWSER_GGUF_SPLIT_DIRECT_LOAD_MAX_BYTES) {
      const record = await this.assetStore.downloadRemote(
        metadata,
        'model',
        options.signal,
        options.onProgress
      );
      return [
        {
          record,
          file: await this.assetStore.getFile(record),
        },
      ];
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
    if (file.size <= BROWSER_GGUF_SPLIT_DIRECT_LOAD_MAX_BYTES) {
      return [await this.installLocalAsset(file, 'model', manifest, options)];
    }

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

  private async filesForEntry(
    entry: ModelEntry,
    manifest: RegistryManifest
  ): Promise<{ modelFiles: File[]; projectorFile: File | null }> {
    const modelFiles: File[] = [];
    for (const assetId of entry.modelAssetIds) {
      const asset = manifest.assets[assetId];
      if (asset == null) {
        await this.markBroken(entry.id);
        throw new QueryError('MODEL_BROKEN', `Installed model "${entry.id}" references a missing asset.`);
      }
      modelFiles.push(await this.getAssetFileForEntry(entry, asset));
    }
    let projectorFile: File | null = null;
    if (entry.projectorAssetId != null) {
      const projector = manifest.assets[entry.projectorAssetId];
      if (projector == null) {
        await this.markBroken(entry.id);
        throw new QueryError('MODEL_BROKEN', `Installed model "${entry.id}" references a missing projector.`);
      }
      projectorFile = await this.getAssetFileForEntry(entry, projector);
    }
    return { modelFiles, projectorFile };
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

  private buildDescriptor(
    modelFiles: File[],
    projectorFile: File | null,
    entry: ModelEntry,
    manifest: RegistryManifest
  ): InternalBundleDescriptor {
    const detection = this.detectionForEntry(entry, manifest);
    const projector: ModelBundleFileProjectorDescriptor | undefined =
      projectorFile == null
        ? undefined
        : {
            kind: 'file',
            file: projectorFile,
          };
    if (modelFiles.length === 1) {
      return {
        kind: 'file',
        file: modelFiles[0],
        projector,
        ...(detection == null ? {} : { detection }),
      };
    }
    return {
      kind: 'files',
      files: modelFiles,
      projector,
      ...(detection == null ? {} : { detection }),
    };
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

function isChatInputObject(input: ChatInput): input is Extract<ChatInput, { messages: unknown }> {
  return !Array.isArray(input);
}
