import test from 'node:test';
import assert from 'node:assert/strict';
import { ModelService } from '../../src/models/model-service.js';
import {
  AssetStore,
  type RemoteAssetMetadata,
  type RemoteStoreReceipt,
} from '../../src/models/asset-store.js';
import { ModelRegistryStore } from '../../src/models/model-registry-store.js';
import {
  type ClassifiedAsset,
  type ClassifiedAssetFile,
  type PairingPlan,
  RuntimePairingValidationError,
  type ModelDetectionResult,
  type ModelAddSource,
  QueryError,
  type AssetRecord,
  type BrowserBackendPreference,
  type CatalogModelInfo,
  type CatalogObservabilityEvent,
  type CatalogObservabilitySnapshot,
  type ModelEntry,
  type ModelInfo,
  type ModelLoadOptions,
  type ObservabilityEvent,
  type ObservabilitySnapshot,
  type RegistryManifest,
  type RuntimeBundleDescriptor,
  type RuntimeSessionDescriptor,
  type RuntimeSessionSnapshot,
} from '../../src/models/types.js';
import type {
  EngineRuntime,
  RuntimeActivation,
  RuntimeActivationResult,
} from '../../src/runtime/engine-runtime.js';
import type { RuntimeBackendConstraint } from '../../src/engine/runtime-assets.js';
import type {
  RustLifecycleBridge,
  type RustLifecycleInstallSource,
  type RustLifecycleInstallValue,
  type RustLifecycleLoadSource,
  type RustLifecyclePrepareLoadValue,
  RustRemoteAction,
  RustRemoteCommand,
  RustRemoteCommandValue,
} from '../../src/wasm/wasm-bridge.js';
import type {
  BackendObservability,
  ChatMessage,
  EmbedRuntimeOptions,
  GenerateRequestHandle,
  GenerateRequestId,
  GenerateResponse,
  NativeRuntimeConfig,
  PromptOptions,
  RequestObservabilityMetrics,
  TransportObservability,
} from '../../src/engine/inference-types.js';
import type { ChatBoundaryInfo } from '../../src/engine/chat-boundary-sanitizer.js';

function file(name: string, contents = name): File {
  return new File([contents], name);
}

function localSource(name: string, contents = name) {
  return {
    kind: 'local' as const,
    files: [file(name, contents)],
  };
}

async function installAndLoad(
  service: ModelService,
  source: ModelAddSource,
  options: ModelLoadOptions = {}
): Promise<ModelInfo> {
  const model = await service.add(source);
  return await service.load(model.id, options);
}

async function withGlobalFetch<T>(
  fetchImpl: typeof globalThis.fetch,
  callback: () => Promise<T>
): Promise<T> {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'fetch');
  Object.defineProperty(globalThis, 'fetch', {
    configurable: true,
    value: fetchImpl,
  });
  try {
    return await callback();
  } finally {
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'fetch');
    } else {
      Object.defineProperty(globalThis, 'fetch', descriptor);
    }
  }
}

async function withNavigatorGpu<T>(
  requestAdapter: () => Promise<{
    readonly features?: { has(feature: string): boolean };
  } | null>,
  callback: () => Promise<T>
): Promise<T> {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    enumerable: true,
    value: {
      ...(globalThis.navigator ?? {}),
      gpu: { requestAdapter },
    },
  });
  try {
    return await callback();
  } finally {
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'navigator');
    } else {
      Object.defineProperty(globalThis, 'navigator', descriptor);
    }
  }
}

async function withNavigatorHardwareConcurrency<T>(
  hardwareConcurrency: number,
  callback: () => Promise<T>
): Promise<T> {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    enumerable: true,
    value: {
      ...(globalThis.navigator ?? {}),
      hardwareConcurrency,
    },
  });
  try {
    return await callback();
  } finally {
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'navigator');
    } else {
      Object.defineProperty(globalThis, 'navigator', descriptor);
    }
  }
}

function cloneManifest(manifest: RegistryManifest): RegistryManifest {
  return JSON.parse(JSON.stringify(manifest)) as RegistryManifest;
}

class MemoryRegistryStore {
  public manifest: RegistryManifest = {
    version: 7,
    projectorIndexRevision: 0,
    assets: {},
    models: {},
  };

  public async read(): Promise<RegistryManifest> {
    return cloneManifest(this.manifest);
  }

  public async write(
    update: (manifest: RegistryManifest) => void | Promise<void>
  ): Promise<RegistryManifest> {
    await update(this.manifest);
    return this.read();
  }
}

class FakeAssetStore {
  private static readonly directLoadMaxBytes = 2 * 1024 * 1024 * 1024;
  public readonly files = new Map<string, File>();
  public readonly deleted: string[] = [];
  public localSplitCount = 0;
  public cleanupCount = 0;
  public forceBrowserSplit = false;
  public syncHandleOpenCount = 0;
  public syncHandleCloseCount = 0;
  public readonly exclusiveLockFailureCalls = new Set<number>();
  public readonly syncHandleFailures = new Map<number, unknown>();
  public async installFile(input: {
    kind: AssetRecord['kind'];
    file: File;
    sourceUrl?: string;
    sourceEtag?: string;
    sourceLastModified?: string;
  }): Promise<AssetRecord> {
    const id = `asset-${input.kind}-${input.file.name}-${input.file.size}`;
    this.files.set(id, input.file);
    return {
      id,
      kind: input.kind,
      name: input.file.name,
      bytes: input.file.size,
      storagePath: id,
      sourceUrl: input.sourceUrl,
      sourceEtag: input.sourceEtag,
      sourceLastModified: input.sourceLastModified,
      refCount: 0,
      createdAt: new Date(0).toISOString(),
    };
  }

  public async installLocalGguf(
    file: File,
    _runtime: unknown,
    _manifest: RegistryManifest,
    _signal?: AbortSignal,
    _onProgress?: unknown
  ): Promise<AssetRecord[]> {
    if (!this.forceBrowserSplit && file.size <= FakeAssetStore.directLoadMaxBytes) {
      return [await this.installFile({ kind: 'model', file })];
    }

    await this.cleanupBrowserSplitArtifacts();
    this.localSplitCount += 1;
    const sourceFileName = file.name.replace(/[\\/:*?"<>|]+/g, '-');
    return [0, 1].map((index) => {
      const id = `asset-shard-${file.name}-${file.size}-${file.lastModified}-${index}`;
      const shard = new File(
        [`${file.name}:${index}`],
        `${sourceFileName.replace(/\.gguf$/i, '')}-${String(index + 1).padStart(5, '0')}-of-00002.gguf`
      );
      this.files.set(id, shard);
      return {
        id,
        kind: 'shard',
        name: shard.name,
        bytes: shard.size,
        storagePath: id,
        sourceBytes: file.size,
        sourcePartIndex: index,
        sourcePartCount: 2,
        sourceFileName,
        sourceFileLastModified: file.lastModified,
        refCount: 0,
        createdAt: new Date(0).toISOString(),
      };
    });
  }

  public async downloadRemote(
    metadata: RemoteAssetMetadata,
    kind: AssetRecord['kind'],
    response: Response
  ): Promise<RemoteStoreReceipt> {
    const payload = await response.arrayBuffer();
    const id = `asset-${kind}-${metadata.name}-${metadata.bytes}`;
    const stored = new File([payload], metadata.name);
    this.files.set(id, stored);
    return {
      records: [
        {
          id,
          kind,
          name: metadata.name,
          bytes: metadata.bytes,
          storagePath: id,
          sourceUrl: metadata.canonicalUrl,
          sourceEtag: metadata.etag,
          sourceLastModified: metadata.lastModified,
          sourceBytes: metadata.bytes,
          refCount: 0,
          createdAt: new Date(0).toISOString(),
        },
      ],
      createdAssetIds: [id],
    };
  }

  public async downloadRemoteGguf(
    metadata: RemoteAssetMetadata,
    _runtime: unknown,
    response: Response
  ): Promise<RemoteStoreReceipt> {
    return await this.downloadRemote(metadata, 'model', response);
  }

  public async getFile(record: AssetRecord): Promise<File> {
    const stored = this.files.get(record.id);
    if (stored == null) {
      throw new QueryError('MODEL_BROKEN', `Missing fake asset ${record.id}.`);
    }
    return stored;
  }

  public async openSyncHandle(
    record: AssetRecord
  ): Promise<{ name: string; handle: import('../../src/engine/file-system-storage.js').OpfsSyncAccessHandle; size: number }> {
    this.syncHandleOpenCount += 1;
    if (this.syncHandleFailures.has(this.syncHandleOpenCount)) {
      const failure = this.syncHandleFailures.get(this.syncHandleOpenCount);
      this.syncHandleFailures.delete(this.syncHandleOpenCount);
      throw failure;
    }
    if (this.exclusiveLockFailureCalls.delete(this.syncHandleOpenCount)) {
      throw new DOMException('The file is exclusively locked.', 'NoModificationAllowedError');
    }
    const stored = this.files.get(record.id);
    if (stored == null) {
      throw new QueryError('MODEL_BROKEN', `Missing fake asset ${record.id}.`);
    }
    const bytes = new Uint8Array(await stored.arrayBuffer());
    const handle: import('../../src/engine/file-system-storage.js').OpfsSyncAccessHandle = {
      read: (target, options) => {
        const at = options?.at ?? 0;
        const available = Math.max(0, bytes.byteLength - at);
        const toRead = Math.min(target.byteLength, available);
        target.set(bytes.subarray(at, at + toRead));
        return toRead;
      },
      write: () => {
        throw new Error('write not supported in fake');
      },
      truncate: () => {},
      flush: () => {},
      close: () => {
        this.syncHandleCloseCount += 1;
      },
      getSize: () => bytes.byteLength,
    };
    return { name: record.name, handle, size: bytes.byteLength };
  }

  public async delete(record: AssetRecord): Promise<void> {
    this.deleted.push(record.id);
    this.files.delete(record.id);
  }

  public async cleanupBrowserSplitArtifacts(): Promise<void> {
    this.cleanupCount += 1;
  }

  public openAcquisitionJournal(): {
    recordStoragePath(storagePath: string): Promise<void>;
    recordStoragePaths(storagePaths: readonly string[]): Promise<void>;
    cleanupUncommitted(manifest: RegistryManifest): Promise<void>;
    clear(): Promise<void>;
  } {
    return {
      recordStoragePath: async () => {},
      recordStoragePaths: async () => {},
      cleanupUncommitted: async () => {},
      clear: async () => {},
    };
  }
}

class FakeAssetClassifier {
  public async classify(assetId: string, input: File): Promise<ClassifiedAssetFile> {
    const isProjector = /mmproj|projector/i.test(input.name);
    const visionCapable = !isProjector && /vision|llava/i.test(input.name);
    return {
      assetId,
      file: input,
      inspection: {
        version: 4,
        role: isProjector ? 'projector' : 'model',
        architecture: visionCapable ? 'vision-test' : 'text-test',
        trainedContextSize: isProjector ? null : 8192,
        visionCapable,
        audioCapable: false,
        audioGenerationCapable: false,
        compatibleVisionProjectorTypes: visionCapable ? ['vision-merger'] : [],
        compatibleAudioProjectorTypes: [],
        compatibleAudioGenerationProjectorTypes: [],
        providedVisionProjectorType: isProjector ? 'vision-merger' : null,
        providedAudioProjectorType: null,
        providedAudioGenerationProjectorType: null,
      },
      name: input.name,
    };
  }
}

class IncompatibleProjectorClassifier extends FakeAssetClassifier {
  public override async classify(assetId: string, input: File): Promise<ClassifiedAssetFile> {
    const classified = await super.classify(assetId, input);
    if (/bad-mmproj/i.test(input.name)) {
      classified.inspection.providedVisionProjectorType = 'other-merger';
    }
    return classified;
  }
}

class FailingAssetClassifier extends FakeAssetClassifier {
  public override async classify(): Promise<ClassifiedAssetFile> {
    throw new QueryError('INVALID_MODEL_SOURCE', 'classification failed');
  }
}

class AbortingAssetClassifier extends FakeAssetClassifier {
  public constructor(private readonly controller: AbortController) {
    super();
  }

  public override async classify(): Promise<ClassifiedAssetFile> {
    this.controller.abort();
    throw new DOMException('classification aborted', 'AbortError');
  }
}

function resolveFakePairing(files: readonly ClassifiedAsset[]): PairingPlan {
  if (files.length === 0) {
    throw new RuntimePairingValidationError(
      'INVALID_MODEL_SOURCE',
      'No model assets were provided.'
    );
  }

  const projectors = files.filter((file) => file.inspection.role === 'projector');
  if (projectors.length > 1) {
    throw new RuntimePairingValidationError(
      'INVALID_MODEL_PAIRING',
      `Multiple projector assets were provided: ${projectors.map((file) => file.name).join(', ')}.`
    );
  }

  const projector = projectors[0] ?? null;

  const modelFiles = files
    .filter((file) => file.assetId !== projector?.assetId)
    .sort((left, right) => left.name.localeCompare(right.name));
  if (modelFiles.length === 0) {
    throw new RuntimePairingValidationError(
      'INVALID_MODEL_PAIRING',
      'Projector assets are not runnable models.'
    );
  }

  const modelCandidates = modelFiles.filter((file) => file.inspection.role !== 'projector');
  const visionCandidates = modelCandidates.filter((file) => file.inspection.visionCapable);
  const compatibilitySources = visionCandidates.filter(
    (file) => file.inspection.compatibleVisionProjectorTypes.length > 0
  );
  if (!compatibleVisionTypesAgree(compatibilitySources)) {
    throw new RuntimePairingValidationError(
      'INVALID_MODEL_SOURCE',
      'Model assets disagree on compatible vision projector types.'
    );
  }

  const base = visionCandidates[0] ?? modelCandidates[0];
  if (base == null) {
    throw new RuntimePairingValidationError(
      'INVALID_MODEL_PAIRING',
      'Projector assets are not runnable models.'
    );
  }
  const compatibleVisionProjectorTypes =
    compatibilitySources[0]?.inspection.compatibleVisionProjectorTypes ?? [];
  if (projector != null) {
    const providedType = projector.inspection.providedVisionProjectorType;
    if (
      providedType != null &&
      compatibleVisionProjectorTypes.length > 0 &&
      !compatibleVisionProjectorTypes.includes(providedType)
    ) {
      throw new RuntimePairingValidationError(
        'INVALID_MODEL_PAIRING',
        `Projector type "${providedType}" is not compatible with this model.`
      );
    }
    return {
      modelAssetIds: modelFiles.map((file) => file.assetId),
      projectorAssetId: projector.assetId,
      name: base.name,
      modality: 'vision',
      status: 'ready',
      compatibleVisionProjectorTypes,
    };
  }

  return {
    modelAssetIds: modelFiles.map((file) => file.assetId),
    name: base.name,
    modality: base.inspection.visionCapable ? 'vision' : 'text',
    status: base.inspection.visionCapable ? 'needs_projector' : 'ready',
    compatibleVisionProjectorTypes,
  };
}

function compatibleVisionTypesAgree(files: readonly ClassifiedAsset[]): boolean {
  if (files.length < 2) {
    return true;
  }
  const expected = stableTypeList(files[0].inspection.compatibleVisionProjectorTypes);
  return files
    .slice(1)
    .every((file) => expected === stableTypeList(file.inspection.compatibleVisionProjectorTypes));
}

function stableTypeList(values: readonly string[]): string {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right)).join('\u0000');
}

class FakeRuntime implements EngineRuntime {
  public closeCount = 0;
  public loadCount = 0;
  public wasmThreadingMode: 'single-thread' | 'pthread' = 'single-thread';
  public backendConstraint: RuntimeBackendConstraint | null = null;
  public lastPrompt: string | null = null;
  public lastAudio: Uint8Array | null = null;
  public lastMaxDurationMs: number | undefined;
  public lastContextKey: string | null = null;
  public mediaMarker: string | null = null;
  public nextOutputText: string | null = null;
  public streamedTokens: string[] = ['token'];
  public enqueuedOptions: Array<number | PromptOptions | EmbedRuntimeOptions | undefined> = [];
  public wasmRunLoopCalls = 0;
  public wasmRunLoopMs = 0;
  private runtimeSession: RuntimeSessionSnapshot | null = null;
  private runtimeMetricsEnabled = false;
  private backendProfilingEnabled = false;
  private nextRequestId = 1;
  private readonly queuedRequests = new Map<
    GenerateRequestId,
    {
      promptText: string;
      options?: number | PromptOptions | EmbedRuntimeOptions;
      embedding?: boolean;
      audio?: boolean;
      normalize?: boolean;
    }
  >();

  public getWasmThreadingMode(): 'single-thread' | 'pthread' {
    return this.wasmThreadingMode;
  }

  public getTransportObservability(): TransportObservability {
    return {
      executionMode: 'worker',
      workerBacked: true,
      enabled: this.runtimeMetricsEnabled,
      wasmRunLoopCalls: this.wasmRunLoopCalls,
      wasmRunLoopMs: this.wasmRunLoopMs,
      activeTokenTransport: 'none',
    };
  }

  public async initModule(): Promise<void> {}

  public async detectModelFromGgufFile(file: Blob & { name?: string }): Promise<ModelDetectionResult> {
    const name = file.name ?? 'model.gguf';
    const isProjector = /mmproj|projector/i.test(name);
    const visionCapable = !isProjector && /vision|llava/i.test(name);
    const inspection = {
      version: 4 as const,
      role: isProjector ? 'projector' as const : 'model' as const,
      architecture: visionCapable ? 'vision-test' : 'text-test',
      trainedContextSize: isProjector ? null : 8192,
      visionCapable,
      audioCapable: false,
      audioGenerationCapable: false,
      compatibleVisionProjectorTypes: visionCapable ? ['vision-merger'] : [],
      compatibleAudioProjectorTypes: [],
      compatibleAudioGenerationProjectorTypes: [],
      providedVisionProjectorType: isProjector ? 'vision-merger' : null,
      providedAudioProjectorType: null,
      providedAudioGenerationProjectorType: null,
    };
    return {
      inspection,
      detectionMethod: 'gguf-metadata',
      modelName: name,
      modelType: null,
      modelArchitecture: inspection.architecture,
    };
  }

  public async resolvePairing(classified: readonly ClassifiedAsset[]): Promise<PairingPlan> {
    return resolveFakePairing(classified);
  }

  public async activateRuntime<TCommit>(
    descriptor: RuntimeBundleDescriptor,
    activation: RuntimeActivation<TCommit>
  ): Promise<RuntimeActivationResult<TCommit>> {
    if (this.runtimeSession != null) {
      throw new Error('FakeRuntime supports one activation per Worker.');
    }
    for (const file of descriptor.modelFiles) {
      file.handle.close();
    }
    descriptor.projector?.handle.close();

    this.loadCount += 1;
    this.runtimeMetricsEnabled = activation.config.observability?.runtime_metrics === true;
    this.backendProfilingEnabled = activation.config.observability?.backend_profiling === true;
    this.runtimeSession = null;
    this.mediaMarker = null;
    this.mediaMarker = descriptor.projector == null ? null : '<image>';
    this.runtimeSession = {
      ...activation.session,
      generation: this.loadCount,
      capabilities: {
        modelClass: 'decoder_only',
        supportsTextGeneration: true,
        supportsEmbeddings: true,
        supportsVision: this.mediaMarker != null,
        audioSampleRateHz: 16_000,
        generatedAudioSampleRateHz: 24_000,
        hasChatTemplate: true,
        embedding: { dimensions: 3, pooling: 'mean' },
        operations: {
          query: true,
          chat: true,
          embed: true,
          listen: true,
          speak: true,
        },
      },
      chatTemplate: 'fake-template',
      bosText: '<s>',
      eosText: '</s>',
      mediaMarker: this.mediaMarker,
    };
    try {
      const committed = await activation.commit({
        session: this.runtimeSession,
        runtimeObservability: this.getRuntimeObservability(),
        backendObservability: await this.getBackendObservability(),
      });
      return { session: this.runtimeSession, committed };
    } catch (error) {
      this.runtimeSession = null;
      this.mediaMarker = null;
      throw error;
    }
  }

  public currentRuntimeSession(): RuntimeSessionSnapshot | null {
    return this.runtimeSession;
  }

  private renderNativeChatPrompt(
    messages: readonly { role: string; content: string }[],
    addAssistant: boolean
  ): string {
    const rendered = messages
      .map((message) => `<${message.role}>\n${message.content}</${message.role}>\n`)
      .join('');
    return `${rendered}${addAssistant ? '<assistant>\n' : ''}`;
  }

  public async probeChatTemplateBoundaryInfo(): Promise<ChatBoundaryInfo> {
    return {
      assistantPrefix: '<assistant>\n',
      assistantSuffix: '</assistant>\n',
      nextTurnPrefixes: ['<system>\n', '<user>\n', '<assistant>\n'],
      eosText: '</s>',
    };
  }

  public getChatTemplate(): string | null {
    return 'fake-template';
  }

  public getBosText(): string {
    return '<s>';
  }

  public getEosText(): string {
    return '</s>';
  }

  public async browserCacheLayout(): Promise<'single-file' | 'split-gguf'> {
    return 'single-file';
  }

  public async planGgufSplitCount(): Promise<number> {
    return 1;
  }

  public async splitGgufStream(): Promise<void> {}

  public async close(): Promise<void> {
    this.closeCount += 1;
    this.runtimeSession = null;
    this.mediaMarker = null;
  }

  public readMediaMarker(): string | null {
    return this.mediaMarker;
  }

  public async cancelQuery(_request: GenerateRequestHandle): Promise<boolean> {
    return true;
  }

  public async enqueueQuery(
    contextKey: string,
    promptText: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestHandle> {
    const requestId = this.nextRequestId++;
    assert.ok(this.runtimeSession);
    const request = { generation: this.runtimeSession.generation, requestId };
    this.lastContextKey = contextKey;
    this.lastPrompt = promptText;
    this.enqueuedOptions.push(options);
    this.queuedRequests.set(requestId, { promptText, options });
    if (typeof options === 'object' && this.streamedTokens.length > 0) {
      const text = this.streamedTokens.join('');
      options.tokenBatchSink?.({
        requestId: `${request.generation}:${requestId}`,
        streamId: requestId,
        sequenceStart: 0,
        text,
        frameCount: this.streamedTokens.length,
        byteCount: new TextEncoder().encode(text).byteLength,
        stats: {
          framesSent: this.streamedTokens.length,
          bytesSent: new TextEncoder().encode(text).byteLength,
          batchesSent: 1,
        },
      });
    }
    return request;
  }

  public async enqueueChat(
    contextKey: string,
    messages: readonly ChatMessage[],
    options?: number | PromptOptions
  ): Promise<GenerateRequestHandle> {
    return this.enqueueQuery(contextKey, this.renderNativeChatPrompt(messages, true), options);
  }

  public async enqueueEmbedding(
    contextKey: string,
    input: string,
    options?: EmbedRuntimeOptions
  ): Promise<GenerateRequestHandle> {
    const requestId = this.nextRequestId++;
    assert.ok(this.runtimeSession);
    this.lastContextKey = contextKey;
    this.lastPrompt = input;
    this.enqueuedOptions.push(options);
    this.queuedRequests.set(requestId, {
      promptText: input,
      options,
      embedding: true,
      normalize: options?.normalize ?? true,
    });
    return { generation: this.runtimeSession.generation, requestId };
  }

  public async enqueueListen(
    audio: Uint8Array,
    language: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestHandle> {
    this.lastAudio = audio;
    return this.enqueueQuery('listen', language, options);
  }

  public async enqueueSpeak(
    text: string,
    language: string,
    speakerAudio: Uint8Array,
    maxDurationMs: number | undefined,
    options?: PromptOptions
  ): Promise<GenerateRequestHandle> {
    const requestId = this.nextRequestId++;
    assert.ok(this.runtimeSession);
    this.lastAudio = speakerAudio;
    this.lastMaxDurationMs = maxDurationMs;
    this.enqueuedOptions.push(options);
    this.queuedRequests.set(requestId, {
      promptText: `${language}:${text}`,
      options,
      audio: true,
    });
    return { generation: this.runtimeSession.generation, requestId };
  }

  public async awaitQuery(handle: GenerateRequestHandle): Promise<GenerateResponse> {
    const requestId = handle.requestId;
    const request = this.queuedRequests.get(requestId);
    if (request == null) {
      return {
        requestId,
        completed: false,
        outputText: '',
        cancelled: false,
        failed: true,
        errorMessage: `Missing fake request ${requestId}.`,
      };
    }
    this.queuedRequests.delete(requestId);
    this.wasmRunLoopCalls += 2;
    this.wasmRunLoopMs += 12.5;
    if (request.embedding === true) {
      return {
        requestId,
        completed: true,
        embedding: {
          values: request.normalize === false ? [3, 4] : [0.6, 0.8],
          pooling: 'mean',
          normalized: request.normalize !== false,
        },
        cancelled: false,
        failed: false,
        observability: this.runtimeMetricsEnabled ? this.createMetrics() : null,
      };
    }
    if (request.audio === true) {
      return {
        requestId,
        completed: true,
        audio: {
          data: new Uint8Array([82, 73, 70, 70]),
          sampleRateHz: 24_000,
          channels: 1,
          durationMs: 80,
        },
        cancelled: false,
        failed: false,
      };
    }
    const outputText = this.nextOutputText ?? `answer:${request.promptText}`;
    this.nextOutputText = null;
    return {
      requestId,
      completed: true,
      outputText,
      cancelled: false,
      failed: false,
      observability: this.runtimeMetricsEnabled ? this.createMetrics() : null,
    };
  }

  public getRuntimeObservability(): RequestObservabilityMetrics | null {
    return this.runtimeMetricsEnabled ? this.createMetrics() : null;
  }

  public async getBackendObservability(): Promise<BackendObservability | null> {
    if (!this.backendProfilingEnabled) {
      return null;
    }
    return {
      profilingEnabled: true,
      webgpuCompiled: false,
      webgpuRegistered: false,
      webgpuDeviceCount: 0,
      gpuOffloadSupported: false,
      engineInitialized: true,
      availableBackends: [{ name: 'cpu', deviceCount: 1 }],
      devices: [],
    };
  }

  private createMetrics(): RequestObservabilityMetrics {
    return {
      ttftMs: 4,
      itlAvgMs: 10, // 100 TPS
      itlP99Ms: 2.0,
      e2eMs: 12,
      prefillMs: 5,
      decodeMs: 50, // 5 tokens * 10ms = 50ms
      nativeGpuMs: 3,
      nativeSyncMs: 1,
      nativeLogicMs: 1,
      inputTokens: 3,
      outputTokens: 5,
      cacheMode: 'live_slot_prefix',
      cacheSource: 'none',
      cacheHits: 0,
      prefillTokens: 3,
    };
  }

  public rustBridge: FakeRustLifecycleBridge | null = null;

  public async createRustLifecycleBridge(): Promise<RustLifecycleBridge> {
    if (this.rustBridge == null) {
      this.rustBridge = new FakeRustLifecycleBridge();
    }
    return this.rustBridge as unknown as RustLifecycleBridge;
  }

}

class FakeRustLifecycleBridge {
  public installCount = 0;
  public prepareCount = 0;
  public commitCount = 0;
  public removeCount = 0;
  public remoteAdvanceCount = 0;
  public remoteCancelCount = 0;
  public remoteCleanupCount = 0;
  public remoteDownloadMode = false;
  public remoteCancelError: Error | null = null;
  public lastOptions: unknown = null;
  private remoteUrl: string | null = null;
  private remoteFailure: { code: string; message: string } | null = null;
  private manifest: RegistryManifest = {
    version: 7,
    projectorIndexRevision: 0,
    assets: {},
    models: {},
  };
  private pendingModelId: string | null = null;

  public list(): CatalogModelInfo[] {
    return Object.values(this.manifest.models).map((entry) => this.toCatalogModel(entry));
  }

  public remoteAcquisition(command: RustRemoteCommand): RustRemoteCommandValue {
    switch (command.command) {
      case 'begin': {
        const url = command.urls[0];
        assert.ok(url);
        this.remoteUrl = url;
        return { kind: 'action', action: this.remoteMetadataAction(1) };
      }
      case 'advance':
        if (this.remoteDownloadMode && command.event.kind === 'metadata_succeeded') {
          assert.ok(this.remoteUrl);
          return {
            kind: 'action',
            action: {
              kind: 'download',
              acquisitionId: command.event.acquisitionId,
              memberId: command.event.memberId,
              attempt: command.event.attempt,
              metadata: {
                url: this.remoteUrl,
                name: 'model.gguf',
                bytes: command.event.headers.contentLength ?? 0,
                etag: command.event.headers.etag,
                lastModified: command.event.headers.lastModified,
              },
            },
          };
        }
        if (this.remoteDownloadMode && command.event.kind === 'cleanup_succeeded') {
          this.remoteCleanupCount += 1;
          return {
            kind: 'failed',
            error: this.remoteFailure ?? {
              code: 'REMOTE_LOAD_FAILED',
              message: 'Remote acquisition failed.',
            },
          };
        }
        if (command.event.kind === 'wait_completed') {
          return {
            kind: 'action',
            action: this.remoteMetadataAction(command.event.attempt + 1),
          };
        }
        assert.equal(command.event.kind, 'operation_failed');
        if (this.remoteDownloadMode) {
          assert.ok(this.remoteUrl);
          this.remoteFailure = {
            code: 'REMOTE_LOAD_FAILED',
            message:
              `remote model download failed for ${this.remoteUrl}: ` +
              command.event.failure.reason,
          };
          return {
            kind: 'action',
            action: {
              kind: 'cleanup',
              acquisitionId: command.event.acquisitionId,
              memberId: command.event.memberId,
              attempt: command.event.attempt,
              assetIds: command.event.createdAssetIds,
            },
          };
        }
        this.remoteAdvanceCount += 1;
        if (this.remoteAdvanceCount < 4) {
          return { kind: 'action', action: this.remoteWaitAction(command.event.attempt) };
        }
        return {
          kind: 'failed',
          error: {
            code: 'REMOTE_METADATA_UNAVAILABLE',
            message: `remote metadata is unavailable for ${this.remoteUrl}: HTTP 503`,
            status: 503,
          },
        };
      case 'cancel':
        this.remoteCancelCount += 1;
        if (this.remoteCancelError != null) {
          throw this.remoteCancelError;
        }
        return { kind: 'cancelled', snapshot: this.snapshot('idle', null, 'off') };
    }
  }

  public install(source: RustLifecycleInstallSource): RustLifecycleInstallValue {
    this.installCount += 1;
    for (const asset of source.assets) {
      this.manifest.assets[asset.id] = {
        ...asset,
        refCount: 1,
        inspection:
          source.classified.find((classified) => classified.assetId === asset.id)?.inspection ??
          asset.inspection,
      };
    }

    const projectorAssetId = source.classified.find(
      (asset) => asset.inspection.role === 'projector'
    )?.assetId;
    const modelAssets = source.assets.filter((asset) => asset.id !== projectorAssetId);
    const primaryAsset = modelAssets[0];
    assert.ok(primaryAsset);
    const modelId = `model-${primaryAsset.id}`;
    const now = new Date(0).toISOString();
    this.manifest.models[modelId] = {
      id: modelId,
      name: primaryAsset.name,
      modality: projectorAssetId == null ? 'text' : 'vision',
      status: 'ready',
      modelAssetIds: modelAssets.map((asset) => asset.id),
      projectorAssetId,
      runtimeFingerprint: 'runtime-fingerprint',
      createdAt: now,
      updatedAt: now,
    };
    const model = this.toCatalogModel(this.manifest.models[modelId]);
    const snapshot = this.snapshot('idle', null, 'off');
    return {
      model,
      manifest: cloneManifest(this.manifest),
      snapshot,
      events: [],
    };
  }

  public prepareLoad(
    source: RustLifecycleLoadSource,
    options: {
      backend?: BrowserBackendPreference;
      runtime?: NativeRuntimeConfig;
      observability?: 'off' | 'runtime' | 'profile';
    }
  ): RustLifecyclePrepareLoadValue {
    this.prepareCount += 1;
    this.lastOptions = options;
    const entry = this.manifest.models[source.modelId];
    assert.ok(entry);
    const assets = entry.modelAssetIds.map((assetId) => {
      const asset = this.manifest.assets[assetId];
      assert.ok(asset);
      return asset;
    });
    this.pendingModelId = entry.id;
    const model = this.toCatalogModel(entry);
    const snapshot = this.snapshot('loading', null, options.observability ?? 'off');
    return {
      loadId: 'load-1',
      model,
      runtimeFingerprint: 'runtime-fingerprint',
      runtimeConfig: {
        ...(options.runtime ?? {}),
        observability: {
          ...(options.runtime?.observability ?? {}),
          runtime_metrics: options.observability === 'runtime' || options.observability === 'profile',
          backend_profiling: options.observability === 'profile',
        },
      },
      assets: assets.map((asset) => ({
        assetId: asset.id,
        kind: asset.kind,
        storagePath: asset.storagePath,
        mountName: asset.name,
        bytes: asset.bytes,
      })),
      projector: null,
      manifest: cloneManifest(this.manifest),
      snapshot,
      events: [{ type: 'load-start', snapshot }],
    };
  }

  public commitLoad(): {
    model: CatalogModelInfo;
    manifest: RegistryManifest;
    snapshot: CatalogObservabilitySnapshot;
    events: CatalogObservabilityEvent[];
  } {
    this.commitCount += 1;
    assert.ok(this.pendingModelId);
    const entry = this.manifest.models[this.pendingModelId];
    assert.ok(entry);
    const loadedAt = new Date(1).toISOString();
    entry.updatedAt = loadedAt;
    entry.lastLoadedAt = loadedAt;
    this.pendingModelId = null;
    const model = this.toCatalogModel(entry);
    const snapshot = this.snapshot('ready', model, 'runtime');
    return {
      model,
      manifest: cloneManifest(this.manifest),
      snapshot,
      events: [{ type: 'load-complete', snapshot }],
    };
  }

  public remove(modelId: string, activeModelId: string | null): {
    removed: ModelEntry;
    orphanedAssets: AssetRecord[];
    manifest: RegistryManifest;
    snapshot: CatalogObservabilitySnapshot;
    events: CatalogObservabilityEvent[];
  } {
    this.removeCount += 1;
    if (activeModelId === modelId) {
      throw new QueryError('MODEL_IN_USE', `Model "${modelId}" is loaded.`);
    }
    const removed = this.manifest.models[modelId];
    assert.ok(removed);
    delete this.manifest.models[modelId];
    const orphanedAssets = removed.modelAssetIds
      .map((assetId) => this.manifest.assets[assetId])
      .filter((asset): asset is AssetRecord => asset != null);
    for (const asset of orphanedAssets) {
      delete this.manifest.assets[asset.id];
    }
    const snapshot = this.snapshot('idle', null, 'off');
    return {
      removed,
      orphanedAssets,
      manifest: cloneManifest(this.manifest),
      snapshot,
      events: [{ type: 'load-complete', snapshot }],
    };
  }

  public close(): void {}

  public drainEvents(): ObservabilityEvent[] {
    return [];
  }

  private remoteMetadataAction(attempt: number): RustRemoteAction {
    assert.ok(this.remoteUrl);
    return {
      kind: 'fetch_metadata',
      acquisitionId: 'remote-1',
      memberId: 0,
      attempt,
      url: this.remoteUrl,
    };
  }

  private remoteWaitAction(attempt: number): RustRemoteAction {
    return {
      kind: 'wait',
      acquisitionId: 'remote-1',
      memberId: 0,
      attempt,
      delayMs: 0,
    };
  }

  private toCatalogModel(entry: ModelEntry): CatalogModelInfo {
    const assets = entry.modelAssetIds
      .map((assetId) => this.manifest.assets[assetId])
      .filter((asset): asset is AssetRecord => asset != null);
    return {
      id: entry.id,
      name: entry.name,
      modality: entry.modality,
      status: entry.status,
      source: assets.some((asset) => asset.sourceUrl != null) ? 'remote' : 'local',
      bytes: assets.reduce((sum, asset) => sum + asset.bytes, 0),
      assetFingerprint: `asset-fingerprint-${entry.id}`,
      createdAt: entry.createdAt,
      updatedAt: entry.updatedAt,
    };
  }

  private snapshot(
    state: ObservabilitySnapshot['state'],
    model: CatalogModelInfo | null,
    mode: CatalogObservabilitySnapshot['mode']
  ): CatalogObservabilitySnapshot {
    return {
      mode,
      state,
      updatedAt: new Date(0).toISOString(),
      model,
      query: null,
    };
  }
}

function createService(overrides: {
  runtime?: FakeRuntime;
  registry?: MemoryRegistryStore;
  assets?: FakeAssetStore;
  classifier?: { classify(assetId: string, file: File, signal?: AbortSignal): Promise<ClassifiedAssetFile> };
  sleep?: (delayMs: number) => Promise<void>;
} = {}): {
  service: ModelService;
  runtime: FakeRuntime;
  registry: MemoryRegistryStore;
  assets: FakeAssetStore;
} {
  const runtime = overrides.runtime ?? new FakeRuntime();
  const registry = overrides.registry ?? new MemoryRegistryStore();
  const assets = overrides.assets ?? new FakeAssetStore();
  return {
    service: new ModelService(
      runtime,
      registry as unknown as ModelRegistryStore,
      assets as unknown as AssetStore,
      overrides.classifier ?? new FakeAssetClassifier(),
      overrides.sleep
    ),
    runtime,
    registry,
    assets,
  };
}

function createRustBackedService(
  runtime: FakeRuntime = new FakeRuntime(),
  overrides: {
    registry?: MemoryRegistryStore;
    assets?: FakeAssetStore;
    classifier?: {
      classify(
        assetId: string,
        file: File,
        signal?: AbortSignal
      ): Promise<ClassifiedAssetFile>;
    };
  } = {}
) {
  const rust = new FakeRustLifecycleBridge();
  (
    runtime as FakeRuntime & {
      createRustLifecycleBridge: () => Promise<RustLifecycleBridge>;
    }
  ).createRustLifecycleBridge = async () => rust as unknown as RustLifecycleBridge;
  return {
    ...createService({ ...overrides, runtime }),
    rust,
  };
}

test('ModelService loads, lists, tracks current, and queries text models', async () => {
  const { service, runtime, assets } = createService();
  const info = await installAndLoad(service, localSource('text-model.gguf'));

  assert.equal(info.status, 'ready');
  assert.equal(info.loaded, true);
  assert.equal(service.current()?.id, info.id);
  assert.equal((await service.list())[0]?.loaded, true);

  const tokens: string[] = [];
  const answer = await service.runQuery(
    'hello',
    {
      tokenBatchSink: (batch) => {
        tokens.push(batch.text);
      },
    }
  );
  assert.equal(answer.text, 'answer:hello');
  assert.deepEqual(tokens, ['token']);
  assert.equal(runtime.lastPrompt, 'hello');
  assert.equal(assets.syncHandleOpenCount, 1);
});

test('ModelService maps common generation options into local prompt options', async () => {
  const { service, runtime } = createService();
  await installAndLoad(service, localSource('text-model.gguf'));

  await service.runQuery('hello', {
    maxTokens: 12,
    temperature: 0.2,
    topP: 0.8,
    sampling: {
      repeat_last_n: 128,
      repeat_penalty: 1.15,
    },
    stop: ['END'],
  });

  const options = runtime.enqueuedOptions.at(-1) as PromptOptions;
  assert.equal(options.nTokens, 12);
  assert.deepEqual(options.sampling, {
    repeat_last_n: 128,
    repeat_penalty: 1.15,
    temperature: 0.2,
    top_p: 0.8,
  });
  assert.deepEqual(options.stop, ['END']);
});

test('ModelService applies default and explicit transcription token limits', async () => {
  const { service, runtime } = createService();
  await installAndLoad(service, localSource('speech.gguf'));

  await service.runListen(new Uint8Array([1, 2, 3]), {});
  assert.equal((runtime.enqueuedOptions.at(-1) as PromptOptions).nTokens, 512);

  await service.runListen(new Uint8Array([4, 5, 6]), { maxTokens: 64 });
  assert.equal((runtime.enqueuedOptions.at(-1) as PromptOptions).nTokens, 64);
  assert.deepEqual(runtime.lastAudio, new Uint8Array([4, 5, 6]));
});

test('ModelService rejects transcription limits outside the native integer range', async () => {
  const { service } = createService();
  await installAndLoad(service, localSource('speech.gguf'));

  await assert.rejects(
    () => service.runListen(new Uint8Array([1]), { maxTokens: 0x80000000 }),
    /Listen maxTokens must be an integer between 1 and 2147483647/
  );
});

test('ModelService forwards and validates the speech duration limit', async () => {
  const { service, runtime } = createService();
  await installAndLoad(service, localSource('speech.gguf'));

  const result = await service.runSpeak('hello', { maxDurationMs: 2_000 });
  assert.equal(runtime.lastMaxDurationMs, 2_000);
  assert.equal(result.durationMs, 80);

  await assert.rejects(
    () => service.runSpeak('hello', { maxDurationMs: 0 }),
    /Speak maxDurationMs must be an integer between 1 and 4294967295/
  );
});

test('ModelService routes operations from the native runtime capability map', async () => {
  const { service, runtime } = createService();
  await installAndLoad(service, localSource('operation-routing.gguf'));
  const session = runtime.currentRuntimeSession();
  assert.ok(session);
  (session.capabilities.operations as { speak: boolean }).speak = false;

  await assert.rejects(
    service.runSpeak('hello', {}),
    (error) =>
      error instanceof QueryError &&
      error.code === 'UNSUPPORTED_OPERATION' &&
      error.message.includes('does not support speak')
  );
  assert.equal(runtime.lastAudio, null);
});

test('ModelService reports request-window browser-to-WASM inference loop time for speech', async () => {
  const { service } = createService();
  await installAndLoad(service, localSource('speech.gguf'), {
    observability: 'runtime',
  });

  await service.runSpeak('hello', {});

  const runtime = service.currentObservability().runtime;
  assert.equal(runtime?.wasmRunLoopCalls, 2);
  assert.equal(runtime?.wasmRunLoopMs, 12.5);
});

test('ModelService uses contextKey as the preferred local text context key', async () => {
  const { service, runtime } = createService();
  await installAndLoad(service, localSource('text-model.gguf'));

  await service.runQuery('hello', {
    contextKey: 'ctx',
  });

  assert.equal(runtime.lastContextKey, 'ctx');
});

test('ModelService.embed returns embedding results without token emission', async () => {
  const { service, runtime } = createService();
  await installAndLoad(service, localSource('embedding-model.gguf'));

  const result = await service.runEmbedding('hello', {
    normalize: false,
    contextKey: 'vectors',
  });

  assert.deepEqual(result.values, [3, 4]);
  assert.equal(result.pooling, 'mean');
  assert.equal(result.normalized, false);
  assert.equal(runtime.lastPrompt, 'hello');
  const options = runtime.enqueuedOptions.at(-1) as { normalize?: boolean; signal?: AbortSignal };
  assert.equal(options.normalize, false);
  assert.equal(options.signal, undefined);
});

test('ModelService routes browser lifecycle through the Rust bridge when available', async () => {
  const runtime = new FakeRuntime();
  const rust = new FakeRustLifecycleBridge();
  (
    runtime as FakeRuntime & {
      createRustLifecycleBridge: () => Promise<RustLifecycleBridge>;
    }
  ).createRustLifecycleBridge = async () => rust as unknown as RustLifecycleBridge;
  const { service, assets } = createService({ runtime });

  const info = await installAndLoad(service, localSource('rust-lifecycle.gguf'), {
    observability: 'runtime',
    runtime: { context: { n_ctx: 1024 } },
  });

  assert.equal(rust.installCount, 1);
  assert.equal(rust.prepareCount, 1);
  assert.deepEqual(rust.lastOptions, {
    backend: 'cpu',
    observability: 'runtime',
    runtime: { context: { n_ctx: 1024, n_threads: 1, n_threads_batch: 1, warmup: false } },
  });
  assert.equal(rust.commitCount, 1);
  assert.equal(info.loaded, true);
  assert.equal(runtime.loadCount, 1);
  assert.equal((await service.list())[0]?.id, info.id);
  assert.deepEqual(assets.deleted, []);
});

test('ModelService keeps a committed runtime when final progress reporting fails', async () => {
  const { service, rust } = createRustBackedService();
  const model = await service.add(localSource('published-runtime.gguf'));

  await assert.rejects(
    service.load(model.id, {
      onProgress: (progress) => {
        if (progress.percent === 100) {
          throw new Error('progress listener failed');
        }
      },
    }),
    /progress listener failed/
  );

  assert.equal(rust.commitCount, 1);
  assert.equal(service.current()?.id, model.id);
});

test('ModelService preserves terminal remote acquisition errors', async () => {
  await withGlobalFetch(
    async () => new Response(null, { status: 503 }),
    async () => {
      const { service, rust } = createRustBackedService();

      await assert.rejects(
        service.add({ kind: 'remote', urls: ['https://example.test/model.gguf'] }),
        (error) =>
          error instanceof QueryError &&
          error.code === 'REMOTE_METADATA_UNAVAILABLE' &&
          error.status === 503 &&
          error.message ===
            'remote metadata is unavailable for https://example.test/model.gguf: HTTP 503'
      );
      assert.equal(rust.remoteAdvanceCount, 4);
      assert.equal(rust.remoteCancelCount, 0);
    }
  );
});

test('ModelService cleans browser remote downloads when classification fails', async () => {
  const remoteBytes = 'remote model bytes';
  await withGlobalFetch(
    async (_input, init) => {
      if (init?.method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: {
            'Content-Length': String(remoteBytes.length),
            ETag: '"remote"',
          },
        });
      }
      return new Response(remoteBytes, { status: 200 });
    },
    async () => {
      const { service, rust, assets } = createRustBackedService(
        new FakeRuntime(),
        { classifier: new FailingAssetClassifier() }
      );
      rust.remoteDownloadMode = true;

      await assert.rejects(
        service.add({ kind: 'remote', urls: ['https://example.test/model.gguf'] }),
        (error) =>
          error instanceof QueryError &&
          error.code === 'REMOTE_LOAD_FAILED' &&
          error.message ===
            'remote model download failed for https://example.test/model.gguf: classification failed'
      );

      assert.equal(rust.remoteCleanupCount, 1);
      assert.deepEqual(assets.deleted, ['asset-model-model.gguf-18']);
      assert.equal(assets.files.size, 0);
    }
  );
});

test('ModelService cleans browser remote downloads when classification is aborted', async () => {
  const remoteBytes = 'remote model bytes';
  await withGlobalFetch(
    async (_input, init) => {
      if (init?.method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: {
            'Content-Length': String(remoteBytes.length),
            ETag: '"remote"',
          },
        });
      }
      return new Response(remoteBytes, { status: 200 });
    },
    async () => {
      const controller = new AbortController();
      const { service, rust, assets } = createRustBackedService(
        new FakeRuntime(),
        { classifier: new AbortingAssetClassifier(controller) }
      );
      rust.remoteDownloadMode = true;

      await assert.rejects(
        service.add(
          { kind: 'remote', urls: ['https://example.test/model.gguf'] },
          { signal: controller.signal }
        ),
        (error) => error instanceof DOMException && error.name === 'AbortError'
      );

      assert.equal(rust.remoteCancelCount, 1);
      assert.deepEqual(assets.deleted, ['asset-model-model.gguf-18']);
      assert.equal(assets.files.size, 0);
    }
  );
});

test('ModelService preserves an aborted install when remote cancellation cleanup fails', async () => {
  const remoteBytes = 'remote model bytes';
  await withGlobalFetch(
    async (_input, init) => {
      if (init?.method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: {
            'Content-Length': String(remoteBytes.length),
            ETag: '"remote"',
          },
        });
      }
      return new Response(remoteBytes, { status: 200 });
    },
    async () => {
      const controller = new AbortController();
      const { service, rust, assets } = createRustBackedService(
        new FakeRuntime(),
        { classifier: new AbortingAssetClassifier(controller) }
      );
      rust.remoteDownloadMode = true;
      rust.remoteCancelError = new Error('cancel failed');

      await assert.rejects(
        service.add(
          { kind: 'remote', urls: ['https://example.test/model.gguf'] },
          { signal: controller.signal }
        ),
        (error) => {
          if (!(error instanceof DOMException) || error.name !== 'AbortError') {
            return false;
          }
          const cleanupFailures = (error as DOMException & {
            readonly cleanupFailures?: AggregateError;
          }).cleanupFailures;
          assert.ok(cleanupFailures instanceof AggregateError);
          assert.equal(cleanupFailures.errors.length, 1);
          assert.match(
            String(cleanupFailures.errors[0]),
            /cancel remote acquisition: cancel failed/
          );
          return true;
        }
      );

      assert.equal(rust.remoteCancelCount, 1);
      assert.deepEqual(assets.deleted, ['asset-model-model.gguf-18']);
      assert.equal(assets.files.size, 0);
    }
  );
});

test('ModelService skips browser split cleanup for direct local loads', async () => {
  const { service, assets } = createService();

  await service.add(localSource('direct-load.gguf'));

  assert.equal(assets.cleanupCount, 0);
});

test('ModelService cleans browser split artifacts before split-capable local loads', async () => {
  const { service, assets } = createService();
  assets.forceBrowserSplit = true;

  await service.add(localSource('split-capable.gguf'));

  assert.equal(assets.cleanupCount, 1);
});

test('ModelService opens each OPFS model handle once on the graceful path', async () => {
  const { service, assets, registry } = createService();
  assets.forceBrowserSplit = true;
  const model = await service.add(localSource('graceful-split.gguf'));
  const manifest = await registry.read();
  const modelFiles = manifest.models[model.id]?.modelAssetIds ?? [];

  await service.load(model.id);

  assert.equal(modelFiles.length, 2);
  assert.equal(assets.syncHandleOpenCount, modelFiles.length);
  assert.equal(assets.syncHandleCloseCount, modelFiles.length);
});

test('ModelService preserves nullish OPFS handle failures', async () => {
  for (const failure of [null, undefined]) {
    const { service, assets } = createService();
    const model = await service.add(localSource('nullish-failure.gguf'));
    assets.syncHandleFailures.set(1, failure);
    const notCaught = Symbol('not caught');
    let caught: unknown = notCaught;

    try {
      await service.load(model.id);
    } catch (error) {
      caught = error;
    }

    assert.notEqual(caught, notCaught);
    assert.equal(caught, failure);
  }
});

test('ModelService retries the whole bundle after a transient OPFS lock', async () => {
  const delays: number[] = [];
  const { service, assets } = createService({
    sleep: async (delayMs) => {
      delays.push(delayMs);
    },
  });
  assets.forceBrowserSplit = true;
  const model = await service.add(localSource('locked-split.gguf'));
  assets.exclusiveLockFailureCalls.add(2);

  await service.load(model.id);

  assert.equal(assets.syncHandleOpenCount, 4);
  assert.equal(assets.syncHandleCloseCount, 3);
  assert.deepEqual(delays, [25]);
});

test('ModelService backs off across repeated OPFS lock failures', async () => {
  const delays: number[] = [];
  const { service, assets } = createService({
    sleep: async (delayMs) => {
      delays.push(delayMs);
    },
  });
  assets.forceBrowserSplit = true;
  const model = await service.add(localSource('locked-twice.gguf'));
  assets.exclusiveLockFailureCalls.add(2);
  assets.exclusiveLockFailureCalls.add(4);

  await service.load(model.id);

  // The production delay table is asserted directly; the injected sleep only
  // removes the waiting, not the policy.
  assert.deepEqual(delays, [25, 50]);
  assert.equal(assets.syncHandleOpenCount, 6);
});

test('ModelService stops retrying an OPFS lock once the delay table is exhausted', async () => {
  const delays: number[] = [];
  const { service, assets } = createService({
    sleep: async (delayMs) => {
      delays.push(delayMs);
    },
  });
  assets.forceBrowserSplit = true;
  const model = await service.add(localSource('locked-forever.gguf'));
  for (let call = 2; call <= 24; call += 2) {
    assets.exclusiveLockFailureCalls.add(call);
  }

  await assert.rejects(service.load(model.id), /exclusively locked/);

  assert.deepEqual(delays, [25, 50, 100, 200, 400]);
});

test('ModelService defaults browser pthread runtime thread counts before Rust prepare', async () => {
  await withNavigatorHardwareConcurrency(12, async () => {
    const runtime = new FakeRuntime();
    runtime.wasmThreadingMode = 'pthread';
    const { service, rust } = createRustBackedService(runtime);

    await installAndLoad(service, localSource('pthread-defaults.gguf'), {
      runtime: { context: { n_ctx: 1024, n_threads: 2 } },
    });

    assert.deepEqual(rust.lastOptions, {
      backend: 'cpu',
      observability: 'off',
      runtime: {
        context: {
          n_ctx: 1024,
          n_threads: 2,
          n_threads_batch: 4,
          warmup: false,
        },
      },
    });
  });
});

test('ModelService auto-selects WebGPU when the browser has a shader-f16 adapter', async () => {
  await withNavigatorGpu(async () => ({ features: { has: () => true } }), async () => {
    const { service, rust } = createRustBackedService();

    await installAndLoad(service, localSource('webgpu-auto.gguf'));

    assert.equal(
      (rust.lastOptions as { backend?: BrowserBackendPreference }).backend,
      'webgpu'
    );
    assert.equal(
      (rust.lastOptions as { runtime?: NativeRuntimeConfig }).runtime?.context?.n_ctx,
      undefined
    );
  });
});

test('ModelService makes the CPU-only runtime constraint dominant over auto-selection', async () => {
  await withNavigatorGpu(async () => ({ features: { has: () => true } }), async () => {
    const runtime = new FakeRuntime();
    runtime.backendConstraint = 'cpu-only';
    const { service, rust } = createRustBackedService(runtime);

    await installAndLoad(service, localSource('cpu-runtime-override.gguf'));

    assert.equal(
      (rust.lastOptions as { backend?: BrowserBackendPreference }).backend,
      'cpu'
    );
  });
});

test('ModelService rejects an explicit WebGPU backend on the CPU-only runtime', async () => {
  let adapterRequests = 0;
  await withNavigatorGpu(async () => {
    adapterRequests += 1;
    return { features: { has: () => true } };
  }, async () => {
    const runtime = new FakeRuntime();
    runtime.backendConstraint = 'cpu-only';
    const { service } = createRustBackedService(runtime);

    await assert.rejects(
      installAndLoad(service, localSource('explicit-webgpu.gguf'), { backend: 'webgpu' }),
      (error) =>
        error instanceof QueryError &&
        error.code === 'UNSUPPORTED_OPERATION' &&
        error.message.includes('did not pass the JSPI suspend/resume probe')
    );
    assert.equal(adapterRequests, 0);
  });
});

test('ModelService leaves omitted CPU context sizing to Rust lifecycle', async () => {
  await withNavigatorGpu(async () => ({ features: { has: () => false } }), async () => {
    const { service, rust } = createRustBackedService();

    await installAndLoad(service, localSource('webgpu-auto-no-f16.gguf'));

    assert.equal(
      (rust.lastOptions as { backend?: BrowserBackendPreference }).backend,
      'cpu'
    );
    assert.equal(
      (rust.lastOptions as { runtime?: NativeRuntimeConfig }).runtime?.context?.n_ctx,
      undefined
    );
  });
});

test('ModelService.chat renders chat templates and sanitizes assistant boundaries', async () => {
  const { service, runtime } = createService();
  await installAndLoad(service, localSource('text-model.gguf'));
  runtime.streamedTokens = ['Hello ', 'there</assistant>\n<user>ignored'];
  runtime.nextOutputText = 'Hello there</assistant>\n<user>ignored';

  const tokens: string[] = [];
  const answer = await service.runChat(
    [
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: 'Say hello.' },
    ],
    {
      tokenBatchSink: (batch) => {
        tokens.push(batch.text);
      },
    }
  );

  assert.equal(answer.text, 'Hello there');
  assert.deepEqual(tokens, ['Hello there']);
  assert.match(runtime.lastPrompt ?? '', /<system>\nBe concise\.<\/system>/);
  assert.match(runtime.lastPrompt ?? '', /<user>\nSay hello\.<\/user>/);
  assert.ok(runtime.lastPrompt?.endsWith('<assistant>\n'));
});

test('ModelService.chat keeps token emission off when a token sink is not requested', async () => {
  const { service, runtime } = createService();
  await installAndLoad(service, localSource('text-model.gguf'));
  runtime.nextOutputText = 'Hello there</assistant>\n<user>ignored';

  const answer = await service.runChat(
    [
      { role: 'user', content: 'Say hello.' },
    ],
    {}
  );

  const options = runtime.enqueuedOptions.at(-1);
  assert.equal(answer.text, 'Hello there');
  assert.equal(typeof options, 'object');
  assert.equal((options as PromptOptions).tokenBatchSink, undefined);
});

test('ModelService passes token sinks to the runtime when token emission is requested', async () => {
  const { service, runtime } = createService();
  await installAndLoad(service, localSource('text-model.gguf'));

  await service.runQuery('hello', {
    tokenBatchSink: () => {},
  });

  const options = runtime.enqueuedOptions.at(-1);
  assert.equal(typeof options, 'object');
  assert.equal(typeof (options as PromptOptions).tokenBatchSink, 'function');
});

test('ModelService supplies the active model fact to the Rust removal policy', async () => {
  const { service, rust } = createRustBackedService();
  const model = await installAndLoad(service, localSource('active-model.gguf'));

  await assert.rejects(
    service.remove(model.id),
    (error) => error instanceof QueryError && error.code === 'MODEL_IN_USE'
  );

  assert.equal(rust.removeCount, 1);
  assert.equal(service.current()?.id, model.id);
});

test('ModelService surfaces OPFS unavailable as a storage error', async () => {
  const service = new ModelService(new FakeRuntime());
  await assert.rejects(
    () => service.add(localSource('requires-opfs.gguf')),
    (error) => error instanceof QueryError && error.code === 'STORAGE_UNAVAILABLE'
  );
});
