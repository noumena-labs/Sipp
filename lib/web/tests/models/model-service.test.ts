import test from 'node:test';
import assert from 'node:assert/strict';
import { ModelService } from '../../src/models/model-service.js';
import { AssetStore } from '../../src/models/asset-store.js';
import { ModelRegistryStore } from '../../src/models/model-registry-store.js';
import {
  type ClassifiedAsset,
  type ClassifiedAssetFile,
  type PairingPlan,
  RuntimePairingValidationError,
  type InternalBundleDescriptor,
  type ModelDetectionResult,
  type StagedModelBundle,
  type StageModelBundleOptions,
  QueryError,
  type AssetRecord,
  type BrowserBackendPreference,
  type ModelEntry,
  type ModelInfo,
  type ObservabilityEvent,
  type ObservabilitySnapshot,
  type RegistryManifest,
} from '../../src/models/types.js';
import type { EngineRuntime } from '../../src/runtime/engine-runtime.js';
import type { RustLifecycleBridge } from '../../src/wasm/wasm-bridge.js';
import type {
  BackendObservability,
  ChatMessage,
  EmbedRuntimeOptions,
  EngineExecutionMode,
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
    version: 3,
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
  public readonly remotes = new Map<
    string,
    {
      etag: string;
      lastModified: string;
      file: File;
    }
  >();

  public async resolveRemoteMetadata(rawUrl: string): Promise<{
    url: string;
    canonicalUrl: string;
    name: string;
    bytes: number;
    etag: string;
    lastModified: string;
  }> {
    const remote = this.remotes.get(rawUrl);
    if (remote == null) {
      throw new QueryError('REMOTE_METADATA_UNAVAILABLE', `No fake remote for ${rawUrl}.`);
    }
    return {
      url: rawUrl,
      canonicalUrl: rawUrl,
      name: remote.file.name,
      bytes: remote.file.size,
      etag: remote.etag,
      lastModified: remote.lastModified,
    };
  }

  public async downloadRemote(
    metadata: { canonicalUrl: string; etag: string; lastModified: string },
    kind: AssetRecord['kind']
  ): Promise<AssetRecord> {
    const remote = this.remotes.get(metadata.canonicalUrl);
    if (remote == null) {
      throw new QueryError('REMOTE_LOAD_FAILED', `No fake remote for ${metadata.canonicalUrl}.`);
    }
    return this.installFile({
      kind,
      file: remote.file,
      sourceUrl: metadata.canonicalUrl,
      sourceEtag: metadata.etag,
      sourceLastModified: metadata.lastModified,
    });
  }

  public async downloadRemoteSplitGguf(metadata: {
    canonicalUrl: string;
    etag: string;
    lastModified: string;
    bytes: number;
  }): Promise<AssetRecord[]> {
    const record = await this.downloadRemote(metadata, 'model');
    if (metadata.bytes <= FakeAssetStore.directLoadMaxBytes) {
      return [record];
    }
    throw new Error('Large fake remote GGUF splitting is not implemented.');
  }

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

  public async installLocalSplitGguf(file: File): Promise<AssetRecord[]> {
    if (file.size <= FakeAssetStore.directLoadMaxBytes) {
      return [await this.installFile({ kind: 'model', file })];
    }

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
      close: () => {},
      getSize: () => bytes.byteLength,
    };
    return { name: record.name, handle, size: bytes.byteLength };
  }

  public async delete(record: AssetRecord): Promise<void> {
    this.deleted.push(record.id);
    this.files.delete(record.id);
  }

  public requiresBrowserSplit(bytes: number): boolean {
    return this.forceBrowserSplit || bytes > FakeAssetStore.directLoadMaxBytes;
  }

  public async cleanupBrowserSplitArtifacts(): Promise<void> {
    this.cleanupCount += 1;
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
        version: 1,
        role: isProjector ? 'projector' : 'model',
        architecture: visionCapable ? 'vision-test' : 'text-test',
        visionCapable,
        compatibleVisionProjectorTypes: visionCapable ? ['vision-merger'] : [],
        providedVisionProjectorType: isProjector ? 'vision-merger' : null,
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

function resolveFakePairing(
  files: readonly ClassifiedAsset[],
  explicitProjectorId: string | null
): PairingPlan {
  if (files.length === 0) {
    throw new RuntimePairingValidationError(
      'INVALID_MODEL_SOURCE',
      'No model assets were provided.'
    );
  }

  const projectors = files.filter((file) => file.inspection.role === 'projector');
  if (explicitProjectorId == null && projectors.length > 1) {
    throw new RuntimePairingValidationError(
      'INVALID_MODEL_PAIRING',
      `Multiple projector assets were provided: ${projectors.map((file) => file.name).join(', ')}.`
    );
  }

  const projector =
    explicitProjectorId == null
      ? projectors[0] ?? null
      : files.find((file) => file.assetId === explicitProjectorId) ?? null;
  if (explicitProjectorId != null && projector == null) {
    throw new RuntimePairingValidationError(
      'INVALID_MODEL_PAIRING',
      'Explicit projector asset was not installed.'
    );
  }
  if (projector != null && projector.inspection.role !== 'projector') {
    throw new RuntimePairingValidationError(
      'INVALID_MODEL_PAIRING',
      `"${projector.name}" is not a projector asset.`
    );
  }

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
    if (explicitProjectorId == null && !base.inspection.visionCapable) {
      throw new RuntimePairingValidationError(
        'INVALID_MODEL_PAIRING',
        'Projector assets can only be auto-paired with vision-capable models.'
      );
    }
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
  public nextLoadError: Error | null = null;
  public stagedDescriptors: InternalBundleDescriptor[] = [];
  public lastPrompt: string | null = null;
  public lastContextKey: string | null = null;
  public mediaMarker: string | null = null;
  public nextOutputText: string | null = null;
  public streamedTokens: string[] = ['token'];
  public enqueuedOptions: Array<number | PromptOptions | EmbedRuntimeOptions | undefined> = [];
  public stageGate: Promise<void> | null = null;
  private runtimeMetricsEnabled = false;
  private backendProfilingEnabled = false;
  private nextRequestId = 1;
  private readonly queuedRequests = new Map<
    GenerateRequestId,
    {
      promptText: string;
      options?: number | PromptOptions | EmbedRuntimeOptions;
      embedding?: boolean;
      normalize?: boolean;
    }
  >();

  public getExecutionMode(): EngineExecutionMode {
    return 'main-thread';
  }

  public getWasmThreadingMode(): 'single-thread' | 'pthread' {
    return this.wasmThreadingMode;
  }

  public getTransportObservability(): TransportObservability {
    return {
      executionMode: 'main-thread',
      workerBacked: false,
      enabled: this.runtimeMetricsEnabled,
      activeTokenTransport: 'none',
    };
  }

  public async initModule(): Promise<void> {}

  public async detectModelFromGgufFile(file: Blob & { name?: string }): Promise<ModelDetectionResult> {
    const name = file.name ?? 'model.gguf';
    const isProjector = /mmproj|projector/i.test(name);
    const visionCapable = !isProjector && /vision|llava/i.test(name);
    const inspection = {
      version: 1 as const,
      role: isProjector ? 'projector' as const : 'model' as const,
      architecture: visionCapable ? 'vision-test' : 'text-test',
      visionCapable,
      compatibleVisionProjectorTypes: visionCapable ? ['vision-merger'] : [],
      providedVisionProjectorType: isProjector ? 'vision-merger' : null,
    };
    return {
      inspection,
      detectionMethod: 'gguf-metadata',
      modelName: name,
      modelType: null,
      modelArchitecture: inspection.architecture,
    };
  }

  public async resolvePairing(
    classified: readonly ClassifiedAsset[],
    explicitProjectorId?: string | null
  ): Promise<PairingPlan> {
    return resolveFakePairing(classified, explicitProjectorId ?? null);
  }

  public async stageModelBundle(
    descriptor: InternalBundleDescriptor,
    _options?: StageModelBundleOptions
  ): Promise<StagedModelBundle> {
    this.stagedDescriptors.push(descriptor);
    if (this.stageGate != null) {
      await this.stageGate;
    }
    for (const shard of descriptor.shards) {
      try {
        shard.handle.close();
      } catch {}
    }
    const projector = descriptor.projector;
    return {
      sourceKind: 'installed',
      modelPath: `/models/${this.stagedDescriptors.length}.gguf`,
      projectorPath: projector == null ? null : '/models/mmproj.gguf',
      isVisionModel: descriptor.detection.inspection.visionCapable,
      projectorStatus: projector == null
        ? descriptor.detection.inspection.visionCapable
          ? 'missing'
          : 'not-required'
        : 'paired',
      modelName: descriptor.detection.modelName,
      detectionMethod: descriptor.detection.detectionMethod,
      modelType: descriptor.detection.modelType,
      modelArchitecture: descriptor.detection.modelArchitecture,
    };
  }

  public async loadRuntimeModel(
    modelPathOrBundle: string | StagedModelBundle,
    config?: NativeRuntimeConfig
  ): Promise<void> {
    this.loadCount += 1;
    this.runtimeMetricsEnabled = config?.observability?.runtime_metrics === true;
    this.backendProfilingEnabled = config?.observability?.backend_profiling === true;
    if (this.nextLoadError != null) {
      const error = this.nextLoadError;
      this.nextLoadError = null;
      this.mediaMarker = null;
      throw error;
    }
    this.mediaMarker =
      typeof modelPathOrBundle === 'string' || modelPathOrBundle.projectorPath == null
        ? null
        : '<image>';
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

  public close(): void {
    this.closeCount += 1;
    this.mediaMarker = null;
  }

  public readMediaMarker(): string | null {
    return this.mediaMarker;
  }

  public async cancelQuery(_requestId: GenerateRequestId): Promise<boolean> {
    return true;
  }

  public async enqueueQuery(
    contextKey: string,
    promptText: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestId> {
    const requestId = this.nextRequestId++;
    this.lastContextKey = contextKey;
    this.lastPrompt = promptText;
    this.enqueuedOptions.push(options);
    this.queuedRequests.set(requestId, { promptText, options });
    if (typeof options === 'object' && this.streamedTokens.length > 0) {
      const text = this.streamedTokens.join('');
      options.tokenBatchSink?.({
        requestId: String(requestId),
        streamId: requestId,
        sequenceStart: 0,
        text,
        frameCount: this.streamedTokens.length,
        byteCount: new TextEncoder().encode(text).byteLength,
        stats: {
          framesSent: this.streamedTokens.length,
          bytesSent: new TextEncoder().encode(text).byteLength,
          batchesSent: 1,
          drainMs: 0,
          drainCalls: 0,
        },
      });
    }
    return requestId;
  }

  public async enqueueChat(
    contextKey: string,
    messages: readonly ChatMessage[],
    options?: number | PromptOptions
  ): Promise<GenerateRequestId> {
    return this.enqueueQuery(contextKey, this.renderNativeChatPrompt(messages, true), options);
  }

  public async enqueueEmbedding(
    contextKey: string,
    input: string,
    options?: EmbedRuntimeOptions
  ): Promise<GenerateRequestId> {
    const requestId = this.nextRequestId++;
    this.lastContextKey = contextKey;
    this.lastPrompt = input;
    this.enqueuedOptions.push(options);
    this.queuedRequests.set(requestId, {
      promptText: input,
      options,
      embedding: true,
      normalize: options?.normalize ?? true,
    });
    return requestId;
  }

  public async awaitQuery(requestId: GenerateRequestId): Promise<GenerateResponse> {
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
  public prepareCount = 0;
  public commitCount = 0;
  public removeCount = 0;
  public lastSource: unknown = null;
  public lastOptions: unknown = null;
  private manifest: RegistryManifest = {
    version: 3,
    projectorIndexRevision: 0,
    assets: {},
    models: {},
  };
  private currentModelId: string | null = null;

  public list(): ModelInfo[] {
    return Object.values(this.manifest.models).map((entry) =>
      this.toModelInfo(entry, this.currentModelId === entry.id)
    );
  }

  public prepareLoad(
    source: {
      kind: 'assets';
      assets: AssetRecord[];
      classified: ClassifiedAsset[];
    },
    options: {
      backend?: BrowserBackendPreference;
      runtime?: NativeRuntimeConfig;
      observability?: 'off' | 'runtime' | 'profile';
    }
  ): {
    loadId: string;
    model: ModelInfo;
    runtimeFingerprint: string;
    runtimeConfig: NativeRuntimeConfig;
    loadRequired: boolean;
    assets: Array<{ assetId: string; kind: AssetRecord['kind']; storagePath: string; mountName: string; bytes: number }>;
    projector: null;
    manifest: RegistryManifest;
    snapshot: ObservabilitySnapshot;
    events: ObservabilityEvent[];
  } {
    this.prepareCount += 1;
    this.lastSource = source;
    this.lastOptions = options;
    const asset = source.assets[0];
    assert.ok(asset);
    this.manifest.assets[asset.id] = {
      ...asset,
      refCount: 1,
      inspection: source.classified[0]?.inspection ?? asset.inspection,
    };
    const modelId = `model-${asset.id}`;
    const now = new Date(0).toISOString();
    this.manifest.models[modelId] = {
      id: modelId,
      name: asset.name,
      modality: 'text',
      status: 'ready',
      modelAssetIds: [asset.id],
      runtimeFingerprint: 'runtime-fingerprint',
      createdAt: now,
      updatedAt: now,
    };
    const model = this.toModelInfo(this.manifest.models[modelId], false);
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
      loadRequired: true,
      assets: [
        {
          assetId: asset.id,
          kind: asset.kind,
          storagePath: asset.storagePath,
          mountName: asset.name,
          bytes: asset.bytes,
        },
      ],
      projector: null,
      manifest: cloneManifest(this.manifest),
      snapshot,
      events: [{ type: 'load-start', snapshot }],
    };
  }

  public commitLoad(): {
    model: ModelInfo;
    manifest: RegistryManifest;
    snapshot: ObservabilitySnapshot;
    events: ObservabilityEvent[];
  } {
    this.commitCount += 1;
    const entry = Object.values(this.manifest.models)[0];
    assert.ok(entry);
    const loadedAt = new Date(1).toISOString();
    entry.updatedAt = loadedAt;
    entry.lastLoadedAt = loadedAt;
    this.currentModelId = entry.id;
    const model = this.toModelInfo(entry, true);
    const snapshot = this.snapshot('ready', model, 'runtime');
    return {
      model,
      manifest: cloneManifest(this.manifest),
      snapshot,
      events: [{ type: 'load-complete', snapshot }],
    };
  }

  public abortLoad(error: { message?: string }): ObservabilitySnapshot {
    return {
      ...this.snapshot('error', null, 'off'),
      query: {
        contextKey: null,
        status: 'failed',
        wallMs: null,
        ttftMs: null,
        outputTokens: null,
        errorMessage: error.message,
      },
    };
  }

  public remove(modelId: string): {
    removed: ModelEntry;
    orphanedAssets: AssetRecord[];
    manifest: RegistryManifest;
    snapshot: ObservabilitySnapshot;
    events: ObservabilityEvent[];
  } {
    this.removeCount += 1;
    const removed = this.manifest.models[modelId];
    assert.ok(removed);
    delete this.manifest.models[modelId];
    const orphanedAssets = removed.modelAssetIds
      .map((assetId) => this.manifest.assets[assetId])
      .filter((asset): asset is AssetRecord => asset != null);
    for (const asset of orphanedAssets) {
      delete this.manifest.assets[asset.id];
    }
    this.currentModelId = null;
    const snapshot = this.snapshot('idle', null, 'off');
    return {
      removed,
      orphanedAssets,
      manifest: cloneManifest(this.manifest),
      snapshot,
      events: [{ type: 'load-complete', snapshot }],
    };
  }

  public unload(): ObservabilitySnapshot {
    this.currentModelId = null;
    return this.snapshot('idle', null, 'off');
  }

  public close(): void {}

  public drainEvents(): ObservabilityEvent[] {
    return [];
  }

  private toModelInfo(entry: ModelEntry, loaded: boolean): ModelInfo {
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
      loaded,
      chatTemplate: loaded ? 'fake-template' : null,
      bosText: loaded ? '<s>' : '',
      eosText: loaded ? '</s>' : '',
      mediaMarker: null,
      createdAt: entry.createdAt,
      updatedAt: entry.updatedAt,
    };
  }

  private snapshot(
    state: ObservabilitySnapshot['state'],
    model: ModelInfo | null,
    mode: ObservabilitySnapshot['mode']
  ): ObservabilitySnapshot {
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
      overrides.classifier ?? new FakeAssetClassifier()
    ),
    runtime,
    registry,
    assets,
  };
}

test('ModelService loads, lists, tracks current, and queries text models', async () => {
  const { service, runtime } = createService();
  const info = await service.load(file('text-model.gguf'));

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
});

test('ModelService maps common generation options into local prompt options', async () => {
  const { service, runtime } = createService();
  await service.load(file('text-model.gguf'));

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

test('ModelService uses contextKey as the preferred local text context key', async () => {
  const { service, runtime } = createService();
  await service.load(file('text-model.gguf'));

  await service.runQuery('hello', {
    contextKey: 'ctx',
  });

  assert.equal(runtime.lastContextKey, 'ctx');
});

test('ModelService.embed returns embedding results without token emission', async () => {
  const { service, runtime } = createService();
  await service.load(file('embedding-model.gguf'));

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

  const info = await service.load(file('rust-lifecycle.gguf'), {
    observability: 'runtime',
    runtime: { context: { n_ctx: 1024 } },
  });

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
  await service.remove(info.id);
  assert.equal(rust.removeCount, 1);
  assert.deepEqual(assets.deleted, ['asset-model-rust-lifecycle.gguf-19']);
});

test('ModelService skips browser split cleanup for direct local loads', async () => {
  const { service, assets } = createService();

  await service.load(file('direct-load.gguf'));

  assert.equal(assets.cleanupCount, 0);
});

test('ModelService cleans browser split artifacts before split-capable local loads', async () => {
  const { service, assets } = createService();
  assets.forceBrowserSplit = true;

  await service.load(file('split-capable.gguf'));

  assert.equal(assets.cleanupCount, 1);
});

test('ModelService defaults browser pthread runtime thread counts before Rust prepare', async () => {
  await withNavigatorHardwareConcurrency(12, async () => {
    const runtime = new FakeRuntime();
    runtime.wasmThreadingMode = 'pthread';
    const rust = new FakeRustLifecycleBridge();
    (
      runtime as FakeRuntime & {
        createRustLifecycleBridge: () => Promise<RustLifecycleBridge>;
      }
    ).createRustLifecycleBridge = async () => rust as unknown as RustLifecycleBridge;
    const { service } = createService({ runtime });

    await service.load(file('pthread-defaults.gguf'), {
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
    const runtime = new FakeRuntime();
    const rust = new FakeRustLifecycleBridge();
    (
      runtime as FakeRuntime & {
        createRustLifecycleBridge: () => Promise<RustLifecycleBridge>;
      }
    ).createRustLifecycleBridge = async () => rust as unknown as RustLifecycleBridge;
    const { service } = createService({ runtime });

    await service.load(file('webgpu-auto.gguf'));

    assert.equal(
      (rust.lastOptions as { backend?: BrowserBackendPreference }).backend,
      'webgpu'
    );
  });
});

test('ModelService auto-selects CPU when the adapter lacks shader-f16', async () => {
  await withNavigatorGpu(async () => ({ features: { has: () => false } }), async () => {
    const runtime = new FakeRuntime();
    const rust = new FakeRustLifecycleBridge();
    (
      runtime as FakeRuntime & {
        createRustLifecycleBridge: () => Promise<RustLifecycleBridge>;
      }
    ).createRustLifecycleBridge = async () => rust as unknown as RustLifecycleBridge;
    const { service } = createService({ runtime });

    await service.load(file('webgpu-auto-no-f16.gguf'));

    assert.equal(
      (rust.lastOptions as { backend?: BrowserBackendPreference }).backend,
      'cpu'
    );
  });
});

test('ModelService.chat renders chat templates and sanitizes assistant boundaries', async () => {
  const { service, runtime } = createService();
  await service.load(file('text-model.gguf'));
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
  await service.load(file('text-model.gguf'));
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
  await service.load(file('text-model.gguf'));

  await service.runQuery('hello', {
    tokenBatchSink: () => {},
  });

  const options = runtime.enqueuedOptions.at(-1);
  assert.equal(typeof options, 'object');
  assert.equal(typeof (options as PromptOptions).tokenBatchSink, 'function');
});

test('ModelService removes current models and deletes orphaned assets', async () => {
  const { service, runtime, assets } = createService();
  const info = await service.load(file('remove-me.gguf'));

  await service.remove(info.id);
  assert.equal(service.current(), null);
  assert.equal(runtime.closeCount, 1);
  assert.equal(assets.deleted.length, 1);
  assert.deepEqual(await service.list(), []);
});

test('ModelService rejects queries during lifecycle transitions and serializes concurrent loads', async () => {
  let releaseStage!: () => void;
  const runtime = new FakeRuntime();
  runtime.stageGate = new Promise<void>((resolve) => {
    releaseStage = resolve;
  });
  const { service } = createService({ runtime });

  const firstLoad = service.load(file('slow.gguf'));
  await new Promise((resolve) => setTimeout(resolve, 0));
  await assert.rejects(
    () => service.runQuery('too early', {}),
    (error) => error instanceof QueryError && error.code === 'MODEL_NOT_READY'
  );

  const secondLoad = service.load(file('next.gguf'));
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(runtime.stagedDescriptors.length, 1);

  runtime.stageGate = null;
  releaseStage();
  await firstLoad;
  await secondLoad;
  assert.equal(runtime.stagedDescriptors.length, 2);
});

test('ModelService surfaces OPFS unavailable as a storage error', async () => {
  const service = new ModelService(new FakeRuntime());
  await assert.rejects(
    () => service.load(file('requires-opfs.gguf')),
    (error) => error instanceof QueryError && error.code === 'STORAGE_UNAVAILABLE'
  );
});
