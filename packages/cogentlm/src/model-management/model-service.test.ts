import test from 'node:test';
import assert from 'node:assert/strict';
import { ModelService } from './model-service.js';
import { AssetStore } from './asset-store.js';
import { ModelRegistryStore } from './model-registry-store.js';
import { PairingValidator } from './pairing-validator.js';
import {
  QueryError,
  type AssetRecord,
  type ModelEntry,
  type RegistryManifest,
} from './model-types.js';
import type { EngineRuntime } from '../runtime/engine-runtime.js';
import type {
  BackendObservability,
  EngineExecutionMode,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  InternalBundleDescriptor,
  PromptOptions,
  RequestObservabilityMetrics,
  RuntimeAggregateObservabilityMetrics,
  StagedModelBundle,
  StageModelBundleOptions,
  TransportObservability,
} from '../types.js';

function file(name: string, contents = name): File {
  return new File([contents], name);
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
  public readonly files = new Map<string, File>();
  public readonly deleted: string[] = [];
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
      hash: id,
      bytes: input.file.size,
      storagePath: id,
      sourceUrl: input.sourceUrl,
      sourceEtag: input.sourceEtag,
      sourceLastModified: input.sourceLastModified,
      refCount: 0,
      createdAt: new Date(0).toISOString(),
    };
  }

  public async getFile(record: AssetRecord): Promise<File> {
    const stored = this.files.get(record.id);
    if (stored == null) {
      throw new QueryError('MODEL_BROKEN', `Missing fake asset ${record.id}.`);
    }
    return stored;
  }

  public async delete(record: AssetRecord): Promise<void> {
    this.deleted.push(record.id);
    this.files.delete(record.id);
  }
}

class FakePairingValidator extends PairingValidator {
  public override async classify(assetId: string, input: File): Promise<{
    assetId: string;
    file: File;
    inspection: {
      version: 1;
      role: 'model' | 'projector';
      architecture: string | null;
      visionCapable: boolean;
      compatibleVisionProjectorTypes: string[];
      providedVisionProjectorType: string | null;
    };
    name: string;
  }> {
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

class MetadataLimitedPairingValidator extends PairingValidator {
  public override async classify(assetId: string, input: File): Promise<{
    assetId: string;
    file: File;
    inspection: {
      version: 1;
      role: 'model' | 'projector';
      architecture: string | null;
      visionCapable: boolean;
      compatibleVisionProjectorTypes: string[];
      providedVisionProjectorType: string | null;
    };
    name: string;
  }> {
    const isProjector = /mmproj|projector/i.test(input.name);
    return {
      assetId,
      file: input,
      inspection: {
        version: 1,
        role: isProjector ? 'projector' : 'model',
        architecture: isProjector ? 'clip' : 'llama',
        visionCapable: false,
        compatibleVisionProjectorTypes: [],
        providedVisionProjectorType: null,
      },
      name: input.name,
    };
  }
}

class IncompatibleProjectorValidator extends FakePairingValidator {
  public override async classify(assetId: string, input: File): Promise<{
    assetId: string;
    file: File;
    inspection: {
      version: 1;
      role: 'model' | 'projector';
      architecture: string | null;
      visionCapable: boolean;
      compatibleVisionProjectorTypes: string[];
      providedVisionProjectorType: string | null;
    };
    name: string;
  }> {
    const classified = await super.classify(assetId, input);
    if (/bad-mmproj/i.test(input.name)) {
      classified.inspection.providedVisionProjectorType = 'other-merger';
    }
    return classified;
  }
}

class FakeRuntime implements EngineRuntime {
  public closeCount = 0;
  public loadCount = 0;
  public nextLoadError: Error | null = null;
  public stagedDescriptors: InternalBundleDescriptor[] = [];
  public lastPrompt: string | null = null;
  public mediaMarker: string | null = null;
  public nextOutputText: string | null = null;
  public streamedTokens: string[] = ['token'];
  public stageGate: Promise<void> | null = null;
  private runtimeMetricsEnabled = false;
  private backendProfilingEnabled = false;
  private nextRequestId = 1;
  private readonly queuedRequests = new Map<
    GenerateRequestId,
    {
      promptText: string;
      options?: number | PromptOptions;
    }
  >();

  public getExecutionMode(): EngineExecutionMode {
    return 'main-thread';
  }

  public getTransportObservability(): TransportObservability {
    return {
      executionMode: 'main-thread',
      workerBacked: false,
      enabled: this.runtimeMetricsEnabled,
      bufferedTokenLimit: 0,
      flushIntervalMs: 0,
      flushCount: 0,
      coalescedTokenCount: 0,
      maxObservedBufferedTokenCount: 0,
      activeTokenTransport: 'none',
    };
  }

  public async initModule(): Promise<void> {}

  public async stageModelBundle(
    descriptor: InternalBundleDescriptor,
    _options?: StageModelBundleOptions
  ): Promise<StagedModelBundle> {
    this.stagedDescriptors.push(descriptor);
    if (this.stageGate != null) {
      await this.stageGate;
    }
    const projector = 'projector' in descriptor ? descriptor.projector : undefined;
    return {
      sourceKind: descriptor.kind,
      modelPath: `/models/${this.stagedDescriptors.length}.gguf`,
      multimodalProjectorPath: projector == null ? null : '/models/mmproj.gguf',
      isVisionModel: projector != null,
      projectorStatus: projector == null ? 'not-required' : 'explicit',
      modelName:
        descriptor.kind === 'file'
          ? descriptor.file.name
          : descriptor.kind === 'files'
            ? descriptor.files[0]?.name ?? 'model.gguf'
            : 'model.gguf',
      detectionMethod: 'gguf-metadata',
      modelType: null,
      modelArchitecture: null,
    };
  }

  public async loadRuntimeModel(
    modelPathOrBundle: string | StagedModelBundle,
    config?: InferenceInitConfig
  ): Promise<void> {
    this.loadCount += 1;
    this.runtimeMetricsEnabled = config?.enableRuntimeObservability === true;
    this.backendProfilingEnabled = config?.enableBackendProfiling === true;
    if (this.nextLoadError != null) {
      const error = this.nextLoadError;
      this.nextLoadError = null;
      this.mediaMarker = null;
      throw error;
    }
    this.mediaMarker =
      typeof modelPathOrBundle === 'string' || modelPathOrBundle.multimodalProjectorPath == null
        ? null
        : '<image>';
  }

  public async applyChatTemplate(
    messages: Array<{ role: string; content: string }>,
    addAssistant: boolean
  ): Promise<string> {
    const rendered = messages
      .map((message) => `<${message.role}>\n${message.content}</${message.role}>\n`)
      .join('');
    return `${rendered}${addAssistant ? '<assistant>\n' : ''}`;
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
    _contextKey: string,
    promptText: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestId> {
    const requestId = this.nextRequestId++;
    this.lastPrompt = promptText;
    this.queuedRequests.set(requestId, { promptText, options });
    if (typeof options === 'object' && this.streamedTokens.length > 0) {
      options.onToken?.(this.streamedTokens);
    }
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

  public getRuntimeObservability(): RuntimeAggregateObservabilityMetrics | null {
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
      cacheHits: 0,
    };
  }
}

function createService(overrides: {
  runtime?: FakeRuntime;
  registry?: MemoryRegistryStore;
  assets?: FakeAssetStore;
  validator?: PairingValidator;
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
      overrides.validator ?? new FakePairingValidator()
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
  assert.equal(service.currentModel()?.id, info.id);
  assert.equal((await service.list())[0]?.loaded, true);

  const tokens: string[] = [];
  const answer = await service.query('hello', {
    onToken: (batch) => tokens.push(...batch),
  });
  assert.equal(answer, 'answer:hello');
  assert.deepEqual(tokens, ['token']);
  assert.equal(runtime.lastPrompt, 'hello');
});

test('ModelService.chat renders chat templates and sanitizes assistant boundaries', async () => {
  const { service, runtime } = createService();
  await service.load(file('text-model.gguf'));
  runtime.streamedTokens = ['Hello ', 'there</assistant>\n<user>ignored'];
  runtime.nextOutputText = 'Hello there</assistant>\n<user>ignored';

  const tokens: string[] = [];
  const answer = await service.chat(
    [
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: 'Say hello.' },
    ],
    {
      onToken: (batch) => tokens.push(...batch),
    }
  );

  assert.equal(answer, 'Hello there');
  assert.deepEqual(tokens, ['Hello ', 'there']);
  assert.match(runtime.lastPrompt ?? '', /<system>\nBe concise\.<\/system>/);
  assert.match(runtime.lastPrompt ?? '', /<user>\nSay hello\.<\/user>/);
  assert.ok(runtime.lastPrompt?.endsWith('<assistant>\n'));
});

test('ModelService keeps observability off by default', async () => {
  const { service } = createService();
  await service.load(file('text-model.gguf'));
  await service.query('hello');

  const snapshot = service.currentObservability();
  assert.equal(snapshot.mode, 'off');
  assert.equal(snapshot.state, 'ready');
  assert.equal(snapshot.query?.status, 'success');
  assert.equal(snapshot.runtime, undefined);
  assert.equal(snapshot.profile, undefined);
});

test('ModelService captures runtime observability without backend profile data', async () => {
  const { service } = createService();
  const loaded = await service.load(file('runtime-model.gguf'), { observability: 'runtime' });
  await service.query('hello');
  await service.load(loaded.id, { observability: 'runtime' });

  const snapshot = service.currentObservability();
  assert.equal(snapshot.mode, 'runtime');
  assert.equal(snapshot.runtime?.outputTokens, 5);
  assert.equal(snapshot.profile, undefined);
});

test('ModelService emits lifecycle observability and captures runtime/profile modes', async () => {
  const { service } = createService();
  const events: string[] = [];
  const unsubscribe = service.subscribeObservability((event) => {
    events.push(event.type);
  });

  await service.load(file('profiled-model.gguf'), { observability: 'profile' });
  await service.query('hello');

  const snapshot = service.currentObservability();
  assert.equal(snapshot.mode, 'profile');
  assert.equal(snapshot.state, 'ready');
  assert.equal(snapshot.query?.status, 'success');
  assert.equal(snapshot.runtime?.tokensPerSecond, 100);
  assert.equal(snapshot.runtime?.execution.mode, 'main-thread');
  assert.equal(snapshot.profile?.profilingEnabled, true);
  assert.deepEqual(events, ['load-start', 'load-complete', 'query-start', 'query-complete']);

  service.close();
  unsubscribe();
  assert.equal(service.currentObservability().state, 'closed');
  assert.deepEqual(events, [
    'load-start',
    'load-complete',
    'query-start',
    'query-complete',
    'close',
  ]);
});

test('ModelService switches models and reuses identical runtime fingerprints as no-ops', async () => {
  const { service, runtime } = createService();
  const first = await service.load(file('first.gguf'), { runtime: { nCtx: 1024 } });
  await service.load(first.id, { runtime: { nCtx: 1024 } });
  assert.equal(runtime.loadCount, 1);

  await service.load(first.id, { runtime: { nCtx: 2048 } });
  assert.equal(runtime.loadCount, 2);

  const second = await service.load(file('second.gguf'));
  assert.notEqual(second.id, first.id);
  assert.equal(service.currentModel()?.id, second.id);
  assert.equal(runtime.loadCount, 3);
});

test('ModelService attaches explicit projectors and rejects metadata-proven mismatches', async () => {
  const { service, runtime } = createService();
  const pendingVision = await service.load(file('vision-base.gguf'));
  assert.equal(pendingVision.status, 'needs_projector');
  assert.equal(pendingVision.modality, 'vision');
  assert.equal(pendingVision.loaded, false);
  assert.equal(service.currentModel(), null);

  const vision = await service.load({
    model: pendingVision.id,
    projector: file('mmproj.gguf'),
  });
  assert.equal(vision.id, pendingVision.id);
  assert.equal(vision.modality, 'vision');
  assert.equal(vision.status, 'ready');
  assert.equal(vision.loaded, true);

  const answer = await service.query({
    prompt: 'describe',
    media: [new Uint8Array([1, 2, 3])],
  });
  assert.equal(answer, 'answer:<image>\ndescribe');
  assert.equal(runtime.lastPrompt, '<image>\ndescribe');

  const mismatch = createService({
    validator: new IncompatibleProjectorValidator(),
  });
  const text = await mismatch.service.load(file('vision-base.gguf'));
  await assert.rejects(
    () =>
      mismatch.service.load({
        model: text.id,
        projector: file('bad-mmproj.gguf'),
      }),
    (error) => error instanceof QueryError && error.code === 'INVALID_MODEL_PAIRING'
  );
});

test('ModelService switches from text to explicit multimodal loads when metadata pairing is inconclusive', async () => {
  const { service, runtime } = createService({
    validator: new MetadataLimitedPairingValidator(),
  });

  const text = await service.load(file('plain-text.gguf'));
  assert.equal(text.modality, 'text');
  assert.equal(service.currentModel()?.id, text.id);

  const vision = await service.load({
    model: file('ambiguous-vision-base.gguf'),
    projector: file('ambiguous-mmproj.gguf'),
  });

  assert.equal(vision.modality, 'vision');
  assert.equal(vision.status, 'ready');
  assert.equal(vision.loaded, true);
  assert.equal(runtime.loadCount, 2);
  assert.equal(service.currentModel()?.id, vision.id);

  const answer = await service.query({
    prompt: 'describe',
    media: [new Uint8Array([1, 2, 3])],
  });
  assert.equal(answer, 'answer:<image>\ndescribe');
  assert.equal(runtime.lastPrompt, '<image>\ndescribe');
});

test('ModelService auto-retries unresolved vision bases when the projector index changes', async () => {
  const { service } = createService();
  const pending = await service.load(file('vision-base.gguf'));
  assert.equal(pending.status, 'needs_projector');

  await service.load({
    model: file('other-vision.gguf'),
    projector: file('mmproj.gguf'),
  });

  const resolved = await service.load(pending.id);
  assert.equal(resolved.id, pending.id);
  assert.equal(resolved.status, 'ready');
  assert.equal(resolved.loaded, true);
});

test('ModelService persists validated projector pairings across service instances', async () => {
  const registry = new MemoryRegistryStore();
  const assets = new FakeAssetStore();

  const first = createService({ registry, assets });
  const installed = await first.service.load({
    model: file('vision-base.gguf'),
    projector: file('mmproj.gguf'),
  });
  assert.equal(installed.status, 'ready');
  assert.equal(installed.loaded, true);

  const second = createService({ registry, assets });
  const reloaded = await second.service.load(installed.id);
  assert.equal(reloaded.id, installed.id);
  assert.equal(reloaded.status, 'ready');
  assert.equal(reloaded.loaded, true);
});

test('ModelService replaces the projector on an installed model without reusing the old one', async () => {
  const { service, runtime } = createService();
  const first = await service.load({
    model: file('vision-base.gguf'),
    projector: file('mmproj-a.gguf'),
  });
  assert.equal(runtime.loadCount, 1);

  const second = await service.load({
    model: first.id,
    projector: file('mmproj-b.gguf'),
  });
  assert.equal(second.id, first.id);
  assert.equal(second.status, 'ready');
  assert.equal(runtime.loadCount, 2);
});

test('ModelService restores the previous installed pairing when a replacement projector fails to load', async () => {
  const { service, runtime } = createService();
  const installed = await service.load({
    model: file('vision-base.gguf'),
    projector: file('mmproj-a.gguf'),
  });
  assert.equal(service.currentModel()?.id, installed.id);

  runtime.nextLoadError = new Error('multimodal init failed');
  await assert.rejects(
    () =>
      service.load({
        model: installed.id,
        projector: file('mmproj-b.gguf'),
      }),
    /multimodal init failed/
  );

  assert.equal(service.currentModel(), null);

  const reloaded = await service.load(installed.id);
  assert.equal(reloaded.id, installed.id);
  assert.equal(reloaded.status, 'ready');
  assert.equal(reloaded.loaded, true);
});

test('ModelService updates remote models when validators change', async () => {
  const { service, assets } = createService();
  assets.remotes.set('https://models.test/model.gguf', {
    etag: '"one"',
    lastModified: 'Mon, 01 Jan 2024 00:00:00 GMT',
    file: file('remote-one.gguf'),
  });
  const first = await service.load('https://models.test/model.gguf');

  assets.remotes.set('https://models.test/model.gguf', {
    etag: '"two"',
    lastModified: 'Tue, 02 Jan 2024 00:00:00 GMT',
    file: file('remote-two.gguf'),
  });
  const second = await service.load('https://models.test/model.gguf');
  assert.notEqual(second.id, first.id);
  assert.equal(service.currentModel()?.id, second.id);
});

test('ModelService removes current models and deletes orphaned assets', async () => {
  const { service, runtime, assets } = createService();
  const info = await service.load(file('remove-me.gguf'));

  await service.remove(info.id);
  assert.equal(service.currentModel(), null);
  assert.equal(runtime.closeCount, 1);
  assert.equal(assets.deleted.length, 1);
  assert.deepEqual(await service.list(), []);
});

test('ModelService marks installed entries broken when assets are missing', async () => {
  const registry = new MemoryRegistryStore();
  const broken: ModelEntry = {
    id: 'model-broken',
    name: 'broken.gguf',
    modality: 'text',
    status: 'ready',
    modelAssetIds: ['asset-missing'],
    createdAt: new Date(0).toISOString(),
    updatedAt: new Date(0).toISOString(),
  };
  registry.manifest.models[broken.id] = broken;
  const { service } = createService({ registry });

  await assert.rejects(
    () => service.load(broken.id),
    (error) => error instanceof QueryError && error.code === 'MODEL_BROKEN'
  );
  assert.equal(registry.manifest.models[broken.id]?.status, 'broken');
});

test('ModelService marks installed entries broken when cached asset files are missing', async () => {
  const registry = new MemoryRegistryStore();
  const asset: AssetRecord = {
    id: 'asset-corrupt-file',
    kind: 'model',
    name: 'corrupt.gguf',
    hash: 'asset-corrupt-file',
    bytes: 12,
    storagePath: 'asset-corrupt-file-corrupt.gguf',
    refCount: 1,
    createdAt: new Date(0).toISOString(),
  };
  const broken: ModelEntry = {
    id: 'model-corrupt-file',
    name: 'corrupt.gguf',
    modality: 'text',
    status: 'ready',
    modelAssetIds: [asset.id],
    createdAt: new Date(0).toISOString(),
    updatedAt: new Date(0).toISOString(),
  };
  registry.manifest.assets[asset.id] = asset;
  registry.manifest.models[broken.id] = broken;
  const { service } = createService({ registry });

  await assert.rejects(
    () => service.load(broken.id),
    (error) => error instanceof QueryError && error.code === 'MODEL_BROKEN'
  );
  assert.equal(registry.manifest.models[broken.id]?.status, 'broken');
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
    () => service.query('too early'),
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
