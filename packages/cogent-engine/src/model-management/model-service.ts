import type { EngineRuntime } from '../runtime/engine-runtime.js';
import type {
  GenerateResponse,
  InferenceInitConfig,
  ModelBundleFileProjectorDescriptor,
  InternalBundleDescriptor,
  PromptOptions,
} from '../types.js';
import { AssetStore, type RemoteAssetMetadata } from './asset-store.js';
import { sha256Text } from './hash.js';
import { ModelRegistryStore } from './model-registry-store.js';
import { PairingValidator, type ClassifiedAssetFile, type PairingPlan } from './pairing-validator.js';
import {
  QueryError,
  toPromptFormatMode,
  type AssetRecord,
  type LoadedModelState,
  type ModelEntry,
  type ModelInfo,
  type ModelLoadOptions,
  type ModelRuntimeOptions,
  type ModelSource,
  type ObservabilityEvent,
  type ObservabilityMode,
  type ObservabilitySnapshot,
  type QueryObservation,
  type QueryInput,
  type QueryOptions,
  type RegistryManifest,
} from './model-types.js';
import type { ModelLifecycleService } from './model-service-contract.js';
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

function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(',')}]`;
  }
  if (value != null && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => `${JSON.stringify(key)}:${stableJson(entry)}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
}

function isSourceObject(source: ModelSource): source is Extract<ModelSource, { model: BaseSource }> {
  return typeof source === 'object' && source != null && !isFile(source) && !Array.isArray(source);
}

function toRuntimeConfig(
  options: ModelRuntimeOptions | undefined,
  mode: ObservabilityMode
): InferenceInitConfig {
  const effectiveOptions = applyObservabilityMode(options, mode);
  return {
    nCtx: effectiveOptions.nCtx,
    nBatch: effectiveOptions.nBatch,
    nUbatch: effectiveOptions.nUbatch,
    nSeqMax: effectiveOptions.nSeqMax,
    nThreads: effectiveOptions.nThreads,
    nThreadsBatch: effectiveOptions.nThreadsBatch,
    nGpuLayers: effectiveOptions.nGpuLayers,
    flashAttention: effectiveOptions.flashAttention,
    kvUnified: effectiveOptions.kvUnified,
    maxCachedSessions: effectiveOptions.maxCachedSessions,
    retainedPrefixTokens: effectiveOptions.retainedPrefixTokens,
    prefillChunkSize: effectiveOptions.prefillChunkSize,
    prefixCacheIntervalTokens: effectiveOptions.prefixCacheIntervalTokens,
    maxPrefixCacheEntries: effectiveOptions.maxPrefixCacheEntries,
    schedulerPolicy: effectiveOptions.schedulerPolicy,
    decodeTokenReserve: effectiveOptions.decodeTokenReserve,
    adaptivePrefillChunking: effectiveOptions.adaptivePrefillChunking,
    enableRuntimeObservability: effectiveOptions.enableRuntimeObservability,
    enableBackendProfiling: effectiveOptions.enableBackendProfiling,
    multimodalUseGpu: effectiveOptions.multimodalUseGpu,
    debugCompareMultimodalEmbeddings: effectiveOptions.debugCompareMultimodalEmbeddings,
    imageMinTokens: effectiveOptions.imageMinTokens,
    imageMaxTokens: effectiveOptions.imageMaxTokens,
    sampling: effectiveOptions.sampling,
  };
}

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

export class ModelService implements ModelLifecycleService {
  private current: LoadedModelState | null = null;
  private operationChain: Promise<void> = Promise.resolve();
  private transitioning = false;
  private readonly observability = new ObservabilityController();

  constructor(
    private readonly runtime: EngineRuntime,
    private readonly registry = new ModelRegistryStore(),
    private readonly assetStore = new AssetStore(),
    private readonly pairingValidator = new PairingValidator()
  ) {}

  public currentModel(): ModelInfo | null {
    const current = this.current;
    if (current == null) {
      return null;
    }
    return this.currentSnapshot ?? null;
  }

  private currentSnapshot: ModelInfo | null = null;

  public async list(): Promise<ModelInfo[]> {
    const manifest = await this.registry.read();
    return Object.values(manifest.models)
      .sort((left, right) => left.createdAt.localeCompare(right.createdAt))
      .map((entry) => this.toModelInfo(entry, manifest));
  }

  public currentObservability(): ObservabilitySnapshot {
    return this.observability.current();
  }

  public subscribeObservability(listener: (event: ObservabilityEvent) => void): () => void {
    return this.observability.subscribe(listener);
  }

  public async load(source: ModelSource, options: ModelLoadOptions = {}): Promise<ModelInfo> {
    return this.withLifecycleLock(async () => {
      if (options.signal?.aborted) {
        throw new DOMException('Model load aborted.', 'AbortError');
      }

      const observabilityMode = resolveObservabilityMode(options.observability);
      const runtimeFingerprint = sha256Text(
        stableJson({
          runtime: options.runtime ?? {},
          observability: observabilityMode,
        })
      );
      this.observability.emit('load-start', {
        mode: observabilityMode,
        state: 'loading',
        query: null,
        runtime: undefined,
        profile: undefined,
      });
      try {
        const manifest = await this.registry.read();
        const existing = this.resolveInstalledModel(manifest, source);
        if (existing != null && !isSourceObject(source)) {
          return await this.loadEntry(existing, runtimeFingerprint, options, observabilityMode);
        }

        const installed = await this.installSource(source, manifest, options);
        await this.registerAssets(installed.assets);
        const classified = await this.classifyAssets(installed.assets, options.signal);
        const plan = await this.resolvePairingPlan(classified, installed.assets, options.signal);
        const entry = await this.upsertModelEntry(plan, runtimeFingerprint);
        const modelInfo = await this.loadEntry(entry, runtimeFingerprint, options, observabilityMode);
        if (installed.source === 'remote') {
          return modelInfo;
        }
        return modelInfo;
      } catch (error) {
        this.observability.emit('error', {
          state: 'error',
          query: {
            session: null,
            status: isAbortError(error) ? 'cancelled' : 'failed',
            wallMs: null,
            ttftMs: null,
            outputTokenCount: null,
            errorCode: error instanceof QueryError ? error.code : undefined,
            errorMessage: error instanceof Error ? error.message : String(error),
          },
        });
        throw error;
      }
    });
  }

  public async remove(id: string): Promise<void> {
    await this.withLifecycleLock(async () => {
      const manifest = await this.registry.read();
      const entry = manifest.models[id];
      if (entry == null) {
        throw new QueryError('MODEL_NOT_FOUND', `Model "${id}" is not installed.`);
      }
      if (this.current?.id === id) {
        this.runtime.close();
        this.current = null;
        this.currentSnapshot = null;
      }

      const orphanedAssets: AssetRecord[] = [];
      await this.registry.write((draft) => {
        const removed = draft.models[id];
        if (removed == null) {
          return;
        }
        delete draft.models[id];
        for (const assetId of [...removed.modelAssetIds, removed.projectorAssetId].filter(
          (value): value is string => typeof value === 'string'
        )) {
          const asset = draft.assets[assetId];
          if (asset == null) {
            continue;
          }
          asset.refCount = Math.max(0, asset.refCount - 1);
          if (asset.refCount === 0) {
            orphanedAssets.push(asset);
            delete draft.assets[assetId];
          }
        }
      });
      for (const asset of orphanedAssets) {
        await this.assetStore.delete(asset);
      }
      this.observability.update({
        state: this.currentSnapshot == null ? 'idle' : 'ready',
        model: this.currentSnapshot,
        query: null,
      });
    });
  }

  public async query(input: QueryInput, options: QueryOptions = {}): Promise<string> {
    if (this.transitioning) {
      throw new QueryError('MODEL_NOT_READY', 'A model lifecycle transition is in progress.');
    }
    if (this.current == null) {
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
    const promptOptions: PromptOptions = {
      nTokens: options.maxTokens,
      promptFormat: toPromptFormatMode(options.format),
      signal: options.signal,
      onToken: options.onToken,
      media,
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
        outputTokenCount: null,
      },
    });
    let failureRecorded = false;
    try {
      const requestId = await this.runtime.enqueueQuery(session, prompt, promptOptions);
      const response = await this.runtime.awaitQuery(requestId, { signal: options.signal });
      if (response.cancelled) {
        const error = new DOMException(response.errorMessage ?? 'Queued request cancelled.', 'AbortError');
        this.recordQueryFailure(session, start, error, response);
        failureRecorded = true;
        throw error;
      }
      if (response.failed) {
        const error = new Error(response.errorMessage ?? 'Queued prompt failed.');
        this.recordQueryFailure(session, start, error, response);
        failureRecorded = true;
        throw error;
      }
      this.recordQuerySuccess(session, start, response);
      return response.outputText;
    } catch (error) {
      if (!failureRecorded) {
        this.recordQueryFailure(session, start, error);
      }
      if (error instanceof QueryError) {
        throw error;
      }
      throw new QueryError(
        'QUERY_FAILED',
        error instanceof Error && error.message.trim().length > 0
          ? `Model query failed: ${error.message}`
          : 'Model query failed.',
        { cause: error }
      );
    }
  }

  public close(): void {
    this.runtime.close();
    this.current = null;
    this.currentSnapshot = null;
    this.observability.markClosed();
  }

  private recordQuerySuccess(
    session: string,
    start: number,
    response: GenerateResponse
  ): void {
    const metrics = response.requestObservability ?? response.runtimeObservability ?? null;
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
    const metrics = response?.requestObservability ?? response?.runtimeObservability ?? null;
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
    const metrics = response?.requestObservability ?? response?.runtimeObservability ?? null;
    return {
      session,
      status,
      wallMs: Math.max(0, nowMs() - start),
      ttftMs: metrics?.ttftMs ?? null,
      outputTokenCount: metrics?.outputTokenCount ?? null,
    };
  }

  private async installSource(
    source: ModelSource,
    manifest: RegistryManifest,
    options: ModelLoadOptions
  ): Promise<SourceInstallResult> {
    if (isSourceObject(source)) {
      const base = await this.installBaseSource(source.model, manifest, options);
      const projector =
        source.projector == null
          ? null
          : await this.installProjectorSource(source.projector, manifest, options);
      return {
        assets: [...base.assets, ...(projector?.assets ?? [])],
        source: base.source,
      };
    }
    return this.installBaseSource(source, manifest, options);
  }

  private async installBaseSource(
    source: BaseSource,
    manifest: RegistryManifest,
    options: ModelLoadOptions
  ): Promise<SourceInstallResult> {
    if (typeof source === 'string') {
      const installed = manifest.models[source];
      if (installed != null) {
        return await this.assetsForEntry(installed, manifest);
      }
      return {
        assets: [await this.installRemoteAsset(source, 'model', manifest, options)],
        source: 'remote',
      };
    }
    if (isFile(source)) {
      return {
        assets: [await this.installLocalAsset(source, 'model', manifest, options)],
        source: 'local',
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
      };
    }
    throw new QueryError('INVALID_MODEL_SOURCE', 'Unsupported model source.');
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
      };
    }
    if (isFile(source)) {
      return {
        assets: [await this.installLocalAsset(source, 'projector', manifest, options)],
        source: 'local',
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

  private async installLocalAsset(
    file: File,
    kind: AssetRecord['kind'],
    manifest: RegistryManifest,
    options: ModelLoadOptions
  ): Promise<InstalledAsset> {
    const record = await this.assetStore.installFile({
      kind,
      file,
      onProgress: options.onProgress,
    });
    const existing = manifest.assets[record.id];
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

  private async assetsForEntry(
    entry: ModelEntry,
    manifest: RegistryManifest
  ): Promise<SourceInstallResult> {
    const assetIds = [...entry.modelAssetIds, entry.projectorAssetId].filter(
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
    };
  }

  private async classifyAssets(
    assets: InstalledAsset[],
    signal?: AbortSignal
  ): Promise<ClassifiedAssetFile[]> {
    return Promise.all(
      assets.map((asset) => this.pairingValidator.classify(asset.record.id, asset.file, signal))
    );
  }

  private async registerAssets(assets: InstalledAsset[]): Promise<void> {
    await this.registry.write((draft) => {
      for (const installed of assets) {
        const existing = draft.assets[installed.record.id];
        if (existing == null) {
          draft.assets[installed.record.id] = installed.record;
          continue;
        }
        draft.assets[installed.record.id] = {
          ...existing,
          kind: installed.record.kind === 'projector' ? 'projector' : existing.kind,
          sourceUrl: installed.record.sourceUrl ?? existing.sourceUrl,
          sourceEtag: installed.record.sourceEtag ?? existing.sourceEtag,
          sourceLastModified: installed.record.sourceLastModified ?? existing.sourceLastModified,
        };
      }
    });
  }

  private async resolvePairingPlan(
    classified: ClassifiedAssetFile[],
    assets: InstalledAsset[],
    signal?: AbortSignal
  ): Promise<PairingPlan> {
    const explicitProjector = classified.find((file) =>
      assets.some((asset) => asset.record.id === file.assetId && asset.record.kind === 'projector')
    );
    const plan = this.pairingValidator.resolve(classified, explicitProjector?.assetId);
    if (plan.status === 'needs_projector') {
      const manifest = await this.registry.read();
      const installedProjectors = Object.values(manifest.assets).filter(
        (asset) => asset.kind === 'projector'
      );
      if (installedProjectors.length === 1) {
        const projector = installedProjectors[0];
        const file = await this.assetStore.getFile(projector);
        return this.pairingValidator.resolve(
          [...classified, await this.pairingValidator.classify(projector.id, file, signal)],
          projector.id
        );
      }
      if (installedProjectors.length > 1) {
        throw new QueryError(
          'INVALID_MODEL_PAIRING',
          'Multiple installed projectors are available. Provide an explicit projector.'
        );
      }
    }
    return plan;
  }

  private async upsertModelEntry(
    plan: PairingPlan,
    runtimeFingerprint: string
  ): Promise<ModelEntry> {
    const id = `model-${sha256Text(
      stableJson({
        modelAssetIds: plan.modelAssetIds,
        projectorAssetId: plan.projectorAssetId ?? null,
      })
    ).slice(0, 24)}`;
    const now = new Date().toISOString();
    let entry!: ModelEntry;
    await this.registry.write((draft) => {
      for (const assetId of [...plan.modelAssetIds, plan.projectorAssetId].filter(
        (value): value is string => typeof value === 'string'
      )) {
        const asset = draft.assets[assetId];
        if (asset == null) {
          continue;
        }
        draft.assets[assetId] = asset;
      }

      const existing = draft.models[id];
      if (existing == null) {
        entry = {
          id,
          name: plan.name,
          modality: plan.modality,
          status: plan.status,
          modelAssetIds: plan.modelAssetIds,
          projectorAssetId: plan.projectorAssetId,
          runtimeFingerprint,
          createdAt: now,
          updatedAt: now,
        };
        draft.models[id] = entry;
        for (const assetId of [...plan.modelAssetIds, plan.projectorAssetId].filter(
          (value): value is string => typeof value === 'string'
        )) {
          const asset = draft.assets[assetId];
          if (asset != null) {
            asset.refCount += 1;
          }
        }
      } else {
        existing.name = plan.name;
        existing.modality = plan.modality;
        existing.status = plan.status;
        existing.runtimeFingerprint = runtimeFingerprint;
        existing.updatedAt = now;
        entry = existing;
      }
    });
    return entry;
  }

  private async loadEntry(
    entry: ModelEntry,
    runtimeFingerprint: string,
    options: ModelLoadOptions,
    observabilityMode: ObservabilityMode
  ): Promise<ModelInfo> {
    if (entry.status === 'broken') {
      throw new QueryError('MODEL_BROKEN', `Installed model "${entry.id}" is broken.`);
    }
    if (entry.status === 'needs_projector') {
      const manifest = await this.registry.read();
      const info = this.toModelInfo(entry, manifest);
      this.observability.emit('load-complete', {
        mode: observabilityMode,
        state: 'ready',
        model: info,
      });
      return info;
    }
    if (this.current?.id === entry.id && this.current.runtimeFingerprint === runtimeFingerprint) {
      const manifest = await this.registry.read();
      const info = this.toModelInfo(entry, manifest);
      const runtime = toRuntimeObservation(
        this.runtime.getRuntimeObservability(),
        this.runtime.getTransportObservability()
      );
      const profile =
        observabilityMode === 'profile'
          ? toBackendProfileObservation(await this.runtime.getBackendObservability())
          : undefined;
      this.observability.emit('load-complete', {
        mode: observabilityMode,
        state: 'ready',
        model: info,
        ...(runtime == null ? {} : { runtime }),
        ...(profile == null ? {} : { profile }),
      });
      return info;
    }

    const manifest = await this.registry.read();
    const files = await this.filesForEntry(entry, manifest);
    const descriptor = this.buildDescriptor(files.modelFiles, files.projectorFile);
    options.onProgress?.({
      phase: 'load',
      loadedBytes: 0,
      totalBytes: null,
      percent: null,
      assetName: entry.name,
    });
    const staged = await this.runtime.stageModelBundle(descriptor, {
      signal: options.signal,
    });
    await this.runtime.loadRuntimeModel(staged, toRuntimeConfig(options.runtime, observabilityMode));

    const loadedAt = new Date().toISOString();
    const updated = await this.registry.write((draft) => {
      const next = draft.models[entry.id];
      if (next != null) {
        next.lastLoadedAt = loadedAt;
        next.runtimeFingerprint = runtimeFingerprint;
        next.updatedAt = loadedAt;
      }
    });
    const loadedEntry = updated.models[entry.id] ?? entry;
    this.current = {
      id: loadedEntry.id,
      runtimeFingerprint,
    };
    this.currentSnapshot = this.toModelInfo(loadedEntry, updated);
    options.onProgress?.({
      phase: 'load',
      loadedBytes: 1,
      totalBytes: 1,
      percent: 100,
      assetName: entry.name,
    });
    const runtime = toRuntimeObservation(
      this.runtime.getRuntimeObservability(),
      this.runtime.getTransportObservability()
    );
    const profile =
      observabilityMode === 'profile'
        ? toBackendProfileObservation(await this.runtime.getBackendObservability())
        : undefined;
    this.observability.emit('load-complete', {
      mode: observabilityMode,
      state: 'ready',
      model: this.currentSnapshot,
      ...(runtime == null ? {} : { runtime }),
      ...(profile == null ? {} : { profile }),
    });
    return this.currentSnapshot;
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

  private buildDescriptor(modelFiles: File[], projectorFile: File | null): InternalBundleDescriptor {
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
      };
    }
    return {
      kind: 'files',
      files: modelFiles,
      projector,
    };
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

  private toModelInfo(entry: ModelEntry, manifest: RegistryManifest): ModelInfo {
    const assets = [...entry.modelAssetIds, entry.projectorAssetId]
      .filter((assetId): assetId is string => typeof assetId === 'string')
      .map((assetId) => manifest.assets[assetId])
      .filter((asset): asset is AssetRecord => asset != null);
    return {
      id: entry.id,
      name: entry.name,
      modality: entry.modality,
      status: entry.status,
      source: assets.some((asset) => asset.sourceUrl != null) ? 'remote' : 'local',
      bytes: assets.reduce((sum, asset) => sum + asset.bytes, 0),
      loaded: this.current?.id === entry.id,
      createdAt: entry.createdAt,
      updatedAt: entry.updatedAt,
    };
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
}
