import type { EngineRuntime } from '../runtime/engine-runtime.js';
import {
  ChatTemplatePromptRuntime,
  StreamingBoundaryTextSanitizer,
} from '../core/chat-template-boundaries.js';
import { sliceUnstreamedSuffix } from '../core/streaming-output.js';
import type {
  GenerateResponse,
  InferenceInitConfig,
  ModelBundleFileProjectorDescriptor,
  ModelDetectionResult,
  InternalBundleDescriptor,
  PromptOptions,
} from '../types.js';
import { createLinkedAbortController, isAbortError } from '../utils/abort.js';
import { stableJson } from '../utils/stable-json.js';
import { AssetStore, type RemoteAssetMetadata } from './asset-store.js';
import { sha256Text } from './hash.js';
import { ModelRegistryStore } from './model-registry-store.js';
import { PairingValidator, type ClassifiedAssetFile, type PairingPlan } from './pairing-validator.js';
import {
  QueryError,
  type AssetRecord,
  type ChatInput,
  type ChatOptions,
  type LoadedModelState,
  type ModelEntry,
  type ModelInfo,
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

function entryAssetFingerprint(entry: Pick<ModelEntry, 'modelAssetIds' | 'projectorAssetId'>): string {
  return sha256Text(
    stableJson({
      modelAssetIds: [...entry.modelAssetIds].sort((left, right) => left.localeCompare(right)),
      projectorAssetId: entry.projectorAssetId ?? null,
    })
  );
}

function entryAssetIds(entry: Pick<ModelEntry, 'modelAssetIds' | 'projectorAssetId'>): string[] {
  return [...entry.modelAssetIds, entry.projectorAssetId].filter(
    (value): value is string => typeof value === 'string'
  );
}

function cloneModelEntry(entry: ModelEntry): ModelEntry {
  return JSON.parse(JSON.stringify(entry)) as ModelEntry;
}

function normalizeVisionProjectorTypes(projectorTypes: readonly string[]): string[] {
  return [...new Set(projectorTypes)].sort((left, right) => left.localeCompare(right));
}

function sameVisionProjectorTypes(left: readonly string[], right: readonly string[]): boolean {
  const normalizedLeft = normalizeVisionProjectorTypes(left);
  const normalizedRight = normalizeVisionProjectorTypes(right);
  return (
    normalizedLeft.length === normalizedRight.length &&
    normalizedLeft.every((value, index) => value === normalizedRight[index])
  );
}

export class ModelService implements ModelLifecycleService {
  private current: LoadedModelState | null = null;
  private chatRuntime: ChatTemplatePromptRuntime | null = null;
  private chatRuntimeKey: string | null = null;
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
          const basePlan = await this.deriveBasePlanForEntry(existing, manifest, options.signal);
          const prepared = await this.resolveEntryForLoading(existing, basePlan, options.signal);
          return await this.loadEntry(prepared, runtimeFingerprint, options, observabilityMode);
        }

        const installed = await this.installSource(source, manifest, options);
        const classified = await this.classifyAssets(installed.assets, options.signal);
        await this.registerAssets(installed.assets, classified);
        const sourceProjectorAssetId = this.resolveSourceProjectorAssetId(
          classified,
          installed.explicitProjectorAssetId
        );
        const basePlan = this.pairingValidator.resolve(
          classified.filter((file) => file.assetId !== sourceProjectorAssetId)
        );
        let entry = await this.upsertBaseModelEntry(basePlan, runtimeFingerprint);

        if (sourceProjectorAssetId != null) {
          const previousEntry = cloneModelEntry(entry);
          try {
            const explicitPlan = this.pairingValidator.resolveExplicit(
              classified,
              sourceProjectorAssetId
            );
            entry = await this.setResolvedProjector(
              entry.id,
              explicitPlan.projectorAssetId!,
              explicitPlan.compatibleVisionProjectorTypes
            );
            return await this.loadEntry(entry, runtimeFingerprint, options, observabilityMode);
          } catch (error) {
            await this.restoreEntry(previousEntry);
            throw error;
          }
        }

        const prepared = await this.resolveEntryForLoading(entry, basePlan, options.signal);
        return await this.loadEntry(prepared, runtimeFingerprint, options, observabilityMode);
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
        let projectorIndexChanged = false;
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
            if (asset.kind === 'projector') {
              projectorIndexChanged = true;
            }
            delete draft.assets[assetId];
          }
        }
        if (projectorIndexChanged) {
          draft.projectorIndexRevision += 1;
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
      signal: options.signal,
      onToken: options.onToken,
      media,
      grammar: options.grammar,
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

  public async chat(input: ChatInput, options: ChatOptions = {}): Promise<string> {
    if (this.transitioning) {
      throw new QueryError('MODEL_NOT_READY', 'A model lifecycle transition is in progress.');
    }
    if (this.current == null) {
      throw new QueryError('MODEL_NOT_READY', 'No model is loaded. Call engine.models.load(...) first.');
    }

    const messages = isChatInputObject(input) ? input.messages : input;
    const media = isChatInputObject(input) ? input.media : undefined;
    const promptContext = await this.getChatRuntime(this.current).render(messages);
    const outputSanitizer = new StreamingBoundaryTextSanitizer(promptContext.boundaryMarkers);
    const linkedAbort = createLinkedAbortController(options.signal);
    let streamedOutputText = '';
    let assistantText = '';
    let stoppedAtBoundary = false;

    const consumeOutputText = (text: string): void => {
      if (text.length === 0 || outputSanitizer.reachedBoundary) {
        return;
      }
      streamedOutputText += text;
      const result = outputSanitizer.consume(text);
      if (result.safeText.length > 0) {
        assistantText += result.safeText;
        options.onToken?.(result.safeText);
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
        options.onToken?.(safeText);
      }
    };

    try {
      const rawText = await this.query(
        {
          prompt: promptContext.promptText,
          ...(media != null && media.length > 0 ? { media } : {}),
        },
        {
          ...options,
          signal: linkedAbort.signal,
          onToken: consumeOutputText,
        }
      );
      const unseenOutputSuffix = sliceUnstreamedSuffix(streamedOutputText, rawText);
      if (!outputSanitizer.reachedBoundary && unseenOutputSuffix.length > 0) {
        consumeOutputText(unseenOutputSuffix);
      }
      flushOutputText();
      return assistantText.trim();
    } catch (error) {
      if (stoppedAtBoundary && options.signal?.aborted !== true) {
        flushOutputText();
        return assistantText.trim();
      }
      throw error;
    } finally {
      linkedAbort.dispose();
    }
  }

  public async applyChatTemplate(
    messages: Array<{ role: string; content: string }>,
    addAssistant: boolean
  ): Promise<string> {
    return await this.runtime.applyChatTemplate(messages, addAssistant);
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
        assets: [await this.installRemoteAsset(source, 'model', manifest, options)],
        source: 'remote',
        explicitProjectorAssetId: null,
      };
    }
    if (isFile(source)) {
      return {
        assets: [await this.installLocalAsset(source, 'model', manifest, options)],
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
        return await this.pairingValidator.classify(asset.record.id, asset.file, signal);
      })
    );
  }

  private async registerAssets(
    assets: InstalledAsset[],
    classified: ClassifiedAssetFile[]
  ): Promise<void> {
    const classifiedById = new Map(classified.map((file) => [file.assetId, file]));
    await this.registry.write((draft) => {
      let projectorIndexChanged = false;
      for (const installed of assets) {
        const existing = draft.assets[installed.record.id];
        const inspection = classifiedById.get(installed.record.id)?.inspection;
        const nextKind =
          inspection?.role === 'projector' || installed.record.kind === 'projector'
            ? 'projector'
            : existing?.kind ?? installed.record.kind;
        if (existing == null) {
          draft.assets[installed.record.id] = {
            ...installed.record,
            kind: nextKind,
            ...(inspection == null ? {} : { inspection }),
          };
          if (nextKind === 'projector') {
            projectorIndexChanged = true;
          }
          continue;
        }
        if (existing.kind !== nextKind && (existing.kind === 'projector' || nextKind === 'projector')) {
          projectorIndexChanged = true;
        }
        draft.assets[installed.record.id] = {
          ...existing,
          kind: nextKind,
          sourceUrl: installed.record.sourceUrl ?? existing.sourceUrl,
          sourceEtag: installed.record.sourceEtag ?? existing.sourceEtag,
          sourceLastModified: installed.record.sourceLastModified ?? existing.sourceLastModified,
          ...(inspection == null ? {} : { inspection }),
        };
      }
      if (projectorIndexChanged) {
        draft.projectorIndexRevision += 1;
      }
    });
  }

  private async upsertBaseModelEntry(
    plan: PairingPlan,
    runtimeFingerprint: string
  ): Promise<ModelEntry> {
    const id = `model-${sha256Text(
      stableJson({
        modelAssetIds: [...plan.modelAssetIds].sort((left, right) => left.localeCompare(right)),
      })
    ).slice(0, 24)}`;
    const now = new Date().toISOString();
    let entry!: ModelEntry;
    await this.registry.write((draft) => {
      const existing = draft.models[id];
      if (existing == null) {
        entry = {
          id,
          name: plan.name,
          modality: plan.modality,
          status: plan.status,
          modelAssetIds: plan.modelAssetIds,
          runtimeFingerprint,
          createdAt: now,
          updatedAt: now,
        };
        draft.models[id] = entry;
        for (const assetId of plan.modelAssetIds) {
          const asset = draft.assets[assetId];
          if (asset != null) {
            asset.refCount += 1;
          }
        }
      } else {
        this.updateAssetReferences(
          draft,
          [...existing.modelAssetIds, existing.projectorAssetId].filter(
            (value): value is string => typeof value === 'string'
          ),
          [...plan.modelAssetIds, existing.projectorAssetId].filter(
            (value): value is string => typeof value === 'string'
          )
        );
        existing.name = plan.name;
        existing.modelAssetIds = plan.modelAssetIds;
        if (existing.projectorAssetId == null) {
          existing.modality = plan.modality;
          existing.status = plan.status;
        }
        existing.runtimeFingerprint = runtimeFingerprint;
        existing.updatedAt = now;
        entry = existing;
      }
    });
    return entry;
  }

  private async deriveBasePlanForEntry(
    entry: ModelEntry,
    manifest: RegistryManifest,
    signal?: AbortSignal
  ): Promise<PairingPlan> {
    const installed = await this.assetsForEntry(entry, manifest, false);
    const classified = await this.classifyAssets(installed.assets, signal);
    return this.pairingValidator.resolve(classified);
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

  private async resolveEntryForLoading(
    entry: ModelEntry,
    basePlan: PairingPlan,
    signal?: AbortSignal
  ): Promise<ModelEntry> {
    let manifest = await this.registry.read();
    let current = manifest.models[entry.id] ?? entry;

    if (current.projectorAssetId != null) {
      const projector = manifest.assets[current.projectorAssetId];
      if (projector == null) {
        current = await this.detachProjector(current.id, basePlan);
        manifest = await this.registry.read();
        current = manifest.models[current.id] ?? current;
      } else if (basePlan.compatibleVisionProjectorTypes.length > 0) {
        const inspectedProjector = await this.ensureProjectorInspection(projector, signal);
        const providedType = inspectedProjector?.inspection?.providedVisionProjectorType ?? null;
        if (
          providedType == null ||
          !basePlan.compatibleVisionProjectorTypes.includes(providedType)
        ) {
          current = await this.detachProjector(current.id, basePlan);
          manifest = await this.registry.read();
          current = manifest.models[current.id] ?? current;
        } else if (
          current.pairing?.state !== 'resolved' ||
          !sameVisionProjectorTypes(
            current.pairing.compatibleVisionProjectorTypes,
            basePlan.compatibleVisionProjectorTypes
          )
        ) {
          current = await this.setResolvedProjector(
            current.id,
            projector.id,
            basePlan.compatibleVisionProjectorTypes
          );
        } else {
          return current;
        }
      } else {
        return current;
      }
    }

    if (basePlan.modality !== 'vision') {
      if (
        current.pairing?.state === 'unresolved' &&
        current.pairing.reasonCode === 'BASE_NOT_VISION' &&
        current.pairing.checkedProjectorIndexRevision === manifest.projectorIndexRevision
      ) {
        return current;
      }
      return await this.setUnresolvedPairing(current.id, basePlan, 'BASE_NOT_VISION');
    }

    if (basePlan.compatibleVisionProjectorTypes.length === 0) {
      if (
        current.pairing?.state === 'unresolved' &&
        current.pairing.reasonCode === 'MISSING_METADATA' &&
        current.pairing.checkedProjectorIndexRevision === manifest.projectorIndexRevision
      ) {
        return current;
      }
      return await this.setUnresolvedPairing(current.id, basePlan, 'MISSING_METADATA');
    }

    if (
      current.pairing?.state === 'unresolved' &&
      current.pairing.checkedProjectorIndexRevision === manifest.projectorIndexRevision &&
      sameVisionProjectorTypes(
        current.pairing.compatibleVisionProjectorTypes,
        basePlan.compatibleVisionProjectorTypes
      )
    ) {
      return current;
    }

    const matches = await this.findCompatibleInstalledProjectorIds(
      manifest,
      basePlan.compatibleVisionProjectorTypes,
      signal
    );
    if (matches.length === 1) {
      return await this.setResolvedProjector(
        current.id,
        matches[0],
        basePlan.compatibleVisionProjectorTypes
      );
    }

    return await this.setUnresolvedPairing(
      current.id,
      basePlan,
      matches.length === 0 ? 'NO_MATCH' : 'MULTIPLE_MATCHES'
    );
  }

  private async findCompatibleInstalledProjectorIds(
    manifest: RegistryManifest,
    compatibleVisionProjectorTypes: readonly string[],
    signal?: AbortSignal
  ): Promise<string[]> {
    const compatible = new Set(compatibleVisionProjectorTypes);
    const matches: string[] = [];
    for (const asset of Object.values(manifest.assets)) {
      if (asset.kind !== 'projector' || asset.refCount <= 0) {
        continue;
      }
      const inspected = await this.ensureProjectorInspection(asset, signal);
      const providedType = inspected?.inspection?.providedVisionProjectorType ?? null;
      if (providedType != null && compatible.has(providedType)) {
        matches.push(asset.id);
      }
    }
    return matches.sort((left, right) => left.localeCompare(right));
  }

  private async ensureProjectorInspection(
    asset: AssetRecord,
    signal?: AbortSignal
  ): Promise<AssetRecord | null> {
    if (asset.inspection?.version === 1) {
      return asset;
    }
    try {
      const file = await this.assetStore.getFile(asset);
      const classified = await this.pairingValidator.classify(asset.id, file, signal);
      const updated = await this.registry.write((draft) => {
        const next = draft.assets[asset.id];
        if (next == null) {
          return;
        }
        const nextKind =
          classified.inspection.role === 'projector' || next.kind === 'projector'
            ? 'projector'
            : next.kind;
        if (next.kind !== nextKind && (next.kind === 'projector' || nextKind === 'projector')) {
          draft.projectorIndexRevision += 1;
        }
        next.kind = nextKind;
        next.inspection = classified.inspection;
      });
      return updated.assets[asset.id] ?? null;
    } catch (error) {
      if (error instanceof QueryError && error.code === 'MODEL_BROKEN') {
        return null;
      }
      throw error;
    }
  }

  private async setResolvedProjector(
    id: string,
    projectorAssetId: string,
    compatibleVisionProjectorTypes: readonly string[]
  ): Promise<ModelEntry> {
    const now = new Date().toISOString();
    let entry!: ModelEntry;
    await this.registry.write((draft) => {
      const existing = draft.models[id];
      if (existing == null) {
        throw new QueryError('MODEL_NOT_FOUND', `Model "${id}" is not installed.`);
      }
      this.updateAssetReferences(
        draft,
        entryAssetIds(existing),
        [...existing.modelAssetIds, projectorAssetId]
      );
      existing.projectorAssetId = projectorAssetId;
      existing.modality = 'vision';
      existing.status = 'ready';
      existing.pairing = {
        state: 'resolved',
        checkedProjectorIndexRevision: draft.projectorIndexRevision,
        compatibleVisionProjectorTypes: normalizeVisionProjectorTypes(
          compatibleVisionProjectorTypes
        ),
        updatedAt: now,
      };
      existing.updatedAt = now;
      entry = existing;
    });
    return entry;
  }

  private async setUnresolvedPairing(
    id: string,
    plan: PairingPlan,
    reasonCode: ModelPairingReasonCode
  ): Promise<ModelEntry> {
    const now = new Date().toISOString();
    let entry!: ModelEntry;
    await this.registry.write((draft) => {
      const existing = draft.models[id];
      if (existing == null) {
        throw new QueryError('MODEL_NOT_FOUND', `Model "${id}" is not installed.`);
      }
      this.updateAssetReferences(draft, entryAssetIds(existing), [...existing.modelAssetIds]);
      existing.projectorAssetId = undefined;
      existing.modality = plan.modality;
      existing.status = plan.status;
      existing.pairing = {
        state: 'unresolved',
        checkedProjectorIndexRevision: draft.projectorIndexRevision,
        compatibleVisionProjectorTypes: normalizeVisionProjectorTypes(
          plan.compatibleVisionProjectorTypes
        ),
        reasonCode,
        updatedAt: now,
      };
      existing.updatedAt = now;
      entry = existing;
    });
    return entry;
  }

  private async detachProjector(id: string, basePlan: PairingPlan): Promise<ModelEntry> {
    const now = new Date().toISOString();
    let entry!: ModelEntry;
    await this.registry.write((draft) => {
      const existing = draft.models[id];
      if (existing == null) {
        throw new QueryError('MODEL_NOT_FOUND', `Model "${id}" is not installed.`);
      }
      this.updateAssetReferences(draft, entryAssetIds(existing), [...existing.modelAssetIds]);
      existing.projectorAssetId = undefined;
      existing.modality = basePlan.modality;
      existing.status = basePlan.status;
      existing.pairing = undefined;
      existing.updatedAt = now;
      entry = existing;
    });
    return entry;
  }

  private async restoreEntry(snapshot: ModelEntry): Promise<ModelEntry> {
    const restored = cloneModelEntry(snapshot);
    let entry!: ModelEntry;
    await this.registry.write((draft) => {
      const existing = draft.models[restored.id];
      if (existing == null) {
        throw new QueryError('MODEL_NOT_FOUND', `Model "${restored.id}" is not installed.`);
      }
      this.updateAssetReferences(draft, entryAssetIds(existing), entryAssetIds(restored));
      draft.models[restored.id] = restored;
      entry = restored;
    });
    return entry;
  }

  private updateAssetReferences(
    manifest: RegistryManifest,
    previousAssetIds: readonly string[],
    nextAssetIds: readonly string[]
  ): void {
    const previous = new Set(previousAssetIds);
    const next = new Set(nextAssetIds);

    for (const assetId of previous) {
      if (next.has(assetId)) {
        continue;
      }
      const asset = manifest.assets[assetId];
      if (asset != null) {
        asset.refCount = Math.max(0, asset.refCount - 1);
      }
    }

    for (const assetId of next) {
      if (previous.has(assetId)) {
        continue;
      }
      const asset = manifest.assets[assetId];
      if (asset != null) {
        asset.refCount += 1;
      }
    }
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
    if (
      this.current?.id === entry.id &&
      this.current.assetFingerprint === entryAssetFingerprint(entry) &&
      this.current.runtimeFingerprint === runtimeFingerprint
    ) {
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
    const descriptor = this.buildDescriptor(files.modelFiles, files.projectorFile, entry, manifest);
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
    const switchingCurrent =
      this.current != null &&
      (this.current.id !== entry.id ||
        this.current.assetFingerprint !== entryAssetFingerprint(entry) ||
        this.current.runtimeFingerprint !== runtimeFingerprint);
    if (switchingCurrent) {
      this.observability.update({
        state: 'loading',
        model: null,
        query: null,
        runtime: undefined,
        profile: undefined,
      });
    }
    try {
      await this.runtime.loadRuntimeModel(staged, toRuntimeConfig(options.runtime, observabilityMode));
    } catch (error) {
      this.runtime.close();
      this.current = null;
      this.currentSnapshot = null;
      throw error;
    }

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
      assetFingerprint: entryAssetFingerprint(loadedEntry),
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
      chatTemplate: this.current?.id === entry.id ? this.runtime.getChatTemplate() : null,
      bosText: this.current?.id === entry.id ? this.runtime.getBosText() : '',
      eosText: this.current?.id === entry.id ? this.runtime.getEosText() : '',
      mediaMarker: this.current?.id === entry.id ? this.runtime.readMediaMarker() : null,
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

  private getChatRuntime(current: LoadedModelState): ChatTemplatePromptRuntime {
    const key = `${current.id}:${current.assetFingerprint}`;
    if (this.chatRuntime == null || this.chatRuntimeKey !== key) {
      this.chatRuntime = new ChatTemplatePromptRuntime(this);
      this.chatRuntimeKey = key;
    }
    return this.chatRuntime;
  }
}

function isChatInputObject(input: ChatInput): input is Extract<ChatInput, { messages: unknown }> {
  return !Array.isArray(input);
}
