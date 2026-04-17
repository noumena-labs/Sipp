import { CogentConfig } from '../cogent-config.js';
import { ModelLoadInfo } from '../core/inference-types.js';
import {
  detectModel,
  detectModelFromGgufFile,
  discoverProjector,
  resolveLocalModelAndProjectorFiles,
} from '../model-bundle/model-bundle-detection.js';
import {
  ModelBundleDescriptor,
  ModelBundleProjectorDescriptor,
  ModelBundleProjectorStatus,
  PreparedModelBundle,
  PrepareModelBundleOptions,
} from '../model-bundle/model-bundle-types.js';
import {
  BrowserModelCache,
  BrowserModelCacheLookupResult,
} from '../storage/browser-model-cache.js';
import { FileSystemStorage } from '../storage/file-system-storage.js';
import {
  DEFAULT_MAX_MODEL_BYTES,
  MountableModelFile,
  URL_DOWNLOAD_CONCURRENCY_MEMORY,
  URL_DOWNLOAD_CONCURRENCY_OPFS,
  URL_METADATA_FETCH_CONCURRENCY,
  UrlShardMetadata,
  createMountableModelFile,
  normalizeModelFileName,
} from './main-thread-runtime-constants.js';
import { EngineModule } from '../wasm/engine-module.js';
import {
  createAbortError,
  createLinkedAbortController,
  isAbortError,
} from '../utils/abort.js';
import { mapWithConcurrency } from '../utils/async.js';
import { asErrorMessage } from '../utils/error.js';

interface UrlAssetRequest {
  url: string;
  mountFileName: string;
}

interface FileAssetRequest {
  file: File;
  mountFileName: string;
}

interface LoadedAssetSet {
  files: MountableModelFile[];
  loadInfo: ModelLoadInfo;
}

interface ResolvedUrlAssetMetadata extends UrlShardMetadata {
  mountFileName: string;
}

interface ResolvedAssetRequest {
  kind: 'url' | 'file';
  mountFileName: string;
  url?: string;
  file?: File;
}

interface ResolvedBundlePlan {
  detection: ReturnType<typeof detectModel>;
  modelAssets: ResolvedAssetRequest[];
  projectorAsset: ResolvedAssetRequest | null;
  projectorStatus: ModelBundleProjectorStatus;
}

export class MainThreadModelLoader {
  private loadedAssetPaths: string[] = [];
  private activeMountPath: string | null = null;

  constructor(
    private readonly config: CogentConfig,
    private readonly opfs: FileSystemStorage,
    private readonly browserModelCache: BrowserModelCache,
    private readonly parseConfiguredUrl: (rawUrl: string, fieldName: string) => URL,
    private readonly onModelLoadInfo: (info: ModelLoadInfo) => void
  ) {}

  public cleanupAfterEngineInit(module: EngineModule): void {
    this.removeAllLoadedAssets(module);
  }

  public cleanupAfterClose(module: EngineModule): void {
    this.unmountActiveAssetSet(module);
    this.loadedAssetPaths = [];
  }

  public async loadModelFromUrl(
    module: EngineModule,
    url: string,
    destFileName = 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    void destFileName;
    return this.loadModelFromUrls(module, [url], onProgress, signal);
  }

  public async loadModelFromReadableStream(
    module: EngineModule,
    stream: ReadableStream<Uint8Array>,
    destFileName = 'model.gguf',
    options: {
      expectedBytes?: number;
      onProgress?: (pct: number) => void;
      signal?: AbortSignal;
    } = {}
  ): Promise<string> {
    void options.expectedBytes;

    const opfsEnabled =
      FileSystemStorage.isSupported() &&
      this.config.persistentModelCache?.enabled !== false;

    let modelFile: Blob;
    if (opfsEnabled) {
      modelFile = await this.opfs.streamToDisk(
        destFileName,
        stream,
        options.onProgress,
        options.signal
      );
    } else {
      modelFile = await this.readStreamToMountableModelFile(
        stream,
        destFileName,
        undefined,
        options.signal
      );
    }

    const mountedPaths = await this.mountAssetFiles(module, [modelFile]);
    const modelPath = mountedPaths[0];
    this.onModelLoadInfo({
      sourceKind: 'buffer',
      reuseMode: 'buffer',
      modelPath,
      fileName: destFileName,
      byteLength: modelFile.size,
      persistentCacheEnabled: opfsEnabled,
      persistentCacheKey: null,
      persistentCacheHit: false,
      persistentCacheStored: opfsEnabled,
    });
    return modelPath;
  }

  public async loadModelFromFile(
    module: EngineModule,
    file: File,
    destFileName = file.name || 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    const assetSet = await this.loadFileAssetSet(
      [
        {
          file,
          mountFileName: destFileName,
        },
      ],
      onProgress,
      signal
    );
    const mountedPaths = await this.mountAssetFiles(module, assetSet.files);
    const modelPath = mountedPaths[0];
    const loadInfo = {
      ...assetSet.loadInfo,
      modelPath,
    };
    this.onModelLoadInfo(loadInfo);
    return modelPath;
  }

  public loadModelFromBuffer(
    module: EngineModule,
    buffer: Uint8Array,
    destFileName = 'model.gguf'
  ): string {
    const maxModelBytes = this.resolveMaxModelBytes();
    if (buffer.byteLength === 0) {
      throw new Error('Model buffer is empty.');
    }
    if (buffer.byteLength > maxModelBytes) {
      throw new Error(`Model exceeds configured maxModelBytes (${maxModelBytes} bytes).`);
    }

    const modelPath = this.prepareModelPath(module, destFileName);
    module.FS.writeFile(modelPath, buffer);
    this.commitLoadedAssetPaths(module, [modelPath]);
    this.onModelLoadInfo({
      sourceKind: 'buffer',
      reuseMode: 'buffer',
      modelPath,
      fileName: normalizeModelFileName(destFileName),
      byteLength: buffer.byteLength,
      persistentCacheEnabled: false,
      persistentCacheKey: null,
      persistentCacheHit: false,
      persistentCacheStored: false,
    });
    return modelPath;
  }

  public async loadModelFromFileShards(
    module: EngineModule,
    files: File[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    if (!files || files.length === 0) {
      throw new Error('No shard files provided.');
    }
    if (files.length === 1) {
      return this.loadModelFromFile(module, files[0], files[0].name, onProgress, signal);
    }

    try {
      const assetSet = await this.loadFileAssetSet(
        files.map((file) => ({
          file,
          mountFileName: file.name,
        })),
        onProgress,
        signal
      );
      const mountedPaths = await this.mountAssetFiles(module, assetSet.files);
      const loadInfo = {
        ...assetSet.loadInfo,
        modelPath: mountedPaths[0],
      };
      this.onModelLoadInfo(loadInfo);
      return mountedPaths[0];
    } catch (error) {
      if (isAbortError(error) || signal?.aborted) {
        throw createAbortError('Model load aborted.');
      }
      throw new Error(`Failed while loading model shards: ${asErrorMessage(error)}`);
    }
  }

  public async loadModelFromUrls(
    module: EngineModule,
    urls: string[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    if (!urls || urls.length === 0) {
      throw new Error('No shard URLs provided.');
    }

    try {
      const assetSet = await this.loadUrlAssetSet(
        urls.map((url) => ({
          url,
          mountFileName: this.deriveFileNameFromUrl(url, 'model.gguf'),
        })),
        onProgress,
        signal
      );
      const mountedPaths = await this.mountAssetFiles(module, assetSet.files);
      const loadInfo = {
        ...assetSet.loadInfo,
        modelPath: mountedPaths[0],
      };
      this.onModelLoadInfo(loadInfo);
      return mountedPaths[0];
    } catch (error) {
      if (isAbortError(error) || signal?.aborted) {
        throw createAbortError();
      }
      throw new Error(`Model load from URLs failed: ${asErrorMessage(error)}`);
    }
  }

  public async prepareModelBundle(
    module: EngineModule,
    descriptor: ModelBundleDescriptor,
    options: PrepareModelBundleOptions = {}
  ): Promise<PreparedModelBundle> {
    const plan = await this.resolveBundlePlan(descriptor, options.signal);
    const modelAssetSet = await this.loadResolvedAssetSet(plan.modelAssets, options.signal);
    const projectorAssetSet =
      plan.projectorAsset == null
        ? null
        : await this.loadResolvedAssetSet([plan.projectorAsset], options.signal);

    const mountedPaths = await this.mountAssetFiles(module, [
      ...modelAssetSet.files,
      ...(projectorAssetSet?.files ?? []),
    ]);
    const modelPath = mountedPaths[0];
    const projectorPath =
      projectorAssetSet == null ? null : mountedPaths[modelAssetSet.files.length] ?? null;

    const modelLoadInfo = {
      ...modelAssetSet.loadInfo,
      modelPath,
    };
    const projectorLoadInfo =
      projectorAssetSet == null || projectorPath == null
        ? null
        : {
            ...projectorAssetSet.loadInfo,
            modelPath: projectorPath,
          };

    this.onModelLoadInfo(modelLoadInfo);

    return {
      sourceKind: descriptor.kind,
      modelPath,
      multimodalProjectorPath: projectorPath,
      isVisionModel: plan.detection.isVisionModel,
      projectorStatus: plan.projectorStatus,
      modelName: plan.detection.modelName,
      detectionMethod: plan.detection.detectionMethod,
      modelType: plan.detection.modelType,
      modelArchitecture: plan.detection.modelArchitecture,
      modelLoadInfo,
      projectorLoadInfo,
    };
  }

  private async resolveBundlePlan(
    descriptor: ModelBundleDescriptor,
    signal?: AbortSignal
  ): Promise<ResolvedBundlePlan> {
    switch (descriptor.kind) {
      case 'url': {
        const detection = detectModel('url', descriptor.url);
        this.ensureNotProjectorSource(detection.modelName, detection.isProjector);
        return {
          detection,
          modelAssets: [
            {
              kind: 'url',
              url: descriptor.url,
              mountFileName:
                descriptor.destFileName ?? this.deriveFileNameFromUrl(descriptor.url, 'model.gguf'),
            },
          ],
          ...(await this.resolveProjectorAssetForUrlDescriptor(
            detection,
            descriptor.projector,
            descriptor.url
          )),
        };
      }
      case 'urls': {
        if (!descriptor.urls || descriptor.urls.length === 0) {
          throw new Error('Model bundle URL list must not be empty.');
        }
        const detection = detectModel('url', descriptor.urls[0]);
        this.ensureNotProjectorSource(detection.modelName, detection.isProjector);
        return {
          detection,
          modelAssets: descriptor.urls.map((url) => ({
            kind: 'url',
            url,
            mountFileName: this.deriveFileNameFromUrl(url, 'model.gguf'),
          })),
          ...(await this.resolveProjectorAssetForUrlDescriptor(
            detection,
            descriptor.projector,
            descriptor.urls[0]
          )),
        };
      }
      case 'file': {
        const detection = await detectModelFromGgufFile(descriptor.file, signal);
        this.ensureNotProjectorSource(detection.modelName, detection.isProjector);
        return {
          detection,
          modelAssets: [
            {
              kind: 'file',
              file: descriptor.file,
              mountFileName: descriptor.destFileName ?? descriptor.file.name,
            },
          ],
          projectorAsset: this.resolveExplicitProjectorAsset(descriptor.projector),
          projectorStatus: this.resolveProjectorStatus(
            detection.isVisionModel,
            descriptor.projector == null ? null : 'explicit'
          ),
        };
      }
      case 'files': {
        if (!descriptor.files || descriptor.files.length === 0) {
          throw new Error('Model bundle file list must not be empty.');
        }
        const explicitProjectorAsset = this.resolveExplicitProjectorAsset(descriptor.projector);
        const localResolution =
          explicitProjectorAsset == null
            ? await resolveLocalModelAndProjectorFiles(descriptor.files, signal)
            : {
                modelFiles: [...descriptor.files],
                projectorFile: null,
                candidateFileNames: [],
                errorMessage: null,
              };

        if (localResolution.errorMessage != null) {
          throw new Error(localResolution.errorMessage);
        }
        if (localResolution.modelFiles.length === 0) {
          throw new Error('Model bundle file list does not contain any model GGUF files.');
        }

        const sortedModelFiles = [...localResolution.modelFiles].sort((left, right) =>
          normalizeModelFileName(left.name).localeCompare(normalizeModelFileName(right.name))
        );
        const detectionFile = sortedModelFiles[0];
        const detection = await detectModelFromGgufFile(detectionFile, signal);
        this.ensureNotProjectorSource(detection.modelName, detection.isProjector);

        return {
          detection,
          modelAssets: sortedModelFiles.map((file) => ({
            kind: 'file',
            file,
            mountFileName: file.name,
          })),
          projectorAsset:
            explicitProjectorAsset ??
            (localResolution.projectorFile == null
              ? null
              : {
                  kind: 'file',
                  file: localResolution.projectorFile,
                  mountFileName: localResolution.projectorFile.name,
                }),
          projectorStatus:
            explicitProjectorAsset != null
              ? this.resolveProjectorStatus(detection.isVisionModel, 'explicit')
              : localResolution.projectorFile != null
                ? 'paired'
                : this.resolveProjectorStatus(detection.isVisionModel, null),
        };
      }
    }
  }

  private async resolveProjectorAssetForUrlDescriptor(
    detection: ReturnType<typeof detectModel>,
    explicitProjector: ModelBundleProjectorDescriptor | undefined,
    primaryModelUrl: string
  ): Promise<{
    projectorAsset: ResolvedAssetRequest | null;
    projectorStatus: ModelBundleProjectorStatus;
  }> {
    const explicitProjectorAsset = this.resolveExplicitProjectorAsset(explicitProjector);
    if (explicitProjectorAsset != null) {
      return {
        projectorAsset: explicitProjectorAsset,
        projectorStatus: 'explicit',
      };
    }

    if (!detection.isVisionModel) {
      return {
        projectorAsset: null,
        projectorStatus: 'not-required',
      };
    }

    const discovery = await discoverProjector(primaryModelUrl);
    if (!discovery.projectorUrl) {
      return {
        projectorAsset: null,
        projectorStatus: 'missing',
      };
    }

    return {
      projectorAsset: {
        kind: 'url',
        url: discovery.projectorUrl,
        mountFileName: this.deriveFileNameFromUrl(discovery.projectorUrl, 'mmproj.gguf'),
      },
      projectorStatus: 'discovered',
    };
  }

  private resolveExplicitProjectorAsset(
    projector: ModelBundleProjectorDescriptor | undefined
  ): ResolvedAssetRequest | null {
    if (projector == null) {
      return null;
    }
    if (projector.kind === 'url') {
      return {
        kind: 'url',
        url: projector.url,
        mountFileName:
          projector.destFileName ?? this.deriveFileNameFromUrl(projector.url, 'mmproj.gguf'),
      };
    }
    return {
      kind: 'file',
      file: projector.file,
      mountFileName: projector.destFileName ?? projector.file.name,
    };
  }

  private resolveProjectorStatus(
    isVisionModel: boolean,
    resolvedStatus: Extract<ModelBundleProjectorStatus, 'explicit'> | null
  ): ModelBundleProjectorStatus {
    if (resolvedStatus != null) {
      return resolvedStatus;
    }
    return isVisionModel ? 'missing' : 'not-required';
  }

  private ensureNotProjectorSource(modelName: string, isProjector: boolean): void {
    if (isProjector) {
      throw new Error(
        `Model source "${modelName}" looks like a projector GGUF. Provide the main model GGUF instead.`
      );
    }
  }

  private async loadResolvedAssetSet(
    assets: ResolvedAssetRequest[],
    signal?: AbortSignal
  ): Promise<LoadedAssetSet> {
    if (assets.length === 0) {
      throw new Error('Asset set must not be empty.');
    }
    const firstAsset = assets[0];
    if (firstAsset.kind === 'url') {
      return this.loadUrlAssetSet(
        assets.map((asset) => ({
          url: asset.url as string,
          mountFileName: asset.mountFileName,
        })),
        undefined,
        signal
      );
    }
    return this.loadFileAssetSet(
      assets.map((asset) => ({
        file: asset.file as File,
        mountFileName: asset.mountFileName,
      })),
      undefined,
      signal
    );
  }

  private async loadFileAssetSet(
    assets: FileAssetRequest[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<LoadedAssetSet> {
    if (!assets || assets.length === 0) {
      throw new Error('No file assets provided.');
    }
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    const files = assets.map((asset) => createMountableModelFile(asset.file, asset.mountFileName));
    const byteLength = files.reduce((sum, file) => sum + file.size, 0);
    if (byteLength <= 0) {
      throw new Error('Model assets are empty.');
    }
    const maxModelBytes = this.resolveMaxModelBytes();
    if (byteLength > maxModelBytes) {
      throw new Error(
        `Total model size (${byteLength} bytes) exceeds configured maxModelBytes (${maxModelBytes} bytes).`
      );
    }

    onProgress?.(100);
    return {
      files,
      loadInfo: {
        sourceKind: 'file',
        reuseMode: 'file-read',
        modelPath: '',
        fileName: normalizeModelFileName(assets[0].mountFileName),
        byteLength,
        persistentCacheEnabled: false,
        persistentCacheKey: null,
        persistentCacheHit: false,
        persistentCacheStored: false,
      },
    };
  }

  private async loadUrlAssetSet(
    assets: UrlAssetRequest[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<LoadedAssetSet> {
    if (!assets || assets.length === 0) {
      throw new Error('No URL assets provided.');
    }

    const opfsSupported =
      FileSystemStorage.isSupported() &&
      this.config.persistentModelCache?.enabled !== false;
    const linkedAbort = createLinkedAbortController(signal);
    const loadSignal = linkedAbort.signal;
    const downloadConcurrency = opfsSupported
      ? URL_DOWNLOAD_CONCURRENCY_OPFS
      : URL_DOWNLOAD_CONCURRENCY_MEMORY;

    try {
      const assetMeta = await this.resolveUrlAssetMetadata(assets, loadSignal);
      const totalBytes = assetMeta.reduce((sum, asset) => sum + asset.contentLength, 0);
      const loadedBytes = new Array<number>(assetMeta.length).fill(0);
      let totalLoadedBytes = 0;

      const reportAssetProgress = (index: number, byteCount: number) => {
        const normalizedBytes = Math.max(0, byteCount);
        const previousBytes = loadedBytes[index];
        if (normalizedBytes <= previousBytes) {
          return;
        }
        loadedBytes[index] = normalizedBytes;
        totalLoadedBytes += normalizedBytes - previousBytes;
        if (onProgress != null && totalBytes > 0) {
          onProgress(Math.min(100, Math.round((totalLoadedBytes / totalBytes) * 100)));
        }
      };

      const assetResults = await mapWithConcurrency(
        assetMeta,
        downloadConcurrency,
        async (asset, index) => {
          if (loadSignal.aborted) {
            throw createAbortError('Model load aborted.');
          }

          const cachedEntry: BrowserModelCacheLookupResult | null = opfsSupported
            ? await this.browserModelCache.get(asset.cacheIdentity)
            : null;
          if (cachedEntry != null) {
            reportAssetProgress(index, cachedEntry.file.size);
            return {
              file: createMountableModelFile(cachedEntry.file, asset.mountFileName),
              cacheKey: cachedEntry.key,
              cacheHit: true,
              cacheStored: false,
            };
          }

          const response = await fetch(asset.url, { signal: loadSignal });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status} for ${asset.mountFileName}`);
          }

          if (opfsSupported) {
            if (!response.body) {
              throw new Error(`Empty body for ${asset.mountFileName}`);
            }
            const storedEntry = await this.browserModelCache.storeStream(
              asset.cacheIdentity,
              response.body,
              (written) => {
                reportAssetProgress(index, written);
              },
              loadSignal
            );
            reportAssetProgress(index, storedEntry.file.size);
            return {
              file: createMountableModelFile(storedEntry.file, asset.mountFileName),
              cacheKey: storedEntry.key,
              cacheHit: false,
              cacheStored: true,
            };
          }

          if (!response.body) {
            const buffer = await response.arrayBuffer();
            reportAssetProgress(index, buffer.byteLength);
            return {
              file: createMountableModelFile(new Blob([buffer]), asset.mountFileName),
              cacheKey: null,
              cacheHit: false,
              cacheStored: false,
            };
          }

          return {
            file: await this.readStreamToMountableModelFile(
              response.body,
              asset.mountFileName,
              (written) => {
                reportAssetProgress(index, written);
              },
              loadSignal
            ),
            cacheKey: null,
            cacheHit: false,
            cacheStored: false,
          };
        },
        () => {
          linkedAbort.controller.abort();
        }
      );

      if (onProgress != null && totalBytes === 0) {
        onProgress(100);
      }

      const cacheKeys = assetResults.map(
        (result, index) =>
          result.cacheKey ?? this.browserModelCache.buildEntryKey(assetMeta[index].cacheIdentity)
      );
      const allCacheHits = opfsSupported && assetResults.every((result) => result.cacheHit);
      const anyCacheStored = opfsSupported && assetResults.some((result) => result.cacheStored);

      return {
        files: assetResults.map((result) => result.file),
        loadInfo: {
          sourceKind: 'url',
          reuseMode: allCacheHits ? 'persistent-cache' : 'network',
          modelPath: '',
          fileName: assetMeta[0].mountFileName,
          byteLength: assetResults.reduce((sum, result) => sum + result.file.size, 0),
          persistentCacheEnabled: opfsSupported,
          persistentCacheKey: opfsSupported ? cacheKeys.join(',') : null,
          persistentCacheHit: allCacheHits,
          persistentCacheStored: anyCacheStored,
        },
      };
    } catch (error) {
      linkedAbort.controller.abort();
      if (isAbortError(error) || signal?.aborted || loadSignal.aborted) {
        throw createAbortError();
      }
      throw new Error(asErrorMessage(error));
    } finally {
      linkedAbort.dispose();
    }
  }

  private resolveMaxModelBytes(): number {
    const maxModelBytes = this.config.maxModelBytes ?? DEFAULT_MAX_MODEL_BYTES;
    if (!Number.isInteger(maxModelBytes) || maxModelBytes <= 0) {
      throw new Error('"maxModelBytes" must be a positive integer.');
    }
    return maxModelBytes;
  }

  private removeFileIfExists(module: EngineModule, path: string): void {
    if (module.FS.analyzePath(path).exists) {
      module.FS.unlink(path);
    }
  }

  private removeAllLoadedAssets(module: EngineModule): void {
    for (const path of this.loadedAssetPaths) {
      if (this.activeMountPath && path.startsWith(this.activeMountPath)) {
        continue;
      }
      this.removeFileIfExists(module, path);
    }
    this.loadedAssetPaths = [];
    this.unmountActiveAssetSet(module);
  }

  private commitLoadedAssetPaths(module: EngineModule, paths: string[]): void {
    const newSet = new Set(paths);
    for (const path of this.loadedAssetPaths) {
      if (!newSet.has(path)) {
        if (this.activeMountPath && path.startsWith(this.activeMountPath)) {
          continue;
        }
        this.removeFileIfExists(module, path);
      }
    }
    this.loadedAssetPaths = [...paths];
  }

  private prepareModelPath(module: EngineModule, destFileName: string): string {
    this.unmountActiveAssetSet(module);
    const safeName = normalizeModelFileName(destFileName);
    const modelPath = `/models/${safeName}`;
    this.ensureModelsDir(module);
    this.removeFileIfExists(module, modelPath);
    return modelPath;
  }

  private async readStreamToMountableModelFile(
    stream: ReadableStream<Uint8Array>,
    fileName: string,
    onProgress?: (bytes: number) => void,
    signal?: AbortSignal
  ): Promise<MountableModelFile> {
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];
    let bytesRead = 0;
    const abortListener =
      signal == null
        ? null
        : () => {
            void reader.cancel(createAbortError('Model load aborted.'));
          };
    const abortSignal = signal;
    if (abortListener != null && abortSignal != null) {
      abortSignal.addEventListener('abort', abortListener, { once: true });
    }
    try {
      while (true) {
        if (signal?.aborted) {
          throw createAbortError('Model load aborted.');
        }
        const { done, value } = await reader.read();
        if (done) {
          if (signal?.aborted) {
            throw createAbortError('Model load aborted.');
          }
          break;
        }
        if (value != null) {
          chunks.push(value);
          bytesRead += value.byteLength;
          onProgress?.(bytesRead);
        }
      }
    } catch (error) {
      if (isAbortError(error) || signal?.aborted) {
        throw createAbortError('Model load aborted.');
      }
      throw error;
    } finally {
      if (abortListener != null && abortSignal != null) {
        abortSignal.removeEventListener('abort', abortListener);
      }
      reader.releaseLock();
    }

    return createMountableModelFile(new Blob(chunks as BlobPart[]), fileName);
  }

  private async resolveUrlAssetMetadata(
    assets: UrlAssetRequest[],
    signal: AbortSignal
  ): Promise<ResolvedUrlAssetMetadata[]> {
    return mapWithConcurrency(
      assets,
      URL_METADATA_FETCH_CONCURRENCY,
      async (asset) => {
        const parsed = this.parseConfiguredUrl(asset.url, 'modelUrl');
        const canonicalUrl = parsed.toString();
        const sourceFileName = normalizeModelFileName(
          parsed.pathname.split('/').pop() || 'model.gguf'
        );
        try {
          const headResp = await fetch(asset.url, { method: 'HEAD', signal });
          const contentLength =
            Number.parseInt(headResp.headers.get('Content-Length') ?? '0', 10) || 0;
          return {
            url: asset.url,
            fileName: sourceFileName,
            mountFileName: normalizeModelFileName(asset.mountFileName),
            contentLength,
            cacheIdentity: {
              canonicalUrl,
              fileName: sourceFileName,
              etag: headResp.headers.get('ETag')?.trim() ?? '',
              lastModified: headResp.headers.get('Last-Modified')?.trim() ?? '',
              contentLength,
            },
          };
        } catch (error) {
          if (isAbortError(error) || signal.aborted) {
            throw createAbortError('Model load aborted.');
          }
          return {
            url: asset.url,
            fileName: sourceFileName,
            mountFileName: normalizeModelFileName(asset.mountFileName),
            contentLength: 0,
            cacheIdentity: {
              canonicalUrl,
              fileName: sourceFileName,
              etag: '',
              lastModified: '',
              contentLength: 0,
            },
          };
        }
      }
    );
  }

  private ensureModelsDir(module: EngineModule): void {
    const modelsPath = '/models';
    if (!module.FS.analyzePath(modelsPath).exists) {
      module.FS.mkdir(modelsPath);
    }
  }

  private async mountAssetFiles(
    module: EngineModule,
    files: MountableModelFile[],
    mountDir = '/workerfs_model'
  ): Promise<string[]> {
    const fs = module.FS;
    const normalizedFiles = files.map((file) =>
      createMountableModelFile(file, file.name || 'model.gguf')
    );

    if (!fs.analyzePath(mountDir).exists) {
      fs.mkdir(mountDir);
    }
    this.unmountActiveAssetSet(module);

    if (!module.WORKERFS) {
      throw new Error(
        'WORKERFS is not available in the Emscripten module. Ensure the module was linked with -lworkerfs.js and WORKERFS is exported.'
      );
    }

    fs.mount(module.WORKERFS, { files: normalizedFiles }, mountDir);
    this.activeMountPath = mountDir;

    const mountedPaths = normalizedFiles.map(
      (file) => `${mountDir}/${file.name || 'model.gguf'}`
    );
    this.commitLoadedAssetPaths(module, mountedPaths);
    return mountedPaths;
  }

  private unmountActiveAssetSet(module: EngineModule): void {
    if (this.activeMountPath == null) {
      return;
    }
    try {
      module.FS.unmount(this.activeMountPath);
    } catch {
      // Ignore stale unmount cleanup failures.
    }
    this.activeMountPath = null;
  }

  private deriveFileNameFromUrl(url: string, fallbackName: string): string {
    const parsed = this.parseConfiguredUrl(url, 'modelUrl');
    const rawName = parsed.pathname.split('/').pop();
    return normalizeModelFileName(rawName && rawName.length > 0 ? rawName : fallbackName);
  }
}
