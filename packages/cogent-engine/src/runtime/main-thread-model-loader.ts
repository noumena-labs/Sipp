import { CogentConfig } from '../cogent-config.js';
import { ModelLoadInfo } from '../types.js';
import {
  BrowserModelCache,
  BrowserModelCacheLookupResult,
} from '../storage/browser-model-cache.js';
import { FileSystemStorage } from '../storage/file-system-storage.js';
import {
  DEFAULT_MAX_MODEL_BYTES,
  EngineModule,
  MountableModelFile,
  URL_DOWNLOAD_CONCURRENCY_MEMORY,
  URL_DOWNLOAD_CONCURRENCY_OPFS,
  URL_METADATA_FETCH_CONCURRENCY,
  UrlShardMetadata,
  createMountableModelFile,
  mapWithConcurrency,
  normalizeModelFileName,
} from './main-thread-runtime-shared.js';
import {
  asErrorMessage,
  createAbortError,
  createLinkedAbortController,
  isAbortError,
} from './runtime-shared.js';

export class MainThreadModelLoader {
  private loadedModelPaths: string[] = [];
  private workerFsMountPath: string | null = null;

  constructor(
    private readonly config: CogentConfig,
    private readonly opfs: FileSystemStorage,
    private readonly browserModelCache: BrowserModelCache,
    private readonly parseConfiguredUrl: (rawUrl: string, fieldName: string) => URL,
    private readonly onModelLoadInfo: (info: ModelLoadInfo) => void
  ) {}

  public cleanupAfterEngineInit(module: EngineModule): void {
    this.removeAllLoadedModelFiles(module);
  }

  public cleanupAfterClose(module: EngineModule): void {
    if (this.workerFsMountPath) {
      try {
        module.FS.unmount(this.workerFsMountPath);
      } catch {
        // Ignore stale mount cleanup failures on close.
      }
      this.workerFsMountPath = null;
    }
    this.loadedModelPaths = [];
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

    const modelPath = await this.mountModelFiles(module, [modelFile]);
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
    void signal;
    if (onProgress) {
      onProgress(100);
    }

    const modelPath = await this.mountModelFiles(module, [file]);
    this.onModelLoadInfo({
      sourceKind: 'file',
      reuseMode: 'file-read',
      modelPath,
      fileName: normalizeModelFileName(destFileName || file.name),
      byteLength: file.size,
      persistentCacheEnabled: false,
      persistentCacheKey: null,
      persistentCacheHit: false,
      persistentCacheStored: false,
    });
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
    this.commitLoadedModelPaths(module, [modelPath]);
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

    const maxModelBytes = this.resolveMaxModelBytes();
    this.ensureModelsDir(module);

    const totalBytes = files.reduce((sum, f) => sum + f.size, 0);
    if (totalBytes <= 0) {
      throw new Error('Model shards are empty.');
    }
    if (totalBytes > maxModelBytes) {
      throw new Error(
        `Total model size (${totalBytes} bytes) exceeds configured maxModelBytes (${maxModelBytes} bytes).`
      );
    }

    try {
      const modelPath = await this.mountModelFiles(module, files);
      this.commitLoadedModelPaths(module, files.map((file) => `/workerfs_model/${file.name}`));
      this.onModelLoadInfo({
        sourceKind: 'file',
        reuseMode: 'file-read',
        modelPath,
        fileName: normalizeModelFileName(files[0].name),
        byteLength: totalBytes,
        persistentCacheEnabled: false,
        persistentCacheKey: null,
        persistentCacheHit: false,
        persistentCacheStored: false,
      });

      if (onProgress) {
        onProgress(100);
      }
      return modelPath;
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

    const opfsSupported =
      FileSystemStorage.isSupported() &&
      this.config.persistentModelCache?.enabled !== false;
    const linkedAbort = createLinkedAbortController(signal);
    const loadSignal = linkedAbort.signal;
    const downloadConcurrency = opfsSupported
      ? URL_DOWNLOAD_CONCURRENCY_OPFS
      : URL_DOWNLOAD_CONCURRENCY_MEMORY;

    try {
      const shardMeta = await this.resolveUrlShardMetadata(urls, loadSignal);
      const totalBytes = shardMeta.reduce((sum, shard) => sum + shard.contentLength, 0);
      const shardLoadedBytes = new Array<number>(shardMeta.length).fill(0);
      let totalLoadedBytes = 0;

      const reportShardProgress = (index: number, loadedBytes: number) => {
        const normalizedBytes = Math.max(0, loadedBytes);
        const previousBytes = shardLoadedBytes[index];
        if (normalizedBytes <= previousBytes) {
          return;
        }
        shardLoadedBytes[index] = normalizedBytes;
        totalLoadedBytes += normalizedBytes - previousBytes;
        if (onProgress != null && totalBytes > 0) {
          onProgress(Math.min(100, Math.round((totalLoadedBytes / totalBytes) * 100)));
        }
      };

      const shardResults = await mapWithConcurrency(
        shardMeta,
        downloadConcurrency,
        async (shard, index) => {
          if (loadSignal.aborted) {
            throw createAbortError('Model load aborted.');
          }

          const cachedEntry: BrowserModelCacheLookupResult | null = opfsSupported
            ? await this.browserModelCache.get(shard.cacheIdentity)
            : null;
          if (cachedEntry != null) {
            reportShardProgress(index, cachedEntry.file.size);
            return {
              file: createMountableModelFile(cachedEntry.file, shard.fileName),
              cacheKey: cachedEntry.key,
              cacheHit: true,
              cacheStored: false,
            };
          }

          const response = await fetch(shard.url, { signal: loadSignal });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status} for ${shard.fileName}`);
          }

          if (opfsSupported) {
            if (!response.body) {
              throw new Error(`Empty body for ${shard.fileName}`);
            }
            const storedEntry = await this.browserModelCache.storeStream(
              shard.cacheIdentity,
              response.body,
              (written) => {
                reportShardProgress(index, written);
              },
              loadSignal
            );
            reportShardProgress(index, storedEntry.file.size);
            return {
              file: createMountableModelFile(storedEntry.file, shard.fileName),
              cacheKey: storedEntry.key,
              cacheHit: false,
              cacheStored: true,
            };
          }

          if (!response.body) {
            const buffer = await response.arrayBuffer();
            reportShardProgress(index, buffer.byteLength);
            return {
              file: createMountableModelFile(new Blob([buffer]), shard.fileName),
              cacheKey: null,
              cacheHit: false,
              cacheStored: false,
            };
          }

          return {
            file: await this.readStreamToMountableModelFile(
              response.body,
              shard.fileName,
              (written) => {
                reportShardProgress(index, written);
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

      const shardBlobs = shardResults.map((result) => result.file);
      const modelPath = await this.mountModelFiles(module, shardBlobs);
      if (onProgress != null && totalBytes === 0) {
        onProgress(100);
      }

      const cacheKeys = shardResults.map(
        (result, index) =>
          result.cacheKey ?? this.browserModelCache.buildEntryKey(shardMeta[index].cacheIdentity)
      );
      const allCacheHits = opfsSupported && shardResults.every((result) => result.cacheHit);
      const anyCacheStored = opfsSupported && shardResults.some((result) => result.cacheStored);

      this.onModelLoadInfo({
        sourceKind: 'url',
        reuseMode: allCacheHits ? 'persistent-cache' : 'network',
        modelPath,
        fileName: shardMeta[0].fileName,
        byteLength: shardBlobs.reduce((sum, blob) => sum + blob.size, 0),
        persistentCacheEnabled: opfsSupported,
        persistentCacheKey: opfsSupported ? cacheKeys.join(',') : null,
        persistentCacheHit: allCacheHits,
        persistentCacheStored: anyCacheStored,
      });

      return modelPath;
    } catch (error) {
      linkedAbort.controller.abort();
      if (isAbortError(error) || signal?.aborted || loadSignal.aborted) {
        throw createAbortError();
      }
      throw new Error(`Model load from URLs failed: ${asErrorMessage(error)}`);
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

  private removeAllLoadedModelFiles(module: EngineModule): void {
    for (const path of this.loadedModelPaths) {
      if (this.workerFsMountPath && path.startsWith(this.workerFsMountPath)) {
        continue;
      }
      this.removeFileIfExists(module, path);
    }
    this.loadedModelPaths = [];
  }

  private commitLoadedModelPaths(module: EngineModule, paths: string[]): void {
    const newSet = new Set(paths);
    for (const path of this.loadedModelPaths) {
      if (!newSet.has(path)) {
        this.removeFileIfExists(module, path);
      }
    }
    this.loadedModelPaths = [...paths];
  }

  private prepareModelPath(module: EngineModule, destFileName: string): string {
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

    return createMountableModelFile(new Blob(chunks as any), fileName);
  }

  private async resolveUrlShardMetadata(
    urls: string[],
    signal: AbortSignal
  ): Promise<UrlShardMetadata[]> {
    return mapWithConcurrency(
      urls,
      URL_METADATA_FETCH_CONCURRENCY,
      async (url) => {
        const parsed = this.parseConfiguredUrl(url, 'modelUrl');
        const canonicalUrl = parsed.toString();
        const fileName = normalizeModelFileName(
          parsed.pathname.split('/').pop() || 'model.gguf'
        );
        try {
          const headResp = await fetch(url, { method: 'HEAD', signal });
          const contentLength =
            Number.parseInt(headResp.headers.get('Content-Length') ?? '0', 10) || 0;
          return {
            url,
            fileName,
            contentLength,
            cacheIdentity: {
              canonicalUrl,
              fileName,
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
            url,
            fileName,
            contentLength: 0,
            cacheIdentity: {
              canonicalUrl,
              fileName,
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

  private async mountModelFiles(
    module: EngineModule,
    files: MountableModelFile[],
    mountDir = '/workerfs_model'
  ): Promise<string> {
    const fs = module.FS;

    if (!fs.analyzePath(mountDir).exists) {
      fs.mkdir(mountDir);
    } else if (this.workerFsMountPath) {
      try {
        fs.unmount(this.workerFsMountPath);
      } catch {
        // Ignore stale unmount failures before remounting.
      }
    }

    if (!module.WORKERFS) {
      throw new Error(
        'WORKERFS is not available in the Emscripten module. Ensure the module was linked with -lworkerfs.js and WORKERFS is exported.'
      );
    }

    fs.mount(module.WORKERFS, { files }, mountDir);
    this.workerFsMountPath = mountDir;

    const firstFileName = files[0].name || 'model.gguf';
    const firstModelPath = `${mountDir}/${firstFileName}`;

    this.commitLoadedModelPaths(
      module,
      files.map((file) => `${mountDir}/${file.name || 'model.gguf'}`)
    );

    return firstModelPath;
  }
}
