import { FileSystemStorage, type OpfsSyncAccessHandle } from '../storage/file-system-storage.js';
import {
  QueryError,
  type AssetRecord,
  type ModelAssetKind,
  type ModelLoadProgress,
  type QueryErrorCode,
  type RegistryManifest,
} from './types.js';
import { sha256Blob, sha256Text, type AssetHashProvider } from './hash.js';
import { currentLocationHref, resolveUrl } from '../utils/url.js';

const DEFAULT_BROWSER_DIRECT_LOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024;
const DEFAULT_BROWSER_SHARD_MAX_BYTES = 512 * 1024 * 1024;
const BROWSER_SPLIT_TEMP_PREFIXES = ['tmp-source-', 'tmp-local-source-'];
const BROWSER_SPLIT_SHARD_PREFIXES = ['split-', 'split-local-'];

export interface GgufSplitRuntime {
  browserCacheLayout(
    sourceBytes: number,
    sourceBytesKnown: boolean,
    directLoadMaxBytes: number,
    shardMaxBytes: number
  ): Promise<'single-file' | 'split-gguf'>;
  planGgufSplitCount(
    sourceBytes: number,
    shardMaxBytes: number,
    callbacks: { readAt(offset: number, target: Uint8Array): number | void }
  ): Promise<number>;
  splitGgufStream(
    sourceBytes: number,
    outputPrefix: string,
    shardMaxBytes: number,
    callbacks: {
      readAt(offset: number, target: Uint8Array): number | void;
      openShard(path: string, index: number, count: number): number | void;
      writeShard(bytes: Uint8Array): number | void;
      closeShard(): number | void;
    }
  ): Promise<void>;
}

export interface RemoteAssetMetadata {
  url: string;
  canonicalUrl: string;
  name: string;
  bytes: number;
  etag: string;
  lastModified: string;
}

export interface InstallAssetInput {
  kind: ModelAssetKind;
  file: File;
  sourceUrl?: string;
  sourceEtag?: string;
  sourceLastModified?: string;
  signal?: AbortSignal;
  onProgress?: (progress: ModelLoadProgress) => void;
}

// Browser-only OPFS ingest path. Native Rust, Python, and Node should load
// normal files or explicit shard arrays directly rather than using this
// sync-access callback bridge.
interface SplitStoredGgufInput {
  sourcePath: string;
  sourceName: string;
  sourceBytes: number;
  outputPrefix: string;
  runtime: GgufSplitRuntime;
  signal?: AbortSignal;
  onProgress?: (progress: ModelLoadProgress) => void;
  failureCode: QueryErrorCode;
  failureMessage: string;
  shardMetadata: (index: number, count: number) => {
    sourceUrl?: string;
    sourceEtag?: string;
    sourceLastModified?: string;
    sourceBytes?: number;
    sourcePartIndex?: number;
    sourcePartCount?: number;
    sourceFileName?: string;
    sourceFileLastModified?: number;
  };
}

function normalizeAssetName(name: string): string {
  const trimmed = name.trim();
  const defaultValue = trimmed.length > 0 ? trimmed : 'model.gguf';
  return defaultValue.replace(/[\\/:*?"<>|]+/g, '-');
}

function fileNameFromUrl(url: string): string {
  try {
    const parsed = new URL(url, currentLocationHref());
    return normalizeAssetName(decodeURIComponent(parsed.pathname.split('/').pop() || 'model.gguf'));
  } catch {
    return normalizeAssetName(url.split('/').pop()?.split('?')[0] ?? 'model.gguf');
  }
}

function toFile(blob: Blob, name: string): File {
  if (blob instanceof File && blob.name === name) {
    return blob;
  }
  const contents = blob instanceof File ? blob.slice(0, blob.size, blob.type) : blob;
  return new File([contents], name, {
    type: blob.type,
    lastModified: blob instanceof File ? blob.lastModified : Date.now(),
  });
}

function isQuotaExceededError(error: unknown): boolean {
  if (error == null || typeof error !== 'object' || !('name' in error)) {
    return false;
  }
  const name = String((error as { name?: unknown }).name);
  return name === 'QuotaExceededError' || name === 'NS_ERROR_DOM_QUOTA_REACHED';
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return 'unknown size';
  }
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

function stripGgufExtension(name: string): string {
  return name.toLowerCase().endsWith('.gguf') ? name.slice(0, -5) : name;
}

function splitShardPath(prefix: string, index: number, count: number): string {
  return `${prefix}-${String(index + 1).padStart(5, '0')}-of-${String(count).padStart(5, '0')}.gguf`;
}

function quotaExceededError(name: string, bytes: number, cause: unknown): QueryError {
  return new QueryError(
    'STORAGE_QUOTA_EXCEEDED',
    `Not enough browser storage quota to cache "${name}" (${formatBytes(bytes)}). Clear site data, use a smaller model, or choose an origin with more persistent storage.`,
    { cause }
  );
}

export class AssetStore {
  private hashProvider: AssetHashProvider = {
    sha256Text,
    sha256Blob,
  };

  constructor(private readonly storage = new FileSystemStorage()) {}

  public setHashProvider(provider: AssetHashProvider): void {
    this.hashProvider = provider;
  }

  public ensureAvailable(): void {
    if (!FileSystemStorage.isSupported()) {
      throw new QueryError(
        'STORAGE_UNAVAILABLE',
        'Managed model storage requires OPFS, but navigator.storage.getDirectory() is unavailable.'
      );
    }
  }

  public async resolveRemoteMetadata(rawUrl: string, signal?: AbortSignal): Promise<RemoteAssetMetadata> {
    this.ensureAvailable();
    const canonicalUrl = this.parseUrl(rawUrl).toString();
    let response: Response;
    try {
      response = await fetch(canonicalUrl, { method: 'HEAD', signal });
    } catch (error) {
      throw new QueryError(
        'REMOTE_METADATA_UNAVAILABLE',
        `Unable to read model metadata for "${canonicalUrl}".`,
        { cause: error }
      );
    }
    if (!response.ok) {
      throw new QueryError(
        'REMOTE_METADATA_UNAVAILABLE',
        `Unable to read model metadata for "${canonicalUrl}" (HTTP ${response.status}).`
      );
    }

    const bytes = Number.parseInt(response.headers.get('Content-Length') ?? '', 10);
    const etag = response.headers.get('ETag')?.trim() ?? '';
    const lastModified = response.headers.get('Last-Modified')?.trim() ?? '';
    if (!Number.isFinite(bytes) || bytes <= 0 || (etag.length === 0 && lastModified.length === 0)) {
      throw new QueryError(
        'REMOTE_METADATA_UNAVAILABLE',
        `Remote model "${canonicalUrl}" must provide Content-Length and either ETag or Last-Modified.`
      );
    }

    return {
      url: rawUrl,
      canonicalUrl,
      name: fileNameFromUrl(canonicalUrl),
      bytes,
      etag,
      lastModified,
    };
  }

  public async downloadRemote(
    metadata: RemoteAssetMetadata,
    kind: ModelAssetKind,
    signal?: AbortSignal,
    onProgress?: (progress: ModelLoadProgress) => void
  ): Promise<AssetRecord> {
    this.ensureAvailable();
    let response: Response;
    try {
      response = await fetch(metadata.canonicalUrl, { signal });
    } catch (error) {
      throw new QueryError('REMOTE_LOAD_FAILED', `Failed to download "${metadata.canonicalUrl}".`, {
        cause: error,
      });
    }
    if (!response.ok || response.body == null) {
      throw new QueryError(
        'REMOTE_LOAD_FAILED',
        `Failed to download "${metadata.canonicalUrl}" (HTTP ${response.status}).`
      );
    }

    const tempName = `tmp-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${metadata.name}`;
    let written = 0;
    let tempFile: File;
    try {
      tempFile = await this.storage.streamToDisk(
        tempName,
        response.body,
        (bytes) => {
          written = bytes;
          onProgress?.({
            phase: 'download',
            loadedBytes: written,
            totalBytes: metadata.bytes,
            percent: Math.min(100, Math.round((written / metadata.bytes) * 100)),
            assetName: metadata.name,
          });
        },
        signal
      );
    } catch (error) {
      if (isQuotaExceededError(error)) {
        throw quotaExceededError(metadata.name, metadata.bytes, error);
      }
      throw error;
    }

    try {
      return await this.registerDownloadedFile({
        kind,
        name: metadata.name,
        file: tempFile,
        storagePath: tempName,
        sourceUrl: metadata.canonicalUrl,
        sourceEtag: metadata.etag,
        sourceLastModified: metadata.lastModified,
        onProgress,
      });
    } catch (error) {
      await this.storage.deleteFile(tempName);
      throw error;
    }
  }

  public async downloadRemoteSplitGguf(
    metadata: RemoteAssetMetadata,
    runtime: GgufSplitRuntime,
    signal?: AbortSignal,
    onProgress?: (progress: ModelLoadProgress) => void
  ): Promise<AssetRecord[]> {
    this.ensureAvailable();
    if (!(await FileSystemStorage.isSyncAccessSupported())) {
      throw new QueryError(
        'STORAGE_UNAVAILABLE',
        'Browser-only large GGUF splitting requires OPFS sync access handles. Run model loading in a browser worker with createSyncAccessHandle() support.'
      );
    }

    let layout: 'single-file' | 'split-gguf';
    try {
      layout = await runtime.browserCacheLayout(
        metadata.bytes,
        true,
        DEFAULT_BROWSER_DIRECT_LOAD_MAX_BYTES,
        DEFAULT_BROWSER_SHARD_MAX_BYTES
      );
    } catch (error) {
      throw new QueryError(
        'REMOTE_LOAD_FAILED',
        'Browser-only large GGUF splitting requires the wasm32 Rust ingest browser build.',
        { cause: error }
      );
    }
    if (layout !== 'split-gguf') {
      return [
        await this.downloadRemote(
          metadata,
          'model',
          signal,
          onProgress
        ),
      ];
    }

    let response: Response;
    try {
      response = await fetch(metadata.canonicalUrl, { signal });
    } catch (error) {
      throw new QueryError('REMOTE_LOAD_FAILED', `Failed to download "${metadata.canonicalUrl}".`, {
        cause: error,
      });
    }
    if (!response.ok || response.body == null) {
      throw new QueryError(
        'REMOTE_LOAD_FAILED',
        `Failed to download "${metadata.canonicalUrl}" (HTTP ${response.status}).`
      );
    }

    const sourceKey = this.hashProvider.sha256Text(
      `${metadata.canonicalUrl}\n${metadata.etag}\n${metadata.lastModified}\n${metadata.bytes}`
    ).slice(0, 24);
    const sourceTempPath = `tmp-source-${Date.now().toString(36)}-${Math.random()
      .toString(36)
      .slice(2)}-${metadata.name}`;
    const outputPrefix = `split-${sourceKey}-${stripGgufExtension(metadata.name)}`;

    try {
      await this.storage.streamToDisk(
        sourceTempPath,
        response.body,
        (bytes) => {
          onProgress?.({
            phase: 'download',
            loadedBytes: bytes,
            totalBytes: metadata.bytes,
            percent: Math.min(100, Math.round((bytes / metadata.bytes) * 100)),
            assetName: metadata.name,
          });
        },
        signal
      );
    } catch (error) {
      await this.storage.deleteFile(sourceTempPath);
      if (isQuotaExceededError(error)) {
        throw quotaExceededError(metadata.name, metadata.bytes, error);
      }
      throw error;
    }

    return await this.splitStoredGguf({
      sourcePath: sourceTempPath,
      sourceName: metadata.name,
      sourceBytes: metadata.bytes,
      outputPrefix,
      runtime,
      signal,
      onProgress,
      failureCode: 'REMOTE_LOAD_FAILED',
      failureMessage: `Failed to split "${metadata.canonicalUrl}".`,
      shardMetadata: (index, count) => ({
        sourceUrl: metadata.canonicalUrl,
        sourceEtag: metadata.etag,
        sourceLastModified: metadata.lastModified,
        sourceBytes: metadata.bytes,
        sourcePartIndex: index,
        sourcePartCount: count,
      }),
    });
  }

  public async installLocalSplitGguf(
    file: File,
    runtime: GgufSplitRuntime,
    signal?: AbortSignal,
    onProgress?: (progress: ModelLoadProgress) => void
  ): Promise<AssetRecord[]> {
    this.ensureAvailable();
    if (!(await FileSystemStorage.isSyncAccessSupported())) {
      throw new QueryError(
        'STORAGE_UNAVAILABLE',
        'Browser-only large local GGUF splitting requires OPFS sync access handles. Run model loading in a browser worker with createSyncAccessHandle() support.'
      );
    }

    const name = normalizeAssetName(file.name || 'model.gguf');
    let layout: 'single-file' | 'split-gguf';
    try {
      layout = await runtime.browserCacheLayout(
        file.size,
        true,
        DEFAULT_BROWSER_DIRECT_LOAD_MAX_BYTES,
        DEFAULT_BROWSER_SHARD_MAX_BYTES
      );
    } catch (error) {
      throw new QueryError(
        'INVALID_MODEL_SOURCE',
        'Browser-only large local GGUF splitting requires the wasm32 Rust ingest browser build.',
        { cause: error }
      );
    }
    if (layout !== 'split-gguf') {
      return [await this.installFile({ kind: 'model', file, signal, onProgress })];
    }

    const sourceKey = this.hashProvider
      .sha256Text(`local\n${name}\n${file.lastModified}\n${file.size}`)
      .slice(0, 24);
    const sourceTempPath = `tmp-local-source-${Date.now().toString(36)}-${Math.random()
      .toString(36)
      .slice(2)}-${name}`;
    const outputPrefix = `split-local-${sourceKey}-${stripGgufExtension(name)}`;

    try {
      await this.storage.streamToDisk(
        sourceTempPath,
        file.stream(),
        (bytes) => {
          this.emitStoreProgress(
            onProgress,
            name,
            bytes,
            file.size,
            Math.min(100, Math.round((bytes / file.size) * 100))
          );
        },
        signal
      );
    } catch (error) {
      await this.storage.deleteFile(sourceTempPath);
      if (isQuotaExceededError(error)) {
        throw quotaExceededError(name, file.size, error);
      }
      throw error;
    }

    return await this.splitStoredGguf({
      sourcePath: sourceTempPath,
      sourceName: name,
      sourceBytes: file.size,
      outputPrefix,
      runtime,
      signal,
      onProgress,
      failureCode: 'INVALID_MODEL_SOURCE',
      failureMessage: `Failed to split local GGUF "${name}".`,
      shardMetadata: (index, count) => ({
        sourceBytes: file.size,
        sourcePartIndex: index,
        sourcePartCount: count,
        sourceFileName: name,
        sourceFileLastModified: file.lastModified,
      }),
    });
  }

  public async installFile(input: InstallAssetInput): Promise<AssetRecord> {
    this.ensureAvailable();
    const name = normalizeAssetName(input.file.name || 'model.gguf');
    this.emitStoreProgress(input.onProgress, name, 0, input.file.size, 0);
    const hash = await this.hashProvider.sha256Blob(input.file, input.signal);
    const id = `asset-${hash}`;
    const storagePath = `${id}-${name}`;
    const existing = await this.storage.getFile(storagePath);
    if (existing == null || existing.size !== input.file.size) {
      try {
        await this.storage.streamToDisk(storagePath, input.file.stream(), undefined, input.signal);
      } catch (error) {
        if (isQuotaExceededError(error)) {
          throw quotaExceededError(name, input.file.size, error);
        }
        throw error;
      }
    }
    this.emitStoreProgress(input.onProgress, name, input.file.size, input.file.size, 100);
    return this.buildAssetRecord({
      id,
      kind: input.kind,
      name,
      hash,
      bytes: input.file.size,
      storagePath,
      sourceUrl: input.sourceUrl,
      sourceEtag: input.sourceEtag,
      sourceLastModified: input.sourceLastModified,
    });
  }

  public async getFile(record: AssetRecord): Promise<File> {
    this.ensureAvailable();
    const file = await this.storage.getFile(record.storagePath);
    if (file == null || file.size !== record.bytes) {
      throw new QueryError(
        'MODEL_BROKEN',
        `Installed model asset "${record.name}" is missing or corrupt.`
      );
    }
    return toFile(file, record.name);
  }

  public async delete(record: AssetRecord): Promise<void> {
    await this.storage.deleteFile(record.storagePath);
  }

  public async cleanupBrowserSplitArtifacts(manifest: RegistryManifest): Promise<void> {
    this.ensureAvailable();
    const protectedPaths = new Set(
      Object.values(manifest.assets).map((asset) => asset.storagePath)
    );
    const fileNames = await this.storage.listFileNames();
    for (const fileName of fileNames) {
      const isTempSource = BROWSER_SPLIT_TEMP_PREFIXES.some((prefix) => fileName.startsWith(prefix));
      const isUnregisteredSplitShard =
        BROWSER_SPLIT_SHARD_PREFIXES.some((prefix) => fileName.startsWith(prefix)) &&
        !protectedPaths.has(fileName);
      if (isTempSource || isUnregisteredSplitShard) {
        await this.storage.deleteFile(fileName);
      }
    }
  }

  public async registerStoredFile(input: {
    kind: ModelAssetKind;
    name: string;
    storagePath: string;
    sourceUrl?: string;
    sourceEtag?: string;
    sourceLastModified?: string;
    sourceBytes?: number;
    sourcePartIndex?: number;
    sourcePartCount?: number;
    sourceFileName?: string;
    sourceFileLastModified?: number;
    signal?: AbortSignal;
    onProgress?: (progress: ModelLoadProgress) => void;
  }): Promise<AssetRecord> {
    const name = normalizeAssetName(input.name);
    const file = await this.storage.getFile(input.storagePath);
    if (file == null || file.size <= 0) {
      throw new QueryError('MODEL_BROKEN', `Stored asset "${name}" is missing or empty.`);
    }
    this.emitStoreProgress(input.onProgress, name, 0, file.size, 0);
    const hash = await this.hashProvider.sha256Blob(file, input.signal);
    this.emitStoreProgress(input.onProgress, name, file.size, file.size, 100);
    return this.buildAssetRecord({
      id: `asset-${hash}`,
      kind: input.kind,
      name,
      hash,
      bytes: file.size,
      storagePath: input.storagePath,
      sourceUrl: input.sourceUrl,
      sourceEtag: input.sourceEtag,
      sourceLastModified: input.sourceLastModified,
      sourceBytes: input.sourceBytes,
      sourcePartIndex: input.sourcePartIndex,
      sourcePartCount: input.sourcePartCount,
      sourceFileName: input.sourceFileName,
      sourceFileLastModified: input.sourceFileLastModified,
    });
  }

  private parseUrl(rawUrl: string): URL {
    try {
      return resolveUrl(rawUrl, 'model URL');
    } catch {
      throw new QueryError('INVALID_MODEL_SOURCE', `Invalid model URL "${rawUrl}".`);
    }
  }

  private async registerDownloadedFile(input: {
    kind: ModelAssetKind;
    name: string;
    file: File;
    storagePath: string;
    sourceUrl: string;
    sourceEtag: string;
    sourceLastModified: string;
    onProgress?: (progress: ModelLoadProgress) => void;
  }): Promise<AssetRecord> {
    const name = normalizeAssetName(input.name || input.file.name || 'model.gguf');
    this.emitStoreProgress(input.onProgress, name, 0, input.file.size, 0);
    const hash = await this.hashProvider.sha256Blob(input.file);
    const id = `asset-${hash}`;
    const canonicalStoragePath = `${id}-${name}`;
    const existing = await this.storage.getFile(canonicalStoragePath);
    const storagePath =
      existing != null && existing.size === input.file.size
        ? canonicalStoragePath
        : input.storagePath;
    if (storagePath !== input.storagePath) {
      await this.storage.deleteFile(input.storagePath);
    }
    this.emitStoreProgress(input.onProgress, name, input.file.size, input.file.size, 100);
    return this.buildAssetRecord({
      id,
      kind: input.kind,
      name,
      hash,
      bytes: input.file.size,
      storagePath,
      sourceUrl: input.sourceUrl,
      sourceEtag: input.sourceEtag,
      sourceLastModified: input.sourceLastModified,
    });
  }

  private async splitStoredGguf(input: SplitStoredGgufInput): Promise<AssetRecord[]> {
    const shardPaths: string[] = [];
    let sourceHandle: OpfsSyncAccessHandle | null = null;
    const shardHandles = new Map<string, OpfsSyncAccessHandle>();
    let activeShard:
      | {
          path: string;
          handle: OpfsSyncAccessHandle;
          offset: number;
        }
      | null = null;
    let splitBytesWritten = 0;

    try {
      sourceHandle = await this.storage.createSyncAccessHandle(input.sourcePath);
      const readAt = (offset: number, target: Uint8Array): number => {
        if (input.signal?.aborted === true || sourceHandle == null) {
          return -1;
        }
        const read = sourceHandle.read(target, { at: offset });
        return read === target.byteLength ? 0 : -1;
      };

      const shardCount = await input.runtime.planGgufSplitCount(
        input.sourceBytes,
        DEFAULT_BROWSER_SHARD_MAX_BYTES,
        { readAt }
      );
      if (!Number.isInteger(shardCount) || shardCount <= 0) {
        throw new QueryError(input.failureCode, `${input.failureMessage} Invalid shard count ${shardCount}.`);
      }

      for (let index = 0; index < shardCount; index += 1) {
        const path = splitShardPath(input.outputPrefix, index, shardCount);
        shardPaths.push(path);
        const handle = await this.storage.createSyncAccessHandle(path, { create: true });
        handle.truncate(0);
        shardHandles.set(path, handle);
      }

      await input.runtime.splitGgufStream(
        input.sourceBytes,
        input.outputPrefix,
        DEFAULT_BROWSER_SHARD_MAX_BYTES,
        {
          readAt,
          openShard: (path, index, count) => {
            if (count !== shardCount || path !== shardPaths[index]) {
              return -1;
            }
            const handle = shardHandles.get(path);
            if (handle == null) {
              return -1;
            }
            activeShard = { path, handle, offset: 0 };
            return 0;
          },
          writeShard: (bytes) => {
            if (input.signal?.aborted === true || activeShard == null) {
              return -1;
            }
            const written = activeShard.handle.write(bytes, { at: activeShard.offset });
            if (written !== bytes.byteLength) {
              return -1;
            }
            activeShard.offset += written;
            splitBytesWritten += written;
            input.onProgress?.({
              phase: 'split',
              loadedBytes: splitBytesWritten,
              totalBytes: input.sourceBytes,
              percent: Math.min(100, Math.round((splitBytesWritten / input.sourceBytes) * 100)),
              assetName: input.sourceName,
            });
            return 0;
          },
          closeShard: () => {
            if (activeShard == null) {
              return -1;
            }
            activeShard.handle.flush();
            activeShard.handle.close();
            shardHandles.delete(activeShard.path);
            activeShard = null;
            return 0;
          },
        }
      );

      sourceHandle.close();
      sourceHandle = null;
      await this.storage.deleteFile(input.sourcePath);

      const records: AssetRecord[] = [];
      for (let index = 0; index < shardPaths.length; index += 1) {
        const path = shardPaths[index];
        records.push(
          await this.registerStoredFile({
            kind: 'shard',
            name: path,
            storagePath: path,
            ...input.shardMetadata(index, shardPaths.length),
            signal: input.signal,
            onProgress: input.onProgress,
          })
        );
      }
      return records;
    } catch (error) {
      for (const path of shardPaths) {
        await this.storage.deleteFile(path);
      }
      if (error instanceof QueryError) {
        throw error;
      }
      if (isQuotaExceededError(error)) {
        throw quotaExceededError(input.sourceName, input.sourceBytes, error);
      }
      throw new QueryError(input.failureCode, input.failureMessage, { cause: error });
    } finally {
      try {
        const shardToClose = activeShard as unknown as { handle: OpfsSyncAccessHandle } | null;
        shardToClose?.handle.close();
      } catch {}
      try {
        sourceHandle?.close();
      } catch {}
      for (const handle of shardHandles.values()) {
        try {
          handle.close();
        } catch {}
      }
      await this.storage.deleteFile(input.sourcePath);
    }
  }

  private emitStoreProgress(
    onProgress: ((progress: ModelLoadProgress) => void) | undefined,
    assetName: string,
    loadedBytes: number,
    totalBytes: number,
    percent: number
  ): void {
    onProgress?.({
      phase: 'store',
      loadedBytes,
      totalBytes,
      percent,
      assetName,
    });
  }

  private buildAssetRecord(input: Omit<AssetRecord, 'refCount' | 'createdAt'>): AssetRecord {
    return {
      ...input,
      refCount: 0,
      createdAt: new Date().toISOString(),
    };
  }
}
