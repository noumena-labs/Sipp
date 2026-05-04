import { FileSystemStorage } from '../storage/file-system-storage.js';
import { QueryError, type AssetRecord, type ModelAssetKind, type ModelLoadProgress } from './model-types.js';
import { sha256Blob } from './hash.js';
import { currentLocationHref, resolveUrl } from '../utils/url.js';

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
  onProgress?: (progress: ModelLoadProgress) => void;
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

function quotaExceededError(name: string, bytes: number, cause: unknown): QueryError {
  return new QueryError(
    'STORAGE_QUOTA_EXCEEDED',
    `Not enough browser storage quota to cache "${name}" (${formatBytes(bytes)}). Clear site data, use a smaller model, or choose an origin with more persistent storage.`,
    { cause }
  );
}

export class AssetStore {
  constructor(private readonly storage = new FileSystemStorage()) {}

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

  public async installFile(input: InstallAssetInput): Promise<AssetRecord> {
    this.ensureAvailable();
    const name = normalizeAssetName(input.file.name || 'model.gguf');
    this.emitStoreProgress(input.onProgress, name, 0, input.file.size, 0);
    const hash = await sha256Blob(input.file);
    const id = `asset-${hash}`;
    const storagePath = `${id}-${name}`;
    const existing = await this.storage.getFile(storagePath);
    if (existing == null || existing.size !== input.file.size) {
      try {
        await this.storage.streamToDisk(storagePath, input.file.stream());
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
    const hash = await sha256Blob(input.file);
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
