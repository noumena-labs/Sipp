import { FileSystemStorage } from '../storage/file-system-storage.js';
import { QueryError, type AssetRecord, type ModelAssetKind, type ModelLoadProgress } from './model-types.js';
import { sha256Blob } from './hash.js';

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

function currentLocationHref(): string | undefined {
  return typeof globalThis.location?.href === 'string' ? globalThis.location.href : undefined;
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
  return new File([blob], name, {
    type: blob.type,
    lastModified: blob instanceof File ? blob.lastModified : Date.now(),
  });
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
    const tempFile = await this.storage.streamToDisk(
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

    try {
      const record = await this.installFile({
        kind,
        file: toFile(tempFile, metadata.name),
        sourceUrl: metadata.canonicalUrl,
        sourceEtag: metadata.etag,
        sourceLastModified: metadata.lastModified,
        onProgress,
      });
      await this.storage.deleteFile(tempName);
      return record;
    } catch (error) {
      await this.storage.deleteFile(tempName);
      throw error;
    }
  }

  public async installFile(input: InstallAssetInput): Promise<AssetRecord> {
    this.ensureAvailable();
    const name = normalizeAssetName(input.file.name || 'model.gguf');
    input.onProgress?.({
      phase: 'store',
      loadedBytes: 0,
      totalBytes: input.file.size,
      percent: 0,
      assetName: name,
    });
    const hash = await sha256Blob(input.file);
    const id = `asset-${hash}`;
    const storagePath = `${id}-${name}`;
    const existing = await this.storage.getFile(storagePath);
    if (existing == null || existing.size !== input.file.size) {
      await this.storage.streamToDisk(storagePath, input.file.stream());
    }
    input.onProgress?.({
      phase: 'store',
      loadedBytes: input.file.size,
      totalBytes: input.file.size,
      percent: 100,
      assetName: name,
    });

    return {
      id,
      kind: input.kind,
      name,
      hash,
      bytes: input.file.size,
      storagePath,
      sourceUrl: input.sourceUrl,
      sourceEtag: input.sourceEtag,
      sourceLastModified: input.sourceLastModified,
      refCount: 0,
      createdAt: new Date().toISOString(),
    };
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
      const baseHref = currentLocationHref();
      return baseHref == null ? new URL(rawUrl) : new URL(rawUrl, baseHref);
    } catch {
      throw new QueryError('INVALID_MODEL_SOURCE', `Invalid model URL "${rawUrl}".`);
    }
  }
}
