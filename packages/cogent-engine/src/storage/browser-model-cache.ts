import { FileSystemStorage } from './file-system-storage.js';

export interface BrowserModelCacheIdentity {
  canonicalUrl: string;
  fileName: string;
  etag: string;
  lastModified: string;
  contentLength: number;
}

export interface BrowserModelCacheLookupResult {
  key: string;
  file: File;
}

export interface BrowserModelCacheStoreResult {
  key: string;
  file: File;
}

export interface BrowserModelCacheStorage {
  getFile(fileName: string): Promise<File | null>;
  readText(fileName: string): Promise<string | null>;
  writeText(fileName: string, contents: string): Promise<void>;
  streamToDisk(
    fileName: string,
    stream: ReadableStream<Uint8Array>,
    onProgress?: (bytes: number) => void,
    signal?: AbortSignal
  ): Promise<File>;
  deleteFile(fileName: string): Promise<void>;
  listFiles(): Promise<string[]>;
  estimate(): Promise<{ usageBytes: number | null; quotaBytes: number | null }>;
}

interface BrowserModelCacheManifestEntry {
  key: string;
  canonicalUrl: string;
  fileName: string;
  storageFileName: string;
  byteLength: number;
  etag: string;
  lastModified: string;
  contentLength: number;
  createdAt: string;
  lastAccessedAt: string;
}

interface BrowserModelCacheManifest {
  version: 1;
  entries: Record<string, BrowserModelCacheManifestEntry>;
}

const CACHE_MANIFEST_FILE_NAME = 'browser-model-cache-manifest.v1.json';
const STORAGE_FILE_PREFIX = 'browser-model-cache-entry-';
const STORAGE_UTILIZATION_TARGET = 0.9;

function hashCacheIdentity(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let hash = 0x811c9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

function createStorageFileName(key: string): string {
  const randomSuffix = Math.random().toString(36).slice(2, 10);
  return `${STORAGE_FILE_PREFIX}${key}-${Date.now().toString(36)}-${randomSuffix}.bin`;
}

function createEmptyManifest(): BrowserModelCacheManifest {
  return {
    version: 1,
    entries: {},
  };
}

export class BrowserModelCache {
  private manifest: BrowserModelCacheManifest | null = null;
  private initPromise: Promise<void> | null = null;
  private operationChain: Promise<void> = Promise.resolve();

  constructor(
    private readonly storage: BrowserModelCacheStorage = new FileSystemStorage()
  ) {}

  public buildEntryKey(identity: BrowserModelCacheIdentity): string {
    const identityText = [
      identity.canonicalUrl,
      identity.etag.trim(),
      identity.lastModified.trim(),
      identity.contentLength > 0 ? String(identity.contentLength) : '',
    ].join('\n');
    return `${hashCacheIdentity(identityText)}-${identity.fileName}`;
  }

  public async get(
    identity: BrowserModelCacheIdentity
  ): Promise<BrowserModelCacheLookupResult | null> {
    const key = this.buildEntryKey(identity);
    return await this.withLock(async () => {
      await this.ensureInitialized();
      const manifest = this.getManifest();
      const entry = manifest.entries[key];
      if (entry == null) {
        return null;
      }

      const file = await this.storage.getFile(entry.storageFileName);
      if (file == null || (entry.byteLength > 0 && file.size !== entry.byteLength)) {
        delete manifest.entries[key];
        await this.writeManifest();
        if (file != null) {
          await this.storage.deleteFile(entry.storageFileName);
        }
        return null;
      }

      entry.lastAccessedAt = new Date().toISOString();
      await this.writeManifest();
      return {
        key,
        file,
      };
    });
  }

  public async storeStream(
    identity: BrowserModelCacheIdentity,
    stream: ReadableStream<Uint8Array>,
    onProgress?: (bytes: number) => void,
    signal?: AbortSignal
  ): Promise<BrowserModelCacheStoreResult> {
    const key = this.buildEntryKey(identity);
    const storageFileName = createStorageFileName(key);
    const previousEntry = await this.withLock(async () => {
      await this.ensureInitialized();
      await this.evictForWrite(identity.contentLength);
      return this.getManifest().entries[key] ?? null;
    });

    const file = await this.storage.streamToDisk(
      storageFileName,
      stream,
      onProgress,
      signal
    );

    await this.withLock(async () => {
      await this.ensureInitialized();
      const manifest = this.getManifest();
      const now = new Date().toISOString();
      manifest.entries[key] = {
        key,
        canonicalUrl: identity.canonicalUrl,
        fileName: identity.fileName,
        storageFileName,
        byteLength: file.size,
        etag: identity.etag,
        lastModified: identity.lastModified,
        contentLength: identity.contentLength,
        createdAt: previousEntry?.createdAt ?? now,
        lastAccessedAt: now,
      };
      await this.writeManifest();
    });

    if (
      previousEntry != null &&
      previousEntry.storageFileName !== storageFileName
    ) {
      await this.storage.deleteFile(previousEntry.storageFileName);
    }

    return {
      key,
      file,
    };
  }

  private async ensureInitialized(): Promise<void> {
    if (this.manifest != null) {
      return;
    }
    if (this.initPromise == null) {
      this.initPromise = (async () => {
        const manifestText = await this.storage.readText(CACHE_MANIFEST_FILE_NAME);
        if (manifestText == null) {
          this.manifest = createEmptyManifest();
        } else {
          try {
          const parsed = JSON.parse(manifestText) as Partial<BrowserModelCacheManifest>;
          if (parsed.version !== 1 || typeof parsed.entries !== 'object' || parsed.entries == null) {
            this.manifest = createEmptyManifest();
          } else {
            this.manifest = {
              version: 1,
              entries: { ...parsed.entries },
            };
          }
          } catch {
            this.manifest = createEmptyManifest();
          }
        }
        await this.cleanupOrphans();
        await this.writeManifest();
      })().finally(() => {
        this.initPromise = null;
      });
    }
    await this.initPromise;
  }

  private getManifest(): BrowserModelCacheManifest {
    if (this.manifest == null) {
      throw new Error('Browser model cache manifest is not initialized.');
    }
    return this.manifest;
  }

  private async cleanupOrphans(): Promise<void> {
    const manifest = this.getManifest();
    const referencedStorageNames = new Set<string>();
    let manifestChanged = false;

    for (const [key, entry] of Object.entries(manifest.entries)) {
      const file = await this.storage.getFile(entry.storageFileName);
      if (file == null || (entry.byteLength > 0 && file.size !== entry.byteLength)) {
        delete manifest.entries[key];
        manifestChanged = true;
        if (file != null) {
          await this.storage.deleteFile(entry.storageFileName);
        }
        continue;
      }
      referencedStorageNames.add(entry.storageFileName);
    }

    const fileNames = await this.storage.listFiles();
    for (const fileName of fileNames) {
      if (fileName === CACHE_MANIFEST_FILE_NAME) {
        continue;
      }
      if (referencedStorageNames.has(fileName)) {
        continue;
      }
      if (fileName.startsWith(STORAGE_FILE_PREFIX)) {
        await this.storage.deleteFile(fileName);
      }
    }

    if (manifestChanged) {
      await this.writeManifest();
    }
  }

  private async evictForWrite(requiredBytes: number): Promise<void> {
    if (requiredBytes <= 0) {
      return;
    }

    const estimate = await this.storage.estimate();
    if (estimate.usageBytes == null || estimate.quotaBytes == null || estimate.quotaBytes <= 0) {
      return;
    }

    const targetUsageBytes = Math.floor(
      estimate.quotaBytes * STORAGE_UTILIZATION_TARGET
    );
    if (estimate.usageBytes + requiredBytes <= targetUsageBytes) {
      return;
    }

    const manifest = this.getManifest();
    const entriesByAge = Object.values(manifest.entries).sort((left, right) =>
      left.lastAccessedAt.localeCompare(right.lastAccessedAt)
    );

    let usageBytes = estimate.usageBytes;
    let manifestChanged = false;
    for (const entry of entriesByAge) {
      if (usageBytes + requiredBytes <= targetUsageBytes) {
        break;
      }
      delete manifest.entries[entry.key];
      usageBytes = Math.max(0, usageBytes - entry.byteLength);
      manifestChanged = true;
      await this.storage.deleteFile(entry.storageFileName);
    }

    if (manifestChanged) {
      await this.writeManifest();
    }
  }

  private async writeManifest(): Promise<void> {
    await this.storage.writeText(
      CACHE_MANIFEST_FILE_NAME,
      JSON.stringify(this.getManifest(), null, 2)
    );
  }

  private async withLock<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.operationChain;
    let release!: () => void;
    this.operationChain = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }
}
