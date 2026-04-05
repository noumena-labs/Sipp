import { ModelLoadSourceKind } from '../types.js';

const CACHE_METADATA_VERSION = 1;

export interface BrowserModelCacheConfig {
  enabled: boolean;
  namespace: string;
  cacheLocalFiles: boolean;
  maxEntryBytes: number;
}

export interface ModelCacheSourceDescriptor {
  kind: ModelLoadSourceKind;
  identity: string;
  fileName: string;
}

export interface BrowserModelCacheRestoreResult {
  persistentCacheKey: string;
  fileName: string;
  byteLength: number;
  stream: ReadableStream<Uint8Array>;
}

interface BrowserModelCacheMetadata {
  version: number;
  sourceIdentity: string;
  sourceKind: ModelLoadSourceKind;
  fileName: string;
  sizeBytes: number;
  storedAt: string;
}

interface PersistentModelCacheFileEntry {
  dataFileName: string;
  metadataFileName: string;
}

export interface BrowserModelCacheWriter {
  persistentCacheKey: string;
  write(chunk: Uint8Array): Promise<void>;
  close(finalSizeBytes: number): Promise<void>;
  abort(): Promise<void>;
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function fnv1a64(value: string): string {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= BigInt(value.charCodeAt(i));
    hash = BigInt.asUintN(64, hash * prime);
  }
  return hash.toString(16).padStart(16, '0');
}

function buildCacheEntryNames(persistentCacheKey: string): PersistentModelCacheFileEntry {
  return {
    dataFileName: `${persistentCacheKey}.bin`,
    metadataFileName: `${persistentCacheKey}.json`,
  };
}

async function readJsonFile<T>(
  directory: FileSystemDirectoryHandle,
  fileName: string
): Promise<T | null> {
  try {
    const handle = await directory.getFileHandle(fileName);
    const file = await handle.getFile();
    return JSON.parse(await file.text()) as T;
  } catch {
    return null;
  }
}

async function removeEntryIfExists(
  directory: FileSystemDirectoryHandle,
  fileName: string
): Promise<void> {
  try {
    await directory.removeEntry(fileName);
  } catch {
    // Ignore best-effort cleanup failures.
  }
}

export class BrowserModelCache {
  constructor(private readonly config: BrowserModelCacheConfig) {}

  public isEnabledForSource(source: ModelCacheSourceDescriptor): boolean {
    if (!this.config.enabled) {
      return false;
    }
    if (source.kind === 'file' && !this.config.cacheLocalFiles) {
      return false;
    }
    return true;
  }

  public async isSupported(): Promise<boolean> {
    return (await this.getCacheDirectory()) != null;
  }

  public buildPersistentCacheKey(source: ModelCacheSourceDescriptor): string {
    return fnv1a64(`${source.kind}:${source.identity}`);
  }

  public async restore(
    source: ModelCacheSourceDescriptor
  ): Promise<BrowserModelCacheRestoreResult | null> {
    if (!this.isEnabledForSource(source)) {
      return null;
    }

    const directory = await this.getCacheDirectory();
    if (directory == null) {
      return null;
    }

    const persistentCacheKey = this.buildPersistentCacheKey(source);
    const entry = buildCacheEntryNames(persistentCacheKey);
    const metadata = await readJsonFile<BrowserModelCacheMetadata>(
      directory,
      entry.metadataFileName
    );
    if (
      metadata == null ||
      metadata.version !== CACHE_METADATA_VERSION ||
      metadata.sourceIdentity !== source.identity ||
      metadata.sourceKind !== source.kind ||
      metadata.sizeBytes <= 0
    ) {
      return null;
    }

    try {
      const dataHandle = await directory.getFileHandle(entry.dataFileName);
      const dataFile = await dataHandle.getFile();
      if (dataFile.size !== metadata.sizeBytes) {
        return null;
      }

      return {
        persistentCacheKey,
        fileName: metadata.fileName,
        byteLength: metadata.sizeBytes,
        stream: dataFile.stream() as ReadableStream<Uint8Array>,
      };
    } catch {
      return null;
    }
  }

  public async createWriter(
    source: ModelCacheSourceDescriptor
  ): Promise<BrowserModelCacheWriter | null> {
    if (!this.isEnabledForSource(source)) {
      return null;
    }

    const directory = await this.getCacheDirectory();
    if (directory == null) {
      return null;
    }

    const persistentCacheKey = this.buildPersistentCacheKey(source);
    const entry = buildCacheEntryNames(persistentCacheKey);
    const dataHandle = await directory.getFileHandle(entry.dataFileName, {
      create: true,
    });
    const writable = await dataHandle.createWritable({ keepExistingData: false });

    let closed = false;

    const closeAndFinalize = async (finalSizeBytes: number) => {
      if (closed) {
        return;
      }
      closed = true;
      await writable.close();

      const metadata: BrowserModelCacheMetadata = {
        version: CACHE_METADATA_VERSION,
        sourceIdentity: source.identity,
        sourceKind: source.kind,
        fileName: source.fileName,
        sizeBytes: finalSizeBytes,
        storedAt: new Date().toISOString(),
      };

      const metadataHandle = await directory.getFileHandle(entry.metadataFileName, {
        create: true,
      });
      const metadataWriter = await metadataHandle.createWritable({
        keepExistingData: false,
      });
      await metadataWriter.write(JSON.stringify(metadata));
      await metadataWriter.close();
    };

    const abortAndCleanup = async () => {
      if (!closed) {
        closed = true;
        try {
          await writable.abort();
        } catch {
          try {
            await writable.close();
          } catch {
            // Ignore close failure during abort cleanup.
          }
        }
      }

      await removeEntryIfExists(directory, entry.dataFileName);
      await removeEntryIfExists(directory, entry.metadataFileName);
    };

    return {
      persistentCacheKey,
      async write(chunk: Uint8Array) {
        if (closed) {
          throw new Error('Persistent cache writer is already closed.');
        }
        const safeChunk = new Uint8Array(chunk.byteLength);
        safeChunk.set(chunk);
        await writable.write(safeChunk);
      },
      async close(finalSizeBytes: number) {
        await closeAndFinalize(finalSizeBytes);
      },
      async abort() {
        await abortAndCleanup();
      },
    };
  }

  private async getCacheDirectory(): Promise<FileSystemDirectoryHandle | null> {
    if (!this.config.enabled) {
      return null;
    }

    const storage = globalThis.navigator?.storage;
    if (storage == null || typeof storage.getDirectory !== 'function') {
      return null;
    }

    try {
      const root = await storage.getDirectory();
      return await root.getDirectoryHandle(this.config.namespace, {
        create: true,
      });
    } catch (error) {
      console.warn(
        `[cogent-engine] Browser model cache unavailable: ${toErrorMessage(error)}`
      );
      return null;
    }
  }
}
