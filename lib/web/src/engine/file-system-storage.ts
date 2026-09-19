import { createAbortError } from '../utils/abort.js';

interface WritableFileSink {
  write(chunk: Uint8Array): Promise<void>;
  close(): Promise<void>;
  abort(): Promise<void>;
  release(): void;
}

interface StreamToDiskOptions {
  readonly onProgress?: (bytes: number) => void;
  readonly signal?: AbortSignal;
  readonly startOffset?: number;
  readonly stallTimeoutMs?: number;
  readonly keepFileOnFailure?: boolean;
}

/** Error raised when a streamed OPFS write receives no data before its deadline. */
export class StreamStallError extends Error {
  public readonly code = 'STALL_TIMEOUT' as const;

  public constructor(stallTimeoutMs: number) {
    super(`Download stream stalled: no data received for ${stallTimeoutMs}ms.`);
    this.name = 'TimeoutError';
  }
}

const STREAM_WRITE_BUFFER_BYTES = 4 * 1024 * 1024;
const DEFAULT_OPFS_ROOT = 'sipp-models';

function storageRootSegments(storageRoot: string): readonly string[] {
  const trimmed = storageRoot.trim();
  if (trimmed.length === 0) {
    throw new Error('OPFS storage root must not be empty.');
  }
  const segments = trimmed.split(/[\\/]+/);
  if (
    segments.some((segment) =>
      segment.length === 0 ||
      segment === '.' ||
      segment === '..' ||
      segment.trim() !== segment
    )
  ) {
    throw new Error('OPFS storage root must be a relative directory path.');
  }
  return segments;
}

function toFileSystemWriteChunk(chunk: Uint8Array): Uint8Array<ArrayBuffer> {
  if (chunk.buffer instanceof ArrayBuffer) {
    return chunk as Uint8Array<ArrayBuffer>;
  }
  const copy = new Uint8Array(chunk.byteLength);
  copy.set(chunk);
  return copy;
}

export interface OpfsSyncAccessHandle {
  read(buffer: Uint8Array, options?: { at?: number }): number;
  write(buffer: Uint8Array, options?: { at?: number }): number;
  truncate(size: number): void;
  flush(): void;
  close(): void;
  getSize(): number;
}

/**
 * Streams large assets into OPFS and exposes both sync access handles for the
 * model load path and File objects for incidental reads (projector, metadata
 * detection). The model load path mounts shards via the
 * SyncAccessHandleFS provider in `wasm/sync-access-handle-fs.ts`.
 */
export class FileSystemStorage {
  private root: FileSystemDirectoryHandle | null = null;
  private readonly rootPath: readonly string[];

  public constructor(storageRoot = DEFAULT_OPFS_ROOT) {
    this.rootPath = storageRootSegments(storageRoot);
  }

  /**
   * Check if OPFS is supported in the current environment.
   */
  public static isSupported(): boolean {
    return (
      typeof navigator !== 'undefined' &&
      typeof navigator.storage !== 'undefined' &&
      typeof navigator.storage.getDirectory === 'function'
    );
  }

  public static async isSyncAccessSupported(): Promise<boolean> {
    if (!FileSystemStorage.isSupported()) {
      return false;
    }
    try {
      const root = await navigator.storage.getDirectory();
      const handle = await root.getFileHandle(
        `sipp-sync-access-probe-${Date.now().toString(36)}`,
        { create: true }
      );
      const createSyncAccessHandle = (handle as unknown as {
        createSyncAccessHandle?: () => Promise<OpfsSyncAccessHandle>;
      }).createSyncAccessHandle;
      await root.removeEntry(handle.name).catch(() => {});
      return typeof createSyncAccessHandle === 'function';
    } catch {
      return false;
    }
  }

  private async ensureRoot(): Promise<FileSystemDirectoryHandle> {
    if (this.root) return this.root;
    let directory = await navigator.storage.getDirectory();
    for (const segment of this.rootPath) {
      directory = await directory.getDirectoryHandle(segment, { create: true });
    }
    this.root = directory;
    return this.root;
  }

  private async getDirectory(
    path: readonly string[],
    options: { create?: boolean } = {}
  ): Promise<FileSystemDirectoryHandle> {
    let directory = await this.ensureRoot();
    for (const segment of path) {
      directory = await directory.getDirectoryHandle(segment, {
        create: options.create === true,
      });
    }
    return directory;
  }

  private async getFileHandleAt(
    path: readonly string[],
    options: { create?: boolean } = {}
  ): Promise<FileSystemFileHandle> {
    if (path.length === 0) {
      throw new Error('OPFS file path must not be empty.');
    }
    const directory = await this.getDirectory(path.slice(0, -1), options);
    const fileName = path[path.length - 1];
    return await directory.getFileHandle(fileName, { create: options.create === true });
  }

  private isNotFoundError(error: unknown): boolean {
    return typeof DOMException === 'function' && error instanceof DOMException && error.name === 'NotFoundError';
  }

  private async toWritableFileSink(
    writable: FileSystemWritableFileStream,
    startOffset = 0
  ): Promise<WritableFileSink> {
    const seek = (writable as unknown as {
      seek?: (offset: number) => Promise<void>;
    }).seek;
    if (startOffset > 0) {
      if (typeof seek !== 'function') {
        throw new Error('OPFS writable stream does not support resumed writes.');
      }
      await seek.call(writable, startOffset);
    }
    if (
      typeof writable.write === 'function' &&
      typeof writable.close === 'function' &&
      typeof writable.abort === 'function'
    ) {
      return {
        write: (chunk) => writable.write(toFileSystemWriteChunk(chunk)),
        close: () => writable.close(),
        abort: () => writable.abort(),
        release: () => {},
      };
    }

    const writer = (writable as WritableStream<Uint8Array>).getWriter();
    return {
      write: (chunk) => writer.write(chunk),
      close: () => writer.close(),
      abort: () => writer.abort(),
      release: () => {
        writer.releaseLock();
      },
    };
  }

  private async createSyncWritableFileSink(
    handle: FileSystemFileHandle,
    startOffset = 0
  ): Promise<WritableFileSink | null> {
    const createSyncAccessHandle = (handle as unknown as {
      createSyncAccessHandle?: () => Promise<OpfsSyncAccessHandle>;
    }).createSyncAccessHandle;
    if (typeof createSyncAccessHandle !== 'function') {
      return null;
    }

    const access = await createSyncAccessHandle.call(handle);
    let offset = startOffset;
    if (startOffset === 0) {
      access.truncate(0);
    }
    return {
      write: async (chunk) => {
        const written = access.write(toFileSystemWriteChunk(chunk), { at: offset });
        if (written !== chunk.byteLength) {
          throw new Error(`OPFS write failed: expected ${chunk.byteLength} bytes, wrote ${written}.`);
        }
        // Each buffered write is a durable resume checkpoint if the Worker is terminated.
        access.flush();
        offset += written;
      },
      close: async () => {
        access.flush();
        access.close();
      },
      abort: async () => {
        try {
          access.flush();
        } catch {}
        access.close();
      },
      release: () => {},
    };
  }

  /**
   * Get a File handle for an existing file in storage.
   */
  public async getFile(fileName: string): Promise<File | null> {
    try {
      const root = await this.ensureRoot();
      const handle = await root.getFileHandle(fileName);
      return await handle.getFile();
    } catch (error) {
      if (!this.isNotFoundError(error)) {
        throw error;
      }
      return null;
    }
  }

  public async listFileNames(): Promise<string[]> {
    const root = await this.ensureRoot();
    const names: string[] = [];
    const entries = (root as unknown as {
      entries: () => AsyncIterable<[string, FileSystemFileHandle | FileSystemDirectoryHandle]>;
    }).entries;
    if (typeof entries !== 'function') {
      return names;
    }
    for await (const [name, handle] of entries.call(root)) {
      if (handle.kind === 'file') {
        names.push(name);
      }
    }
    return names;
  }

  public async listFileNamesAt(path: readonly string[]): Promise<string[]> {
    try {
      const directory = await this.getDirectory(path);
      const names: string[] = [];
      const entries = (directory as unknown as {
        entries: () => AsyncIterable<[string, FileSystemFileHandle | FileSystemDirectoryHandle]>;
      }).entries;
      if (typeof entries !== 'function') {
        return names;
      }
      for await (const [name, handle] of entries.call(directory)) {
        if (handle.kind === 'file') {
          names.push(name);
        }
      }
      return names;
    } catch (error) {
      if (!this.isNotFoundError(error)) {
        throw error;
      }
      return [];
    }
  }

  public async createSyncAccessHandle(
    fileName: string,
    options: { create?: boolean } = {}
  ): Promise<OpfsSyncAccessHandle> {
    const root = await this.ensureRoot();
    const handle = await root.getFileHandle(fileName, { create: options.create === true });
    const createSyncAccessHandle = (handle as unknown as {
      createSyncAccessHandle?: () => Promise<OpfsSyncAccessHandle>;
    }).createSyncAccessHandle;
    if (typeof createSyncAccessHandle !== 'function') {
      throw new Error(
        'OPFS sync access handles are unavailable. Large GGUF splitting must run in a browser worker that supports createSyncAccessHandle().'
      );
    }
    return await createSyncAccessHandle.call(handle);
  }

  public async readText(fileName: string): Promise<string | null> {
    const file = await this.getFile(fileName);
    if (file == null) {
      return null;
    }
    return await file.text();
  }

  public async readTextAt(path: readonly string[]): Promise<string | null> {
    try {
      const handle = await this.getFileHandleAt(path);
      return await (await handle.getFile()).text();
    } catch (error) {
      if (!this.isNotFoundError(error)) {
        throw error;
      }
      return null;
    }
  }

  public async writeText(fileName: string, contents: string): Promise<void> {
    const root = await this.ensureRoot();
    const handle = await root.getFileHandle(fileName, { create: true });
    const writable = await handle.createWritable();
    try {
      await writable.write(contents);
      await writable.close();
    } catch (error) {
      try {
        await writable.abort();
      } catch {}
      throw error;
    }
  }

  public async writeTextAt(path: readonly string[], contents: string): Promise<void> {
    const handle = await this.getFileHandleAt(path, { create: true });
    const writable = await handle.createWritable();
    try {
      await writable.write(contents);
      await writable.close();
    } catch (error) {
      try {
        await writable.abort();
      } catch {}
      throw error;
    }
  }

  /**
   * Stream a web response body directly to OPFS.
   */
  public async streamToDisk(
    fileName: string,
    stream: ReadableStream<Uint8Array>,
    options: StreamToDiskOptions = {}
  ): Promise<File> {
    const {
      onProgress,
      signal,
      startOffset = 0,
      stallTimeoutMs = 0,
      keepFileOnFailure = false,
    } = options;

    if (signal?.aborted) {
      throw createAbortError('File write aborted.');
    }

    const root = await this.ensureRoot();
    const handle = await root.getFileHandle(fileName, { create: true });

    const sink =
      (await this.createSyncWritableFileSink(handle, startOffset)) ??
      (await this.toWritableFileSink(
        await handle.createWritable(startOffset > 0 ? { keepExistingData: true } : undefined),
        startOffset
      ));
    const reader = stream.getReader();
    let closed = false;
    let bytesWritten = startOffset;
    let pendingBytes = 0;
    const pendingChunks: Uint8Array[] = [];

    const flushPending = async (): Promise<void> => {
      if (pendingBytes === 0) {
        return;
      }
      const chunk =
        pendingChunks.length === 1
          ? pendingChunks[0]
          : (() => {
              const merged = new Uint8Array(pendingBytes);
              let offset = 0;
              for (const part of pendingChunks) {
                merged.set(part, offset);
                offset += part.byteLength;
              }
              return merged;
            })();
      await sink.write(chunk);
      bytesWritten += pendingBytes;
      pendingChunks.length = 0;
      pendingBytes = 0;
      onProgress?.(bytesWritten);
    };

    try {
      if (startOffset > 0) {
        onProgress?.(startOffset);
      }

      while (true) {
        if (signal?.aborted) {
          throw createAbortError('File write aborted.');
        }

        let readResult: Awaited<ReturnType<ReadableStreamDefaultReader<Uint8Array>['read']>>;
        if (stallTimeoutMs > 0) {
          let timerId: ReturnType<typeof setTimeout> | undefined;
          const timeoutPromise = new Promise<never>((_, reject) => {
            timerId = setTimeout(() => {
              reject(new StreamStallError(stallTimeoutMs));
            }, stallTimeoutMs);
          });
          try {
            readResult = await Promise.race([reader.read(), timeoutPromise]);
          } finally {
            if (timerId !== undefined) {
              clearTimeout(timerId);
            }
          }
        } else {
          readResult = await reader.read();
        }

        const { done, value } = readResult;
        if (done) {
          break;
        }
        if (value == null) {
          continue;
        }

        if (value.byteLength >= STREAM_WRITE_BUFFER_BYTES) {
          await flushPending();
          await sink.write(value);
          bytesWritten += value.byteLength;
          onProgress?.(bytesWritten);
          continue;
        }

        pendingChunks.push(value);
        pendingBytes += value.byteLength;
        if (pendingBytes >= STREAM_WRITE_BUFFER_BYTES) {
          await flushPending();
        }
      }

      await flushPending();
      await sink.close();
      closed = true;
      return await handle.getFile();
    } catch (e) {
      try {
        if (!closed) {
          if (keepFileOnFailure) {
            try {
              await flushPending();
            } catch {}
            try {
              await sink.close();
              closed = true;
            } catch {
              await sink.abort();
            }
          } else {
            await sink.abort();
          }
        }
      } catch {}
      try {
        await reader.cancel(e);
      } catch {}
      if (!keepFileOnFailure) {
        try {
          await root.removeEntry(fileName);
        } catch {}
      }
      throw e;
    } finally {
      try {
        reader.releaseLock();
      } catch {}
      try {
        sink.release();
      } catch {}
    }
  }

  /**
   * Delete a file from storage.
   */
  public async deleteFile(fileName: string): Promise<void> {
    try {
      const root = await this.ensureRoot();
      await root.removeEntry(fileName);
    } catch (error) {
      if (!this.isNotFoundError(error)) {
        throw error;
      }
    }
  }

  public async deleteFileAt(path: readonly string[]): Promise<void> {
    try {
      if (path.length === 0) {
        throw new Error('OPFS file path must not be empty.');
      }
      const directory = await this.getDirectory(path.slice(0, -1));
      await directory.removeEntry(path[path.length - 1]);
    } catch (error) {
      if (!this.isNotFoundError(error)) {
        throw error;
      }
    }
  }
}
