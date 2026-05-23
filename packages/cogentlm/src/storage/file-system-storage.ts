import { createAbortError } from '../utils/abort.js';

interface WritableFileSink {
  write(chunk: Uint8Array): Promise<void>;
  close(): Promise<void>;
  abort(): Promise<void>;
  release(): void;
}

const STREAM_WRITE_BUFFER_BYTES = 4 * 1024 * 1024;

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
  private readonly dirName = 'cogent-models';

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
        `cogent-sync-access-probe-${Date.now().toString(36)}`,
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
    const opfsRoot = await navigator.storage.getDirectory();
    this.root = await opfsRoot.getDirectoryHandle(this.dirName, { create: true });
    return this.root;
  }

  private isNotFoundError(error: unknown): boolean {
    return typeof DOMException === 'function' && error instanceof DOMException && error.name === 'NotFoundError';
  }

  private toWritableFileSink(writable: FileSystemWritableFileStream): WritableFileSink {
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
    handle: FileSystemFileHandle
  ): Promise<WritableFileSink | null> {
    const createSyncAccessHandle = (handle as unknown as {
      createSyncAccessHandle?: () => Promise<OpfsSyncAccessHandle>;
    }).createSyncAccessHandle;
    if (typeof createSyncAccessHandle !== 'function') {
      return null;
    }

    const access = await createSyncAccessHandle.call(handle);
    let offset = 0;
    access.truncate(0);
    return {
      write: async (chunk) => {
        const written = access.write(toFileSystemWriteChunk(chunk), { at: offset });
        if (written !== chunk.byteLength) {
          throw new Error(`OPFS write failed: expected ${chunk.byteLength} bytes, wrote ${written}.`);
        }
        offset += written;
      },
      close: async () => {
        access.flush();
        access.close();
      },
      abort: async () => {
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

  /**
   * Stream a web response body directly to OPFS.
   */
  public async streamToDisk(
    fileName: string,
    stream: ReadableStream<Uint8Array>,
    onProgress?: (bytes: number) => void,
    signal?: AbortSignal
  ): Promise<File> {
    if (signal?.aborted) {
      throw createAbortError('File write aborted.');
    }

    const root = await this.ensureRoot();
    const handle = await root.getFileHandle(fileName, { create: true });

    const sink =
      (await this.createSyncWritableFileSink(handle)) ??
      this.toWritableFileSink(await handle.createWritable());
    const reader = stream.getReader();
    let closed = false;
    try {
      let bytesWritten = 0;
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

      while (true) {
        if (signal?.aborted) {
          throw createAbortError('File write aborted.');
        }

        const { done, value } = await reader.read();
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
      // Cleanup on failure
      try {
        if (!closed) {
          await sink.abort();
        }
      } catch {}
      try {
        await reader.cancel(e);
      } catch {}
      try {
        await root.removeEntry(fileName);
      } catch {}
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
}
