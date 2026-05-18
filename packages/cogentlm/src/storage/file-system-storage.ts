import { createAbortError } from '../utils/abort.js';

export interface OpfsSyncAccessHandle {
  read(buffer: Uint8Array, options?: { at?: number }): number;
  write(buffer: Uint8Array, options?: { at?: number }): number;
  truncate(size: number): void;
  flush(): void;
  close(): void;
}

/**
 * FileSystemStorage provides an abstraction for the Origin Private File System (OPFS),
 * allowing large assets to be streamed directly to browser-managed persistent storage.
 *
 * This enables zero-copy loading of models >2GB from URLs by:
 * 1. Streaming the download directly to a file on disk.
 * 2. Retrieving a native File handle from the stored file.
 * 3. Mounting that File into the WASM filesystem via WORKERFS.
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

    // We use createWritable() which returns a FileSystemWritableFileStream.
    // In some browsers (Firefox), this might be behind a flag or limited.
    // Use the modern piping API if possible.
    const writable = await handle.createWritable();
    try {
      let bytesWritten = 0;
      const progressTransformer = new TransformStream({
        transform(chunk, controller) {
          bytesWritten += chunk.byteLength;
          if (onProgress) onProgress(bytesWritten);
          controller.enqueue(chunk);
        }
      });

      await stream.pipeThrough(progressTransformer).pipeTo(writable, { signal });
      return await handle.getFile();
    } catch (e) {
      // Cleanup on failure
      try { await writable.abort(); } catch {}
      try {
        await root.removeEntry(fileName);
      } catch {}
      throw e;
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
