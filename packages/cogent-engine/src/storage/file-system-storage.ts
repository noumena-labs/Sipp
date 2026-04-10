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

  private async ensureRoot(): Promise<FileSystemDirectoryHandle> {
    if (this.root) return this.root;
    const opfsRoot = await navigator.storage.getDirectory();
    this.root = await opfsRoot.getDirectoryHandle(this.dirName, { create: true });
    return this.root;
  }

  /**
   * Get a File handle for an existing file in storage.
   */
  public async getFile(fileName: string): Promise<File | null> {
    try {
      const root = await this.ensureRoot();
      const handle = await root.getFileHandle(fileName);
      return await handle.getFile();
    } catch (e) {
      return null;
    }
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
      try {
        await root.removeEntry(fileName);
      } catch {}
      throw error;
    }
  }

  public async estimate(): Promise<{
    usageBytes: number | null;
    quotaBytes: number | null;
  }> {
    if (
      typeof navigator === 'undefined' ||
      typeof navigator.storage?.estimate !== 'function'
    ) {
      return {
        usageBytes: null,
        quotaBytes: null,
      };
    }

    try {
      const estimate = await navigator.storage.estimate();
      return {
        usageBytes:
          typeof estimate.usage === 'number' && Number.isFinite(estimate.usage)
            ? estimate.usage
            : null,
        quotaBytes:
          typeof estimate.quota === 'number' && Number.isFinite(estimate.quota)
            ? estimate.quota
            : null,
      };
    } catch {
      return {
        usageBytes: null,
        quotaBytes: null,
      };
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
    const root = await this.ensureRoot();
    const handle = await root.getFileHandle(fileName, { create: true });

    // We use createWritable() which returns a FileSystemWritableFileStream.
    // In some browsers (Firefox), this might be behind a flag or limited.
    // Use the modern piping API if possible.
    const writable = await handle.createWritable();
    const abortListener =
      signal == null
        ? null
        : () => {
            void writable.abort();
          };
    
    try {
      let bytesWritten = 0;
      const progressTransformer = new TransformStream({
        transform(chunk, controller) {
          bytesWritten += chunk.byteLength;
          if (onProgress) onProgress(bytesWritten);
          controller.enqueue(chunk);
        }
      });

      if (abortListener != null) {
        signal?.addEventListener('abort', abortListener, { once: true });
      }

      await stream.pipeThrough(progressTransformer).pipeTo(writable);
      return await handle.getFile();
    } catch (e) {
      // Cleanup on failure
      try { await writable.abort(); } catch {}
      try {
        await root.removeEntry(fileName);
      } catch {}
      throw e;
    } finally {
      if (abortListener != null) {
        signal?.removeEventListener('abort', abortListener);
      }
    }
  }

  /**
   * Delete a file from storage.
   */
  public async deleteFile(fileName: string): Promise<void> {
    try {
      const root = await this.ensureRoot();
      await root.removeEntry(fileName);
    } catch (e) {}
  }

  /**
   * List all cached model files.
   */
  public async listFiles(): Promise<string[]> {
    try {
      const root = await this.ensureRoot();
      const names: string[] = [];
      // @ts-ignore - async iterator on entries()
      for await (const name of root.keys()) {
        names.push(name);
      }
      return names;
    } catch (e) {
      return [];
    }
  }

  /**
   * Clear all cached files.
   */
  public async clear(): Promise<void> {
    try {
      const opfsRoot = await navigator.storage.getDirectory();
      await opfsRoot.removeEntry(this.dirName, { recursive: true });
      this.root = null;
    } catch (e) {}
  }
}
