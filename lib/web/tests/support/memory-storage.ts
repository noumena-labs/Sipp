import {
  StreamStallError,
  type OpfsSyncAccessHandle,
} from '../../src/engine/file-system-storage.js';

class MemorySyncAccessHandle implements OpfsSyncAccessHandle {
  private buffer: Uint8Array;

  public constructor(
    private readonly files: Map<string, File>,
    private readonly fileName: string,
    bytes: Uint8Array
  ) {
    this.buffer = bytes.slice();
  }

  public read(target: Uint8Array, options: { at?: number } = {}): number {
    const offset = options.at ?? 0;
    const source = this.buffer.subarray(offset, offset + target.byteLength);
    target.set(source);
    return source.byteLength;
  }

  public write(source: Uint8Array, options: { at?: number } = {}): number {
    const offset = options.at ?? 0;
    const end = offset + source.byteLength;
    if (end > this.buffer.byteLength) {
      const next = new Uint8Array(end);
      next.set(this.buffer);
      this.buffer = next;
    }
    this.buffer.set(source, offset);
    return source.byteLength;
  }

  public truncate(size: number): void {
    this.buffer = this.buffer.slice(0, size);
  }

  public flush(): void {}

  public close(): void {
    this.files.set(this.fileName, new File([this.buffer], this.fileName));
  }
}

interface MemoryStreamOptions {
  readonly onProgress?: (bytes: number) => void;
  readonly signal?: AbortSignal;
  readonly startOffset?: number;
  readonly stallTimeoutMs?: number;
  readonly keepFileOnFailure?: boolean;
}

/** In-memory OPFS substitute shared by browser storage and acquisition tests. */
export class MemoryStorage {
  public readonly files = new Map<string, File>();
  public readonly texts = new Map<string, string>();
  public readonly operations: string[] = [];
  public readonly writes: string[] = [];
  public readonly deleted: string[] = [];
  public failWith: unknown = null;

  public async streamToDisk(
    fileName: string,
    stream: ReadableStream<Uint8Array>,
    options: MemoryStreamOptions = {}
  ): Promise<File> {
    if (this.failWith != null) {
      throw this.failWith;
    }
    const {
      keepFileOnFailure = false,
      onProgress,
      startOffset = 0,
      stallTimeoutMs = 0,
    } = options;
    this.operations.push(`stream:${fileName}`);
    this.writes.push(fileName);
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];
    const existing = this.files.get(fileName);
    if (startOffset > 0 && existing != null) {
      chunks.push(new Uint8Array(await existing.arrayBuffer()).slice(0, startOffset));
    }
    let bytes = startOffset;
    if (startOffset > 0) {
      onProgress?.(startOffset);
    }

    try {
      while (true) {
        const readResult = await readWithTimeout(reader, stallTimeoutMs);
        const { done, value } = readResult;
        if (done) {
          break;
        }
        if (value != null) {
          chunks.push(value);
          bytes += value.byteLength;
          onProgress?.(bytes);
        }
      }
    } catch (error) {
      if (keepFileOnFailure && chunks.length > 0) {
        this.files.set(fileName, new File(chunks, fileName));
      }
      throw error;
    } finally {
      reader.releaseLock();
    }

    const file = new File(chunks, fileName);
    this.files.set(fileName, file);
    return file;
  }

  public async getFile(fileName: string): Promise<File | null> {
    return this.files.get(fileName) ?? null;
  }

  public async listFileNames(): Promise<string[]> {
    return [...this.files.keys()];
  }

  public async listFileNamesAt(path: readonly string[]): Promise<string[]> {
    const prefix = `${path.join('/')}/`;
    return [...this.texts.keys()]
      .filter((key) => key.startsWith(prefix))
      .map((key) => key.slice(prefix.length))
      .filter((key) => !key.includes('/'));
  }

  public async readTextAt(path: readonly string[]): Promise<string | null> {
    return this.texts.get(path.join('/')) ?? null;
  }

  public async writeTextAt(path: readonly string[], contents: string): Promise<void> {
    const key = path.join('/');
    this.operations.push(`journal:${key}`);
    this.texts.set(key, contents);
  }

  public async createSyncAccessHandle(
    fileName: string,
    options: { readonly create?: boolean } = {}
  ): Promise<OpfsSyncAccessHandle> {
    const file = this.files.get(fileName);
    if (file == null && options.create !== true) {
      throw new DOMException('Missing file', 'NotFoundError');
    }
    const bytes = file == null ? new Uint8Array() : new Uint8Array(await file.arrayBuffer());
    return new MemorySyncAccessHandle(this.files, fileName, bytes);
  }

  public async deleteFile(fileName: string): Promise<void> {
    this.deleted.push(fileName);
    this.operations.push(`delete:${fileName}`);
    this.files.delete(fileName);
  }

  public async deleteFileAt(path: readonly string[]): Promise<void> {
    this.texts.delete(path.join('/'));
  }
}

async function readWithTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  stallTimeoutMs: number
): Promise<ReadableStreamReadResult<Uint8Array>> {
  if (stallTimeoutMs === 0) {
    return await reader.read();
  }
  let timerId: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timerId = setTimeout(() => reject(new StreamStallError(stallTimeoutMs)), stallTimeoutMs);
  });
  try {
    return await Promise.race([reader.read(), timeout]);
  } finally {
    if (timerId !== undefined) {
      clearTimeout(timerId);
    }
  }
}
