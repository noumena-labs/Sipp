import test from 'node:test';
import assert from 'node:assert/strict';
import { AssetStore, type RemoteAssetMetadata } from './asset-store.js';
import { QueryError } from './model-types.js';
import { FileSystemStorage } from '../storage/file-system-storage.js';

class MemoryStorage {
  public readonly files = new Map<string, File>();
  public readonly writes: string[] = [];
  public readonly deleted: string[] = [];
  public failWith: unknown = null;

  public async streamToDisk(
    fileName: string,
    stream: ReadableStream<Uint8Array>,
    onProgress?: (bytes: number) => void
  ): Promise<File> {
    if (this.failWith != null) {
      throw this.failWith;
    }
    this.writes.push(fileName);
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];
    let bytes = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (value != null) {
          chunks.push(value);
          bytes += value.byteLength;
          onProgress?.(bytes);
        }
      }
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

  public async deleteFile(fileName: string): Promise<void> {
    this.deleted.push(fileName);
    this.files.delete(fileName);
  }
}

const metadata: RemoteAssetMetadata = {
  url: 'https://models.test/model.gguf',
  canonicalUrl: 'https://models.test/model.gguf',
  name: 'model.gguf',
  bytes: 11,
  etag: '"v1"',
  lastModified: 'Wed, 01 May 2024 00:00:00 GMT',
};

async function withSupportedStorage<T>(fn: () => Promise<T>): Promise<T> {
  const original = FileSystemStorage.isSupported;
  FileSystemStorage.isSupported = () => true;
  try {
    return await fn();
  } finally {
    FileSystemStorage.isSupported = original;
  }
}

async function withFetchResponse<T>(body: string, fn: () => Promise<T>): Promise<T> {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(new Blob([body]).stream(), { status: 200 });
  try {
    return await fn();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test('AssetStore registers remote downloads without copying the OPFS temp file', async () => {
  await withSupportedStorage(async () => {
    await withFetchResponse('model-bytes', async () => {
      const storage = new MemoryStorage();
      const store = new AssetStore(storage as unknown as FileSystemStorage);

      const record = await store.downloadRemote(metadata, 'model');
      const file = await store.getFile(record);

      assert.equal(storage.writes.length, 1);
      assert.match(storage.writes[0], /^tmp-/);
      assert.equal(record.storagePath, storage.writes[0]);
      assert.equal(record.name, 'model.gguf');
      assert.equal(record.bytes, 11);
      assert.equal(await file.text(), 'model-bytes');
    });
  });
});

test('AssetStore surfaces quota failures with a storage-specific error code', async () => {
  await withSupportedStorage(async () => {
    await withFetchResponse('model-bytes', async () => {
      const storage = new MemoryStorage();
      storage.failWith = new DOMException('quota full', 'QuotaExceededError');
      const store = new AssetStore(storage as unknown as FileSystemStorage);

      await assert.rejects(
        () => store.downloadRemote(metadata, 'model'),
        (error) =>
          error instanceof QueryError &&
          error.code === 'STORAGE_QUOTA_EXCEEDED' &&
          error.message.includes('model.gguf')
      );
    });
  });
});
