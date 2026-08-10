import test from 'node:test';
import assert from 'node:assert/strict';
import { AssetStore, type GgufSplitRuntime, type RemoteAssetMetadata } from '../../src/models/asset-store.js';
import {
  BrowserAcquisitionJournal,
  recoverBrowserAcquisitionJournals,
} from '../../src/models/acquisition-journal.js';
import { QueryError } from '../../src/models/types.js';
import { FileSystemStorage, type OpfsSyncAccessHandle } from '../../src/engine/file-system-storage.js';

class MemorySyncAccessHandle implements OpfsSyncAccessHandle {
  private buffer: Uint8Array;

  constructor(
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

class MemoryStorage {
  public readonly files = new Map<string, File>();
  public readonly texts = new Map<string, string>();
  public readonly operations: string[] = [];
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
    this.operations.push(`stream:${fileName}`);
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
    options: { create?: boolean } = {}
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
    this.files.delete(fileName);
  }

  public async deleteFileAt(path: readonly string[]): Promise<void> {
    this.texts.delete(path.join('/'));
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

const emptyManifest = {
  version: 7,
  projectorIndexRevision: 0,
  assets: {},
  models: {},
} as const;

async function withSupportedStorage<T>(fn: () => Promise<T>): Promise<T> {
  const original = FileSystemStorage.isSupported;
  FileSystemStorage.isSupported = () => true;
  try {
    return await fn();
  } finally {
    FileSystemStorage.isSupported = original;
  }
}

function createTestAssetStore(storage: MemoryStorage): AssetStore {
  return new AssetStore(storage as unknown as FileSystemStorage);
}

async function withSyncAccessSupported<T>(fn: () => Promise<T>): Promise<T> {
  const original = FileSystemStorage.isSyncAccessSupported;
  FileSystemStorage.isSyncAccessSupported = async () => true;
  try {
    return await fn();
  } finally {
    FileSystemStorage.isSyncAccessSupported = original;
  }
}

test('AssetStore registers remote downloads without copying the OPFS temp file', async () => {
  await withSupportedStorage(async () => {
    const storage = new MemoryStorage();
    const store = createTestAssetStore(storage);
    const response = new Response(new Blob(['model-bytes']).stream(), { status: 200 });

    const receipt = await store.downloadRemote(metadata, 'model', response);
    const record = receipt.records[0];
    assert.ok(record);
    const file = await store.getFile(record);

    assert.equal(storage.writes.length, 1);
    assert.match(storage.writes[0], /^asset-[0-9a-f]{64}-model\.gguf$/);
    assert.equal(record.storagePath, storage.writes[0]);
    assert.match(record.id, /^asset-[0-9a-f]{64}$/);
    assert.equal(record.name, 'model.gguf');
    assert.equal(record.bytes, 11);
    assert.equal(record.sourceBytes, 11);
    assert.equal(await file.text(), 'model-bytes');
  });
});

test('AssetStore records remote download journal before creating the asset file', async () => {
  await withSupportedStorage(async () => {
    const storage = new MemoryStorage();
    const store = createTestAssetStore(storage);
    const journal = new BrowserAcquisitionJournal(
      storage as unknown as FileSystemStorage,
      'lease-1'
    );
    const response = new Response(new Blob(['model-bytes']).stream(), { status: 200 });

    await store.downloadRemote(metadata, 'model', response, undefined, undefined, journal);

    assert.equal(storage.operations.length, 2);
    assert.equal(storage.operations[0], 'journal:.incoming/journals/lease-1.json');
    assert.match(storage.operations[1], /^stream:asset-[0-9a-f]{64}-model\.gguf$/);
  });
});

test('Browser acquisition journal recovery preserves registered assets', async () => {
  const storage = new MemoryStorage();
  storage.files.set('asset-orphan', new File(['orphan'], 'asset-orphan'));
  storage.files.set('asset-keep', new File(['keep'], 'asset-keep'));
  storage.texts.set(
    '.incoming/journals/lease-2.json',
    JSON.stringify({
      version: 1,
      acquisitionId: 'lease-2',
      entries: [
        { storagePath: 'asset-orphan' },
        { storagePath: 'asset-keep' },
      ],
    })
  );

  await recoverBrowserAcquisitionJournals(storage as unknown as FileSystemStorage, {
    version: 7,
    projectorIndexRevision: 0,
    models: {},
    assets: {
      keep: {
        id: 'keep',
        kind: 'model',
        name: 'model.gguf',
        bytes: 4,
        storagePath: 'asset-keep',
        refCount: 1,
        createdAt: new Date(0).toISOString(),
      },
    },
  });

  assert.equal(storage.files.has('asset-orphan'), false);
  assert.equal(storage.files.has('asset-keep'), true);
  assert.equal(storage.texts.has('.incoming/journals/lease-2.json'), false);
});

test('AssetStore replaces wrong-sized deterministic remote files', async () => {
  await withSupportedStorage(async () => {
    const storage = new MemoryStorage();
    const store = createTestAssetStore(storage);
    const initial = new Response(new Blob(['model-bytes']).stream(), { status: 200 });
    const first = await store.downloadRemote(metadata, 'model', initial);
    const record = first.records[0];
    assert.ok(record);
    storage.files.set(record.storagePath, new File(['short'], record.storagePath));

    const replacement = new Response(new Blob(['model-bytes']).stream(), { status: 200 });
    const second = await store.downloadRemote(metadata, 'model', replacement);

    assert.deepEqual(storage.deleted, [record.storagePath]);
    assert.equal(second.records[0]?.storagePath, record.storagePath);
    assert.equal(second.records[0]?.bytes, 11);
    assert.equal(await storage.files.get(record.storagePath)?.text(), 'model-bytes');
  });
});

test('AssetStore surfaces quota failures with a storage-specific error code', async () => {
  await withSupportedStorage(async () => {
    const storage = new MemoryStorage();
    storage.failWith = new DOMException('quota full', 'QuotaExceededError');
    const store = createTestAssetStore(storage);
    const response = new Response(new Blob(['model-bytes']).stream(), { status: 200 });

    await assert.rejects(
      () => store.downloadRemote(metadata, 'model', response),
      (error) =>
        error instanceof QueryError &&
        error.code === 'STORAGE_QUOTA_EXCEEDED' &&
        error.message.includes('model.gguf')
    );
  });
});

test('AssetStore splits large local GGUF files through sync OPFS callbacks', async () => {
  await withSupportedStorage(async () => {
    await withSyncAccessSupported(async () => {
      const storage = new MemoryStorage();
      const store = createTestAssetStore(storage);
      const source = new File(['source'], 'local-model.gguf', { lastModified: 123456 });
      Object.defineProperty(source, 'size', { value: 3 * 1024 * 1024 * 1024 });
      const encoder = new TextEncoder();
      const runtime: GgufSplitRuntime = {
        async browserCacheLayout() {
          return 'split-gguf';
        },
        async planGgufSplitCount() {
          return 2;
        },
        async splitGgufStream(_sourceBytes, outputPrefix, _shardMaxBytes, callbacks) {
          for (let index = 0; index < 2; index += 1) {
            const path = `${outputPrefix}-${String(index + 1).padStart(5, '0')}-of-00002.gguf`;
            assert.equal(callbacks.openShard(path, index, 2), 0);
            assert.equal(callbacks.writeShard(encoder.encode(`shard-${index}`)), 0);
            assert.equal(callbacks.closeShard(), 0);
          }
        },
      };

      const records = await store.installLocalGguf(source, runtime, emptyManifest);

      assert.equal(records.length, 2);
      assert.deepEqual(
        records.map((record) => record.sourcePartIndex),
        [0, 1]
      );
      assert.ok(records.every((record) => record.kind === 'shard'));
      assert.ok(records.every((record) => record.sourceBytes === source.size));
      assert.ok(records.every((record) => record.sourceFileName === 'local-model.gguf'));
      assert.ok(records.every((record) => record.sourceFileLastModified === 123456));
      assert.equal(await (await store.getFile(records[0])).text(), 'shard-0');
      assert.ok(storage.deleted.some((path) => path.startsWith('tmp-local-source-')));
    });
  });
});

test('AssetStore cleans browser split temp files and unregistered shards', async () => {
  await withSupportedStorage(async () => {
    const storage = new MemoryStorage();
    storage.files.set('tmp-source-leftover.gguf', new File(['tmp'], 'tmp-source-leftover.gguf'));
    storage.files.set('tmp-local-source-leftover.gguf', new File(['tmp'], 'tmp-local-source-leftover.gguf'));
    storage.files.set('split-orphan-00001-of-00002.gguf', new File(['orphan'], 'split-orphan-00001-of-00002.gguf'));
    storage.files.set('split-keep-00001-of-00002.gguf', new File(['keep'], 'split-keep-00001-of-00002.gguf'));
    storage.files.set('asset-model.gguf', new File(['model'], 'asset-model.gguf'));
    const store = createTestAssetStore(storage);

    await store.cleanupBrowserSplitArtifacts({
      version: 7,
      projectorIndexRevision: 0,
      models: {},
      assets: {
        keep: {
          id: 'keep',
          kind: 'shard',
          name: 'split-keep-00001-of-00002.gguf',
          bytes: 4,
          storagePath: 'split-keep-00001-of-00002.gguf',
          refCount: 0,
          createdAt: new Date(0).toISOString(),
        },
      },
    });

    assert.equal(storage.files.has('tmp-source-leftover.gguf'), false);
    assert.equal(storage.files.has('tmp-local-source-leftover.gguf'), false);
    assert.equal(storage.files.has('split-orphan-00001-of-00002.gguf'), false);
    assert.equal(storage.files.has('split-keep-00001-of-00002.gguf'), true);
    assert.equal(storage.files.has('asset-model.gguf'), true);
  });
});
