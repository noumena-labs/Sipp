import test from 'node:test';
import assert from 'node:assert/strict';
import { AssetStore, type GgufSplitRuntime, type RemoteAssetMetadata } from '../../src/models/asset-store.js';
import {
  BrowserAcquisitionJournal,
  recoverBrowserAcquisitionState,
} from '../../src/models/acquisition-journal.js';
import { QueryError, type ModelLoadProgress } from '../../src/models/types.js';
import { FileSystemStorage } from '../../src/engine/file-system-storage.js';
import { MemoryStorage } from '../support/memory-storage.js';

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

const singleFileRuntime: GgufSplitRuntime = {
  browserCacheLayout: async () => 'single-file',
  planGgufSplitCount: async () => 1,
  splitGgufStream: async () => {},
};

async function downloadRemote(
  store: AssetStore,
  body: ReadableStream<Uint8Array> | null,
  options: {
    readonly metadata?: RemoteAssetMetadata;
    readonly journal?: BrowserAcquisitionJournal;
    readonly onProgress?: (progress: ModelLoadProgress) => void;
  } = {}
) {
  const remoteMetadata = options.metadata ?? metadata;
  const plan = await store.prepareRemoteDownload(remoteMetadata, singleFileRuntime);
  return await store.downloadRemoteGguf(
    remoteMetadata,
    singleFileRuntime,
    {
      plan,
      body,
      onProgress: options.onProgress,
      journal: options.journal,
    }
  );
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
    const body = new Blob(['model-bytes']).stream() as unknown as ReadableStream<Uint8Array>;

    const receipt = await downloadRemote(store, body);
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

test('AssetStore records resumable download metadata before creating the asset file', async () => {
  await withSupportedStorage(async () => {
    const storage = new MemoryStorage();
    const store = createTestAssetStore(storage);
    const journal = new BrowserAcquisitionJournal(
      storage as unknown as FileSystemStorage,
      'lease-1'
    );
    const body = new Blob(['model-bytes']).stream() as unknown as ReadableStream<Uint8Array>;

    await downloadRemote(store, body, { journal });

    assert.equal(storage.operations.length, 2);
    assert.match(storage.operations[0], /^journal:\.incoming\/partials\/[0-9a-f]{64}\.json$/);
    assert.match(storage.operations[1], /^stream:asset-[0-9a-f]{64}-model\.gguf$/);
  });
});

test('Browser acquisition recovery retains resumable bytes across page reloads', async () => {
  const storage = new MemoryStorage();
  const store = createTestAssetStore(storage);
  const plan = await store.prepareRemoteDownload(metadata, singleFileRuntime);
  const journal = new BrowserAcquisitionJournal(
    storage as unknown as FileSystemStorage,
    'lease-resume'
  );
  await journal.recordResumableDownload(plan.storagePath, metadata.bytes);
  await journal.recordTemporaryPaths(['split-uncommitted.gguf']);
  storage.files.set(plan.storagePath, new File(['partial'], plan.storagePath));
  storage.files.set(
    'split-uncommitted.gguf',
    new File(['temporary'], 'split-uncommitted.gguf')
  );

  await recoverBrowserAcquisitionState(
    storage as unknown as FileSystemStorage,
    emptyManifest
  );

  assert.equal(storage.files.get(plan.storagePath)?.size, 7);
  assert.equal(storage.files.has('split-uncommitted.gguf'), false);
  assert.equal(storage.texts.has('.incoming/journals/lease-resume.json'), false);
  assert.ok([...storage.texts.keys()].some((path) => path.startsWith('.incoming/partials/')));
  assert.equal(
    (await store.prepareRemoteDownload(metadata, singleFileRuntime)).startOffset,
    7
  );
});

test('Browser acquisition rollback preserves resumable bytes and removes generated artifacts', async () => {
  const storage = new MemoryStorage();
  const store = createTestAssetStore(storage);
  const plan = await store.prepareRemoteDownload(metadata, singleFileRuntime);
  const journal = new BrowserAcquisitionJournal(
    storage as unknown as FileSystemStorage,
    'lease-rollback'
  );
  await journal.recordResumableDownload(plan.storagePath, metadata.bytes);
  await journal.recordTemporaryPaths(['split-rollback.gguf']);
  storage.files.set(plan.storagePath, new File(['partial'], plan.storagePath));
  storage.files.set('split-rollback.gguf', new File(['temporary'], 'split-rollback.gguf'));

  await journal.cleanupUncommitted(emptyManifest);

  assert.equal(storage.files.get(plan.storagePath)?.size, 7);
  assert.equal(storage.files.has('split-rollback.gguf'), false);
  assert.ok([...storage.texts.keys()].some((path) => path.startsWith('.incoming/partials/')));
  assert.equal(storage.texts.has('.incoming/journals/lease-rollback.json'), false);
});

test('Browser acquisition cleanup discards stale resumable downloads', async () => {
  const storage = new MemoryStorage();
  const store = createTestAssetStore(storage);
  const plan = await store.prepareRemoteDownload(metadata, singleFileRuntime);
  const journal = new BrowserAcquisitionJournal(
    storage as unknown as FileSystemStorage,
    'lease-stale'
  );
  await journal.recordResumableDownload(plan.storagePath, metadata.bytes);
  storage.files.set(plan.storagePath, new File(['partial'], plan.storagePath));
  const markerPath = [...storage.texts.keys()].find((path) =>
    path.startsWith('.incoming/partials/')
  );
  assert.ok(markerPath != null);
  const marker = JSON.parse(storage.texts.get(markerPath) ?? '{}') as Record<string, unknown>;
  marker.updatedAt = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();
  storage.texts.set(markerPath, JSON.stringify(marker));

  await recoverBrowserAcquisitionState(
    storage as unknown as FileSystemStorage,
    emptyManifest
  );

  assert.equal(storage.files.has(plan.storagePath), false);
  assert.equal(storage.texts.has(markerPath), false);
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

  await recoverBrowserAcquisitionState(storage as unknown as FileSystemStorage, {
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
    const initial = new Blob(['model-bytes']).stream() as unknown as ReadableStream<Uint8Array>;
    const first = await downloadRemote(store, initial);
    const record = first.records[0];
    assert.ok(record);
    storage.files.set(record.storagePath, new File(['short'], record.storagePath));

    const replacement = new Blob(['model-bytes']).stream() as unknown as ReadableStream<Uint8Array>;
    const stalePlan = await store.prepareRemoteDownload(metadata, singleFileRuntime);
    const plan = await store.resetRemoteDownload(stalePlan);
    const second = await store.downloadRemoteGguf(
      metadata,
      singleFileRuntime,
      { plan, body: replacement }
    );

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
    const body = new Blob(['model-bytes']).stream() as unknown as ReadableStream<Uint8Array>;

    await assert.rejects(
      () => downloadRemote(store, body),
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

test('AssetStore cleans local split temp files and unregistered shards', async () => {
  await withSupportedStorage(async () => {
    const storage = new MemoryStorage();
    storage.files.set('tmp-source-leftover.gguf', new File(['tmp'], 'tmp-source-leftover.gguf'));
    storage.files.set('tmp-local-source-leftover.gguf', new File(['tmp'], 'tmp-local-source-leftover.gguf'));
    storage.files.set('split-orphan-00001-of-00002.gguf', new File(['orphan'], 'split-orphan-00001-of-00002.gguf'));
    storage.files.set('split-local-orphan-00001-of-00002.gguf', new File(['orphan'], 'split-local-orphan-00001-of-00002.gguf'));
    storage.files.set('split-local-keep-00001-of-00002.gguf', new File(['keep'], 'split-local-keep-00001-of-00002.gguf'));
    storage.files.set('asset-model.gguf', new File(['model'], 'asset-model.gguf'));
    const store = createTestAssetStore(storage);

    await store.cleanupLocalSplitArtifacts({
      version: 7,
      projectorIndexRevision: 0,
      models: {},
      assets: {
        keep: {
          id: 'keep',
          kind: 'shard',
          name: 'split-local-keep-00001-of-00002.gguf',
          bytes: 4,
          storagePath: 'split-local-keep-00001-of-00002.gguf',
          refCount: 0,
          createdAt: new Date(0).toISOString(),
        },
      },
    });

    assert.equal(storage.files.has('tmp-source-leftover.gguf'), true);
    assert.equal(storage.files.has('tmp-local-source-leftover.gguf'), false);
    assert.equal(storage.files.has('split-orphan-00001-of-00002.gguf'), true);
    assert.equal(storage.files.has('split-local-orphan-00001-of-00002.gguf'), false);
    assert.equal(storage.files.has('split-local-keep-00001-of-00002.gguf'), true);
    assert.equal(storage.files.has('asset-model.gguf'), true);
  });
});

test('AssetStore supports resumable downloads with 206 Partial Content', async () => {
  await withSupportedStorage(async () => {
    const storage = new MemoryStorage();
    const store = createTestAssetStore(storage);
    const metadata: RemoteAssetMetadata = {
      url: 'https://example.com/model.gguf',
      canonicalUrl: 'https://example.com/model.gguf',
      name: 'model.gguf',
      bytes: 5,
    };
    const plan = await store.prepareRemoteDownload(metadata, singleFileRuntime);
    const { storagePath } = plan;

    storage.files.set(storagePath, new File([Uint8Array.from([1, 2, 3])], storagePath));

    const progress: number[] = [];
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(Uint8Array.from([4, 5]));
        controller.close();
      },
    });

    const resumedPlan = await store.prepareRemoteDownload(metadata, singleFileRuntime);
    const receipt = await store.downloadRemoteGguf(
      metadata,
      singleFileRuntime,
      {
        plan: resumedPlan,
        body,
        onProgress: (p) => progress.push(p.loadedBytes),
      }
    );

    assert.equal(receipt.records.length, 1);
    assert.equal(receipt.records[0].bytes, 5);
    const finalFile = await store.getFile(receipt.records[0]);
    assert.equal(finalFile.size, 5);
    const content = new Uint8Array(await finalFile.arrayBuffer());
    assert.deepEqual([...content], [1, 2, 3, 4, 5]);
    assert.ok(progress.includes(3));
    assert.ok(progress.includes(5));
  });
});
