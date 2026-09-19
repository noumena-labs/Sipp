import test from 'node:test';
import assert from 'node:assert/strict';
import { RemoteAcquisitionHost } from '../../src/models/remote-acquisition-host.js';
import { AssetStore, type GgufSplitRuntime, type RemoteAssetMetadata } from '../../src/models/asset-store.js';
import { FileSystemStorage } from '../../src/engine/file-system-storage.js';
import type { RustRemoteAction } from '../../src/wasm/wasm-bridge.js';
import type { FallbackEvent, RegistryManifest } from '../../src/models/types.js';
import { MemoryStorage } from '../support/memory-storage.js';

const fakeRuntime: GgufSplitRuntime = {
  browserCacheLayout: async () => 'single-file',
  planGgufSplitCount: async () => 1,
  splitGgufStream: async () => {},
};

const fakeManifest: RegistryManifest = {
  version: 7,
  projectorIndexRevision: 0,
  models: {},
  assets: {},
};

const fakeClassify = async (assetId: string) => ({
  assetId,
  name: 'model.gguf',
  inspection: {
    format: 'gguf' as const,
    architecture: 'llama',
    alignment: 32,
    contextLength: 2048,
    embeddingLength: 4096,
    blockCount: 32,
    feedForwardLength: 11008,
    headCount: 32,
    headCountKv: 32,
    fileType: 1,
    chatTemplate: null,
    bosTokenId: 1,
    eosTokenId: 2,
  },
});

async function withSupportedStorage<T>(fn: () => Promise<T>): Promise<T> {
  const original = FileSystemStorage.isSupported;
  FileSystemStorage.isSupported = () => true;
  try {
    return await fn();
  } finally {
    FileSystemStorage.isSupported = original;
  }
}

async function withCustomFetch<T>(
  customFetch: typeof globalThis.fetch,
  fn: () => Promise<T>
): Promise<T> {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = customFetch;
  try {
    return await withSupportedStorage(fn);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test('RemoteAcquisitionHost executes initial download without Range header', async () => {
  const storage = new MemoryStorage();
  const assetStore = new AssetStore(storage as unknown as FileSystemStorage);
  const host = new RemoteAcquisitionHost(
    assetStore,
    fakeRuntime,
    fakeManifest,
    fakeClassify,
    {}
  );

  let capturedHeaders: HeadersInit | undefined;
  await withCustomFetch(
    async (_input, init) => {
      capturedHeaders = init?.headers;
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(Uint8Array.from([1, 2, 3, 4, 5]));
            controller.close();
          },
        }),
        { status: 200, headers: { 'Content-Length': '5' } }
      );
    },
    async () => {
      const action: RustRemoteAction = {
        kind: 'download',
        acquisitionId: 'acq-1',
        memberId: 0,
        attempt: 1,
        metadata: {
          url: 'https://example.com/model.gguf',
          name: 'model.gguf',
          bytes: 5,
        },
      };

      const result = await host.execute(action);
      assert.equal(result.event.kind, 'download_succeeded');
      assert.equal(capturedHeaders, undefined);

      const files = [...storage.files.values()];
      assert.equal(files.length, 1);
      assert.equal(files[0].size, 5);
      const content = new Uint8Array(await files[0].arrayBuffer());
      assert.deepEqual([...content], [1, 2, 3, 4, 5]);
      assert.equal(storage.texts.size, 1);
      await host.commitJournal();
      assert.equal(storage.texts.size, 0);
    }
  );
});

test('RemoteAcquisitionHost resumes partial download with Range and If-Range headers', async () => {
  const storage = new MemoryStorage();
  const assetStore = new AssetStore(storage as unknown as FileSystemStorage);
  const host = new RemoteAcquisitionHost(
    assetStore,
    fakeRuntime,
    fakeManifest,
    fakeClassify,
    {}
  );

  const metadata: RemoteAssetMetadata = {
    url: 'https://example.com/model.gguf',
    canonicalUrl: 'https://example.com/model.gguf',
    name: 'model.gguf',
    bytes: 5,
    etag: '"model-etag-123"',
  };

  const { storagePath } = await assetStore.prepareRemoteDownload(metadata, fakeRuntime);
  storage.files.set(storagePath, new File([Uint8Array.from([1, 2, 3])], storagePath));

  let capturedHeaders: Record<string, string> | undefined;
  await withCustomFetch(
    async (_input, init) => {
      capturedHeaders = init?.headers as Record<string, string>;
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(Uint8Array.from([4, 5]));
            controller.close();
          },
        }),
        {
          status: 206,
          headers: {
            'Content-Range': 'bytes 3-4/5',
            'Content-Length': '2',
          },
        }
      );
    },
    async () => {
      const action: RustRemoteAction = {
        kind: 'download',
        acquisitionId: 'acq-2',
        memberId: 0,
        attempt: 2,
        metadata: {
          url: metadata.url,
          name: metadata.name,
          bytes: metadata.bytes,
          etag: metadata.etag,
        },
      };

      const result = await host.execute(action);
      assert.equal(result.event.kind, 'download_succeeded');
      assert.equal(capturedHeaders?.['Range'], 'bytes=3-');
      assert.equal(capturedHeaders?.['If-Range'], '"model-etag-123"');

      const file = storage.files.get(storagePath);
      assert.ok(file != null);
      assert.equal(file.size, 5);
      const content = new Uint8Array(await file.arrayBuffer());
      assert.deepEqual([...content], [1, 2, 3, 4, 5]);
    }
  );
});

test('RemoteAcquisitionHost resets offset when server returns 416 Range Not Satisfiable', async () => {
  const storage = new MemoryStorage();
  const assetStore = new AssetStore(storage as unknown as FileSystemStorage);
  const warnings: FallbackEvent[] = [];
  const host = new RemoteAcquisitionHost(
    assetStore,
    fakeRuntime,
    fakeManifest,
    fakeClassify,
    { onWarning: (event) => warnings.push(event) }
  );

  const metadata: RemoteAssetMetadata = {
    url: 'https://example.com/model.gguf',
    canonicalUrl: 'https://example.com/model.gguf',
    name: 'model.gguf',
    bytes: 5,
  };

  const { storagePath } = await assetStore.prepareRemoteDownload(metadata, fakeRuntime);
  storage.files.set(storagePath, new File([Uint8Array.from([1, 2, 3])], storagePath));

  let callCount = 0;
  await withCustomFetch(
    async () => {
      callCount += 1;
      if (callCount === 1) {
        return new Response(null, { status: 416 });
      }
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(Uint8Array.from([10, 20, 30, 40, 50]));
            controller.close();
          },
        }),
        { status: 200, headers: { 'Content-Length': '5' } }
      );
    },
    async () => {
      const action: RustRemoteAction = {
        kind: 'download',
        acquisitionId: 'acq-3',
        memberId: 0,
        attempt: 2,
        metadata: {
          url: metadata.url,
          name: metadata.name,
          bytes: metadata.bytes,
        },
      };

      const result = await host.execute(action);
      assert.equal(result.event.kind, 'download_succeeded');
      assert.equal(callCount, 2);
      assert.deepEqual(warnings, [{
        type: 'fallback-warning',
        kind: 'transfer',
        detail:
          'Resumable download fallback for "model.gguf": range request was not satisfiable.',
        fallbackTo: 'full-download',
      }]);

      const file = storage.files.get(storagePath);
      assert.ok(file != null);
      assert.equal(file.size, 5);
      const content = new Uint8Array(await file.arrayBuffer());
      assert.deepEqual([...content], [10, 20, 30, 40, 50]);
    }
  );
});

test('RemoteAcquisitionHost discards an ignored range response before a full retry', async () => {
  const storage = new MemoryStorage();
  const assetStore = new AssetStore(storage as unknown as FileSystemStorage);
  const warnings: FallbackEvent[] = [];
  const host = new RemoteAcquisitionHost(
    assetStore,
    fakeRuntime,
    fakeManifest,
    fakeClassify,
    { onWarning: (event) => warnings.push(event) }
  );
  const metadata: RemoteAssetMetadata = {
    url: 'https://example.com/model.gguf',
    canonicalUrl: 'https://example.com/model.gguf',
    name: 'model.gguf',
    bytes: 5,
  };
  const { storagePath } = await assetStore.prepareRemoteDownload(metadata, fakeRuntime);
  storage.files.set(storagePath, new File([Uint8Array.from([1, 2, 3])], storagePath));

  const requestHeaders: Array<HeadersInit | undefined> = [];
  let ignoredBodyCancelled = false;
  await withCustomFetch(
    async (_input, init) => {
      requestHeaders.push(init?.headers);
      if (requestHeaders.length === 1) {
        return new Response(
          new ReadableStream<Uint8Array>({
            cancel() {
              ignoredBodyCancelled = true;
            },
          }),
          { status: 200, headers: { 'Content-Length': '5' } }
        );
      }
      return new Response(Uint8Array.from([10, 20, 30, 40, 50]), {
        status: 200,
        headers: { 'Content-Length': '5' },
      });
    },
    async () => {
      const result = await host.execute({
        kind: 'download',
        acquisitionId: 'acq-range-ignored',
        memberId: 0,
        attempt: 2,
        metadata: {
          url: metadata.url,
          name: metadata.name,
          bytes: metadata.bytes,
        },
      });

      assert.equal(result.event.kind, 'download_succeeded');
      assert.equal(ignoredBodyCancelled, true);
      assert.deepEqual(requestHeaders, [{ Range: 'bytes=3-' }, undefined]);
      assert.deepEqual(warnings, [{
        type: 'fallback-warning',
        kind: 'transfer',
        detail:
          'Resumable download fallback for "model.gguf": server did not honor the requested byte range.',
        fallbackTo: 'full-download',
      }]);
      const file = storage.files.get(storagePath);
      assert.ok(file != null);
      assert.deepEqual(
        [...new Uint8Array(await file.arrayBuffer())],
        [10, 20, 30, 40, 50]
      );
    }
  );
});

test('RemoteAcquisitionHost preserves a partial download when a range retry returns 503', async () => {
  const storage = new MemoryStorage();
  const assetStore = new AssetStore(storage as unknown as FileSystemStorage);
  const warnings: FallbackEvent[] = [];
  const host = new RemoteAcquisitionHost(
    assetStore,
    fakeRuntime,
    fakeManifest,
    fakeClassify,
    { onWarning: (event) => warnings.push(event) }
  );

  const metadata: RemoteAssetMetadata = {
    url: 'https://example.com/model.gguf',
    canonicalUrl: 'https://example.com/model.gguf',
    name: 'model.gguf',
    bytes: 5,
  };
  const { storagePath } = await assetStore.prepareRemoteDownload(metadata, fakeRuntime);
  storage.files.set(storagePath, new File([Uint8Array.from([1, 2, 3])], storagePath));

  let callCount = 0;
  await withCustomFetch(
    async () => {
      callCount += 1;
      return new Response(null, { status: 503, headers: { 'Retry-After': '1' } });
    },
    async () => {
      const result = await host.execute({
        kind: 'download',
        acquisitionId: 'acq-503',
        memberId: 0,
        attempt: 2,
        metadata: {
          url: metadata.url,
          name: metadata.name,
          bytes: metadata.bytes,
        },
      });

      assert.equal(result.event.kind, 'operation_failed');
      assert.equal(callCount, 1);
      assert.deepEqual(warnings, []);
      assert.equal(storage.files.get(storagePath)?.size, 3);
    }
  );
});

test('RemoteAcquisitionHost reports transport failure on download stall timeout so Rust can retry', async () => {
  const storage = new MemoryStorage();
  const assetStore = new AssetStore(storage as unknown as FileSystemStorage);
  const host = new RemoteAcquisitionHost(
    assetStore,
    fakeRuntime,
    fakeManifest,
    fakeClassify,
    { stallTimeoutMs: 50 }
  );

  const metadata: RemoteAssetMetadata = {
    url: 'https://example.com/model.gguf',
    canonicalUrl: 'https://example.com/model.gguf',
    name: 'model.gguf',
    bytes: 5,
  };

  const { storagePath } = await assetStore.prepareRemoteDownload(metadata, fakeRuntime);

  await withCustomFetch(
    async () => {
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(Uint8Array.from([1, 2]));
            // Intentionally stall without closing or providing remaining bytes
          },
        }),
        { status: 200, headers: { 'Content-Length': '5' } }
      );
    },
    async () => {
      const action: RustRemoteAction = {
        kind: 'download',
        acquisitionId: 'acq-4',
        memberId: 0,
        attempt: 1,
        metadata: {
          url: metadata.url,
          name: metadata.name,
          bytes: metadata.bytes,
        },
      };

      const result = await host.execute(action);
      assert.equal(result.event.kind, 'operation_failed');
      if (result.event.kind === 'operation_failed') {
        assert.equal(result.event.failure.phase, 'download');
        assert.equal(result.event.failure.kind, 'transport');
        assert.match(result.event.failure.reason, /stalled/i);
      }

      const partialFile = storage.files.get(storagePath);
      assert.ok(partialFile != null);
      assert.equal(partialFile.size, 2);
    }
  );
});
