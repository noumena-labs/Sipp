import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BrowserModelCache,
  BrowserModelCacheIdentity,
  BrowserModelCacheStorage,
} from './browser-model-cache.js';

class InMemoryBrowserModelCacheStorage implements BrowserModelCacheStorage {
  public readonly deletedFiles: string[] = [];
  private readonly files = new Map<string, File>();

  public async getFile(fileName: string): Promise<File | null> {
    return this.files.get(fileName) ?? null;
  }

  public async readText(fileName: string): Promise<string | null> {
    const file = this.files.get(fileName);
    return file == null ? null : await file.text();
  }

  public async writeText(fileName: string, contents: string): Promise<void> {
    this.files.set(fileName, new File([contents], fileName, { type: 'application/json' }));
  }

  public async streamToDisk(
    fileName: string,
    stream: ReadableStream<Uint8Array>,
    _onProgress?: (bytes: number) => void,
    _signal?: AbortSignal
  ): Promise<File> {
    const reader = stream.getReader();
    const bytes: number[] = [];
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (value != null) {
          bytes.push(...value);
        }
      }
    } finally {
      reader.releaseLock();
    }
    const file = new File([Uint8Array.from(bytes)], fileName);
    this.files.set(fileName, file);
    return file;
  }

  public async deleteFile(fileName: string): Promise<void> {
    this.deletedFiles.push(fileName);
    this.files.delete(fileName);
  }

  public async listFiles(): Promise<string[]> {
    return [...this.files.keys()];
  }

  public async estimate(): Promise<{ usageBytes: number | null; quotaBytes: number | null }> {
    let usageBytes = 0;
    for (const [fileName, file] of this.files.entries()) {
      if (fileName === 'browser-model-cache-manifest.v1.json') {
        continue;
      }
      usageBytes += file.size;
    }
    return {
      usageBytes,
      quotaBytes: 1024,
    };
  }

  public seedFile(fileName: string, contents: string | Uint8Array): void {
    const part =
      typeof contents === 'string' ? contents : Uint8Array.from(contents);
    this.files.set(fileName, new File([part], fileName));
  }
}

function createIdentity(
  overrides: Partial<BrowserModelCacheIdentity> = {}
): BrowserModelCacheIdentity {
  return {
    canonicalUrl: 'https://example.com/model.gguf',
    fileName: 'model.gguf',
    etag: '"abc"',
    lastModified: 'Mon, 01 Jan 2024 00:00:00 GMT',
    contentLength: 4,
    ...overrides,
  };
}

function createStream(bytes: number[]): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(Uint8Array.from(bytes));
      controller.close();
    },
  });
}

test('BrowserModelCache stores manifest-backed entries and serves cache hits by identity', async () => {
  const storage = new InMemoryBrowserModelCacheStorage();
  const cache = new BrowserModelCache(storage);
  const identity = createIdentity();

  const stored = await cache.storeStream(identity, createStream([1, 2, 3, 4]));
  const hit = await cache.get(identity);

  assert.ok(stored.key.includes('model.gguf'));
  assert.ok(hit != null);
  assert.equal(hit?.key, stored.key);
  assert.equal(hit?.file.size, 4);

  const manifestText = await storage.readText('browser-model-cache-manifest.v1.json');
  assert.ok(manifestText != null);
  const manifest = JSON.parse(manifestText as string) as {
    version: number;
    entries: Record<string, { byteLength: number; canonicalUrl: string }>;
  };
  assert.equal(manifest.version, 1);
  assert.equal(manifest.entries[stored.key]?.byteLength, 4);
  assert.equal(manifest.entries[stored.key]?.canonicalUrl, identity.canonicalUrl);
});

test('BrowserModelCache removes orphaned files and stale manifest entries on initialization', async () => {
  const storage = new InMemoryBrowserModelCacheStorage();
  const cache = new BrowserModelCache(storage);
  const identity = createIdentity();
  const cacheKey = cache.buildEntryKey(identity);

  storage.seedFile('browser-model-cache-entry-valid.bin', Uint8Array.from([1, 2, 3, 4]));
  storage.seedFile('browser-model-cache-entry-orphan.bin', Uint8Array.from([9, 9, 9]));
  await storage.writeText(
    'browser-model-cache-manifest.v1.json',
    JSON.stringify({
      version: 1,
      entries: {
        [cacheKey]: {
          key: cacheKey,
          canonicalUrl: identity.canonicalUrl,
          fileName: identity.fileName,
          storageFileName: 'browser-model-cache-entry-valid.bin',
          byteLength: 4,
          etag: identity.etag,
          lastModified: identity.lastModified,
          contentLength: identity.contentLength,
          createdAt: '2024-01-01T00:00:00.000Z',
          lastAccessedAt: '2024-01-01T00:00:00.000Z',
        },
        stale: {
          key: 'stale',
          canonicalUrl: 'https://missing.example.com/model.gguf',
          fileName: 'missing.gguf',
          storageFileName: 'browser-model-cache-entry-missing.bin',
          byteLength: 4,
          etag: '',
          lastModified: '',
          contentLength: 4,
          createdAt: '2024-01-01T00:00:00.000Z',
          lastAccessedAt: '2024-01-01T00:00:00.000Z',
        },
      },
    })
  );

  const hit = await cache.get(identity);
  assert.ok(hit != null);
  assert.ok(storage.deletedFiles.includes('browser-model-cache-entry-orphan.bin'));

  const manifest = JSON.parse(
    (await storage.readText('browser-model-cache-manifest.v1.json')) as string
  ) as {
    entries: Record<string, unknown>;
  };
  assert.ok(manifest.entries[cacheKey] != null);
  assert.equal(manifest.entries.stale, undefined);
});

test('BrowserModelCache evicts least-recently-used entries before a write when storage pressure is high', async () => {
  const storage = new InMemoryBrowserModelCacheStorage();
  const cache = new BrowserModelCache(storage);
  const oldIdentity = createIdentity({
    canonicalUrl: 'https://example.com/old.gguf',
    fileName: 'old.gguf',
  });
  const recentIdentity = createIdentity({
    canonicalUrl: 'https://example.com/recent.gguf',
    fileName: 'recent.gguf',
  });
  const oldKey = cache.buildEntryKey(oldIdentity);
  const recentKey = cache.buildEntryKey(recentIdentity);

  storage.seedFile('browser-model-cache-entry-old.bin', Uint8Array.from(new Array(220).fill(1)));
  storage.seedFile(
    'browser-model-cache-entry-recent.bin',
    Uint8Array.from(new Array(180).fill(2))
  );
  await storage.writeText(
    'browser-model-cache-manifest.v1.json',
    JSON.stringify({
      version: 1,
      entries: {
        [oldKey]: {
          key: oldKey,
          canonicalUrl: oldIdentity.canonicalUrl,
          fileName: oldIdentity.fileName,
          storageFileName: 'browser-model-cache-entry-old.bin',
          byteLength: 220,
          etag: oldIdentity.etag,
          lastModified: oldIdentity.lastModified,
          contentLength: oldIdentity.contentLength,
          createdAt: '2024-01-01T00:00:00.000Z',
          lastAccessedAt: '2024-01-01T00:00:00.000Z',
        },
        [recentKey]: {
          key: recentKey,
          canonicalUrl: recentIdentity.canonicalUrl,
          fileName: recentIdentity.fileName,
          storageFileName: 'browser-model-cache-entry-recent.bin',
          byteLength: 180,
          etag: recentIdentity.etag,
          lastModified: recentIdentity.lastModified,
          contentLength: recentIdentity.contentLength,
          createdAt: '2024-01-01T00:00:00.000Z',
          lastAccessedAt: '2024-02-01T00:00:00.000Z',
        },
      },
    })
  );

  const newIdentity = createIdentity({
    canonicalUrl: 'https://example.com/new.gguf',
    fileName: 'new.gguf',
    contentLength: 600,
  });
  await cache.storeStream(newIdentity, createStream(new Array(600).fill(3)));

  assert.ok(storage.deletedFiles.includes('browser-model-cache-entry-old.bin'));
  assert.equal(storage.deletedFiles.includes('browser-model-cache-entry-recent.bin'), false);
});
