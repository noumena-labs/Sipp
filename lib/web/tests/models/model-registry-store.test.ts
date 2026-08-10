import test from 'node:test';
import assert from 'node:assert/strict';
import { FileSystemStorage } from '../../src/engine/file-system-storage.js';
import { ModelRegistryStore } from '../../src/models/model-registry-store.js';

class MemoryStorage {
  public readonly writes: string[] = [];

  public constructor(private registry: string | null) {}

  public async readText(fileName: string): Promise<string | null> {
    assert.equal(fileName, 'registry.json');
    return this.registry;
  }

  public async writeText(fileName: string, contents: string): Promise<void> {
    assert.equal(fileName, 'registry.json');
    this.registry = contents;
    this.writes.push(contents);
  }

  public async listFileNamesAt(): Promise<string[]> {
    return [];
  }
}

async function withSupportedStorage<T>(fn: () => Promise<T>): Promise<T> {
  const original = FileSystemStorage.isSupported;
  FileSystemStorage.isSupported = () => true;
  try {
    return await fn();
  } finally {
    FileSystemStorage.isSupported = original;
  }
}

test('ModelRegistryStore rejects previous manifest versions', async () => {
  await withSupportedStorage(async () => {
    const storage = new MemoryStorage(
      JSON.stringify({
        version: 6,
        projectorIndexRevision: 0,
        assets: {},
        models: {},
      })
    );
    const store = new ModelRegistryStore(storage as unknown as FileSystemStorage);

    await assert.rejects(store.read(), /Model registry must be manifest version 7\./);
    assert.deepEqual(storage.writes, []);
  });
});

test('ModelRegistryStore creates current manifests for new storage roots', async () => {
  await withSupportedStorage(async () => {
    const storage = new MemoryStorage(null);
    const store = new ModelRegistryStore(storage as unknown as FileSystemStorage);

    const manifest = await store.read();

    assert.equal(manifest.version, 7);
    assert.equal(storage.writes.length, 1);
    assert.equal(JSON.parse(storage.writes[0] ?? '{}').version, 7);
  });
});
