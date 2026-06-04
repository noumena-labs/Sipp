import test from 'node:test';
import assert from 'node:assert/strict';
import { MainThreadModelLoader } from '../../../src/runtime/main-thread/model-loader.js';
import type { EngineModule, EmscriptenFs } from '../../../src/wasm/engine-module.js';
import type { ModelDetectionResult } from '../../../src/models/types.js';
import type { OpfsSyncAccessHandle } from '../../../src/engine/file-system-storage.js';

interface FakeFs extends EmscriptenFs {
  dirs: Set<string>;
  files: Map<string, Uint8Array>;
  mounts: Array<{ mountpoint: string; fileNames: string[] }>;
  unmounts: string[];
  unlinks: string[];
}

interface FakeModule extends EngineModule {
  FS: FakeFs;
}

function createModule(): FakeModule {
  const dirs = new Set<string>();
  const files = new Map<string, Uint8Array>();
  const mounts: Array<{ mountpoint: string; fileNames: string[] }> = [];
  const unmounts: string[] = [];
  const unlinks: string[] = [];

  const fs: FakeFs = {
    dirs,
    files,
    mounts,
    unmounts,
    unlinks,
    analyzePath: (path: string) => ({
      exists: dirs.has(path) || files.has(path) || mounts.some((mount) => mount.mountpoint === path),
    }),
    mkdir: (path: string) => {
      dirs.add(path);
    },
    writeFile: (path: string, data: Uint8Array) => {
      files.set(path, new Uint8Array(data));
    },
    unlink: (path: string) => {
      unlinks.push(path);
      files.delete(path);
    },
    mount: (_type: unknown, opts: { files?: Array<{ name?: string }> }, mountpoint: string) => {
      mounts.push({
        mountpoint,
        fileNames: opts.files?.map((file) => file.name || 'model.gguf') ?? [],
      });
    },
    unmount: (mountpoint: string) => {
      unmounts.push(mountpoint);
    },
  };

  return {
    FS: fs,
    HEAP32: new Int32Array(8),
    HEAPF32: new Float32Array(8),
    HEAPF64: new Float64Array(8),
    HEAPU8: new Uint8Array(8),
    _free: () => {},
    _malloc: () => 0,
    ccall: () => 0,
    UTF8ToString: () => '',
    addFunction: () => 0,
    removeFunction: () => {},
  };
}

function fakeHandle(bytes: Uint8Array): { handle: OpfsSyncAccessHandle; closed: { value: boolean } } {
  const closed = { value: false };
  const handle: OpfsSyncAccessHandle = {
    read: (target, options) => {
      const at = options?.at ?? 0;
      const available = Math.max(0, bytes.byteLength - at);
      const toRead = Math.min(target.byteLength, available);
      target.set(bytes.subarray(at, at + toRead));
      return toRead;
    },
    write: () => {
      throw new Error('write not supported in fake');
    },
    truncate: () => {},
    flush: () => {},
    close: () => {
      closed.value = true;
    },
    getSize: () => bytes.byteLength,
  };
  return { handle, closed };
}

function modelDetection(name: string, vision = false): ModelDetectionResult {
  return {
    inspection: {
      version: 1,
      role: 'model',
      architecture: 'llama',
      visionCapable: vision,
      compatibleVisionProjectorTypes: [],
      providedVisionProjectorType: null,
    },
    detectionMethod: 'gguf-metadata',
    modelName: name,
    modelType: null,
    modelArchitecture: 'llama',
  };
}

test('stageModelBundle mounts shards into sync-access FS and stages projector via MEMFS', async () => {
  const loader = new MainThreadModelLoader({});
  const module = createModule();
  const shardBytes = Uint8Array.from([1, 2, 3, 4, 5]);
  const projectorBytes = Uint8Array.from([9, 8, 7]);
  const { handle } = fakeHandle(shardBytes);

  const staged = await loader.stageModelBundle(module, {
    shards: [{ name: 'model.gguf', handle, size: shardBytes.byteLength }],
    projector: { file: new File([projectorBytes], 'mmproj.gguf') },
    detection: modelDetection('model.gguf', true),
  });

  assert.equal(staged.modelPath, '/sah_model/model.gguf');
  assert.equal(staged.projectorPath, '/memfs_projector/mmproj.gguf');
  assert.equal(staged.projectorStatus, 'paired');
  assert.deepEqual(module.FS.mounts, [
    { mountpoint: '/sah_model', fileNames: ['model.gguf'] },
  ]);
  assert.deepEqual(
    [...(module.FS.files.get('/memfs_projector/mmproj.gguf') ?? [])],
    [...projectorBytes]
  );
});

test('cleanup unmounts the sync-access FS and closes every shard handle', async () => {
  const loader = new MainThreadModelLoader({});
  const module = createModule();
  const shardA = fakeHandle(Uint8Array.from([1, 2]));
  const shardB = fakeHandle(Uint8Array.from([3, 4]));

  await loader.stageModelBundle(module, {
    shards: [
      { name: 'shard-1.gguf', handle: shardA.handle, size: 2 },
      { name: 'shard-2.gguf', handle: shardB.handle, size: 2 },
    ],
    detection: modelDetection('shard-1.gguf'),
  });

  loader.cleanup(module);

  assert.deepEqual(module.FS.unmounts, ['/sah_model']);
  assert.equal(shardA.closed.value, true);
  assert.equal(shardB.closed.value, true);
});

test('stageModelBundle rejects an empty shard list', async () => {
  const loader = new MainThreadModelLoader({});
  const module = createModule();
  await assert.rejects(
    loader.stageModelBundle(module, {
      shards: [],
      detection: modelDetection('empty.gguf'),
    }),
    /at least one shard/
  );
});
