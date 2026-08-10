import test from 'node:test';
import assert from 'node:assert/strict';
import { WasmModelLoader } from '../../../src/runtime/wasm/model-loader.js';
import type { EngineModule, EmscriptenFs } from '../../../src/wasm/engine-module.js';
import type { OpfsSyncAccessHandle } from '../../../src/engine/file-system-storage.js';

interface FakeFs extends EmscriptenFs {
  dirs: Set<string>;
  files: Map<string, Uint8Array>;
  mounts: Array<{ mountpoint: string; fileNames: string[] }>;
  unmounts: string[];
}

interface FakeModule extends EngineModule {
  FS: FakeFs;
}

function createModule(): FakeModule {
  const dirs = new Set<string>();
  const files = new Map<string, Uint8Array>();
  const mounts: Array<{ mountpoint: string; fileNames: string[] }> = [];
  const unmounts: string[] = [];

  const fs: FakeFs = {
    dirs,
    files,
    mounts,
    unmounts,
    analyzePath: (path: string) => ({
      exists: dirs.has(path) || files.has(path) || mounts.some((mount) => mount.mountpoint === path),
    }),
    mkdir: (path: string) => {
      dirs.add(path);
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

test('WasmModelLoader mounts model files and projector through sync-access FS', () => {
  const loader = new WasmModelLoader({});
  const module = createModule();
  const shardBytes = Uint8Array.from([1, 2, 3, 4, 5]);
  const projectorBytes = Uint8Array.from([9, 8, 7]);
  const shard = fakeHandle(shardBytes);
  const projector = fakeHandle(projectorBytes);

  const mounted = loader.mountBundle(module, {
    modelFiles: [{ name: 'model.gguf', handle: shard.handle, size: shardBytes.byteLength }],
    projector: {
      name: 'mmproj.gguf',
      handle: projector.handle,
      size: projectorBytes.byteLength,
    },
  });

  assert.equal(mounted.modelPath, '/sah_model/model.gguf');
  assert.equal(mounted.projectorPath, '/sah_projector/mmproj.gguf');
  assert.deepEqual(module.FS.mounts, [
    { mountpoint: '/sah_model', fileNames: ['model.gguf'] },
    { mountpoint: '/sah_projector', fileNames: ['mmproj.gguf'] },
  ]);
  assert.equal(module.FS.files.size, 0);
  loader.cleanup(module);
  assert.equal(shard.closed.value, true);
  assert.equal(projector.closed.value, true);
});

test('cleanup unmounts the sync-access FS and closes every model handle', () => {
  const loader = new WasmModelLoader({});
  const module = createModule();
  const shardA = fakeHandle(Uint8Array.from([1, 2]));
  const shardB = fakeHandle(Uint8Array.from([3, 4]));

  loader.mountBundle(module, {
    modelFiles: [
      { name: 'shard-1.gguf', handle: shardA.handle, size: 2 },
      { name: 'shard-2.gguf', handle: shardB.handle, size: 2 },
    ],
  });

  loader.cleanup(module);

  assert.deepEqual(module.FS.unmounts, ['/sah_model']);
  assert.equal(shardA.closed.value, true);
  assert.equal(shardB.closed.value, true);
});

test('cleanup attempts every release and clears ownership before throwing', () => {
  const loader = new WasmModelLoader({});
  const module = createModule();
  const shardA = fakeHandle(Uint8Array.from([1, 2]));
  const shardB = fakeHandle(Uint8Array.from([3, 4]));
  const closeCalls: string[] = [];
  shardA.handle.close = () => {
    closeCalls.push('shard-a');
    throw new Error('shard A close failed');
  };
  shardB.handle.close = () => {
    closeCalls.push('shard-b');
  };

  loader.mountBundle(module, {
    modelFiles: [
      { name: 'shard-1.gguf', handle: shardA.handle, size: 2 },
      { name: 'shard-2.gguf', handle: shardB.handle, size: 2 },
    ],
  });
  module.FS.unmount = (mountpoint: string) => {
    module.FS.unmounts.push(mountpoint);
    throw new Error('unmount failed');
  };

  assert.throws(
    () => loader.cleanup(module),
    (error: unknown) => error instanceof AggregateError && error.errors.length === 2
  );
  assert.deepEqual(module.FS.unmounts, ['/sah_model']);
  assert.deepEqual(closeCalls, ['shard-a', 'shard-b']);

  assert.doesNotThrow(() => loader.cleanup(module));
  assert.deepEqual(module.FS.unmounts, ['/sah_model']);
  assert.deepEqual(closeCalls, ['shard-a', 'shard-b']);
});

test('mount failure stays primary when cleanup also fails', () => {
  const loader = new WasmModelLoader({});
  const module = createModule();
  const shard = fakeHandle(Uint8Array.from([1, 2]));
  const projector = fakeHandle(Uint8Array.from([3, 4]));
  const primary = new Error('projector mount failed');
  let mountCount = 0;
  module.FS.mount = (_type, opts, mountpoint) => {
    mountCount += 1;
    if (mountCount === 2) {
      throw primary;
    }
    module.FS.mounts.push({
      mountpoint,
      fileNames: opts.files?.map((file) => file.name || 'model.gguf') ?? [],
    });
  };
  module.FS.unmount = () => {
    throw new Error('unmount failed');
  };
  shard.handle.close = () => {
    throw new Error('shard close failed');
  };

  assert.throws(
    () => loader.mountBundle(module, {
      modelFiles: [{ name: 'model.gguf', handle: shard.handle, size: 2 }],
      projector: { name: 'mmproj.gguf', handle: projector.handle, size: 2 },
    }),
    (error: unknown) => {
      const withCleanup = error as Error & { cleanupFailures?: AggregateError };
      return error === primary && withCleanup.cleanupFailures?.errors.length === 2;
    }
  );
  assert.equal(projector.closed.value, true);
});

test('validateBundle rejects an empty model file list', () => {
  const loader = new WasmModelLoader({});
  assert.throws(
    () => loader.validateBundle({
      modelFiles: [],
    }),
    /at least one model file/
  );
});
