import test from 'node:test';
import assert from 'node:assert/strict';
import { MainThreadModelLoader } from './main-thread-model-loader.js';
import type { EngineModule, EmscriptenFs } from '../wasm/engine-module.js';

interface FakeModule extends EngineModule {
  FS: EmscriptenFs & {
    dirs: Set<string>;
    files: Map<string, Uint8Array>;
    mounts: Array<{ mountpoint: string; fileNames: string[] }>;
    unmounts: string[];
    unlinks: string[];
  };
}

function createModule(): FakeModule {
  const dirs = new Set<string>();
  const files = new Map<string, Uint8Array>();
  const mounts: Array<{ mountpoint: string; fileNames: string[] }> = [];
  const unmounts: string[] = [];
  const unlinks: string[] = [];

  const fs: FakeModule['FS'] = {
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
    mount: (_type: any, opts: { files?: Array<{ name?: string }> }, mountpoint: string) => {
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
    WORKERFS: {},
    HEAP32: new Int32Array(8),
    HEAPF64: new Float64Array(8),
    HEAPU8: new Uint8Array(8),
    _free: () => {},
    _malloc: () => 0,
    ccall: () => 0,
    UTF8ToString: () => '',
  };
}

test('MainThreadModelLoader stages projector in MEMFS and model in WORKERFS', async () => {
  const loader = new MainThreadModelLoader({});
  const module = createModule();
  const projectorBytes = Uint8Array.from([1, 2, 3, 4]);

  const staged = await loader.stageModelBundle(module, {
    kind: 'file',
    file: new File(['not-a-gguf-model'], 'model.gguf'),
    projector: {
      kind: 'file',
      file: new File([projectorBytes], 'mmproj.gguf'),
    },
  });

  assert.equal(staged.modelPath, '/workerfs_model/model.gguf');
  assert.equal(staged.multimodalProjectorPath, '/memfs_projector/mmproj.gguf');
  assert.deepEqual(module.FS.mounts, [
    {
      mountpoint: '/workerfs_model',
      fileNames: ['model.gguf'],
    },
  ]);
  assert.deepEqual(
    [...(module.FS.files.get('/memfs_projector/mmproj.gguf') ?? [])],
    [...projectorBytes]
  );
});

test('MainThreadModelLoader cleanup removes MEMFS projector and unmounts model files', async () => {
  const loader = new MainThreadModelLoader({});
  const module = createModule();

  await loader.stageModelBundle(module, {
    kind: 'file',
    file: new File(['not-a-gguf-model'], 'model.gguf'),
    projector: {
      kind: 'file',
      file: new File(['projector'], 'mmproj.gguf'),
    },
  });

  loader.cleanupAfterEngineInit(module);

  assert.deepEqual(module.FS.unlinks, ['/memfs_projector/mmproj.gguf']);
  assert.deepEqual(module.FS.unmounts, ['/workerfs_model']);
  assert.equal(module.FS.files.has('/memfs_projector/mmproj.gguf'), false);
});
