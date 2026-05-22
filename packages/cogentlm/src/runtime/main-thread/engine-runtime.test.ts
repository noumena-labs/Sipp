import test from 'node:test';
import assert from 'node:assert/strict';
import { MainThreadEngineRuntime } from './engine-runtime.js';
import type { EngineModule } from '../../wasm/engine-module.js';
import type { StagedModelBundle } from '../../bundle/model-bundle-types.js';

function createModule(): EngineModule {
  return {
    FS: {
      analyzePath: () => ({ exists: false }),
      mkdir: () => {},
      writeFile: () => {},
      unlink: () => {},
      mount: () => {},
      unmount: () => {},
    },
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

test('MainThreadEngineRuntime fails projector-backed loads that do not expose a media marker', async () => {
  const runtime = new MainThreadEngineRuntime({
    executionMode: 'main-thread',
    moduleUrl: 'https://example.test/runtime.js',
    wasmUrl: 'https://example.test/runtime.wasm',
  });
  const fakeModule = createModule();
  let bridgeCloseCount = 0;
  let cleanupCount = 0;

  (runtime as unknown as { module: EngineModule }).module = fakeModule;
  (runtime as unknown as {
    wasmBridge: {
      loadRuntimeModel: () => Promise<number>;
      readMediaMarker: () => string | null;
      readNativeChatTemplate: () => string | null;
      getBosText: () => string | null;
      getEosText: () => string | null;
      close: () => void;
    };
  }).wasmBridge = {
    loadRuntimeModel: async () => 0,
    readMediaMarker: () => null,
    readNativeChatTemplate: () => null,
    getBosText: () => null,
    getEosText: () => null,
    close: () => {
      bridgeCloseCount += 1;
    },
  };
  (runtime as unknown as {
    modelLoader: {
      cleanup: () => void;
    };
  }).modelLoader = {
    cleanup: () => {
      cleanupCount += 1;
    },
  };

  const staged: StagedModelBundle = {
    sourceKind: 'file',
    modelPath: '/models/model.gguf',
    projectorPath: '/models/mmproj.gguf',
    isVisionModel: true,
    projectorStatus: 'explicit',
    modelName: 'vision-model.gguf',
    detectionMethod: 'gguf-metadata',
    modelType: 'model',
    modelArchitecture: 'qwen2vl',
  };

  await assert.rejects(
    () => runtime.loadRuntimeModel(staged),
    /did not expose a media marker/
  );
  assert.equal(bridgeCloseCount, 1);
  assert.equal(cleanupCount, 1);
  assert.equal(runtime.readMediaMarker(), null);
});
