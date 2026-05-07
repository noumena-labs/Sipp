import test from 'node:test';
import assert from 'node:assert/strict';
import { MainThreadEngineRuntime } from './engine-runtime-main-thread.js';
import type { EngineModule } from '../wasm/engine-module.js';
import type { StagedModelBundle } from '../types.js';

type TestBridge = {
  startTextRequest?: (
    contextKey: string,
    promptText: string,
    maxOutputTokens: number,
    callbackPtr: number,
    grammar?: string,
    tokenEmissionMode?: number
  ) => number;
  startMediaRequest?: (
    contextKey: string,
    promptText: string,
    maxOutputTokens: number,
    media: Uint8Array[],
    callbackPtr: number,
    grammar?: string,
    tokenEmissionMode?: number
  ) => number;
};

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

function createReadyRuntimeWithBridge(bridge: TestBridge): MainThreadEngineRuntime {
  const runtime = new MainThreadEngineRuntime({ executionMode: 'main-thread' });
  const internals = runtime as unknown as {
    module: EngineModule;
    engineInitialized: boolean;
    wasmBridge: TestBridge;
  };
  internals.module = createModule();
  internals.engineInitialized = true;
  internals.wasmBridge = bridge;
  return runtime;
}

test('MainThreadEngineRuntime allows output token counts above the old artifact limit', async () => {
  let capturedMaxOutputTokens: number | undefined;
  const runtime = createReadyRuntimeWithBridge({
    startTextRequest: (_contextKey, _promptText, maxOutputTokens) => {
      capturedMaxOutputTokens = maxOutputTokens;
      return 101;
    },
  });

  const requestId = await runtime.enqueueQuery('context', 'prompt', {
    nTokens: 2049,
  });

  assert.equal(requestId, 101);
  assert.equal(capturedMaxOutputTokens, 2049);
});

test('MainThreadEngineRuntime rejects non-positive and non-integer nTokens', async () => {
  let enqueueCount = 0;
  const runtime = createReadyRuntimeWithBridge({
    startTextRequest: () => {
      enqueueCount += 1;
      return 102;
    },
  });

  const invalidValues: Array<{ value: number; message: RegExp }> = [
    { value: 0, message: /positive integer/ },
    { value: -1, message: /positive integer/ },
    { value: 1.5, message: /integer/ },
  ];

  for (const invalidValue of invalidValues) {
    await assert.rejects(
      () =>
        runtime.enqueueQuery('context', 'prompt', {
          nTokens: invalidValue.value,
        }),
      invalidValue.message
    );
  }
  assert.equal(enqueueCount, 0);
});

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
    multimodalProjectorPath: '/models/mmproj.gguf',
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
