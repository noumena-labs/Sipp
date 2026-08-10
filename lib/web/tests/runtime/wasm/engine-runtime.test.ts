import test from 'node:test';
import assert from 'node:assert/strict';
import { WasmEngineRuntime } from '../../../src/runtime/wasm/engine-runtime.js';
import type { EngineModule } from '../../../src/wasm/engine-module.js';
import type {
  RuntimeBundleDescriptor,
  RuntimeSessionDescriptor,
  RuntimeSessionSnapshot,
} from '../../../src/models/types.js';

const TEST_SESSION: RuntimeSessionDescriptor = {
  model: {
    id: 'model-1',
    name: 'model.gguf',
    modality: 'vision',
    status: 'ready',
    source: 'local',
    bytes: 1,
    assetFingerprint: 'asset-fingerprint',
    createdAt: '1970-01-01T00:00:00.000Z',
    updatedAt: '1970-01-01T00:00:00.000Z',
  },
  runtimeFingerprint: 'runtime-fingerprint',
};

function runtimeSnapshot(
  session: RuntimeSessionDescriptor,
  generation: number
): RuntimeSessionSnapshot {
  return {
    ...session,
    generation,
    capabilities: {
      modelClass: 'decoder_only',
      supportsTextGeneration: true,
      supportsEmbeddings: false,
      supportsVision: false,
      audioSampleRateHz: null,
      generatedAudioSampleRateHz: null,
      hasChatTemplate: false,
      embedding: null,
      operations: {
        query: true,
        chat: false,
        embed: false,
        listen: false,
        speak: false,
      },
    },
    chatTemplate: null,
    bosText: '',
    eosText: '',
    mediaMarker: null,
  };
}

function runtimeBundle(): RuntimeBundleDescriptor {
  return {
    modelFiles: [{
      name: 'model.gguf',
      handle: {
        read: () => 0,
        write: () => 0,
        truncate: () => {},
        flush: () => {},
        close: () => {},
        getSize: () => 1,
      },
      size: 1,
    }],
  };
}

function createModule(): EngineModule {
  return {
    FS: {
      analyzePath: () => ({ exists: false }),
      mkdir: () => {},
      mount: () => {},
      unmount: () => {},
    },
    HEAP32: new Int32Array(8),
    HEAPF32: new Float32Array(8),
    HEAPF64: new Float64Array(8),
    HEAPU8: new Uint8Array(8),
    _free: () => {},
    _malloc: () => 0,
    ccall: (ident: string) => {
      if (ident === 'CE_RustBrowserEngineAbiVersion') {
        return 15;
      }
      return 0;
    },
    UTF8ToString: () => '',
    addFunction: () => 0,
    removeFunction: () => {},
  };
}

function withWasmPthreadSupport<T>(callback: () => Promise<T>): Promise<T> {
  const crossOriginIsolatedDescriptor = Object.getOwnPropertyDescriptor(
    globalThis,
    'crossOriginIsolated'
  );
  const workerDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  Object.defineProperty(globalThis, 'crossOriginIsolated', {
    configurable: true,
    value: true,
  });
  Object.defineProperty(globalThis, 'Worker', {
    configurable: true,
    value: class FakeWorker {},
  });
  return callback().finally(() => {
    if (crossOriginIsolatedDescriptor == null) {
      Reflect.deleteProperty(globalThis, 'crossOriginIsolated');
    } else {
      Object.defineProperty(globalThis, 'crossOriginIsolated', crossOriginIsolatedDescriptor);
    }
    if (workerDescriptor == null) {
      Reflect.deleteProperty(globalThis, 'Worker');
    } else {
      Object.defineProperty(globalThis, 'Worker', workerDescriptor);
    }
  });
}

test('WasmEngineRuntime points pthread helpers at the selected runtime module', async () => {
  await withWasmPthreadSupport(async () => {
    const moduleUrl = 'https://example.test/wasm/sipp-wasm-pthread.js';
    const wasmUrl = 'https://example.test/wasm/sipp-wasm-pthread.wasm';
    const runtime = new WasmEngineRuntime({
      wasmThreading: 'pthread',
      moduleUrl,
      wasmUrl,
    });
    let capturedOptions: Record<string, unknown> | null = null;
    (runtime as unknown as {
      importModuleFactory: () => Promise<(options: Record<string, unknown>) => Promise<EngineModule>>;
    }).importModuleFactory = async () => async (options) => {
      capturedOptions = options;
      return createModule();
    };

    await runtime.initModule();

    assert.equal(capturedOptions?.mainScriptUrlOrBlob, moduleUrl);
    assert.equal(
      (capturedOptions?.locateFile as (path: string) => string)('Sipp.wasm'),
      wasmUrl
    );
  });
});

test('WasmEngineRuntime rejects stale browser runtime ABI artifacts', async () => {
  const runtime = new WasmEngineRuntime({
    wasmThreading: 'single-thread',
    moduleUrl: 'https://example.test/runtime.js',
    wasmUrl: 'https://example.test/runtime.wasm',
  });
  const staleModule = createModule();
  staleModule.ccall = (ident: string) => {
    if (ident === 'CE_RustBrowserEngineAbiVersion') {
      return 5;
    }
    return 0;
  };
  (runtime as unknown as {
    importModuleFactory: () => Promise<() => Promise<EngineModule>>;
  }).importModuleFactory = async () => async () => staleModule;

  await assert.rejects(
    () => runtime.initModule(),
    /Sipp browser runtime ABI mismatch: expected 15, got 5/
  );
});

test('WasmEngineRuntime rejects invalid JavaScript prompt option containers', () => {
  const runtime = new WasmEngineRuntime();
  const resolvePromptOptions = (
    runtime as unknown as {
      resolvePromptOptions: (options: unknown) => unknown;
    }
  ).resolvePromptOptions.bind(runtime);

  assert.throws(
    () => resolvePromptOptions([]),
    /Prompt options must be an object or token count/
  );
  assert.throws(
    () => resolvePromptOptions({ sampling: [] }),
    /sampling must be an object when provided/
  );
});

test('WasmEngineRuntime publishes its single session only after activation commits', async () => {
  const runtime = new WasmEngineRuntime();
  const module = createModule();
  const events: string[] = [];
  const target: RuntimeSessionDescriptor = {
    ...TEST_SESSION,
    model: {
      ...TEST_SESSION.model,
      id: 'model-2',
      assetFingerprint: 'asset-fingerprint-2',
    },
    runtimeFingerprint: 'runtime-fingerprint-2',
  };
  const newSession = runtimeSnapshot(target, 2);

  (runtime as unknown as { module: EngineModule }).module = module;
  (runtime as unknown as { runtimeObservabilityEnabled: boolean }).runtimeObservabilityEnabled =
    true;
  (runtime as unknown as {
    wasmBridge: {
      getRuntimeSession: () => RuntimeSessionSnapshot;
      close: () => void;
      loadRuntimeModel: () => Promise<number>;
      getBackendObservabilityJson: () => Promise<string | null>;
      getSharedTokenRingDescriptor: () => unknown;
      modelServiceCreate: () => unknown;
      modelServiceList: () => unknown;
    };
  }).wasmBridge = {
    getRuntimeSession: () => newSession,
    close: () => {},
    loadRuntimeModel: async () => {
      events.push('load-new');
      assert.equal(runtime.currentRuntimeSession(), null);
      return 0;
    },
    getBackendObservabilityJson: async () => null,
    getSharedTokenRingDescriptor: () => ({
      buffer: module.HEAPU8.buffer,
      headerOffset: 0,
      bodyOffset: 0,
      bodyCapacity: 0,
    }),
    modelServiceCreate: () => ({ ok: true, value: { handle: 1 } }),
    modelServiceList: () => {
      events.push('commit');
      return { ok: true, value: [] };
    },
  };
  (runtime as unknown as {
    modelLoader: {
      validateBundle: () => void;
      mountBundle: () => { modelPath: string; projectorPath: null };
      cleanup: () => void;
      closeBundle: () => void;
    };
  }).modelLoader = {
    validateBundle: () => {
      events.push('validate');
    },
    mountBundle: () => {
      events.push('mount-new');
      return { modelPath: '/models/model.gguf', projectorPath: null };
    },
    cleanup: () => {
      events.push('cleanup-mounts');
    },
    closeBundle: () => {},
  };
  const lifecycle = await runtime.createRustLifecycleBridge({} as never);

  // Occupy the runtime's serialized Wasm-bridge queue so activation cannot
  // progress until the test releases it.
  let releaseBridge!: () => void;
  const bridgeGate = new Promise<void>((resolve) => {
    releaseBridge = resolve;
  });
  void (runtime as unknown as {
    wasmBridgeOperations: { run: (operation: () => Promise<void>) => Promise<void> };
  }).wasmBridgeOperations.run(() => bridgeGate);
  const activationPromise = runtime.activateRuntime(runtimeBundle(), {
    session: target,
    config: {},
    commit: async ({ session }) => {
      // The lifecycle bridge enters the same serialized Wasm queue. This must
      // run after native activation releases that queue rather than deadlocking
      // behind its own activation operation.
      await lifecycle.list();
      assert.equal(runtime.currentRuntimeSession(), null);
      return session.model.id;
    },
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(runtime.currentRuntimeSession(), null);
  assert.equal(runtime.getRuntimeObservability(), null);
  releaseBridge();
  const activation = await activationPromise;

  // The bundle is released before commit: a cleanup failure after commit would
  // leave the Rust catalog committed while activation reports failure.
  assert.deepEqual(events, [
    'validate',
    'mount-new',
    'load-new',
    'cleanup-mounts',
    'commit',
  ]);
  assert.equal(activation.committed, 'model-2');
  assert.equal(runtime.currentRuntimeSession()?.model.id, 'model-2');
});

test('WasmEngineRuntime leaves no published runtime when projector activation is invalid', async () => {
  const runtime = new WasmEngineRuntime({
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
      getRuntimeSession: () => unknown;
      getBackendObservabilityJson: () => Promise<string | null>;
      close: () => void;
    };
  }).wasmBridge = {
    loadRuntimeModel: async () => 0,
    readMediaMarker: () => null,
    readNativeChatTemplate: () => null,
    getBosText: () => null,
    getEosText: () => null,
    getRuntimeSession: () => ({
      ...TEST_SESSION,
      generation: 1,
      capabilities: {
        modelClass: 'decoder_only',
        supportsTextGeneration: true,
        supportsEmbeddings: false,
        supportsVision: true,
        audioSampleRateHz: null,
        generatedAudioSampleRateHz: null,
        hasChatTemplate: false,
        embedding: null,
        operations: {
          query: true,
          chat: false,
          embed: false,
          listen: false,
          speak: false,
        },
      },
      chatTemplate: null,
      bosText: '',
      eosText: '',
      mediaMarker: null,
    }),
    getBackendObservabilityJson: async () => null,
    close: () => {
      bridgeCloseCount += 1;
    },
  };
  (runtime as unknown as {
    modelLoader: {
      validateBundle: () => void;
      mountBundle: () => { modelPath: string; projectorPath: string };
      cleanup: () => void;
      closeBundle: () => void;
    };
  }).modelLoader = {
    validateBundle: () => {},
    mountBundle: () => ({
      modelPath: '/models/model.gguf',
      projectorPath: '/models/mmproj.gguf',
    }),
    cleanup: () => {
      cleanupCount += 1;
    },
    closeBundle: () => {},
  };

  const bundle = runtimeBundle();

  await assert.rejects(
    () => runtime.activateRuntime(bundle, {
      session: TEST_SESSION,
      config: {},
      commit: async () => undefined,
    }),
    /did not expose a media marker/
  );
  assert.equal(bridgeCloseCount, 1);
  assert.equal(cleanupCount, 1);
  assert.equal(runtime.readMediaMarker(), null);
});

test('WasmEngineRuntime rejects a second activation in the same Worker', async () => {
  const runtime = new WasmEngineRuntime();
  (runtime as unknown as { module: EngineModule }).module = createModule();
  (runtime as unknown as { engineInitialized: boolean }).engineInitialized = true;
  (runtime as unknown as { runtimeSession: RuntimeSessionSnapshot }).runtimeSession =
    runtimeSnapshot(TEST_SESSION, 1);
  let closeBundleCount = 0;
  (runtime as unknown as {
    modelLoader: {
      validateBundle: () => void;
      closeBundle: () => void;
    };
  }).modelLoader = {
    validateBundle: () => {},
    closeBundle: () => {
      closeBundleCount += 1;
    },
  };

  await assert.rejects(
    runtime.activateRuntime(runtimeBundle(), {
      session: TEST_SESSION,
      config: {},
      commit: async () => undefined,
    }),
    /can activate only one runtime session/
  );
  assert.equal(closeBundleCount, 1);
});

test('WasmEngineRuntime preserves activation failure while attempting every cleanup', async () => {
  const runtime = new WasmEngineRuntime();
  const primary = new Error('activation failed');
  const cleanupCalls: string[] = [];
  (runtime as unknown as { module: EngineModule }).module = createModule();
  (runtime as unknown as {
    wasmBridge: {
      loadRuntimeModel: () => Promise<number>;
      close: () => Promise<void>;
    };
  }).wasmBridge = {
    loadRuntimeModel: async () => {
      throw primary;
    },
    close: async () => {
      cleanupCalls.push('native');
      throw new Error('native close failed');
    },
  };
  (runtime as unknown as {
    modelLoader: {
      validateBundle: () => void;
      mountBundle: () => { modelPath: string; projectorPath: null };
      cleanup: () => void;
      closeBundle: () => void;
    };
  }).modelLoader = {
    validateBundle: () => {},
    mountBundle: () => ({ modelPath: '/models/model.gguf', projectorPath: null }),
    cleanup: () => {
      cleanupCalls.push('bundle');
      throw new Error('bundle cleanup failed');
    },
    closeBundle: () => {},
  };

  await assert.rejects(
    runtime.activateRuntime(runtimeBundle(), {
      session: TEST_SESSION,
      config: {},
      commit: async () => undefined,
    }),
    (error: unknown) => {
      const withCleanup = error as Error & { cleanupFailures?: AggregateError };
      return error === primary && withCleanup.cleanupFailures?.errors.length === 2;
    }
  );
  assert.deepEqual(cleanupCalls, ['native', 'bundle']);
  assert.equal(runtime.currentRuntimeSession(), null);
});

test('WasmEngineRuntime rejects a request handle from an older native session', async () => {
  const runtime = new WasmEngineRuntime();
  const activeSession = runtimeSnapshot(TEST_SESSION, 2);
  (runtime as unknown as { module: EngineModule }).module = createModule();
  (runtime as unknown as { engineInitialized: boolean }).engineInitialized = true;
  (runtime as unknown as { runtimeSession: RuntimeSessionSnapshot }).runtimeSession = activeSession;

  await assert.rejects(
    runtime.awaitQuery({ generation: 1, requestId: 1 }),
    /does not belong to the active runtime generation/
  );
});

test('WasmEngineRuntime never commits when releasing the mounted bundle fails', async () => {
  const runtime = new WasmEngineRuntime();
  const module = createModule();
  const events: string[] = [];
  const target: RuntimeSessionDescriptor = {
    ...TEST_SESSION,
    model: { ...TEST_SESSION.model, id: 'model-3', assetFingerprint: 'asset-3' },
    runtimeFingerprint: 'runtime-3',
  };

  (runtime as unknown as { module: EngineModule }).module = module;
  (runtime as unknown as {
    wasmBridge: {
      getRuntimeSession: () => RuntimeSessionSnapshot;
      close: () => void;
      loadRuntimeModel: () => Promise<number>;
      getBackendObservabilityJson: () => Promise<string | null>;
    };
  }).wasmBridge = {
    getRuntimeSession: () => runtimeSnapshot(target, 3),
    close: () => {
      events.push('close-native');
    },
    loadRuntimeModel: async () => 0,
    getBackendObservabilityJson: async () => null,
  };
  (runtime as unknown as {
    modelLoader: {
      validateBundle: () => void;
      mountBundle: () => { modelPath: string; projectorPath: null };
      cleanup: () => void;
      closeBundle: () => void;
    };
  }).modelLoader = {
    validateBundle: () => {},
    mountBundle: () => ({ modelPath: '/models/model.gguf', projectorPath: null }),
    cleanup: () => {
      events.push('cleanup-mounts');
      throw new Error('unmount failed');
    },
    closeBundle: () => {},
  };

  await assert.rejects(
    runtime.activateRuntime(runtimeBundle(), {
      session: target,
      config: {},
      commit: async ({ session }) => {
        events.push('commit');
        return session.model.id;
      },
    }),
    /unmount failed/
  );

  // Native activation succeeded, so the runtime is closed; the catalog must not
  // have committed, because this activation reports failure.
  assert.equal(events.includes('commit'), false);
  assert.deepEqual(events, ['cleanup-mounts', 'close-native', 'cleanup-mounts']);
  assert.equal(runtime.currentRuntimeSession(), null);
});

test('WasmEngineRuntime serializes catalog work behind a suspended inference call', async () => {
  const runtime = new WasmEngineRuntime();
  const module = createModule();
  const order: string[] = [];
  let releaseInference!: () => void;
  const inferenceSuspended = new Promise<void>((resolve) => {
    releaseInference = resolve;
  });

  (runtime as unknown as { module: EngineModule }).module = module;
  (runtime as unknown as { engineInitialized: boolean }).engineInitialized = true;
  (runtime as unknown as { wasmBridge: Record<string, unknown> }).wasmBridge = {
    // Stands in for a JSPI export suspended mid-call.
    runInferenceLoop: async () => {
      order.push('inference:start');
      await inferenceSuspended;
      order.push('inference:end');
      return { stepResult: 0, completedResponseCount: 0 };
    },
    validatePairing: () => {
      order.push('pairing');
      return { ok: true, plan: { pairs: [] } };
    },
    modelServiceCreate: () => {
      order.push('lifecycle-create');
      return { ok: true, value: { handle: 1 } };
    },
    modelServiceList: () => {
      order.push('lifecycle-list');
      return { ok: true, value: [] };
    },
  };

  // Occupy the queue the way a suspended inference loop would.
  const inference = (
    runtime as unknown as {
      withReadyWasmBridge: (
        operation: (bridge: { runInferenceLoop: () => Promise<unknown> }) => Promise<unknown>
      ) => Promise<unknown>;
    }
  ).withReadyWasmBridge((bridge) => bridge.runInferenceLoop());

  const pairing = runtime.resolvePairing([]);
  const lifecycle = runtime
    .createRustLifecycleBridge({} as never)
    .then(async (bridge) => await bridge.list());

  await new Promise((resolve) => setTimeout(resolve, 0));
  // Nothing may enter Wasm while the inference call is suspended.
  assert.deepEqual(order, ['inference:start']);

  releaseInference();
  await Promise.all([inference, pairing, lifecycle]);

  assert.deepEqual(order, [
    'inference:start',
    'inference:end',
    'pairing',
    'lifecycle-create',
    'lifecycle-list',
  ]);
});
