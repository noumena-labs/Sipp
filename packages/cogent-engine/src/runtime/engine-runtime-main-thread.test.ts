import assert from 'node:assert/strict';
import test from 'node:test';

import { FileSystemStorage } from '../storage/file-system-storage.js';
import { MainThreadEngineRuntime } from './engine-runtime-main-thread.js';

const REQUEST_STEP_RESULT_PROGRESSED = 1;
const REQUEST_STEP_RESULT_TERMINAL = 2;
const COMPLETED_REQUEST_STATUS_PENDING = 0;
const COMPLETED_REQUEST_STATUS_COMPLETED = 1;
const COMPLETED_REQUEST_STATUS_CANCELLED = 2;

type ScenarioKind = 'success' | 'callback-error';

class MockMainThreadModule {
  public readonly mountCalls: Array<{ mountDir: string; files: Blob[] }> = [];
  public readonly FS = {
    analyzePath: () => ({ exists: false }),
    mkdir: () => {},
    writeFile: () => {},
    unlink: () => {},
    open: () => ({ fd: 0, position: 0 }),
    write: () => 0,
    close: () => {},
    mount: (_type: unknown, opts: { files: Blob[] }, mountDir: string) => {
      this.mountCalls.push({ mountDir, files: opts.files });
    },
    unmount: () => {},
  };
  public readonly WORKERFS = {};

  public readonly HEAP32: Int32Array;
  public readonly HEAPF64: Float64Array;
  public insideNativeStep = false;
  public cancelCallCount = 0;
  public consumeCallCount = 0;
  public closeCallCount = 0;
  public removedFunctionPtrs: number[] = [];
  public initResult = 0;

  private readonly heapU8: Uint8Array;
  private readonly functionTable = new Map<number, (...args: number[]) => number>();
  private readonly completedOutputText: string;
  private readonly completedErrorText: string;
  private readonly completedStatus: number;
  private nextHeapPtr = 1024;
  private nextFunctionPtr = 1;
  private nextRequestId = 7;
  private queuedCallbackPtr = 0;
  private runStepCount = 0;

  constructor(private readonly scenario: ScenarioKind) {
    const memory = new ArrayBuffer(16 * 1024);
    this.heapU8 = new Uint8Array(memory);
    this.HEAP32 = new Int32Array(memory);
    this.HEAPF64 = new Float64Array(memory);
    this.completedOutputText =
      scenario === 'success' ? 'tok1tok2' : 'tok1';
    this.completedErrorText =
      scenario === 'callback-error' ? 'Request cancelled.' : '';
    this.completedStatus =
      scenario === 'success'
        ? COMPLETED_REQUEST_STATUS_COMPLETED
        : COMPLETED_REQUEST_STATUS_CANCELLED;
  }

  public _free(_ptr: number): void {}

  public _malloc(size: number): number {
    const alignedSize = (size + 7) & ~7;
    const ptr = this.nextHeapPtr;
    this.nextHeapPtr += alignedSize;
    return ptr;
  }

  public addFunction(func: (...args: number[]) => number): number {
    const ptr = this.nextFunctionPtr++;
    this.functionTable.set(ptr, func);
    return ptr;
  }

  public removeFunction(ptr: number): void {
    this.removedFunctionPtrs.push(ptr);
    this.functionTable.delete(ptr);
  }

  public UTF8ToString(ptr: number, maxBytesToRead?: number): string {
    const bytes: number[] = [];
    const maxBytes = maxBytesToRead ?? this.heapU8.length - ptr;
    for (let index = 0; index < maxBytes; index += 1) {
      const byte = this.heapU8[ptr + index];
      if (byte === 0) {
        break;
      }
      bytes.push(byte);
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  public ccall(
    ident: string,
    _returnType: string | null,
    _argTypes: string[],
    args: unknown[],
    opts?: { async?: boolean }
  ): Promise<number> | number {
    const result = this.handleCall(ident, args);
    return opts?.async ? Promise.resolve(result) : result;
  }

  private handleCall(ident: string, args: unknown[]): number {
    switch (ident) {
      case 'CE_EnqueuePrompt':
        this.queuedCallbackPtr = Number(args[3]);
        return this.nextRequestId++;
      case 'CE_ResetRuntimeObservability':
        return 0;
      case 'CE_RunRequestStep':
        return this.runRequestStep();
      case 'CE_CancelQueuedRequest':
        this.cancelCallCount += 1;
        return 1;
      case 'CE_Close':
        this.closeCallCount += 1;
        return 0;
      case 'CE_Init':
        return this.initResult;
      case 'CE_GetCompletedRequestStatus':
        return this.completedStatus;
      case 'CE_GetCompletedRequestOutputSize':
        return this.completedOutputText.length;
      case 'CE_CopyCompletedRequestOutput':
        this.writeCString(this.completedOutputText, args[1] as number);
        return this.completedOutputText.length;
      case 'CE_GetCompletedRequestErrorSize':
        return this.completedErrorText.length;
      case 'CE_CopyCompletedRequestError':
        this.writeCString(this.completedErrorText, args[1] as number);
        return this.completedErrorText.length;
      case 'CE_ConsumeCompletedRequest':
        this.consumeCallCount += 1;
        return 1;
      case 'CE_GetRuntimeObservability':
        this.writeRuntimeObservability(args[0] as number);
        return 0;
      case 'CE_GetCompletedRequestRuntimeObservability':
        this.writeRuntimeObservability(args[1] as number);
        return 0;
      default:
        throw new Error(`Unexpected ccall: ${ident}`);
    }
  }

  private runRequestStep(): number {
    this.insideNativeStep = true;
    const callback = this.functionTable.get(this.queuedCallbackPtr);
    const emitToken = (text: string) => {
      if (callback == null) {
        return;
      }
      const tokenPtr = this.writeTempCString(text);
      callback(tokenPtr, text.length);
    };

    if (this.scenario === 'success') {
      if (this.runStepCount === 0) {
        emitToken('tok1');
        this.runStepCount += 1;
        this.insideNativeStep = false;
        return REQUEST_STEP_RESULT_PROGRESSED;
      }

      emitToken('tok2');
      this.runStepCount += 1;
      this.insideNativeStep = false;
      return REQUEST_STEP_RESULT_TERMINAL;
    }

    if (this.runStepCount === 0) {
      emitToken('tok1');
      this.runStepCount += 1;
      this.insideNativeStep = false;
      return REQUEST_STEP_RESULT_PROGRESSED;
    }

    this.runStepCount += 1;
    this.insideNativeStep = false;
    return REQUEST_STEP_RESULT_TERMINAL;
  }

  private writeTempCString(value: string): number {
    const ptr = this._malloc(value.length + 1);
    this.writeCString(value, ptr);
    return ptr;
  }

  private writeCString(value: string, ptr: number): void {
    const bytes = new TextEncoder().encode(value);
    this.heapU8.set(bytes, ptr);
    this.heapU8[ptr + bytes.length] = 0;
  }

  private writeRuntimeObservability(ptr: number): void {
    const f64Offset = (ptr >> 3);
    const i32Offset = ((ptr + 9 * 8) >> 2);
    const doubles = [12.5, 3.5, 4.5, 1.5, 2.5, 6.5, 0.5, 0.75, 12.5];
    const ints = [9, 7, 2, 2, 2, 3, 1, 1, 0, 4, 0, 0, 1];

    for (let index = 0; index < doubles.length; index += 1) {
      this.HEAPF64[f64Offset + index] = doubles[index];
    }
    for (let index = 0; index < ints.length; index += 1) {
      this.HEAP32[i32Offset + index] = ints[index];
    }
  }
}

function getQueuedPromptState(
  runtime: MainThreadEngineRuntime
): {
  callbacks: number;
  buffers: number;
  callbackPtrs: number;
  callbackErrors: number;
  activeRuns: number;
} {
  const runtimeState = runtime as unknown as {
    queuedPromptCallbacks: Map<number, unknown>;
    queuedPromptTokenBuffers: Map<number, unknown>;
    queuedPromptCallbackPtrs: Map<number, unknown>;
    queuedPromptCallbackErrors: Map<number, unknown>;
    activeQueuedRequestRuns: Set<number>;
  };

  return {
    callbacks: runtimeState.queuedPromptCallbacks.size,
    buffers: runtimeState.queuedPromptTokenBuffers.size,
    callbackPtrs: runtimeState.queuedPromptCallbackPtrs.size,
    callbackErrors: runtimeState.queuedPromptCallbackErrors.size,
    activeRuns: runtimeState.activeQueuedRequestRuns.size,
  };
}

function createMockRuntimeObservability(
  outputTokenCount: number,
  batchParticipationCount = 3
): { doubles: number[]; ints: number[] } {
  return {
    doubles: [12.5, 3.5, 4.5, 1.5, 2.5, 6.5, 0.5, 0.75, 12.5],
    ints: [9, 7, outputTokenCount, outputTokenCount, outputTokenCount, batchParticipationCount, 1, 1, 0, 4, 0, 0, 1],
  };
}

class MockConcurrentObservabilityModule {
  public readonly FS = {
    analyzePath: () => ({ exists: false }),
    mkdir: () => {},
    writeFile: () => {},
    unlink: () => {},
    open: () => ({ fd: 0, position: 0 }),
    write: () => 0,
    close: () => {},
    mount: () => {},
    unmount: () => {},
  };
  public readonly WORKERFS = {};
  public readonly HEAP32: Int32Array;
  public readonly HEAPF64: Float64Array;

  private readonly heapU8: Uint8Array;
  private nextHeapPtr = 1024;
  private readonly stepDeferreds = new Map<
    number,
    { promise: Promise<number>; resolve: (value: number) => void }
  >();
  private readonly completed = new Map<
    number,
    {
      outputText: string;
      errorText: string;
      runtimeObservability: { doubles: number[]; ints: number[] };
    }
  >();
  private currentObservability = createMockRuntimeObservability(0, 0);

  constructor() {
    const memory = new ArrayBuffer(16 * 1024);
    this.heapU8 = new Uint8Array(memory);
    this.HEAP32 = new Int32Array(memory);
    this.HEAPF64 = new Float64Array(memory);
  }

  public _free(_ptr: number): void {}

  public _malloc(size: number): number {
    const alignedSize = (size + 7) & ~7;
    const ptr = this.nextHeapPtr;
    this.nextHeapPtr += alignedSize;
    return ptr;
  }

  public UTF8ToString(ptr: number, maxBytesToRead?: number): string {
    const bytes: number[] = [];
    const maxBytes = maxBytesToRead ?? this.heapU8.length - ptr;
    for (let index = 0; index < maxBytes; index += 1) {
      const byte = this.heapU8[ptr + index];
      if (byte === 0) {
        break;
      }
      bytes.push(byte);
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  public ccall(
    ident: string,
    _returnType: string | null,
    _argTypes: string[],
    args: unknown[],
    opts?: { async?: boolean }
  ): Promise<number> | number {
    const result = this.handleCall(ident, args);
    return opts?.async ? Promise.resolve(result) : result;
  }

  public resolveTerminal(
    requestId: number,
    outputText: string,
    outputTokenCount: number,
    batchParticipationCount: number,
    updateObservability = true
  ): void {
    this.completed.set(requestId, {
      outputText,
      errorText: '',
      runtimeObservability: createMockRuntimeObservability(
        outputTokenCount,
        batchParticipationCount
      ),
    });
    if (updateObservability) {
      this.currentObservability = createMockRuntimeObservability(
        outputTokenCount,
        batchParticipationCount
      );
    }
    this.ensureStepDeferred(requestId).resolve(REQUEST_STEP_RESULT_TERMINAL);
  }

  private handleCall(ident: string, args: unknown[]): Promise<number> | number {
    const requestId = Number(args[0]);
    switch (ident) {
      case 'CE_ResetRuntimeObservability':
        this.currentObservability = createMockRuntimeObservability(0, 0);
        return 0;
      case 'CE_RunRequestStep':
        return this.ensureStepDeferred(requestId).promise;
      case 'CE_GetCompletedRequestStatus':
        return this.completed.has(requestId)
          ? COMPLETED_REQUEST_STATUS_COMPLETED
          : COMPLETED_REQUEST_STATUS_PENDING;
      case 'CE_GetCompletedRequestOutputSize':
        return this.completed.get(requestId)?.outputText.length ?? 0;
      case 'CE_CopyCompletedRequestOutput':
        this.writeCString(
          this.completed.get(requestId)?.outputText ?? '',
          args[1] as number
        );
        return this.completed.get(requestId)?.outputText.length ?? 0;
      case 'CE_GetCompletedRequestErrorSize':
        return this.completed.get(requestId)?.errorText.length ?? 0;
      case 'CE_CopyCompletedRequestError':
        this.writeCString(
          this.completed.get(requestId)?.errorText ?? '',
          args[1] as number
        );
        return this.completed.get(requestId)?.errorText.length ?? 0;
      case 'CE_GetCompletedRequestRuntimeObservability':
        this.writeRuntimeObservabilityFrom(
          this.completed.get(requestId)?.runtimeObservability ??
            createMockRuntimeObservability(0, 0),
          args[1] as number
        );
        return 0;
      case 'CE_ConsumeCompletedRequest':
        this.completed.delete(requestId);
        return 1;
      case 'CE_GetRuntimeObservability':
        this.writeRuntimeObservability(args[0] as number);
        return 0;
      default:
        throw new Error(`Unexpected ccall: ${ident}`);
    }
  }

  private ensureStepDeferred(requestId: number): {
    promise: Promise<number>;
    resolve: (value: number) => void;
  } {
    const existing = this.stepDeferreds.get(requestId);
    if (existing != null) {
      return existing;
    }
    let resolve!: (value: number) => void;
    const promise = new Promise<number>((promiseResolve) => {
      resolve = promiseResolve;
    });
    const deferred = { promise, resolve };
    this.stepDeferreds.set(requestId, deferred);
    return deferred;
  }

  private writeCString(value: string, ptr: number): void {
    const bytes = new TextEncoder().encode(value);
    this.heapU8.set(bytes, ptr);
    this.heapU8[ptr + bytes.length] = 0;
  }

  private writeRuntimeObservability(ptr: number): void {
    this.writeRuntimeObservabilityFrom(this.currentObservability, ptr);
  }

  private writeRuntimeObservabilityFrom(
    observability: { doubles: number[]; ints: number[] },
    ptr: number
  ): void {
    const f64Offset = ptr >> 3;
    const i32Offset = (ptr + 9 * 8) >> 2;
    for (let index = 0; index < observability.doubles.length; index += 1) {
      this.HEAPF64[f64Offset + index] = observability.doubles[index];
    }
    for (let index = 0; index < observability.ints.length; index += 1) {
      this.HEAP32[i32Offset + index] = observability.ints[index];
    }
  }
}

function attachReadyModule(
  runtime: MainThreadEngineRuntime,
  module: MockMainThreadModule
): void {
  const runtimeState = runtime as unknown as {
    module: MockMainThreadModule;
    engineInitialized: boolean;
    runtimeObservabilityEnabled: boolean;
  };
  runtimeState.module = module;
  runtimeState.engineInitialized = true;
  runtimeState.runtimeObservabilityEnabled = true;
}

async function waitForCondition(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('Timed out while waiting for condition.');
}

test('MainThreadEngineRuntime flushes queued tokens outside native steps and reads typed results', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const callbackPhases: boolean[] = [];
  const callbackTokens: string[] = [];
  const requestId = await runtime.queuePrompt('ctx', 'prompt', {
    nTokens: 16,
    onToken: (token) => {
      callbackPhases.push(module.insideNativeStep);
      callbackTokens.push(token);
    },
  });

  const response = await runtime.runQueuedRequest(requestId);

  assert.equal(requestId, 7);
  assert.deepEqual(callbackTokens, ['tok1', 'tok2']);
  assert.deepEqual(callbackPhases, [false, false]);
  assert.equal(response.completed, true);
  assert.equal(response.failed, false);
  assert.equal(response.cancelled, false);
  assert.equal(response.outputText, 'tok1tok2');
  assert.equal(response.runtimeObservability?.promptEvalMs, 3.5);
  assert.equal(response.runtimeObservability?.outputTokenCount, 2);
  assert.equal(module.consumeCallCount, 1);
  assert.deepEqual(module.removedFunctionPtrs, [1]);
});

test('MainThreadEngineRuntime cancels terminal execution after callback failure and still consumes the response', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('callback-error');
  attachReadyModule(runtime, module);

  const requestId = await runtime.queuePrompt('ctx', 'prompt', {
    nTokens: 16,
    onToken: () => {
      throw new Error('token callback failed');
    },
  });

  await assert.rejects(
    runtime.runQueuedRequest(requestId),
    /token callback failed/
  );

  assert.equal(module.cancelCallCount, 1);
  assert.equal(module.consumeCallCount, 1);
  assert.deepEqual(module.removedFunctionPtrs, [1]);
});

test('MainThreadEngineRuntime releases queued callback state when cancelling before execution', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const requestId = await runtime.queuePrompt('ctx-cancel', 'prompt', {
    nTokens: 16,
    onToken: () => {},
  });

  const cancelled = await runtime.cancelQueuedRequest(requestId);

  assert.equal(cancelled, true);
  assert.deepEqual(getQueuedPromptState(runtime), {
    callbacks: 0,
    buffers: 0,
    callbackPtrs: 0,
    callbackErrors: 0,
    activeRuns: 0,
  });
  assert.deepEqual(module.removedFunctionPtrs, [1]);
  assert.equal(module.consumeCallCount, 1);
});

test('MainThreadEngineRuntime consumes completed cancel responses even without queued callbacks', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const requestId = await runtime.queuePrompt('ctx-cancel-no-callback', 'prompt', 16);
  const cancelled = await runtime.cancelQueuedRequest(requestId);

  assert.equal(cancelled, true);
  assert.equal(module.consumeCallCount, 1);
});

test('MainThreadEngineRuntime queue/cancel churn leaves no queued callback residue and still supports a smoke prompt', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const churnCount = 5;
  for (let index = 0; index < churnCount; index += 1) {
    const requestId = await runtime.queuePrompt(`ctx-churn-${index}`, 'prompt', {
      nTokens: 16,
      onToken: () => {},
    });
    const cancelled = await runtime.cancelQueuedRequest(requestId);
    assert.equal(cancelled, true);
  }

  const smokeOutput = await runtime.submitPrompt('ctx-smoke-after-churn', 'prompt', 16);
  assert.equal(smokeOutput, 'tok1tok2');

  assert.deepEqual(getQueuedPromptState(runtime), {
    callbacks: 0,
    buffers: 0,
    callbackPtrs: 0,
    callbackErrors: 0,
    activeRuns: 0,
  });
  assert.equal(module.removedFunctionPtrs.length, churnCount);
  assert.equal(module.consumeCallCount, churnCount + 1);
});

test('MainThreadEngineRuntime close clears queued lifecycle state and stale model metadata', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const requestId = await runtime.queuePrompt('ctx-close', 'prompt', {
    nTokens: 16,
    onToken: () => {},
  });
  const runtimeState = runtime as unknown as {
    activeQueuedRequestRuns: Set<number>;
    lastModelLoadInfo: object | null;
    transportObservability: {
      enabled: boolean;
      flushCount: number;
      coalescedTokenCount: number;
    };
  };
  runtimeState.activeQueuedRequestRuns.add(requestId);
  runtimeState.lastModelLoadInfo = {
    sourceKind: 'buffer',
    reuseMode: 'buffer',
    modelPath: '/models/model.gguf',
    fileName: 'model.gguf',
    byteLength: 4,
    persistentCacheEnabled: false,
    persistentCacheKey: null,
    persistentCacheHit: false,
    persistentCacheStored: false,
  };
  runtimeState.transportObservability.enabled = true;
  runtimeState.transportObservability.flushCount = 3;
  runtimeState.transportObservability.coalescedTokenCount = 4;

  runtime.close();

  assert.equal(module.closeCallCount, 1);
  assert.deepEqual(module.removedFunctionPtrs, [1]);
  assert.deepEqual(getQueuedPromptState(runtime), {
    callbacks: 0,
    buffers: 0,
    callbackPtrs: 0,
    callbackErrors: 0,
    activeRuns: 0,
  });
  assert.equal(runtime.getLastModelLoadInfo(), null);
  assert.deepEqual(runtime.getTransportObservability(), {
    executionMode: 'main-thread',
    workerBacked: false,
    enabled: false,
    bufferedTokenLimit: 0,
    flushIntervalMs: 0,
    flushCount: 0,
    coalescedTokenCount: 0,
    maxObservedBufferedTokenCount: 0,
  });
});

test('MainThreadEngineRuntime clears stale lifecycle state when reinit fails', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);
  module.initResult = 7;

  const requestId = await runtime.queuePrompt('ctx-reinit-fail', 'prompt', {
    nTokens: 16,
    onToken: () => {},
  });
  const runtimeState = runtime as unknown as {
    activeQueuedRequestRuns: Set<number>;
    runtimeObservabilityEnabled: boolean;
    backendProfilingEnabled: boolean;
    engineInitialized: boolean;
    transportObservability: {
      enabled: boolean;
      flushCount: number;
    };
  };
  runtimeState.activeQueuedRequestRuns.add(requestId);

  await assert.rejects(
    runtime.initEngine('/models/failing-model.gguf', {
      enableRuntimeObservability: true,
      enableBackendProfiling: true,
    }),
    /Code: 7/
  );

  assert.equal(module.closeCallCount, 1);
  assert.deepEqual(module.removedFunctionPtrs, [1]);
  assert.deepEqual(getQueuedPromptState(runtime), {
    callbacks: 0,
    buffers: 0,
    callbackPtrs: 0,
    callbackErrors: 0,
    activeRuns: 0,
  });
  assert.equal(runtimeState.engineInitialized, false);
  assert.equal(runtimeState.runtimeObservabilityEnabled, false);
  assert.equal(runtimeState.backendProfilingEnabled, false);
  assert.equal(runtimeState.transportObservability.enabled, false);
  assert.equal(runtimeState.transportObservability.flushCount, 0);
});

test('MainThreadEngineRuntime aborts non-OPFS stream loads without mounting partial files', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const originalIsSupported = FileSystemStorage.isSupported;
  try {
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = () => false;

    let releaseSecondRead!: () => void;
    const secondReadGate = new Promise<void>((resolve) => {
      releaseSecondRead = resolve;
    });
    let chunkIndex = 0;
    const stream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        if (chunkIndex === 0) {
          chunkIndex += 1;
          controller.enqueue(Uint8Array.from([1]));
          return;
        }

        await secondReadGate;
        controller.close();
      },
    });
    const abortController = new AbortController();
    const loadPromise = runtime.loadModelFromReadableStream(stream, 'model.gguf', {
      signal: abortController.signal,
    });

    await Promise.resolve();
    await Promise.resolve();
    abortController.abort();
    releaseSecondRead();

    await assert.rejects(
      loadPromise,
      (error: unknown) => error instanceof Error && error.name === 'AbortError'
    );
    assert.equal(module.mountCalls.length, 0);
  } finally {
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = originalIsSupported;
  }
});

test('MainThreadEngineRuntime preserves shard filenames when loading split model URLs without OPFS', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const originalFetch = globalThis.fetch;
  const originalIsSupported = FileSystemStorage.isSupported;

  try {
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = () => false;
    globalThis.fetch = (async (_url: string, init?: { method?: string }) => {
      if (init?.method === 'HEAD') {
        return {
          headers: {
            get: () => '4',
          },
        } as unknown as Response;
      }
      return {
        ok: true,
        status: 200,
        body: null,
        arrayBuffer: async () => Uint8Array.from([1, 2, 3, 4]).buffer,
      } as Response;
    }) as typeof fetch;

    const modelPath = await runtime.loadModelFromUrls([
      'https://example.com/model-00001-of-00002.gguf',
      'https://example.com/model-00002-of-00002.gguf',
    ]);
    const mountedNames =
      module.mountCalls.at(-1)?.files.map((file) => (file as { name?: string }).name || 'model.gguf') ?? [];

    assert.equal(modelPath, '/workerfs_model/model-00001-of-00002.gguf');
    assert.deepEqual(mountedNames, [
      'model-00001-of-00002.gguf',
      'model-00002-of-00002.gguf',
    ]);
  } finally {
    globalThis.fetch = originalFetch;
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = originalIsSupported;
  }
});

test('MainThreadEngineRuntime bounds concurrent non-OPFS URL shard downloads and preserves order', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const originalFetch = globalThis.fetch;
  const originalIsSupported = FileSystemStorage.isSupported;
  const startedGets: string[] = [];
  const releaseGetResponse = new Map<string, () => void>();
  let activeGets = 0;
  let maxActiveGets = 0;

  try {
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = () => false;
    globalThis.fetch = (async (url: string, init?: { method?: string }) => {
      if (init?.method === 'HEAD') {
        return {
          headers: {
            get: () => '4',
          },
        } as unknown as Response;
      }

      startedGets.push(url);
      activeGets += 1;
      maxActiveGets = Math.max(maxActiveGets, activeGets);

      const ready = new Promise<void>((resolve) => {
        releaseGetResponse.set(url, resolve);
      });
      await ready;
      activeGets -= 1;

      return {
        ok: true,
        status: 200,
        body: new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(Uint8Array.from([1, 2, 3, 4]));
            controller.close();
          },
        }),
      } as Response;
    }) as typeof fetch;

    const loadPromise = runtime.loadModelFromUrls([
      'https://example.com/model-00001-of-00003.gguf',
      'https://example.com/model-00002-of-00003.gguf',
      'https://example.com/model-00003-of-00003.gguf',
    ]);

    await waitForCondition(() => startedGets.length === 2);

    assert.deepEqual(startedGets, [
      'https://example.com/model-00001-of-00003.gguf',
      'https://example.com/model-00002-of-00003.gguf',
    ]);
    assert.equal(maxActiveGets, 2);

    releaseGetResponse.get('https://example.com/model-00001-of-00003.gguf')?.();
    await waitForCondition(() => startedGets.length === 3);

    assert.deepEqual(startedGets, [
      'https://example.com/model-00001-of-00003.gguf',
      'https://example.com/model-00002-of-00003.gguf',
      'https://example.com/model-00003-of-00003.gguf',
    ]);

    releaseGetResponse.get('https://example.com/model-00002-of-00003.gguf')?.();
    releaseGetResponse.get('https://example.com/model-00003-of-00003.gguf')?.();

    const modelPath = await loadPromise;
    const mountedNames =
      module.mountCalls.at(-1)?.files.map((file) => (file as { name?: string }).name || 'model.gguf') ?? [];

    assert.equal(modelPath, '/workerfs_model/model-00001-of-00003.gguf');
    assert.deepEqual(mountedNames, [
      'model-00001-of-00003.gguf',
      'model-00002-of-00003.gguf',
      'model-00003-of-00003.gguf',
    ]);
  } finally {
    globalThis.fetch = originalFetch;
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = originalIsSupported;
  }
});

test('MainThreadEngineRuntime aborts non-OPFS URL loads without mounting partial files', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const originalFetch = globalThis.fetch;
  const originalIsSupported = FileSystemStorage.isSupported;

  try {
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = () => false;

    const readGates = new Map<string, Promise<void>>();
    const releaseReads = new Map<string, () => void>();
    globalThis.fetch = (async (url: string, init?: { method?: string }) => {
      if (init?.method === 'HEAD') {
        return {
          headers: {
            get: () => '4',
          },
        } as unknown as Response;
      }

      if (!readGates.has(url)) {
        readGates.set(
          url,
          new Promise<void>((resolve) => {
            releaseReads.set(url, resolve);
          })
        );
      }

      return {
        ok: true,
        status: 200,
        body: new ReadableStream<Uint8Array>({
          async pull(controller) {
            controller.enqueue(Uint8Array.from([1]));
            await readGates.get(url);
            controller.close();
          },
        }),
      } as Response;
    }) as typeof fetch;

    const abortController = new AbortController();
    const loadPromise = runtime.loadModelFromUrls([
      'https://example.com/model-00001-of-00002.gguf',
      'https://example.com/model-00002-of-00002.gguf',
    ], undefined, abortController.signal);

    await Promise.resolve();
    await Promise.resolve();
    abortController.abort();
    releaseReads.get('https://example.com/model-00001-of-00002.gguf')?.();
    releaseReads.get('https://example.com/model-00002-of-00002.gguf')?.();

    await assert.rejects(
      loadPromise,
      (error: unknown) => error instanceof Error && error.name === 'AbortError'
    );
    assert.equal(module.mountCalls.length, 0);
  } finally {
    globalThis.fetch = originalFetch;
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = originalIsSupported;
  }
});

test('MainThreadEngineRuntime does not treat same-basename OPFS cache entries as valid for a different URL', async () => {
  const runtime = new MainThreadEngineRuntime({
    persistentModelCache: {
      enabled: true,
    },
  });
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const originalFetch = globalThis.fetch;
  const originalIsSupported = FileSystemStorage.isSupported;
  let getRequestCount = 0;
  let requestedCacheKey: string | null = null;

  try {
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = () => true;
    const runtimeState = runtime as unknown as {
      opfs: {
        getFile: (fileName: string) => Promise<File | null>;
        streamToDisk: (
          fileName: string,
          stream: ReadableStream<Uint8Array>
        ) => Promise<File>;
      };
    };
    runtimeState.opfs.getFile = async (fileName: string) => {
      requestedCacheKey = fileName;
      return fileName === 'model.gguf'
        ? new File([Uint8Array.from([9, 9, 9, 9])], fileName)
        : null;
    };
    runtimeState.opfs.streamToDisk = async (fileName: string) =>
      new File([Uint8Array.from([1, 2, 3, 4])], fileName);

    globalThis.fetch = (async (_url: string, init?: { method?: string }) => {
      if (init?.method === 'HEAD') {
        return {
          headers: {
            get: () => '4',
          },
        } as unknown as Response;
      }
      getRequestCount += 1;
      return {
        ok: true,
        status: 200,
        body: new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(Uint8Array.from([1, 2, 3, 4]));
            controller.close();
          },
        }),
        arrayBuffer: async () => Uint8Array.from([1, 2, 3, 4]).buffer,
      } as Response;
    }) as typeof fetch;

    await runtime.loadModelFromUrls(['https://fresh.example.com/model.gguf']);

    assert.notEqual(requestedCacheKey, 'model.gguf');
    assert.equal(getRequestCount, 1);
  } finally {
    globalThis.fetch = originalFetch;
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = originalIsSupported;
  }
});

test('MainThreadEngineRuntime keeps runtime observability isolated across concurrent queued requests', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockConcurrentObservabilityModule();
  attachReadyModule(
    runtime,
    module as unknown as MockMainThreadModule
  );

  const firstPromise = runtime.runQueuedRequest(101);
  await Promise.resolve();
  const secondPromise = runtime.runQueuedRequest(202);
  await Promise.resolve();

  module.resolveTerminal(202, 'second', 2, 20);
  const secondResponse = await secondPromise;

  module.resolveTerminal(101, 'first', 1, 10, false);
  const firstResponse = await firstPromise;

  assert.equal(secondResponse.runtimeObservability?.outputTokenCount, 2);
  assert.equal(secondResponse.runtimeObservability?.batchParticipationCount, 20);
  assert.equal(firstResponse.runtimeObservability?.outputTokenCount, 1);
  assert.equal(firstResponse.runtimeObservability?.batchParticipationCount, 10);
});
