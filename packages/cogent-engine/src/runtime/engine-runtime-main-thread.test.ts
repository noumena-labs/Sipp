import assert from 'node:assert/strict';
import test from 'node:test';

import { MainThreadEngineRuntime } from './engine-runtime-main-thread.js';

const REQUEST_STEP_RESULT_PROGRESSED = 1;
const REQUEST_STEP_RESULT_TERMINAL = 2;
const COMPLETED_REQUEST_STATUS_COMPLETED = 1;
const COMPLETED_REQUEST_STATUS_CANCELLED = 2;

type ScenarioKind = 'success' | 'callback-error';

class MockMainThreadModule {
  public readonly FS = {
    analyzePath: () => ({ exists: false }),
    mkdir: () => {},
    writeFile: () => {},
    unlink: () => {},
    open: () => ({ fd: 0, position: 0 }),
    write: () => 0,
    close: () => {},
  };

  public readonly HEAP32: Int32Array;
  public readonly HEAPF64: Float64Array;
  public insideNativeStep = false;
  public cancelCallCount = 0;
  public consumeCallCount = 0;
  public removedFunctionPtrs: number[] = [];

  private readonly heapU8: Uint8Array;
  private readonly functionTable = new Map<number, (...args: number[]) => number>();
  private readonly requestId = 7;
  private readonly completedOutputText: string;
  private readonly completedErrorText: string;
  private readonly completedStatus: number;
  private nextHeapPtr = 1024;
  private nextFunctionPtr = 1;
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

  public _CE_FreeString(_ptr: number): void {}

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
        this.queuedCallbackPtr = args[3] as number;
        return this.requestId;
      case 'CE_ResetRuntimeObservability':
        return 0;
      case 'CE_RunRequestStep':
        return this.runRequestStep();
      case 'CE_CancelQueuedRequest':
        this.cancelCallCount += 1;
        return 1;
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
    const f64Offset = ptr >> 3;
    const i32Offset = (ptr + 9 * 8) >> 2;
    const doubles = [12.5, 3.5, 4.5, 1.5, 2.5, 6.5, 0.5, 0.75, 12.5];
    const ints = [9, 7, 2, 2, 2, 3, 3, 1, 1, 0, 4, 0, 0, 1];

    for (let index = 0; index < doubles.length; index += 1) {
      this.HEAPF64[f64Offset + index] = doubles[index];
    }
    for (let index = 0; index < ints.length; index += 1) {
      this.HEAP32[i32Offset + index] = ints[index];
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
