import assert from 'node:assert/strict';
import test from 'node:test';

import type { DetailedRequestObservabilityMetrics } from '../observability/runtime-observability-detail.js';
import { FileSystemStorage } from '../storage/file-system-storage.js';
import { MainThreadEngineRuntime } from './engine-runtime-main-thread.js';

const REQUEST_STEP_RESULT_PROGRESSED = 1;
const REQUEST_STEP_RESULT_TERMINAL = 2;
const REQUEST_STEP_RESULT_WAITING = 0;
const COMPLETED_REQUEST_STATUS_PENDING = 0;
const COMPLETED_REQUEST_STATUS_COMPLETED = 1;
const COMPLETED_REQUEST_STATUS_CANCELLED = 2;
const RUNTIME_EVENT_KIND_TOKEN = 1;
const RUNTIME_EVENT_KIND_TERMINAL = 2;
const GGUF_MAGIC = 0x46554747;

enum GgufValueType {
  STRING = 8,
}

type ScenarioKind = 'success' | 'callback-error';
type TransportKind = 'callback' | 'event';
type MockMainThreadModuleOptions = {
  supportsRuntimeEventDrain?: boolean;
};

const ggufTextEncoder = new TextEncoder();

function encodeUint32(value: number): Uint8Array {
  const buffer = new ArrayBuffer(4);
  new DataView(buffer).setUint32(0, value, true);
  return new Uint8Array(buffer);
}

function encodeUint64(value: number): Uint8Array {
  const buffer = new ArrayBuffer(8);
  new DataView(buffer).setBigUint64(0, BigInt(value), true);
  return new Uint8Array(buffer);
}

function encodeString(value: string): Uint8Array {
  const bytes = ggufTextEncoder.encode(value);
  return concatBytes(encodeUint64(bytes.length), bytes);
}

function encodeField(key: string, type: GgufValueType, value: Uint8Array): Uint8Array {
  return concatBytes(encodeString(key), encodeUint32(type), value);
}

function buildGguf(fields: Array<{ key: string; type: GgufValueType; value: Uint8Array }>): Uint8Array {
  return concatBytes(
    encodeUint32(GGUF_MAGIC),
    encodeUint32(3),
    encodeUint64(0),
    encodeUint64(fields.length),
    ...fields.map((field) => encodeField(field.key, field.type, field.value))
  );
}

function concatBytes(...parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const output = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.byteLength;
  }
  return output;
}

function toBlobPart(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

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
  public readonly HEAPU8: Uint8Array;
  public insideNativeStep = false;
  public cancelCallCount = 0;
  public consumeCallCount = 0;
  public closeCallCount = 0;
  public runStepCallCount = 0;
  public removedFunctionPtrs: number[] = [];
  public initResult = 0;
  public lastQueuedCallbackPtr = 0;
  public mediaMarker: string | null = null;
  public chatTemplate: string | null = null;
  public appliedChatTemplateText = '';
  public returnEmptyAppliedChatTemplate = false;
  public lastAppliedChatTemplateMessages: Array<{
    role: string;
    content: string | Array<{ type: string; text: string }>;
  }> | null = null;
  public lastQueuedPromptText = '';
  public lastQueuedEnqueueKind: 'text' | 'media' | null = null;
  public lastQueuedMediaImages: Uint8Array[] = [];
  public lastInitIdent: string | null = null;
  public lastInitArgs: unknown[] | null = null;

  private readonly heapU8: Uint8Array;
  private readonly functionTable = new Map<number, (...args: number[]) => number>();
  private readonly completedOutputText: string;
  private readonly completedErrorText: string;
  private readonly completedStatus: number;
  private readonly supportsRuntimeEventDrain: boolean;
  private nextHeapPtr = 1024;
  private nextFunctionPtr = 1;
  private nextRequestId = 7;
  private queuedCallbackPtr = 0;
  private runStepCount = 0;
  private completedResponseAvailable = false;
  private readonly runtimeEvents: Array<{
    requestId: number;
    kind: number;
    status: number;
    text: string;
  }> = [];

  constructor(
    private readonly scenario: ScenarioKind,
    private readonly transport: TransportKind = 'callback',
    options: MockMainThreadModuleOptions = {}
  ) {
    const memory = new ArrayBuffer(2 * 1024 * 1024);
    this.heapU8 = new Uint8Array(memory);
    this.HEAPU8 = this.heapU8;
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
    this.supportsRuntimeEventDrain =
      options.supportsRuntimeEventDrain ?? transport === 'event';
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

  public queueRuntimeTokenEvent(requestId: number, text: string): void {
    this.runtimeEvents.push({
      requestId,
      kind: RUNTIME_EVENT_KIND_TOKEN,
      status: COMPLETED_REQUEST_STATUS_PENDING,
      text,
    });
  }

  public queueRuntimeTerminalEvent(
    requestId: number,
    status = COMPLETED_REQUEST_STATUS_COMPLETED
  ): void {
    this.runtimeEvents.push({
      requestId,
      kind: RUNTIME_EVENT_KIND_TERMINAL,
      status,
      text: '',
    });
  }

  private handleCall(ident: string, args: unknown[]): number {
    switch (ident) {
      case 'CE_EnqueuePrompt':
        this.lastQueuedEnqueueKind = 'text';
        this.lastQueuedPromptText = String(args[1]);
        this.queuedCallbackPtr = Number(args[3]);
        this.lastQueuedCallbackPtr = this.queuedCallbackPtr;
        return this.nextRequestId++;
      case 'CE_EnqueuePromptWithMedia': {
        this.lastQueuedEnqueueKind = 'media';
        this.lastQueuedPromptText = String(args[1]);
        this.queuedCallbackPtr = Number(args[6]);
        this.lastQueuedCallbackPtr = this.queuedCallbackPtr;
        const imageCount = Number(args[3]);
        const flatPtr = Number(args[4]);
        const sizesPtr = Number(args[5]);
        this.lastQueuedMediaImages = [];
        let offset = 0;
        for (let index = 0; index < imageCount; index += 1) {
          const imageSize = this.HEAP32[(sizesPtr >> 2) + index];
          this.lastQueuedMediaImages.push(
            this.HEAPU8.slice(flatPtr + offset, flatPtr + offset + imageSize)
          );
          offset += imageSize;
        }
        return this.nextRequestId++;
      }
      case 'CE_ResetRuntimeObservability':
        return 0;
      case 'CE_DrainRuntimeEvents':
        if (!this.supportsRuntimeEventDrain) {
          throw new Error(`Unexpected ccall: ${ident}`);
        }
        return this.drainRuntimeEvents(args);
      case 'CE_RunSchedulerTick':
      case 'CE_RunRequestStep':
        this.runStepCallCount += 1;
        return this.runRequestStep();
      case 'CE_CancelQueuedRequest':
        this.cancelCallCount += 1;
        this.completedResponseAvailable = true;
        return 1;
      case 'CE_Close':
        this.closeCallCount += 1;
        return 0;
      case 'CE_Init':
      case 'CE_InitWithMultimodal':
        this.lastInitIdent = ident;
        this.lastInitArgs = [...args];
        return this.initResult;
      case 'CE_GetMediaMarker':
        return this.mediaMarker == null ? 0 : this.writeTempCString(this.mediaMarker);
      case 'CE_GetChatTemplate':
        return this.chatTemplate == null ? 0 : this.writeTempCString(this.chatTemplate);
      case 'CE_ApplyChatTemplate': {
        if (this.chatTemplate == null) {
          return 0;
        }
        const messages = JSON.parse(String(args[0])) as Array<{
          role: string;
          content: string | Array<{ type: string; text: string }>;
        }>;
        this.lastAppliedChatTemplateMessages = messages;
        const content = messages
          .map((message) =>
            typeof message.content === 'string'
              ? message.content
              : message.content.map((part) => `${part.type}:${part.text}`).join(',')
          )
          .join('|');
        if (this.returnEmptyAppliedChatTemplate) {
          return 0;
        }
        return this.writeTempCString(
          this.appliedChatTemplateText || `templated:${content}:${Number(args[1])}`
        );
      }
      case 'CE_FreeString':
        return 0;
      case 'CE_GetCompletedRequestStatus':
        return this.completedResponseAvailable
          ? this.completedStatus
          : COMPLETED_REQUEST_STATUS_PENDING;
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
        this.completedResponseAvailable = false;
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
      if (this.transport === 'event') {
        this.queueRuntimeTokenEvent(this.nextRequestId - 1, text);
        return;
      }
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
      this.completedResponseAvailable = true;
      if (this.supportsRuntimeEventDrain) {
        this.queueRuntimeTerminalEvent(this.nextRequestId - 1, this.completedStatus);
      }
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
    this.completedResponseAvailable = true;
    if (this.supportsRuntimeEventDrain) {
      this.queueRuntimeTerminalEvent(this.nextRequestId - 1, this.completedStatus);
    }
    this.insideNativeStep = false;
    return REQUEST_STEP_RESULT_TERMINAL;
  }

  private drainRuntimeEvents(args: unknown[]): number {
    const eventBufferPtr = Number(args[0]);
    const eventCapacity = Number(args[1]);
    const textBufferPtr = Number(args[2]);
    const textCapacity = Number(args[3]);
    const resultPtr = Number(args[4]);
    const resultOffset = resultPtr >> 2;

    if (eventCapacity <= 0) {
      this.HEAP32[resultOffset] = 0;
      this.HEAP32[resultOffset + 1] = 0;
      return 0;
    }

    let drainedEvents = 0;
    let usedTextBytes = 0;
    while (drainedEvents < eventCapacity && this.runtimeEvents.length > 0) {
      const event = this.runtimeEvents[0];
      const textBytes = event.text.length > 0 ? event.text.length + 1 : 0;
      if (textBytes > 0 && usedTextBytes + textBytes > textCapacity) {
        break;
      }

      const eventOffset = (eventBufferPtr + drainedEvents * 20) >> 2;
      this.HEAP32[eventOffset] = event.requestId;
      this.HEAP32[eventOffset + 1] = event.kind;
      this.HEAP32[eventOffset + 2] = event.status;
      this.HEAP32[eventOffset + 3] = usedTextBytes;
      this.HEAP32[eventOffset + 4] = event.text.length;
      if (event.text.length > 0) {
        this.writeCString(event.text, textBufferPtr + usedTextBytes);
        usedTextBytes += event.text.length + 1;
      }
      this.runtimeEvents.shift();
      drainedEvents += 1;
    }

    this.HEAP32[resultOffset] = drainedEvents;
    this.HEAP32[resultOffset + 1] = usedTextBytes;
    return 0;
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
    const ints = [9, 7, 2, 2, 2, 321, 3, 1, 1, 0, 4, 0, 0, 1];

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
  completions: number;
  activeRuns: number;
} {
  const runtimeState = runtime as unknown as {
    queuedPromptCallbacks: Map<number, unknown>;
    queuedPromptTokenBuffers: Map<number, unknown>;
    queuedPromptCallbackPtrs: Map<number, unknown>;
    queuedPromptCallbackErrors: Map<number, unknown>;
    tracker: {
      completions: Map<number, unknown>;
      activeRuns: Set<number>;
    };
  };

  return {
    callbacks: runtimeState.queuedPromptCallbacks.size,
    buffers: runtimeState.queuedPromptTokenBuffers.size,
    callbackPtrs: runtimeState.queuedPromptCallbackPtrs.size,
    callbackErrors: runtimeState.queuedPromptCallbackErrors.size,
    completions: runtimeState.tracker.completions.size,
    activeRuns: runtimeState.tracker.activeRuns.size,
  };
}

function createMockRuntimeObservability(
  outputTokenCount: number,
  batchParticipationCount = 3
): { doubles: number[]; ints: number[] } {
  return {
    doubles: [12.5, 3.5, 4.5, 1.5, 2.5, 6.5, 0.5, 0.75, 12.5],
    ints: [9, 7, outputTokenCount, outputTokenCount, outputTokenCount, 321, batchParticipationCount, 1, 1, 0, 4, 0, 0, 1],
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
  private schedulerTickDeferred:
    | { promise: Promise<number>; resolve: (value: number) => void }
    | null = null;
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
    if (this.schedulerTickDeferred != null) {
      const deferred = this.schedulerTickDeferred;
      this.schedulerTickDeferred = null;
      deferred.resolve(REQUEST_STEP_RESULT_PROGRESSED);
    }
  }

  private handleCall(ident: string, args: unknown[]): Promise<number> | number {
    const requestId = Number(args[0]);
    switch (ident) {
      case 'CE_ResetRuntimeObservability':
        this.currentObservability = createMockRuntimeObservability(0, 0);
        return 0;
      case 'CE_Close':
        return 0;
      case 'CE_Init':
        return 0;
      case 'CE_RunSchedulerTick':
        if (this.completed.size > 0) {
          return REQUEST_STEP_RESULT_PROGRESSED;
        }
        return this.ensureSchedulerTickDeferred().promise;
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

  private ensureSchedulerTickDeferred(): {
    promise: Promise<number>;
    resolve: (value: number) => void;
  } {
    if (this.schedulerTickDeferred != null) {
      return this.schedulerTickDeferred;
    }
    let resolve!: (value: number) => void;
    const promise = new Promise<number>((promiseResolve) => {
      resolve = promiseResolve;
    });
    this.schedulerTickDeferred = { promise, resolve };
    return this.schedulerTickDeferred;
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

class MockWaitingBurstModule {
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
  public runStepCallCount = 0;

  private readonly heapU8: Uint8Array;
  private nextHeapPtr = 1024;
  private readonly requestId = 61;
  private completedResponseAvailable = false;

  constructor(private readonly waitingCountBeforeTerminal: number) {
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

  public addFunction(): number {
    return 0;
  }

  public removeFunction(): void {}

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
    const requestId = Number(args[0]);
    switch (ident) {
      case 'CE_EnqueuePrompt':
        return this.requestId;
      case 'CE_ResetRuntimeObservability':
      case 'CE_Close':
      case 'CE_Init':
        return 0;
      case 'CE_RunSchedulerTick':
        this.runStepCallCount += 1;
        if (this.runStepCallCount <= this.waitingCountBeforeTerminal) {
          return REQUEST_STEP_RESULT_WAITING;
        }
        this.completedResponseAvailable = true;
        return REQUEST_STEP_RESULT_TERMINAL;
      case 'CE_GetCompletedRequestStatus':
        if (requestId !== this.requestId || !this.completedResponseAvailable) {
          return COMPLETED_REQUEST_STATUS_PENDING;
        }
        return COMPLETED_REQUEST_STATUS_COMPLETED;
      case 'CE_GetCompletedRequestOutputSize':
        return 'wait-burst-output'.length;
      case 'CE_CopyCompletedRequestOutput':
        this.writeCString('wait-burst-output', args[1] as number);
        return 'wait-burst-output'.length;
      case 'CE_GetCompletedRequestErrorSize':
        return 0;
      case 'CE_CopyCompletedRequestError':
        return 0;
      case 'CE_GetCompletedRequestRuntimeObservability':
        this.writeRuntimeObservability(args[1] as number);
        return 0;
      case 'CE_GetRuntimeObservability':
        this.writeRuntimeObservability(args[0] as number);
        return 0;
      case 'CE_ConsumeCompletedRequest':
        this.completedResponseAvailable = false;
        return 1;
      default:
        throw new Error(`Unexpected ccall: ${ident}`);
    }
  }

  private writeCString(value: string, ptr: number): void {
    const bytes = new TextEncoder().encode(value);
    this.heapU8.set(bytes, ptr);
    this.heapU8[ptr + bytes.length] = 0;
  }

  private writeRuntimeObservability(ptr: number): void {
    const observability = createMockRuntimeObservability(1, 4);
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

function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(message));
    }, ms);

    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

test('MainThreadEngineRuntime caches native metadata and formats auto-chat prompts via the loaded template', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  module.mediaMarker = '<__media__>';
  module.chatTemplate = 'native-template';
  module.appliedChatTemplateText = 'templated prompt';
  (runtime as unknown as { module: MockMainThreadModule }).module = module;

  await runtime.initEngine('/models/template.gguf');
  const requestId = await runtime.queuePrompt('ctx', 'hello world', {
    nTokens: 16,
  });

  assert.equal(requestId, 7);
  assert.equal(runtime.getMediaMarker(), '<__media__>');
  assert.equal(module.lastInitIdent, 'CE_Init');
  assert.equal(module.lastQueuedEnqueueKind, 'text');
  assert.equal(module.lastQueuedPromptText, 'templated prompt');
});

test('MainThreadEngineRuntime applies the native chat template for media prompts and validates marker count', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  module.mediaMarker = '<__media__>';
  module.chatTemplate = 'native-template';
  module.appliedChatTemplateText = 'templated <__media__> prompt';
  (runtime as unknown as { module: MockMainThreadModule }).module = module;

  await runtime.initEngine('/models/vision.gguf');
  const requestId = await runtime.queuePrompt('ctx', 'describe <__media__>', {
    nTokens: 16,
    media: [new Uint8Array([1, 2, 3])],
  });

  assert.equal(requestId, 7);
  assert.equal(module.lastQueuedEnqueueKind, 'media');
  assert.equal(module.lastQueuedPromptText, 'templated <__media__> prompt');
  assert.deepEqual(module.lastQueuedMediaImages, [new Uint8Array([1, 2, 3])]);
  assert.deepEqual(module.lastAppliedChatTemplateMessages, [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'describe ' },
        { type: 'media_marker', text: '<__media__>' },
      ],
    },
  ]);

  module.appliedChatTemplateText = 'templated prompt without marker';
  await assert.rejects(
    runtime.queuePrompt('ctx', 'describe nothing', {
      nTokens: 16,
      media: [new Uint8Array([9])],
    }),
    /Prompt contains 0 media marker\(s\) but 1 image\(s\) were provided/
  );
});

test('MainThreadEngineRuntime applies the native chat template when PromptOptions.messages is provided', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  module.chatTemplate = 'native-template';
  module.appliedChatTemplateText = 'templated multi-turn prompt';
  (runtime as unknown as { module: MockMainThreadModule }).module = module;

  await runtime.initEngine('/models/template.gguf');
  const requestId = await runtime.queuePrompt('ctx', '', {
    nTokens: 16,
    messages: [
      { role: 'system', content: 'You are Aria.' },
      { role: 'user', content: 'first question' },
      { role: 'assistant', content: 'first reply' },
      { role: 'user', content: 'second question' },
    ],
  });

  assert.equal(requestId, 7);
  assert.equal(module.lastQueuedEnqueueKind, 'text');
  assert.equal(module.lastQueuedPromptText, 'templated multi-turn prompt');
  assert.deepEqual(module.lastAppliedChatTemplateMessages, [
    { role: 'system', content: 'You are Aria.' },
    { role: 'user', content: 'first question' },
    { role: 'assistant', content: 'first reply' },
    { role: 'user', content: 'second question' },
  ]);
});

test('MainThreadEngineRuntime rejects PromptOptions.messages when the model has no chat template', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  module.chatTemplate = '';
  (runtime as unknown as { module: MockMainThreadModule }).module = module;

  await runtime.initEngine('/models/no-template.gguf');
  await assert.rejects(
    runtime.queuePrompt('ctx', '', {
      nTokens: 16,
      messages: [{ role: 'user', content: 'hi' }],
    }),
    /loaded model does not expose a chat template/
  );
});

test('MainThreadEngineRuntime rejects PromptOptions.messages combined with media attachments', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  module.mediaMarker = '<__media__>';
  module.chatTemplate = 'native-template';
  (runtime as unknown as { module: MockMainThreadModule }).module = module;

  await runtime.initEngine('/models/vision.gguf');
  await assert.rejects(
    runtime.queuePrompt('ctx', '<__media__>', {
      nTokens: 16,
      messages: [{ role: 'user', content: 'hi' }],
      media: [new Uint8Array([1])],
    }),
    /not currently compatible with media attachments/
  );
});

test('MainThreadEngineRuntime fails loudly when native auto-chat formatting returns an empty prompt', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  module.chatTemplate = 'native-template';
  module.returnEmptyAppliedChatTemplate = true;
  (runtime as unknown as { module: MockMainThreadModule }).module = module;

  await runtime.initEngine('/models/template.gguf');

  await assert.rejects(
    runtime.queuePrompt('ctx', 'hello world', {
      nTokens: 16,
    }),
    /Failed to apply the model chat template/i
  );
});

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
  assert.equal(runtime.getRuntimeAggregateObservability()?.outputTokenCount, 2);
  assert.ok((runtime.getRuntimeAggregateObservability()?.tokensPerSecond ?? 0) > 0);
  assert.equal(response.completed, true);
  assert.equal(response.failed, false);
  assert.equal(response.cancelled, false);
  assert.equal(response.outputText, 'tok1tok2');
  const requestObservability =
    response.requestObservability as DetailedRequestObservabilityMetrics | null;
  const runtimeObservability =
    response.runtimeObservability as DetailedRequestObservabilityMetrics | null;
  assert.equal(requestObservability?.promptEvalMs, 3.5);
  assert.equal(requestObservability?.outputTokenCount, 2);
  assert.ok((requestObservability?.tokensPerSecond ?? 0) > 0);
  assert.equal(runtimeObservability?.promptEvalMs, 3.5);
  assert.equal(runtimeObservability?.outputTokenCount, 2);
  assert.equal(module.consumeCallCount, 1);
  assert.deepEqual(module.removedFunctionPtrs, [1]);
});

test('MainThreadEngineRuntime skips callback pointers when native runtime event drain is available', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success', 'event');
  attachReadyModule(runtime, module);

  const tokens: string[] = [];
  const requestId = await runtime.queuePrompt('ctx', 'prompt', {
    nTokens: 16,
    onToken: (token) => {
      tokens.push(token);
    },
  });
  const response = await runtime.runQueuedRequest(requestId);

  assert.equal(requestId, 7);
  assert.equal(module.lastQueuedCallbackPtr, 0);
  assert.equal(response.outputText, 'tok1tok2');
  assert.deepEqual(tokens, ['tok1', 'tok2']);
  const transportObservability = runtime.getTransportObservability();
  assert.equal(transportObservability.executionMode, 'main-thread');
  assert.equal(transportObservability.workerBacked, false);
  assert.equal(transportObservability.tokenTransportPreference, 'auto');
  assert.equal(transportObservability.activeTokenTransport, 'runtime-events');
  assert.equal(transportObservability.tokenCallbackRegistrationCount, 0);
  assert.equal(transportObservability.nativeCallbackTokenCount, 0);
  assert.ok((transportObservability.runtimeEventDrainCount ?? 0) > 0);
  assert.equal(transportObservability.runtimeEventTokenCount, 2);
  assert.equal(transportObservability.runtimeEventTerminalCount, 1);
  assert.equal(transportObservability.runtimeEventTextBytes, 8);
  runtime.close();
});

test('MainThreadEngineRuntime falls back to callback pointers when runtime event drain is unavailable', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success', 'callback', {
    supportsRuntimeEventDrain: false,
  });
  attachReadyModule(runtime, module);

  const tokens: string[] = [];
  const requestId = await runtime.queuePrompt('ctx', 'prompt', {
    nTokens: 16,
    onToken: (token) => {
      tokens.push(token);
    },
  });
  const response = await runtime.runQueuedRequest(requestId);

  assert.equal(requestId, 7);
  assert.equal(module.lastQueuedCallbackPtr, 1);
  assert.equal(response.outputText, 'tok1tok2');
  assert.deepEqual(tokens, ['tok1', 'tok2']);
  const transportObservability = runtime.getTransportObservability();
  assert.equal(transportObservability.executionMode, 'main-thread');
  assert.equal(transportObservability.workerBacked, false);
  assert.equal(transportObservability.tokenTransportPreference, 'auto');
  assert.equal(transportObservability.activeTokenTransport, 'callbacks');
  assert.equal(transportObservability.tokenCallbackRegistrationCount, 1);
  assert.equal(transportObservability.nativeCallbackTokenCount, 2);
  assert.equal(transportObservability.runtimeEventDrainCount, 0);
  assert.equal(transportObservability.runtimeEventTokenCount, 0);
  assert.equal(transportObservability.runtimeEventTerminalCount, 0);
  assert.equal(transportObservability.runtimeEventTextBytes, 0);
  runtime.close();
});

test('MainThreadEngineRuntime rejects forced runtime-events mode when the loaded runtime does not expose event drain', async () => {
  const runtime = new MainThreadEngineRuntime({
    debugTokenTransport: 'runtime-events',
  });
  const module = new MockMainThreadModule('success', 'callback', {
    supportsRuntimeEventDrain: false,
  });
  attachReadyModule(runtime, module);

  await assert.rejects(
    runtime.queuePrompt('ctx', 'prompt', {
      nTokens: 16,
      onToken: () => {},
    }),
    /debugTokenTransport=runtime-events requires CE_DrainRuntimeEvents support/i
  );
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
    completions: 0,
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
    completions: 0,
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
    tracker: {
      track: (requestId: number) => unknown;
    };
    lastModelLoadInfo: object | null;
    transportObservability: {
      enabled: boolean;
      flushCount: number;
      coalescedTokenCount: number;
    };
  };
  runtimeState.tracker.track(requestId);
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
    completions: 0,
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
    tokenTransportPreference: 'auto',
    activeTokenTransport: 'none',
    tokenCallbackRegistrationCount: 0,
    nativeCallbackTokenCount: 0,
    runtimeEventDrainCount: 0,
    runtimeEventTokenCount: 0,
    runtimeEventTerminalCount: 0,
    runtimeEventTextBytes: 0,
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
    tracker: {
      track: (requestId: number) => unknown;
    };
    runtimeObservabilityEnabled: boolean;
    backendProfilingEnabled: boolean;
    engineInitialized: boolean;
    transportObservability: {
      enabled: boolean;
      flushCount: number;
    };
  };
  runtimeState.tracker.track(requestId);

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
    completions: 0,
    activeRuns: 0,
  });
  assert.equal(runtimeState.engineInitialized, false);
  assert.equal(runtimeState.runtimeObservabilityEnabled, false);
  assert.equal(runtimeState.backendProfilingEnabled, false);
  assert.equal(runtimeState.transportObservability.enabled, false);
  assert.equal(runtimeState.transportObservability.flushCount, 0);
});

test('MainThreadEngineRuntime starts queued execution before runQueuedRequest() is awaited', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const tokens: string[] = [];
  await runtime.queuePrompt('ctx-background-progress', 'prompt', {
    nTokens: 16,
    onToken: (token) => {
      tokens.push(token);
    },
  });

  await new Promise((resolve) => setTimeout(resolve, 10));

  assert.ok(module.runStepCallCount > 0);
  assert.deepEqual(tokens, ['tok1', 'tok2']);
});

test('MainThreadEngineRuntime can run queued execution through an external scheduler pump controller', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);
  runtime.setQueuedRequestPumpMode('external');

  const tokens: string[] = [];
  const requestId = await runtime.queuePrompt('ctx-external-pump', 'prompt', {
    nTokens: 16,
    onToken: (token) => {
      tokens.push(token);
    },
  });

  assert.equal(module.runStepCallCount, 0);

  const waiter = runtime.runQueuedRequest(requestId);
  while (runtime.hasActiveQueuedRequests()) {
    await runtime.pumpQueuedRequestsOnce();
  }

  const response = await waiter;
  assert.equal(response.outputText, 'tok1tok2');
  assert.deepEqual(tokens, ['tok1', 'tok2']);
});

test('MainThreadEngineRuntime lets a late runQueuedRequest() waiter observe an already-completed request', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const requestId = await runtime.queuePrompt('ctx-late-waiter', 'prompt', 16);
  await new Promise((resolve) => setTimeout(resolve, 10));

  const stepCallsBeforeWaiter = module.runStepCallCount;
  const response = await runtime.runQueuedRequest(requestId);

  assert.ok(stepCallsBeforeWaiter > 0);
  assert.equal(module.runStepCallCount, stepCallsBeforeWaiter);
  assert.equal(response.outputText, 'tok1tok2');
});

test('MainThreadEngineRuntime does not yield to setTimeout during short WAITING bursts', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockWaitingBurstModule(3);
  attachReadyModule(runtime, module as unknown as MockMainThreadModule);

  const originalSetTimeout = globalThis.setTimeout;
  let zeroDelayTimeoutCount = 0;
  globalThis.setTimeout = (((handler: TimerHandler, timeout?: number, ...args: unknown[]) => {
    if (timeout === 0) {
      zeroDelayTimeoutCount += 1;
    }
    return originalSetTimeout(handler, timeout as number, ...(args as []));
  }) as typeof setTimeout);

  try {
    const requestId = await runtime.queuePrompt('ctx-wait-burst', 'prompt', 16);
    const response = await runtime.runQueuedRequest(requestId);

    assert.equal(response.outputText, 'wait-burst-output');
    assert.equal(module.runStepCallCount, 4);
    assert.equal(zeroDelayTimeoutCount, 0);
  } finally {
    globalThis.setTimeout = originalSetTimeout;
  }
});

test('MainThreadEngineRuntime shares one completed response across concurrent runQueuedRequest() waiters', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockConcurrentObservabilityModule();
  attachReadyModule(runtime, module as unknown as MockMainThreadModule);

  const firstPromise = runtime.runQueuedRequest(303);
  await Promise.resolve();
  const secondPromise = runtime.runQueuedRequest(303);
  await Promise.resolve();

  module.resolveTerminal(303, 'shared-output', 2, 12);
  const [firstResponse, secondResponse] = await Promise.all([
    withTimeout(firstPromise, 25, 'Timed out waiting for first queued waiter.'),
    withTimeout(secondPromise, 25, 'Timed out waiting for second queued waiter.'),
  ]);

  assert.equal(firstResponse.outputText, 'shared-output');
  assert.equal(secondResponse.outputText, 'shared-output');
});

test('MainThreadEngineRuntime rejects outstanding queued waiters when close() is called', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockConcurrentObservabilityModule();
  attachReadyModule(runtime, module as unknown as MockMainThreadModule);

  const waiter = runtime.runQueuedRequest(404);
  await Promise.resolve();
  runtime.close();

  await assert.rejects(
    withTimeout(waiter, 25, 'Timed out waiting for queued waiter rejection after close().'),
    /closed/i
  );
});

test('MainThreadEngineRuntime rejects outstanding queued waiters when initEngine() reinitializes the runtime', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockConcurrentObservabilityModule();
  attachReadyModule(runtime, module as unknown as MockMainThreadModule);

  const waiter = runtime.runQueuedRequest(505);
  await Promise.resolve();
  await runtime.initEngine('/models/reinit.gguf');

  await assert.rejects(
    withTimeout(waiter, 25, 'Timed out waiting for queued waiter rejection after reinit.'),
    /reinit|closed|reset/i
  );
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

test('MainThreadEngineRuntime prepares a discovered vision bundle and mounts model plus projector together', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const originalFetch = globalThis.fetch;
  const originalIsSupported = FileSystemStorage.isSupported;

  try {
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = () => false;
    globalThis.fetch = (async (url: string, init?: { method?: string }) => {
      if (url === 'https://huggingface.co/api/models/org/repo') {
        return new Response(
          JSON.stringify({
            siblings: [
              { rfilename: 'Qwen2-VL-2B-Instruct-Q4_K_M.gguf' },
              { rfilename: 'mmproj-model-f16.gguf' },
            ],
          }),
          { status: 200 }
        );
      }
      if (init?.method === 'HEAD') {
        return {
          ok: true,
          status: 200,
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

    const bundle = await runtime.prepareModelBundle({
      kind: 'url',
      url: 'https://huggingface.co/org/repo/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
    });
    const mountedNames =
      module.mountCalls.at(-1)?.files.map((file) => (file as { name?: string }).name || 'model.gguf') ?? [];

    assert.equal(bundle.isVisionModel, true);
    assert.equal(bundle.projectorStatus, 'discovered');
    assert.equal(bundle.modelPath, '/workerfs_model/Qwen2-VL-2B-Instruct-Q4_K_M.gguf');
    assert.equal(bundle.multimodalProjectorPath, '/workerfs_model/mmproj-model-f16.gguf');
    assert.equal(runtime.getLastModelLoadInfo()?.modelPath, bundle.modelPath);
    assert.deepEqual(mountedNames, [
      'Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
      'mmproj-model-f16.gguf',
    ]);
  } finally {
    globalThis.fetch = originalFetch;
    (FileSystemStorage as unknown as { isSupported: () => boolean }).isSupported = originalIsSupported;
  }
});

test('MainThreadEngineRuntime auto-pairs local projector files and sorts shard names in prepared bundles', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const bundle = await runtime.prepareModelBundle({
    kind: 'files',
    files: [
      new File([Uint8Array.from([2])], 'Qwen2-VL-2B-Instruct-00002-of-00002.gguf'),
      new File([Uint8Array.from([9])], 'mmproj-model-f16.gguf'),
      new File([Uint8Array.from([1])], 'Qwen2-VL-2B-Instruct-00001-of-00002.gguf'),
    ],
  });
  const mountedNames =
    module.mountCalls.at(-1)?.files.map((file) => (file as { name?: string }).name || 'model.gguf') ?? [];

  assert.equal(bundle.projectorStatus, 'paired');
  assert.equal(bundle.modelPath, '/workerfs_model/Qwen2-VL-2B-Instruct-00001-of-00002.gguf');
  assert.equal(bundle.multimodalProjectorPath, '/workerfs_model/mmproj-model-f16.gguf');
  assert.deepEqual(mountedNames, [
    'Qwen2-VL-2B-Instruct-00001-of-00002.gguf',
    'Qwen2-VL-2B-Instruct-00002-of-00002.gguf',
    'mmproj-model-f16.gguf',
  ]);
});

test('MainThreadEngineRuntime pairs a metadata-detected local projector even when the filename is generic', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const bundle = await runtime.prepareModelBundle({
    kind: 'files',
    files: [
      new File(
        [
          toBlobPart(buildGguf([
            {
              key: 'general.type',
              type: GgufValueType.STRING,
              value: encodeString('model'),
            },
            {
              key: 'general.architecture',
              type: GgufValueType.STRING,
              value: encodeString('qwen2vl'),
            },
          ])),
        ],
        'model-00002-of-00002.gguf'
      ),
      new File(
        [
          toBlobPart(buildGguf([
            {
              key: 'general.type',
              type: GgufValueType.STRING,
              value: encodeString('mmproj'),
            },
            {
              key: 'general.architecture',
              type: GgufValueType.STRING,
              value: encodeString('clip'),
            },
          ])),
        ],
        'adapter.gguf'
      ),
      new File(
        [
          toBlobPart(buildGguf([
            {
              key: 'general.type',
              type: GgufValueType.STRING,
              value: encodeString('model'),
            },
            {
              key: 'general.architecture',
              type: GgufValueType.STRING,
              value: encodeString('qwen2vl'),
            },
          ])),
        ],
        'model-00001-of-00002.gguf'
      ),
    ],
  });
  const mountedNames =
    module.mountCalls.at(-1)?.files.map((file) => (file as { name?: string }).name || 'model.gguf') ?? [];

  assert.equal(bundle.isVisionModel, true);
  assert.equal(bundle.projectorStatus, 'paired');
  assert.equal(bundle.modelArchitecture, 'qwen2vl');
  assert.equal(bundle.multimodalProjectorPath, '/workerfs_model/adapter.gguf');
  assert.deepEqual(mountedNames, [
    'model-00001-of-00002.gguf',
    'model-00002-of-00002.gguf',
    'adapter.gguf',
  ]);
});

test('MainThreadEngineRuntime initEngine(bundle) uses bundle projectors by default and respects explicit overrides', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  const preparedBundle = {
    sourceKind: 'url' as const,
    modelPath: '/workerfs_model/model.gguf',
    multimodalProjectorPath: '/workerfs_model/mmproj.gguf',
    isVisionModel: true,
    projectorStatus: 'explicit' as const,
    modelName: 'model.gguf',
    detectionMethod: 'filename' as const,
    modelType: 'model',
    modelArchitecture: 'qwen2vl',
    modelLoadInfo: null,
    projectorLoadInfo: null,
  };

  await runtime.initEngine(preparedBundle);
  assert.equal(module.lastInitIdent, 'CE_InitWithMultimodal');
  assert.equal(module.lastInitArgs?.[20], '/workerfs_model/mmproj.gguf');

  await runtime.initEngine(preparedBundle, {
    multimodalProjectorPath: '/override/mmproj.gguf',
  });
  assert.equal(module.lastInitArgs?.[20], '/override/mmproj.gguf');

  await runtime.initEngine({
    ...preparedBundle,
    multimodalProjectorPath: null,
    projectorStatus: 'missing',
  });
  assert.equal(module.lastInitIdent, 'CE_Init');
});

test('MainThreadEngineRuntime forwards multimodalUseGpu independently from text GPU layers', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  await runtime.initEngine('/models/vision.gguf', {
    nGpuLayers: 99,
    multimodalProjectorPath: '/models/mmproj.gguf',
    multimodalUseGpu: false,
  });

  assert.equal(module.lastInitIdent, 'CE_InitWithMultimodal');
  assert.equal(module.lastInitArgs?.[7], 99);
  assert.equal(module.lastInitArgs?.[20], '/models/mmproj.gguf');
  assert.equal(module.lastInitArgs?.[21], 0);
});

test('MainThreadEngineRuntime forwards configured decode sampling to native init', async () => {
  const runtime = new MainThreadEngineRuntime({});
  const module = new MockMainThreadModule('success');
  attachReadyModule(runtime, module);

  await runtime.initEngine('/models/sampling.gguf', {
    sampling: {
      repeatLastN: 96,
      repeatPenalty: 1.15,
      frequencyPenalty: 0.2,
      presencePenalty: 0.3,
      topK: 24,
      topP: 0.92,
      minP: 0.08,
      temperature: 0.55,
      seed: 1337,
    },
  });

  assert.equal(module.lastInitIdent, 'CE_Init');
  assert.deepEqual(module.lastInitArgs?.slice(-9), [
    96,
    1.15,
    0.2,
    0.3,
    24,
    0.92,
    0.08,
    0.55,
    1337,
  ]);
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
      browserModelCache: {
        buildEntryKey: (identity: {
          canonicalUrl: string;
          fileName: string;
          etag: string;
          lastModified: string;
          contentLength: number;
        }) => string;
        get: (identity: {
          canonicalUrl: string;
          fileName: string;
          etag: string;
          lastModified: string;
          contentLength: number;
        }) => Promise<{ key: string; file: File } | null>;
        storeStream: (
          identity: {
            canonicalUrl: string;
            fileName: string;
            etag: string;
            lastModified: string;
            contentLength: number;
          },
          stream: ReadableStream<Uint8Array>
        ) => Promise<{ key: string; file: File }>;
      };
    };
    runtimeState.browserModelCache.get = async (identity) => {
      const key = runtimeState.browserModelCache.buildEntryKey(identity);
      requestedCacheKey = key;
      return key === 'model.gguf'
        ? { key, file: new File([Uint8Array.from([9, 9, 9, 9])], key) }
        : null;
    };
    runtimeState.browserModelCache.storeStream = async (identity) => {
      const key = runtimeState.browserModelCache.buildEntryKey(identity);
      return {
        key,
        file: new File([Uint8Array.from([1, 2, 3, 4])], key),
      };
    };

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

  const secondRuntimeObservability =
    secondResponse.runtimeObservability as DetailedRequestObservabilityMetrics | null;
  const firstRuntimeObservability =
    firstResponse.runtimeObservability as DetailedRequestObservabilityMetrics | null;
  assert.equal(secondRuntimeObservability?.outputTokenCount, 2);
  assert.equal(secondRuntimeObservability?.batchParticipationCount, 20);
  assert.equal(firstRuntimeObservability?.outputTokenCount, 1);
  assert.equal(firstRuntimeObservability?.batchParticipationCount, 10);
});
