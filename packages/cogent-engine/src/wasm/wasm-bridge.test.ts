import assert from 'node:assert/strict';
import test from 'node:test';

import { WasmBridge, MAX_GRAMMAR_BYTES } from './wasm-bridge.js';
import { EngineModule } from './engine-module.js';

const COMPLETED_REQUEST_STATUS_PENDING = 0;
const COMPLETED_REQUEST_STATUS_COMPLETED = 1;
const COMPLETED_REQUEST_STATUS_CANCELLED = 2;
const REQUEST_STEP_RESULT_PROGRESSED = 1;
const RUNTIME_EVENT_KIND_TOKEN = 1;
const RUNTIME_EVENT_KIND_TERMINAL = 2;

class MockWasmBridgeModule implements EngineModule {
  public readonly FS = {
    analyzePath: () => ({ exists: false }),
    mkdir: () => {},
    writeFile: () => {},
    unlink: () => {},
    mount: () => {},
    unmount: () => {},
  };

  public readonly WORKERFS = {};
  public readonly HEAP32: Int32Array;
  public readonly HEAPF64: Float64Array;
  public readonly HEAPU8: Uint8Array;
  public freedPointers: number[] = [];
  public freedStringPointers: number[] = [];

  public supportsBurst = true;
  public supportsBurstWithDeadline = true;
  public supportsRuntimeEvents = true;
  public schedulerTickResult = REQUEST_STEP_RESULT_PROGRESSED;
  public schedulerBurstStatus = REQUEST_STEP_RESULT_PROGRESSED;
  public schedulerBurstResult = {
    ticksExecuted: 4,
    progressedTicks: 3,
    completedResponseCount: 2,
    emittedTokenCount: 1,
  };
  public completedStatus = COMPLETED_REQUEST_STATUS_COMPLETED;
  public completedOutputText = 'hello';
  public completedErrorText = '';
  public completedConsumed = false;
  public backendJson = '{"adapter":"webgpu"}';
  public mediaMarker: string | null = null;
  public chatTemplate: string | null = null;
  public appliedChatTemplateText = '';
  public lastFreedBackendPtr = 0;
  public lastInitCall:
    | {
        ident: string;
        argTypes: string[];
        args: any[];
      }
    | null = null;
  public lastMediaEnqueue:
    | {
        contextKey: string;
        promptText: string;
        maxOutputTokens: number;
        callbackPtr: number;
        images: Uint8Array[];
        grammar: string;
      }
    | null = null;
  public lastEnqueue:
    | {
        contextKey: string;
        promptText: string;
        maxOutputTokens: number;
        callbackPtr: number;
        grammar: string;
      }
    | null = null;
  public lastBurstWithDeadlineArgs:
    | {
        maxTicks: number;
        maxCompletedResponses: number;
        maxEmittedTokens: number;
        maxDurationUs: number;
      }
    | null = null;
  public runtimeEventBatch: Array<{
    requestId: number;
    kind: number;
    text: string;
  }> = [];
  public closeCallCount = 0;

  private readonly heapU8: Uint8Array;
  private readonly functionTable = new Map<number, (...args: number[]) => number>();
  private nextHeapPtr = 1024;
  private nextFunctionPtr = 1;

  public constructor() {
    const memory = new ArrayBuffer(1024 * 1024);
    this.heapU8 = new Uint8Array(memory);
    this.HEAPU8 = this.heapU8;
    this.HEAP32 = new Int32Array(memory);
    this.HEAPF64 = new Float64Array(memory);
  }

  public _free(ptr: number | bigint): void {
    this.freedPointers.push(Number(ptr));
  }

  public _malloc(size: number | bigint): number | bigint {
    const alignedSize = (Number(size) + 7) & ~7;
    const ptr = this.nextHeapPtr;
    this.nextHeapPtr += alignedSize;
    return ptr;
  }

  public addFunction(func: (...args: any[]) => any, _signature: string): number | bigint {
    const ptr = this.nextFunctionPtr++;
    this.functionTable.set(ptr, func as (...args: number[]) => number);
    return ptr;
  }

  public removeFunction(ptr: number | bigint): void {
    this.functionTable.delete(Number(ptr));
  }

  public UTF8ToString(ptr: number | bigint, maxBytesToRead?: number): string {
    const start = Number(ptr);
    const bytes: number[] = [];
    const maxBytes = maxBytesToRead ?? this.heapU8.length - start;
    for (let index = 0; index < maxBytes; index += 1) {
      const byte = this.heapU8[start + index];
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
    args: any[],
    opts?: { async?: boolean }
  ): Promise<any> | any {
    const result = this.handleCall(ident, _argTypes, args);
    return opts?.async ? Promise.resolve(result) : result;
  }

  public invokeFunction(ptr: number, rawPtr: number, length: number): number {
    const fn = this.functionTable.get(ptr);
    assert.ok(fn);
    return fn(rawPtr, length);
  }

  public writeTempCString(text: string): number {
    const ptr = Number(this._malloc(text.length + 1));
    this.writeCString(text, ptr);
    return ptr;
  }

  private handleCall(ident: string, argTypes: string[], args: any[]): number {
    switch (ident) {
      case 'CE_RunSchedulerBurst':
        if (!this.supportsBurst) {
          throw new Error(`Unexpected ccall: ${ident}`);
        }
        this.writeSchedulerBurstResult(args[3] as number);
        return this.schedulerBurstStatus;
      case 'CE_RunSchedulerBurstWithDeadline':
        if (!this.supportsBurstWithDeadline) {
          throw new Error(`Unexpected ccall: ${ident}`);
        }
        this.lastBurstWithDeadlineArgs = {
          maxTicks: Number(args[0]),
          maxCompletedResponses: Number(args[1]),
          maxEmittedTokens: Number(args[2]),
          maxDurationUs: Number(args[3]),
        };
        this.writeSchedulerBurstResult(args[4] as number);
        return this.schedulerBurstStatus;
      case 'CE_RunSchedulerTick':
        return this.schedulerTickResult;
      case 'CE_DrainRuntimeEvents':
        if (!this.supportsRuntimeEvents) {
          throw new Error(`Unexpected ccall: ${ident}`);
        }
        return this.drainRuntimeEvents(args);
      case 'CE_GetCompletedRequestStatus':
        return this.completedConsumed
          ? COMPLETED_REQUEST_STATUS_PENDING
          : this.completedStatus;
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
      case 'CE_GetCompletedRequestRuntimeObservability':
      case 'CE_GetRuntimeObservability':
        this.writeRuntimeObservability(args.at(-1) as number);
        return 0;
      case 'CE_ConsumeCompletedRequest':
        this.completedConsumed = true;
        return 1;
      case 'CE_GetBackendObservabilityJson':
        return this.writeTempCString(this.backendJson);
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
        const fallbackText = messages
          .map((message) => {
            const content =
              typeof message.content === 'string'
                ? message.content
                : message.content.map((part) => `${part.type}:${part.text}`).join(',');
            return `${message.role}:${content}`;
          })
          .join('|');
        return this.writeTempCString(
          this.appliedChatTemplateText || `templated:${fallbackText}:${Number(args[1])}`
        );
      }
      case 'CE_EnqueuePrompt': {
        this.lastEnqueue = {
          contextKey: String(args[0]),
          promptText: String(args[1]),
          maxOutputTokens: Number(args[2]),
          callbackPtr: Number(args[3]),
          grammar: args[4] == null ? '' : String(args[4]),
        };
        return 77;
      }
      case 'CE_EnqueuePromptWithMedia': {
        const contextKey = String(args[0]);
        const promptText = String(args[1]);
        const maxOutputTokens = Number(args[2]);
        const imageCount = Number(args[3]);
        const flatPtr = Number(args[4]);
        const sizesPtr = Number(args[5]);
        const callbackPtr = Number(args[6]);
        const grammar = args[7] == null ? '' : String(args[7]);
        const images: Uint8Array[] = [];
        let offset = 0;
        for (let index = 0; index < imageCount; index += 1) {
          const byteLength = this.HEAP32[(sizesPtr >> 2) + index];
          images.push(this.HEAPU8.slice(flatPtr + offset, flatPtr + offset + byteLength));
          offset += byteLength;
        }
        this.lastMediaEnqueue = {
          contextKey,
          promptText,
          maxOutputTokens,
          callbackPtr,
          images,
          grammar,
        };
        return 77;
      }
      case 'CE_Init':
      case 'CE_InitWithMultimodal':
        this.lastInitCall = { ident, argTypes: [...argTypes], args };
        return 0;
      case 'CE_FreeString':
        this.lastFreedBackendPtr = Number(args[0]);
        this.freedStringPointers.push(Number(args[0]));
        return 0;
      case 'CE_Close':
        this.closeCallCount += 1;
        return 0;
      default:
        throw new Error(`Unexpected ccall: ${ident}`);
    }
  }

  private writeCString(text: string, ptr: number): void {
    const bytes = new TextEncoder().encode(text);
    this.heapU8.set(bytes, ptr);
    this.heapU8[ptr + bytes.length] = 0;
  }

  private writeSchedulerBurstResult(ptr: number): void {
    const offset = ptr >> 2;
    this.HEAP32[offset] = this.schedulerBurstResult.ticksExecuted;
    this.HEAP32[offset + 1] = this.schedulerBurstResult.progressedTicks;
    this.HEAP32[offset + 2] = this.schedulerBurstResult.completedResponseCount;
    this.HEAP32[offset + 3] = this.schedulerBurstResult.emittedTokenCount;
  }

  private writeRuntimeObservability(ptr: number): void {
    const doubles = [9, 8, 7, 6, 5, 4, 3, 2, 1];
    const ints = [11, 10, 9, 8, 7, 1234, 6, 5, 4, 3, 2, 1, 0, 12];
    const f64Offset = ptr >> 3;
    const i32Offset = (ptr + 9 * 8) >> 2;
    for (let index = 0; index < doubles.length; index += 1) {
      this.HEAPF64[f64Offset + index] = doubles[index];
    }
    for (let index = 0; index < ints.length; index += 1) {
      this.HEAP32[i32Offset + index] = ints[index];
    }
  }

  private drainRuntimeEvents(args: any[]): number {
    const eventBufferPtr = Number(args[0]);
    const textBufferPtr = Number(args[2]);
    const resultPtr = Number(args[4]);

    if (eventBufferPtr === 0 && textBufferPtr === 0) {
      return 0;
    }

    let textOffset = 0;
    for (let index = 0; index < this.runtimeEventBatch.length; index += 1) {
      const event = this.runtimeEventBatch[index];
      const eventOffset = (eventBufferPtr + index * 20) >> 2;
      this.HEAP32[eventOffset] = event.requestId;
      this.HEAP32[eventOffset + 1] = event.kind;
      this.HEAP32[eventOffset + 2] = event.kind === RUNTIME_EVENT_KIND_TERMINAL
        ? this.completedStatus
        : COMPLETED_REQUEST_STATUS_PENDING;

      if (event.text.length > 0) {
        this.writeCString(event.text, textBufferPtr + textOffset);
      }
      this.HEAP32[eventOffset + 3] = textOffset;
      this.HEAP32[eventOffset + 4] = event.text.length;
      textOffset += event.text.length + 1;
    }

    const resultOffset = resultPtr >> 2;
    this.HEAP32[resultOffset] = this.runtimeEventBatch.length;
    this.HEAP32[resultOffset + 1] = textOffset;
    return 0;
  }
}

test('WasmBridge falls back to single scheduler tick when burst API is unavailable', async () => {
  const module = new MockWasmBridgeModule();
  module.supportsBurst = false;
  module.supportsBurstWithDeadline = false;
  module.schedulerTickResult = REQUEST_STEP_RESULT_PROGRESSED;
  const bridge = new WasmBridge(module);

  const result = await bridge.runSchedulerProgress(16, 2, 1);

  assert.equal(result.stepResult, REQUEST_STEP_RESULT_PROGRESSED);
  assert.equal(result.completedResponseCount, 0);
});

test('WasmBridge prefers the deadline burst API when a duration budget is requested', async () => {
  const module = new MockWasmBridgeModule();
  const bridge = new WasmBridge(module);

  const result = await bridge.runSchedulerProgress(16, 2, 8, {
    maxDurationUs: 60_000,
  });

  assert.equal(result.stepResult, REQUEST_STEP_RESULT_PROGRESSED);
  assert.equal(result.completedResponseCount, 2);
  assert.deepEqual(module.lastBurstWithDeadlineArgs, {
    maxTicks: 16,
    maxCompletedResponses: 2,
    maxEmittedTokens: 8,
    maxDurationUs: 60_000,
  });
});

test('WasmBridge drains runtime events into token and terminal batches', () => {
  const module = new MockWasmBridgeModule();
  module.runtimeEventBatch = [
    { requestId: 7, kind: RUNTIME_EVENT_KIND_TOKEN, text: 'tok1' },
    { requestId: 7, kind: RUNTIME_EVENT_KIND_TERMINAL, text: '' },
  ];
  const bridge = new WasmBridge(module);

  const drained = bridge.drainRuntimeEvents(8);

  assert.deepEqual(drained, {
    terminalRequestIds: [7],
    tokenEvents: [{ requestId: 7, token: 'tok1', textLength: 4 }],
    textBytes: 4,
  });
});

test('WasmBridge reuses burst and runtime-event buffers until close()', async () => {
  const module = new MockWasmBridgeModule();
  module.runtimeEventBatch = [
    { requestId: 7, kind: RUNTIME_EVENT_KIND_TOKEN, text: 'tok1' },
  ];
  const bridge = new WasmBridge(module);

  await bridge.runSchedulerProgress(64, 4, 32);
  bridge.drainRuntimeEvents(8);
  module.runtimeEventBatch = [
    { requestId: 7, kind: RUNTIME_EVENT_KIND_TERMINAL, text: '' },
  ];
  bridge.drainRuntimeEvents(4);

  assert.deepEqual(module.freedPointers, []);

  bridge.close();

  assert.equal(module.closeCallCount, 1);
  assert.equal(module.freedPointers.length, 4);
});

test('WasmBridge consumes completed responses and reads request observability', () => {
  const module = new MockWasmBridgeModule();
  module.completedOutputText = 'answer';
  module.completedErrorText = 'warning';
  module.completedStatus = COMPLETED_REQUEST_STATUS_CANCELLED;
  const bridge = new WasmBridge(module);

  const response = bridge.takeCompletedResponse(13);

  assert.equal(response.requestId, 13);
  assert.equal(response.outputText, 'answer');
  assert.equal(response.errorMessage, 'warning');
  assert.equal(response.cancelled, true);
  assert.equal(response.requestObservability?.outputTokenCount, 7);
  assert.equal(module.completedConsumed, true);
});

test('WasmBridge fetches backend observability JSON and frees the returned string', async () => {
  const module = new MockWasmBridgeModule();
  const bridge = new WasmBridge(module);

  const raw = await bridge.getBackendObservabilityJson();

  assert.equal(raw, module.backendJson);
  assert.notEqual(module.lastFreedBackendPtr, 0);
});

test('WasmBridge calls multimodal init when a projector path is configured', async () => {
  const module = new MockWasmBridgeModule();
  const bridge = new WasmBridge(module);

  await bridge.initEngine('/models/model.gguf', {
    nCtx: 4096,
    nBatch: 256,
    nUbatch: 256,
    nSeqMax: 1,
    nThreads: 2,
    nThreadsBatch: 2,
    nGpuLayers: 99,
    flashAttention: -1,
    kvUnified: -1,
    maxCachedSessions: 8,
    retainedPrefixTokens: 100,
    prefillChunkSize: 0,
    prefixCacheIntervalTokens: 128,
    maxPrefixCacheEntries: 32,
    schedulerPolicy: 1,
    decodeTokenReserve: 1,
    adaptivePrefillChunking: 0,
    enableRuntimeObservability: 0,
    enableBackendProfiling: 0,
    multimodalProjectorPath: '/models/mmproj.gguf',
    multimodalUseGpu: 0,
    debugCompareMultimodalEmbeddings: 0,
    imageMinTokens: 64,
    imageMaxTokens: 256,
    samplingRepeatLastN: 96,
    samplingRepeatPenalty: 1.1,
    samplingFrequencyPenalty: 0.2,
    samplingPresencePenalty: 0.3,
    samplingTopK: 24,
    samplingTopP: 0.92,
    samplingMinP: 0.08,
    samplingTemperature: 0.55,
    samplingSeed: 1337,
  });

  assert.deepEqual(module.lastInitCall, {
    ident: 'CE_InitWithMultimodal',
    argTypes: [
      'string',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'string',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
      'number',
    ],
    args: [
      '/models/model.gguf',
      4096,
      256,
      256,
      1,
      2,
      2,
      99,
      -1,
      -1,
      8,
      100,
      0,
      128,
      32,
      1,
      1,
      0,
      0,
      0,
      '/models/mmproj.gguf',
      0,
      0,
      64,
      256,
      96,
      1.1,
      0.2,
      0.3,
      24,
      0.92,
      0.08,
      0.55,
      1337,
    ],
  });
});

test('WasmBridge flattens media buffers and exposes template + marker', () => {
  const module = new MockWasmBridgeModule();
  module.chatTemplate = 'template';
  module.mediaMarker = '<__media__>';
  const bridge = new WasmBridge(module);

  assert.equal(bridge.getMediaMarker(), '<__media__>');
  assert.equal(bridge.getChatTemplate(), 'template');

  const requestId = bridge.enqueuePromptWithMedia(
    'ctx',
    'look <__media__>',
    16,
    [new Uint8Array([1, 2, 3]), new Uint8Array([4, 5])],
    9
  );

  assert.equal(requestId, 77);
  assert.deepEqual(module.lastMediaEnqueue, {
    contextKey: 'ctx',
    promptText: 'look <__media__>',
    maxOutputTokens: 16,
    callbackPtr: 9,
    images: [new Uint8Array([1, 2, 3]), new Uint8Array([4, 5])],
    grammar: '',
  });
});

test('WasmBridge applies model chat template text', () => {
  const module = new MockWasmBridgeModule();
  module.chatTemplate = 'template';
  module.appliedChatTemplateText = 'templated:ok';
  const bridge = new WasmBridge(module);

  const rendered = bridge.applyChatTemplate(
    [
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'hi' },
    ],
    true
  );

  assert.equal(rendered, 'templated:ok');
});

test('WasmBridge decodes token callbacks through the function table', () => {
  const module = new MockWasmBridgeModule();
  const bridge = new WasmBridge(module);
  let seenToken = '';
  const callbackPtr = Number(
    bridge.registerTokenCallback((token) => {
      seenToken = token;
      return 0;
    })
  );
  const tokenPtr = module.writeTempCString('piece');

  const result = module.invokeFunction(callbackPtr, tokenPtr, 5);
  bridge.unregisterCallback(callbackPtr);

  assert.equal(result, 0);
  assert.equal(seenToken, 'piece');
});

test('WasmBridge reads aggregate runtime observability from the module heap', () => {
  const bridge = new WasmBridge(new MockWasmBridgeModule());

  const metrics = bridge.readRuntimeObservability();

  assert.equal(metrics?.totalMs, 9);
  assert.equal(metrics?.outputTokenCount, 7);
  assert.equal(metrics?.firstSampledTokenId, 1234);
  assert.equal(metrics?.prefixCacheStoreCount, 12);
});

test('WasmBridge accepts an optional grammar argument for enqueuePrompt', () => {
  const module = new MockWasmBridgeModule();
  const bridge = new WasmBridge(module);
  const grammarSource = 'root ::= "yes" | "no"';
  // Small grammar under the size cap should be accepted and forwarded.
  const requestId = bridge.enqueuePrompt('ctx', 'hello', 16, 0, grammarSource);
  assert.ok(requestId > 0);
  assert.equal(module.lastEnqueue?.grammar, grammarSource);
});

test('WasmBridge forwards an empty string when no grammar is supplied', () => {
  const module = new MockWasmBridgeModule();
  const bridge = new WasmBridge(module);
  bridge.enqueuePrompt('ctx', 'hello', 16, 0);
  assert.equal(module.lastEnqueue?.grammar, '');
});

test('WasmBridge forwards grammar through enqueuePromptWithMedia', () => {
  const module = new MockWasmBridgeModule();
  module.mediaMarker = '<__media__>';
  const bridge = new WasmBridge(module);
  const grammarSource = 'root ::= "ok"';
  bridge.enqueuePromptWithMedia(
    'ctx',
    'look <__media__>',
    16,
    [new Uint8Array([1, 2])],
    0,
    grammarSource
  );
  assert.equal(module.lastMediaEnqueue?.grammar, grammarSource);
});

test('WasmBridge rejects grammar payloads above the size cap', () => {
  const bridge = new WasmBridge(new MockWasmBridgeModule());
  // Build a grammar string whose UTF-8 byte length exceeds the cap.
  const oversized = 'a'.repeat(MAX_GRAMMAR_BYTES + 1);
  assert.throws(
    () => bridge.enqueuePrompt('ctx', 'hello', 16, 0, oversized),
    /grammar exceeds maximum size/
  );
});

test('WasmBridge grammar cap applies to enqueuePromptWithMedia', () => {
  const module = new MockWasmBridgeModule();
  module.mediaMarker = '<__media__>';
  const bridge = new WasmBridge(module);
  const oversized = 'b'.repeat(MAX_GRAMMAR_BYTES + 1);
  assert.throws(
    () =>
      bridge.enqueuePromptWithMedia(
        'ctx',
        'look <__media__>',
        16,
        [new Uint8Array([1, 2])],
        0,
        oversized
      ),
    /grammar exceeds maximum size/
  );
});
