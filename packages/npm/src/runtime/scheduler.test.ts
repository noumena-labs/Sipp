import test from 'node:test';
import assert from 'node:assert/strict';
import type {
  EngineExecutionMode,
  GenerateResponse,
  TokenBatch,
  TransportObservability,
} from '../engine/inference-types.js';
import { COMPLETED_REQUEST_STATUS_COMPLETED } from '../wasm/wasm-bridge.js';
import { RequestTracker } from './request-tracker.js';
import { QueuedRequestScheduler } from './scheduler.js';
import type { WasmBridge } from '../wasm/wasm-bridge.js';

const TOKEN_BATCH_RECORD_HEADER_BYTES = 16;
const textEncoder = new TextEncoder();

function createTransportObservability(
  executionMode: EngineExecutionMode = 'main-thread'
): TransportObservability {
  return {
    executionMode,
    workerBacked: executionMode === 'worker',
    enabled: false,
    activeTokenTransport: 'none',
    activeTokenEmission: false,
  };
}

function writeU32(heapU8: Uint8Array, offset: number, value: number): void {
  heapU8[offset] = value & 0xff;
  heapU8[offset + 1] = (value >>> 8) & 0xff;
  heapU8[offset + 2] = (value >>> 16) & 0xff;
  heapU8[offset + 3] = (value >>> 24) & 0xff;
}

function writeTokenBatchRecord(
  heapU8: Uint8Array,
  offset: number,
  requestId: number,
  sequenceStart: number,
  frameCount: number,
  text: string
): number {
  const payload = textEncoder.encode(text);
  writeU32(heapU8, offset, requestId);
  writeU32(heapU8, offset + 4, sequenceStart);
  writeU32(heapU8, offset + 8, frameCount);
  writeU32(heapU8, offset + 12, payload.byteLength);
  heapU8.set(payload, offset + TOKEN_BATCH_RECORD_HEADER_BYTES);
  return TOKEN_BATCH_RECORD_HEADER_BYTES + payload.byteLength;
}

test('QueuedRequestScheduler settles completed requests reported by the inference loop', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const finalized: number[] = [];
  const bridge = {
    async runInferenceLoop() {
      return {
        stepResult: 0,
        completedResponseCount: 1,
      };
    },
    getCompletedRequestStatus() {
      return COMPLETED_REQUEST_STATUS_COMPLETED;
    },
    takeCompletedResponse(requestId: number): GenerateResponse {
      return {
        requestId,
        completed: true,
        cancelled: false,
        failed: false,
        outputText: 'done',
      };
    },
  } as unknown as WasmBridge;

  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptTokenBatchSinks: new Map(),
    queuedPromptTokenBatchSinkErrors: new Map(),
    getTransportObservability: () => transport,
    getBridge: () => bridge,
    finalizeRequest: (_bridge, requestId, options) => {
      finalized.push(requestId);
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  const tracked = scheduler.track(1);
  const response = await Promise.race([
    tracked.promise,
    new Promise<GenerateResponse>((_, reject) => {
      setTimeout(() => reject(new Error('scheduler did not settle request')), 100);
    }),
  ]);

  assert.equal(response.outputText, 'done');
  assert.deepEqual(finalized, [1]);
});

test('QueuedRequestScheduler drains token buffer to TokenBatch sinks', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const tokenBatchSinks = new Map<number, (batch: TokenBatch) => void>();
  const tokenBatchSinkErrors = new Map<number, unknown>();
  const batches: TokenBatch[] = [];
  const memory = new ArrayBuffer(256);
  const heapU8 = new Uint8Array(memory);
  const heap32 = new Int32Array(memory);
  const bufferAddr = 64;
  const usedAddr = 4;
  const recordSize = writeTokenBatchRecord(heapU8, bufferAddr, 1, 7, 2, 'hi');
  heap32[usedAddr / 4] = recordSize;

  const bridge = {
    module: { HEAPU8: heapU8, HEAP32: heap32 },
    getTokenBufferPointer() {
      return bufferAddr;
    },
    getTokenBufferUsedAddress() {
      return usedAddr;
    },
    async runInferenceLoop() {
      return {
        stepResult: 0,
        completedResponseCount: 1,
      };
    },
    getCompletedRequestStatus() {
      return COMPLETED_REQUEST_STATUS_COMPLETED;
    },
    takeCompletedResponse(requestId: number): GenerateResponse {
      return {
        requestId,
        completed: true,
        cancelled: false,
        failed: false,
        outputText: 'hi',
      };
    },
  } as unknown as WasmBridge;

  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptTokenBatchSinks: tokenBatchSinks,
    queuedPromptTokenBatchSinkErrors: tokenBatchSinkErrors,
    getTransportObservability: () => transport,
    getBridge: () => bridge,
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  tokenBatchSinks.set(1, (batch) => batches.push(batch));
  const tracked = scheduler.track(1);
  await tracked.promise;

  assert.equal(batches.length, 1);
  assert.equal(batches[0].requestId, '1');
  assert.equal(batches[0].streamId, 1);
  assert.equal(batches[0].sequenceStart, 7);
  assert.equal(batches[0].text, 'hi');
  assert.equal(batches[0].frameCount, 2);
  assert.equal(batches[0].byteCount, 2);
  assert.deepEqual(batches[0].stats, {
    framesSent: 2,
    bytesSent: 2,
    batchesSent: 1,
  });
  assert.equal(tokenBatchSinkErrors.size, 0);
  assert.equal(transport.tokenDrainCount, undefined);
  assert.equal(transport.tokenDrainMs, undefined);
});

test('QueuedRequestScheduler keeps native token budget while emitting tokens', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const tokenBatchSinks = new Map<number, (batch: TokenBatch) => void>();
  const tokenBatchSinkErrors = new Map<number, unknown>();
  const batches: TokenBatch[] = [];
  const loopTokenLimits: number[] = [];
  const memory = new ArrayBuffer(256);
  const heapU8 = new Uint8Array(memory);
  const heap32 = new Int32Array(memory);
  const bufferAddr = 64;
  const usedAddr = 4;
  let loopCount = 0;

  const writeTokenRecord = (text: string) => {
    heap32[usedAddr / 4] = writeTokenBatchRecord(heapU8, bufferAddr, 1, 0, 1, text);
  };

  const bridge = {
    module: { HEAPU8: heapU8, HEAP32: heap32 },
    getTokenBufferPointer() {
      return bufferAddr;
    },
    getTokenBufferUsedAddress() {
      return usedAddr;
    },
    async runInferenceLoop(
      _maxTicks: number,
      _maxCompletedResponses: number,
      maxGeneratedTokens: number
    ) {
      loopTokenLimits.push(maxGeneratedTokens);
      loopCount += 1;
      if (loopCount === 1) {
        writeTokenRecord('a');
        return {
          stepResult: 0,
          completedResponseCount: 0,
        };
      }
      return {
        stepResult: 0,
        completedResponseCount: 1,
      };
    },
    getCompletedRequestStatus() {
      return COMPLETED_REQUEST_STATUS_COMPLETED;
    },
    takeCompletedResponse(requestId: number): GenerateResponse {
      return {
        requestId,
        completed: true,
        cancelled: false,
        failed: false,
        outputText: 'a',
      };
    },
  } as unknown as WasmBridge;

  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptTokenBatchSinks: tokenBatchSinks,
    queuedPromptTokenBatchSinkErrors: tokenBatchSinkErrors,
    getTransportObservability: () => transport,
    getBridge: () => bridge,
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  tokenBatchSinks.set(1, (batch) => batches.push(batch));
  const tracked = scheduler.track(1);
  await tracked.promise;

  assert.deepEqual(loopTokenLimits, [512, 512]);
  assert.equal(batches.length, 1);
  assert.equal(batches[0].text, 'a');
  assert.equal(batches[0].frameCount, 1);
  assert.equal(tokenBatchSinkErrors.size, 0);
});

test('QueuedRequestScheduler time-slices worker loops while emitting tokens', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability('worker');
  const tokenBatchSinks = new Map<number, (batch: TokenBatch) => void>();
  const tokenBatchSinkErrors = new Map<number, unknown>();
  const maxDurationValues: Array<number | undefined> = [];
  const bridge = {
    async runInferenceLoop(
      _maxTicks: number,
      _maxCompletedResponses: number,
      _maxGeneratedTokens: number,
      options?: { maxDurationUs?: number }
    ) {
      maxDurationValues.push(options?.maxDurationUs);
      return {
        stepResult: 0,
        completedResponseCount: 1,
      };
    },
    getTokenBufferUsedAddress() {
      return 0;
    },
    getCompletedRequestStatus() {
      return COMPLETED_REQUEST_STATUS_COMPLETED;
    },
    takeCompletedResponse(requestId: number): GenerateResponse {
      return {
        requestId,
        completed: true,
        cancelled: false,
        failed: false,
        outputText: 'done',
      };
    },
  } as unknown as WasmBridge;

  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptTokenBatchSinks: tokenBatchSinks,
    queuedPromptTokenBatchSinkErrors: tokenBatchSinkErrors,
    getTransportObservability: () => transport,
    getBridge: () => bridge,
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  tokenBatchSinks.set(1, () => {});
  const tracked = scheduler.track(1);
  await tracked.promise;

  assert.deepEqual(maxDurationValues, [16000]);
});

test('QueuedRequestScheduler leaves worker loops unsliced without token emission', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability('worker');
  const maxDurationValues: Array<number | undefined> = [];
  const bridge = {
    async runInferenceLoop(
      _maxTicks: number,
      _maxCompletedResponses: number,
      _maxGeneratedTokens: number,
      options?: { maxDurationUs?: number }
    ) {
      maxDurationValues.push(options?.maxDurationUs);
      return {
        stepResult: 0,
        completedResponseCount: 1,
      };
    },
    getCompletedRequestStatus() {
      return COMPLETED_REQUEST_STATUS_COMPLETED;
    },
    takeCompletedResponse(requestId: number): GenerateResponse {
      return {
        requestId,
        completed: true,
        cancelled: false,
        failed: false,
        outputText: 'done',
      };
    },
  } as unknown as WasmBridge;

  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptTokenBatchSinks: new Map(),
    queuedPromptTokenBatchSinkErrors: new Map(),
    getTransportObservability: () => transport,
    getBridge: () => bridge,
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  const tracked = scheduler.track(1);
  await tracked.promise;

  assert.deepEqual(maxDurationValues, [0]);
});
