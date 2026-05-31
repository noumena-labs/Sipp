import test from 'node:test';
import assert from 'node:assert/strict';
import type { GenerateResponse, TokenBatch, TransportObservability } from '../engine/inference-types.js';
import { COMPLETED_REQUEST_STATUS_COMPLETED } from '../wasm/wasm-bridge.js';
import { RequestTracker } from './request-tracker.js';
import { QueuedRequestScheduler } from './scheduler.js';
import type { WasmBridge } from '../wasm/wasm-bridge.js';

function createTransportObservability(): TransportObservability {
  return {
    executionMode: 'main-thread',
    workerBacked: false,
    enabled: false,
    activeTokenTransport: 'none',
    activeTokenEmission: false,
  };
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
  const payload = new TextEncoder().encode('hi');
  const recordSize = 8 + payload.byteLength;

  heapU8[bufferAddr] = 1;
  heapU8[bufferAddr + 4] = payload.byteLength;
  heapU8.set(payload, bufferAddr + 8);
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
  assert.equal(batches[0].sequenceStart, 0);
  assert.equal(batches[0].text, 'hi');
  assert.equal(batches[0].frameCount, 1);
  assert.equal(batches[0].byteCount, 2);
  assert.deepEqual(batches[0].stats, {
    framesSent: 1,
    bytesSent: 2,
    batchesSent: 1,
  });
  assert.equal(tokenBatchSinkErrors.size, 0);
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
  const encoder = new TextEncoder();
  const bufferAddr = 64;
  const usedAddr = 4;
  let loopCount = 0;

  const writeTokenRecord = (text: string) => {
    const payload = encoder.encode(text);
    heapU8[bufferAddr] = 1;
    heapU8[bufferAddr + 1] = 0;
    heapU8[bufferAddr + 2] = 0;
    heapU8[bufferAddr + 3] = 0;
    heapU8[bufferAddr + 4] = payload.byteLength;
    heapU8[bufferAddr + 5] = payload.byteLength >>> 8;
    heapU8[bufferAddr + 6] = payload.byteLength >>> 16;
    heapU8[bufferAddr + 7] = payload.byteLength >>> 24;
    heapU8.set(payload, bufferAddr + 8);
    heap32[usedAddr / 4] = 8 + payload.byteLength;
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
