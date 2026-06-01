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
import type { SharedTokenRingDescriptor } from './shared-token-ring.js';

const TOKEN_RING_HEADER_INTS = 8;
const TOKEN_RING_HEADER_BYTES = TOKEN_RING_HEADER_INTS * 4;
const TOKEN_RING_WRITE_INDEX = 0;
const TOKEN_RING_CAPACITY = 2;
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

interface TestTokenRing {
  readonly descriptor: SharedTokenRingDescriptor;
  readonly header: Int32Array;
  readonly body: Uint8Array;
}

function createTokenRing(capacity: number, shared = false): TestTokenRing {
  const buffer = shared
    ? new SharedArrayBuffer(TOKEN_RING_HEADER_BYTES + capacity)
    : new ArrayBuffer(TOKEN_RING_HEADER_BYTES + capacity);
  const header = new Int32Array(buffer, 0, TOKEN_RING_HEADER_INTS);
  header[TOKEN_RING_CAPACITY] = capacity;
  return {
    descriptor: {
      buffer,
      headerOffset: 0,
      bodyOffset: TOKEN_RING_HEADER_BYTES,
      bodyCapacity: capacity,
    },
    header,
    body: new Uint8Array(buffer, TOKEN_RING_HEADER_BYTES, capacity),
  };
}

function writeU32(body: Uint8Array, offset: number, value: number): void {
  const index = offset % body.byteLength;
  body[index] = value & 0xff;
  body[(index + 1) % body.byteLength] = (value >>> 8) & 0xff;
  body[(index + 2) % body.byteLength] = (value >>> 16) & 0xff;
  body[(index + 3) % body.byteLength] = (value >>> 24) & 0xff;
}

function writeTokenBatchRecord(
  ring: TestTokenRing,
  requestId: number,
  sequenceStart: number,
  frameCount: number,
  text: string
): void {
  const payload = textEncoder.encode(text);
  const writeIndex = ring.descriptor.buffer instanceof SharedArrayBuffer
    ? Atomics.load(ring.header, TOKEN_RING_WRITE_INDEX)
    : ring.header[TOKEN_RING_WRITE_INDEX];
  const offset = writeIndex % ring.body.byteLength;
  writeU32(ring.body, offset, requestId);
  writeU32(ring.body, offset + 4, sequenceStart);
  writeU32(ring.body, offset + 8, frameCount);
  writeU32(ring.body, offset + 12, payload.byteLength);
  ring.body.set(payload, offset + TOKEN_BATCH_RECORD_HEADER_BYTES);
  const nextWriteIndex = writeIndex + TOKEN_BATCH_RECORD_HEADER_BYTES + payload.byteLength;
  if (ring.descriptor.buffer instanceof SharedArrayBuffer) {
    Atomics.store(ring.header, TOKEN_RING_WRITE_INDEX, nextWriteIndex);
    return;
  }
  ring.header[TOKEN_RING_WRITE_INDEX] = nextWriteIndex;
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

test('QueuedRequestScheduler batches same-turn admissions before the first native loop', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const maxCompletedResponses: number[] = [];
  const bridge = {
    async runInferenceLoop(
      _maxTicks: number,
      maxCompleted: number
    ) {
      maxCompletedResponses.push(maxCompleted);
      return {
        stepResult: 0,
        completedResponseCount: 2,
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
        outputText: `done-${requestId}`,
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

  const first = scheduler.track(1);
  const second = scheduler.track(2);
  await Promise.all([first.promise, second.promise]);

  assert.deepEqual(maxCompletedResponses, [2]);
});

test('QueuedRequestScheduler drains shared token ring to TokenBatch sinks', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const tokenBatchSinks = new Map<number, (batch: TokenBatch) => void>();
  const tokenBatchSinkErrors = new Map<number, unknown>();
  const batches: TokenBatch[] = [];
  const ring = createTokenRing(128, true);
  writeTokenBatchRecord(ring, 1, 7, 2, 'hi');

  const bridge = {
    getSharedTokenRingDescriptor() {
      return ring.descriptor;
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
  assert.equal(batches[0].stats.framesSent, 2);
  assert.equal(batches[0].stats.bytesSent, 2);
  assert.equal(batches[0].stats.batchesSent, 1);
  assert.ok(batches[0].stats.drainMs >= 0);
  assert.equal(batches[0].stats.drainCalls, 1);
  assert.equal(tokenBatchSinkErrors.size, 0);
  assert.equal(transport.tokenDrainCalls, undefined);
  assert.equal(transport.tokenDrainMs, undefined);
});

test('QueuedRequestScheduler keeps native token budget on the main thread while emitting tokens', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const tokenBatchSinks = new Map<number, (batch: TokenBatch) => void>();
  const tokenBatchSinkErrors = new Map<number, unknown>();
  const batches: TokenBatch[] = [];
  const loopTokenLimits: number[] = [];
  const ring = createTokenRing(128);
  let loopCount = 0;

  const writeTokenRecord = (text: string) => {
    writeTokenBatchRecord(ring, 1, 0, 1, text);
  };

  const bridge = {
    getSharedTokenRingDescriptor() {
      return ring.descriptor;
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

test('QueuedRequestScheduler drains shared token ring with bulk native loops', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability('worker');
  const tokenBatchSinks = new Map<number, (batch: TokenBatch) => void>();
  const tokenBatchSinkErrors = new Map<number, unknown>();
  const batches: TokenBatch[] = [];
  const maxDurationValues: Array<number | undefined> = [];
  const tokenLimits: number[] = [];
  const ring = createTokenRing(128);

  const bridge = {
    getSharedTokenRingDescriptor() {
      return ring.descriptor;
    },
    async runInferenceLoop(
      _maxTicks: number,
      _maxCompletedResponses: number,
      maxGeneratedTokens: number,
      options?: { maxDurationUs?: number }
    ) {
      tokenLimits.push(maxGeneratedTokens);
      maxDurationValues.push(options?.maxDurationUs);
      writeTokenBatchRecord(ring, 1, 0, 1, 'w');
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

  assert.deepEqual(maxDurationValues, [0]);
  assert.deepEqual(tokenLimits, [512]);
  assert.equal(batches.length, 1);
  assert.equal(batches[0].text, 'w');
});

test('QueuedRequestScheduler leaves worker loops unsliced without token emission', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability('worker');
  const maxDurationValues: Array<number | undefined> = [];
  const tokenLimits: number[] = [];
  const bridge = {
    async runInferenceLoop(
      _maxTicks: number,
      _maxCompletedResponses: number,
      maxGeneratedTokens: number,
      options?: { maxDurationUs?: number }
    ) {
      tokenLimits.push(maxGeneratedTokens);
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
  assert.deepEqual(tokenLimits, [512]);
});
