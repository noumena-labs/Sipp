import test from 'node:test';
import assert from 'node:assert/strict';
import type {
  GenerateRequestHandle,
  GenerateResponse,
  TokenBatch,
  TransportObservability,
} from '../../src/engine/inference-types.js';
import {
  COMPLETED_REQUEST_STATUS_COMPLETED,
  COMPLETED_REQUEST_STATUS_PENDING,
} from '../../src/wasm/wasm-bridge.js';
import { RequestTracker } from '../../src/runtime/request-tracker.js';
import { QueuedRequestScheduler } from '../../src/runtime/scheduler.js';
import type { WasmBridge } from '../../src/wasm/wasm-bridge.js';
import type { SharedTokenRingDescriptor } from '../../src/runtime/shared-token-ring.js';

const TOKEN_RING_HEADER_INTS = 8;
const TOKEN_RING_HEADER_BYTES = TOKEN_RING_HEADER_INTS * 4;
const TOKEN_RING_WRITE_INDEX = 0;
const TOKEN_RING_CAPACITY = 2;
const TOKEN_BATCH_RECORD_HEADER_BYTES = 16;
const textEncoder = new TextEncoder();

function requestHandle(requestId: number, generation = 1): GenerateRequestHandle {
  return { generation, requestId };
}

function createDeferred<T>(): {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (error: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

async function waitForEventLoopTurn(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function createTransportObservability(): TransportObservability {
  return {
    executionMode: 'worker',
    workerBacked: true,
    enabled: false,
    wasmRunLoopCalls: 0,
    wasmRunLoopMs: 0,
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
    takeCompletedResponse(request: GenerateRequestHandle): GenerateResponse {
      return {
        requestId: request.requestId,
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
    getTransportObservability: () => transport,
    getRuntimeGeneration: () => 1,
    withWasmBridge: (operation) => Promise.resolve(operation(bridge)),
    finalizeRequest: (_bridge, requestId, options) => {
      finalized.push(requestId.requestId);
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  const tracked = scheduler.track(requestHandle(1));
  const response = await Promise.race([
    tracked.promise,
    new Promise<GenerateResponse>((_, reject) => {
      setTimeout(() => reject(new Error('scheduler did not settle request')), 100);
    }),
  ]);

  assert.equal(response.outputText, 'done');
  assert.deepEqual(finalized, [1]);
});

test('QueuedRequestScheduler measures browser-to-WASM inference loop calls when observability is enabled', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  transport.enabled = true;
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
    takeCompletedResponse(request: GenerateRequestHandle): GenerateResponse {
      return {
        requestId: request.requestId,
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
    getTransportObservability: () => transport,
    getRuntimeGeneration: () => 1,
    withWasmBridge: (operation) => Promise.resolve(operation(bridge)),
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  await scheduler.track(requestHandle(1)).promise;

  assert.equal(transport.wasmRunLoopCalls, 1);
  assert.ok(transport.wasmRunLoopMs >= 0);
});

test('QueuedRequestScheduler settles direct completions outside the inference loop', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const bridge = {
    async runInferenceLoop() {
      return {
        stepResult: 0,
        completedResponseCount: 0,
      };
    },
    getCompletedRequestStatus() {
      return COMPLETED_REQUEST_STATUS_COMPLETED;
    },
    takeCompletedResponse(request: GenerateRequestHandle): GenerateResponse {
      return {
        requestId: request.requestId,
        completed: true,
        cancelled: false,
        failed: false,
        audio: {
          data: new Uint8Array([1]),
          sampleRateHz: 24_000,
          channels: 1,
          durationMs: 1,
        },
      };
    },
  } as unknown as WasmBridge;
  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptTokenBatchSinks: new Map(),
    getTransportObservability: () => transport,
    getRuntimeGeneration: () => 1,
    withWasmBridge: (operation) => Promise.resolve(operation(bridge)),
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  const response = await scheduler.track(requestHandle(7)).promise;

  assert.equal(response.audio?.sampleRateHz, 24_000);
});

test('QueuedRequestScheduler batches same-turn admissions before the first native loop', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const maxCompletedResponses: number[] = [];
  const bridge = {
    async runInferenceLoop(
      _generation: number,
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
    takeCompletedResponse(request: GenerateRequestHandle): GenerateResponse {
      return {
        requestId: request.requestId,
        completed: true,
        cancelled: false,
        failed: false,
        outputText: `done-${request.requestId}`,
      };
    },
  } as unknown as WasmBridge;

  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptTokenBatchSinks: new Map(),
    getTransportObservability: () => transport,
    getRuntimeGeneration: () => 1,
    withWasmBridge: (operation) => Promise.resolve(operation(bridge)),
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  const first = scheduler.track(requestHandle(1));
  const second = scheduler.track(requestHandle(2));
  await Promise.all([first.promise, second.promise]);

  assert.deepEqual(maxCompletedResponses, [2]);
});

test('QueuedRequestScheduler drains shared token ring to TokenBatch sinks', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const tokenBatchSinks = new Map<number, (batch: TokenBatch) => void>();
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
    takeCompletedResponse(request: GenerateRequestHandle): GenerateResponse {
      return {
        requestId: request.requestId,
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
    getTransportObservability: () => transport,
    getRuntimeGeneration: () => 1,
    withWasmBridge: (operation) => Promise.resolve(operation(bridge)),
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  tokenBatchSinks.set(1, (batch) => batches.push(batch));
  const tracked = scheduler.track(requestHandle(1));
  await tracked.promise;

  assert.equal(batches.length, 1);
  assert.equal(batches[0].requestId, '1:1');
  assert.equal(batches[0].streamId, 1);
  assert.equal(batches[0].sequenceStart, 7);
  assert.equal(batches[0].text, 'hi');
  assert.equal(batches[0].frameCount, 2);
  assert.equal(batches[0].byteCount, 2);
  assert.equal(batches[0].stats.framesSent, 2);
  assert.equal(batches[0].stats.bytesSent, 2);
  assert.equal(batches[0].stats.batchesSent, 1);
  assert.equal(transport.tokenDrainCalls, undefined);
  assert.equal(transport.tokenDrainMs, undefined);
});

test('QueuedRequestScheduler limits native token budget while emitting tokens', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const tokenBatchSinks = new Map<number, (batch: TokenBatch) => void>();
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
      _generation: number,
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
      return loopCount === 1
        ? COMPLETED_REQUEST_STATUS_PENDING
        : COMPLETED_REQUEST_STATUS_COMPLETED;
    },
    takeCompletedResponse(request: GenerateRequestHandle): GenerateResponse {
      return {
        requestId: request.requestId,
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
    getTransportObservability: () => transport,
    getRuntimeGeneration: () => 1,
    withWasmBridge: (operation) => Promise.resolve(operation(bridge)),
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  tokenBatchSinks.set(1, (batch) => batches.push(batch));
  const tracked = scheduler.track(requestHandle(1));
  await tracked.promise;

  assert.deepEqual(loopTokenLimits, [1, 1]);
  assert.equal(batches.length, 1);
  assert.equal(batches[0].text, 'a');
  assert.equal(batches[0].frameCount, 1);
});

test('QueuedRequestScheduler drains shared token ring with streaming native loops', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const tokenBatchSinks = new Map<number, (batch: TokenBatch) => void>();
  const batches: TokenBatch[] = [];
  const tokenLimits: number[] = [];
  const ring = createTokenRing(128);

  const bridge = {
    getSharedTokenRingDescriptor() {
      return ring.descriptor;
    },
    async runInferenceLoop(
      _generation: number,
      _maxTicks: number,
      _maxCompletedResponses: number,
      maxGeneratedTokens: number
    ) {
      tokenLimits.push(maxGeneratedTokens);
      writeTokenBatchRecord(ring, 1, 0, 1, 'w');
      return {
        stepResult: 0,
        completedResponseCount: 1,
      };
    },
    getCompletedRequestStatus() {
      return COMPLETED_REQUEST_STATUS_COMPLETED;
    },
    takeCompletedResponse(request: GenerateRequestHandle): GenerateResponse {
      return {
        requestId: request.requestId,
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
    getTransportObservability: () => transport,
    getRuntimeGeneration: () => 1,
    withWasmBridge: (operation) => Promise.resolve(operation(bridge)),
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  tokenBatchSinks.set(1, (batch) => batches.push(batch));
  const tracked = scheduler.track(requestHandle(1));
  await tracked.promise;

  assert.deepEqual(tokenLimits, [1]);
  assert.equal(batches.length, 1);
  assert.equal(batches[0].text, 'w');
});

test('QueuedRequestScheduler preserves the first token-sink failure and stops delivery', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const tokenBatchSinks = new Map<number, (batch: TokenBatch) => void>();
  const ring = createTokenRing(128);
  const request = requestHandle(1);
  let sinkCalls = 0;
  let cancellationCalls = 0;
  writeTokenBatchRecord(ring, request.requestId, 0, 1, 'a');
  writeTokenBatchRecord(ring, request.requestId, 1, 1, 'b');

  const bridge = {
    getSharedTokenRingDescriptor() {
      return ring.descriptor;
    },
    async runInferenceLoop() {
      return { stepResult: 0, completedResponseCount: 1 };
    },
    getCompletedRequestStatus() {
      return COMPLETED_REQUEST_STATUS_COMPLETED;
    },
    takeCompletedResponse(): GenerateResponse {
      return {
        requestId: request.requestId,
        completed: true,
        cancelled: false,
        failed: false,
        outputText: 'ab',
      };
    },
  } as unknown as WasmBridge;

  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptTokenBatchSinks: tokenBatchSinks,
    getTransportObservability: () => transport,
    getRuntimeGeneration: () => request.generation,
    withWasmBridge: (operation) => Promise.resolve(operation(bridge)),
    finalizeRequest: (_bridge, handle, options) => {
      tracker.finalize(handle, options);
    },
    cancelQuery: async () => {
      cancellationCalls += 1;
      return true;
    },
  });
  tokenBatchSinks.set(request.requestId, () => {
    sinkCalls += 1;
    throw undefined;
  });

  scheduler.track(request);
  const response = tracker.beginWait(request);
  await response;

  assert.equal(sinkCalls, 1);
  assert.equal(cancellationCalls, 1);
  assert.equal(tokenBatchSinks.has(request.requestId), false);
  assert.equal(tracker.get(request)?.tokenBatchSinkFailed, true);
  assert.equal(tracker.get(request)?.tokenBatchSinkError, undefined);
  tracker.endWait(request);
});

test('QueuedRequestScheduler uses the continuous token budget without token emission', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const tokenLimits: number[] = [];
  const bridge = {
    async runInferenceLoop(
      _generation: number,
      _maxTicks: number,
      _maxCompletedResponses: number,
      maxGeneratedTokens: number
    ) {
      tokenLimits.push(maxGeneratedTokens);
      return {
        stepResult: 0,
        completedResponseCount: 1,
      };
    },
    getCompletedRequestStatus() {
      return COMPLETED_REQUEST_STATUS_COMPLETED;
    },
    takeCompletedResponse(request: GenerateRequestHandle): GenerateResponse {
      return {
        requestId: request.requestId,
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
    getTransportObservability: () => transport,
    getRuntimeGeneration: () => 1,
    withWasmBridge: (operation) => Promise.resolve(operation(bridge)),
    finalizeRequest: (_bridge, requestId, options) => {
      tracker.finalize(requestId, options);
    },
    cancelQuery: async () => true,
  });

  const tracked = scheduler.track(requestHandle(1));
  await tracked.promise;

  assert.deepEqual(tokenLimits, [512]);
});

test('QueuedRequestScheduler rejects current requests when bridge acquisition fails', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const failure = new Error('bridge acquisition failed');
  const request = requestHandle(1);
  const finalized: GenerateRequestHandle[] = [];
  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptTokenBatchSinks: new Map(),
    getTransportObservability: createTransportObservability,
    getRuntimeGeneration: () => request.generation,
    withWasmBridge: async () => {
      throw failure;
    },
    finalizeRequest: (bridge, handle, options) => {
      assert.equal(bridge, null);
      finalized.push(handle);
      tracker.finalize(handle, options);
    },
    cancelQuery: async () => true,
  });

  const tracked = scheduler.track(request);
  await assert.rejects(
    Promise.race([
      tracked.promise,
      new Promise<GenerateResponse>((_, reject) => {
        setTimeout(() => reject(new Error('scheduler did not reject request')), 100);
      }),
    ]),
    (error) => error === failure
  );

  assert.deepEqual(finalized, [request]);
  assert.equal(tracker.activeCount, 0);
});

test('QueuedRequestScheduler ignores stale bridge-acquisition failures', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const bridgeStarted = createDeferred<void>();
  const bridgeResult = createDeferred<never>();
  const request = requestHandle(1);
  const finalized: GenerateRequestHandle[] = [];
  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptTokenBatchSinks: new Map(),
    getTransportObservability: createTransportObservability,
    getRuntimeGeneration: () => request.generation,
    withWasmBridge: async () => {
      bridgeStarted.resolve();
      return await bridgeResult.promise;
    },
    finalizeRequest: (_bridge, handle, options) => {
      finalized.push(handle);
      tracker.finalize(handle, options);
    },
    cancelQuery: async () => true,
  });

  scheduler.track(request);
  await bridgeStarted.promise;
  scheduler.reset();
  bridgeResult.reject(new Error('stale bridge acquisition failed'));
  await waitForEventLoopTurn();
  await waitForEventLoopTurn();

  assert.deepEqual(finalized, []);
  assert.equal(tracker.get(request)?.settled, false);
  assert.equal(tracker.get(request)?.active, true);
  tracker.clear();
});
