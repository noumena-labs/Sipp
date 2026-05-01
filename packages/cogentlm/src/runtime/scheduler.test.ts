import test from 'node:test';
import assert from 'node:assert/strict';
import type { GenerateResponse, TransportObservability } from '../types.js';
import {
  COMPLETED_REQUEST_STATUS_COMPLETED,
  REQUEST_STEP_RESULT_WAITING,
} from './main-thread-runtime-constants.js';
import { RequestTracker } from './request-tracker.js';
import { QueuedRequestScheduler } from './scheduler.js';
import type { WasmBridge } from '../wasm/wasm-bridge.js';

function createTransportObservability(): TransportObservability {
  return {
    executionMode: 'main-thread',
    workerBacked: false,
    enabled: false,
    bufferedTokenLimit: 0,
    flushIntervalMs: 0,
    flushCount: 0,
    coalescedTokenCount: 0,
    maxObservedBufferedTokenCount: 0,
    activeTokenTransport: 'none',
    runtimeEventDrainCount: 0,
    runtimeEventTokenCount: 0,
    runtimeEventTerminalCount: 0,
    runtimeEventTextBytes: 0,
  };
}

test('QueuedRequestScheduler settles completed requests even without terminal events', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const finalized: number[] = [];
  const bridge = {
    async runSchedulerProgress() {
      return {
        stepResult: REQUEST_STEP_RESULT_WAITING,
        completedResponseCount: 1,
      };
    },
    drainRuntimeEvents() {
      return {
        terminalRequestIds: [],
        tokenEvents: [],
        textBytes: 0,
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
    queuedPromptCallbacks: new Map(),
    queuedPromptTokenBuffers: new Map(),
    queuedPromptCallbackErrors: new Map(),
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

test('QueuedRequestScheduler ignores token events for requests without callbacks', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const transport = createTransportObservability();
  const queuedPromptCallbacks = new Map();
  const queuedPromptTokenBuffers = new Map();
  const finalized: number[] = [];
  const bridge = {
    async runSchedulerProgress() {
      return {
        stepResult: REQUEST_STEP_RESULT_WAITING,
        completedResponseCount: 1,
      };
    },
    drainRuntimeEvents() {
      return {
        terminalRequestIds: [],
        tokenEvents: [{ requestId: 1, token: 'ignored' }],
        textBytes: 'ignored'.length,
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
    queuedPromptCallbacks,
    queuedPromptTokenBuffers,
    queuedPromptCallbackErrors: new Map(),
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
  assert.equal(queuedPromptTokenBuffers.size, 0);
  assert.equal(transport.runtimeEventTokenCount, 1);
});
