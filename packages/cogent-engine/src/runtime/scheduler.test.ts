import assert from 'node:assert/strict';
import test from 'node:test';

import type { GenerateResponse } from '../types.js';
import { WasmBridge } from '../wasm/wasm-bridge.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
  REQUEST_STEP_RESULT_PROGRESSED,
} from './main-thread-runtime-constants.js';
import { RequestTracker } from './request-tracker.js';
import { QueuedRequestScheduler } from './scheduler.js';

class MockSchedulerBridge {
  public readonly burstCalls: Array<{
    maxTicks: number;
    maxCompletedResponses: number;
    maxEmittedTokens: number;
    maxDurationUs: number | null;
  }> = [];
  public pendingTokenEvents: Array<{ requestId: number; token: string }> = [];

  public async runSchedulerProgress(
    maxTicks: number,
    maxCompletedResponses: number,
    maxEmittedTokens: number,
    options: {
      maxDurationUs?: number;
    } = {}
  ): Promise<{ stepResult: number; completedResponseCount: number }> {
    this.burstCalls.push({
      maxTicks,
      maxCompletedResponses,
      maxEmittedTokens,
      maxDurationUs: options.maxDurationUs ?? null,
    });
    return {
      stepResult: REQUEST_STEP_RESULT_PROGRESSED,
      completedResponseCount: 0,
    };
  }

  public drainRuntimeEvents(maxEventCount: number): {
    terminalRequestIds: number[];
    tokenEvents: Array<{ requestId: number; token: string; textLength: number }>;
    textBytes: number;
  } {
    const drained = this.pendingTokenEvents.splice(0, maxEventCount);
    return {
      terminalRequestIds: [],
      tokenEvents: drained.map((event) => ({
        requestId: event.requestId,
        token: event.token,
        textLength: event.token.length,
      })),
      textBytes: drained.reduce((total, event) => total + event.token.length, 0),
    };
  }

  public drainCompletedRequestIds(): number[] {
    return [];
  }

  public getCompletedRequestStatus(): number {
    return COMPLETED_REQUEST_STATUS_PENDING;
  }

  public takeCompletedResponse(): GenerateResponse {
    throw new Error('takeCompletedResponse() should not be called in this scheduler test.');
  }
}

test('QueuedRequestScheduler uses smaller interactive bursts for callback-driven streaming requests', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const queuedPromptCallbacks = new Map<number, ((token: string) => void) | undefined>();
  const queuedPromptTokenBuffers = new Map<number, string[]>();
  const queuedPromptCallbackErrors = new Map<number, unknown>();
  const deliveredTokens: string[] = [];
  const bridge = new MockSchedulerBridge();

  queuedPromptCallbacks.set(101, (token) => {
    deliveredTokens.push(token);
  });

  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptCallbacks,
    queuedPromptTokenBuffers,
    queuedPromptCallbackErrors,
    getTransportObservability: () => ({
      ...DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
    }),
    getBridge: () => bridge as unknown as WasmBridge,
    finalizeRequest: () => {},
    cancelQueuedRequest: async () => true,
  });
  scheduler.setPumpMode('external');

  scheduler.track(101);

  await scheduler.pumpOnce();
  assert.deepEqual(bridge.burstCalls[0], {
    maxTicks: 8,
    maxCompletedResponses: 1,
    maxEmittedTokens: 1,
    maxDurationUs: null,
  });

  bridge.pendingTokenEvents.push({ requestId: 101, token: 'tok1' });
  await scheduler.pumpOnce();
  assert.deepEqual(deliveredTokens, ['tok1']);
  assert.deepEqual(bridge.burstCalls[1], {
    maxTicks: 8,
    maxCompletedResponses: 1,
    maxEmittedTokens: 1,
    maxDurationUs: null,
  });

  bridge.pendingTokenEvents.push({ requestId: 101, token: 'tok2' });
  await scheduler.pumpOnce();
  assert.deepEqual(deliveredTokens, ['tok1', 'tok2']);
  assert.deepEqual(bridge.burstCalls[2], {
    maxTicks: 16,
    maxCompletedResponses: 1,
    maxEmittedTokens: 8,
    maxDurationUs: 60_000,
  });
});

test('QueuedRequestScheduler keeps the larger throughput burst for requests without token callbacks', async () => {
  const tracker = new RequestTracker<GenerateResponse>();
  const bridge = new MockSchedulerBridge();
  const scheduler = new QueuedRequestScheduler({
    tracker,
    queuedPromptCallbacks: new Map(),
    queuedPromptTokenBuffers: new Map(),
    queuedPromptCallbackErrors: new Map(),
    getTransportObservability: () => ({
      ...DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
    }),
    getBridge: () => bridge as unknown as WasmBridge,
    finalizeRequest: () => {},
    cancelQueuedRequest: async () => true,
  });
  scheduler.setPumpMode('external');

  scheduler.track(202);
  await scheduler.pumpOnce();

  assert.deepEqual(bridge.burstCalls[0], {
    maxTicks: 64,
    maxCompletedResponses: 1,
    maxEmittedTokens: 32,
    maxDurationUs: null,
  });
});
