import assert from 'node:assert/strict';
import test from 'node:test';

import { WorkerEngineRuntime } from './engine-runtime-worker.js';
import type {
  WorkerLoadModelResult,
  WorkerRequestMessage,
  WorkerResponseMessage,
} from './engine-runtime-worker-protocol.js';

type WorkerMessageHandler = (
  worker: MockWorker,
  message: WorkerRequestMessage
) => void;

class MockWorker {
  public static instances: MockWorker[] = [];
  public static handlerFactory: (() => WorkerMessageHandler) | null = null;

  public onmessage: ((event: MessageEvent<WorkerResponseMessage>) => void) | null = null;
  public onerror: ((event: ErrorEvent) => void) | null = null;
  public onmessageerror: ((event: MessageEvent<unknown>) => void) | null = null;
  public terminated = false;
  public readonly messages: WorkerRequestMessage[] = [];

  private readonly handler: WorkerMessageHandler;

  constructor(_url: string, _options?: WorkerOptions) {
    this.handler = MockWorker.handlerFactory?.() ?? (() => {});
    MockWorker.instances.push(this);
  }

  public postMessage(message: WorkerRequestMessage): void {
    this.messages.push(message);
    queueMicrotask(() => {
      this.handler(this, message);
    });
  }

  public terminate(): void {
    this.terminated = true;
  }

  public emit(message: WorkerResponseMessage): void {
    this.onmessage?.({ data: message } as MessageEvent<WorkerResponseMessage>);
  }

  public triggerError(message: string): void {
    this.onerror?.({ message } as ErrorEvent);
  }
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

function installMockWorker(): () => void {
  const originalWorker = globalThis.Worker;
  MockWorker.instances = [];
  globalThis.Worker = MockWorker as unknown as typeof Worker;
  return () => {
    globalThis.Worker = originalWorker;
    MockWorker.instances = [];
    MockWorker.handlerFactory = null;
  };
}

function getWorkerQueuedState(
  runtime: WorkerEngineRuntime
): {
  queuedCallbacks: number;
  pendingCallbacks: number;
  queuedErrors: number;
  queuedSignals: number;
} {
  const runtimeState = runtime as unknown as {
    queuedTokenCallbacks: Map<number, unknown>;
    pendingQueuedTokenCallbacks: Map<number, unknown>;
    queuedTokenErrors: Map<number, unknown>;
    queuedSignals: Map<number, unknown>;
  };

  return {
    queuedCallbacks: runtimeState.queuedTokenCallbacks.size,
    pendingCallbacks: runtimeState.pendingQueuedTokenCallbacks.size,
    queuedErrors: runtimeState.queuedTokenErrors.size,
    queuedSignals: runtimeState.queuedSignals.size,
  };
}

test('WorkerEngineRuntime rejects pending calls on worker crash and recreates the worker', async () => {
  const restoreWorker = installMockWorker();
  try {
    MockWorker.handlerFactory = () => (worker, message) => {
      if (message.kind === 'init-module') {
        worker.emit({
          kind: 'resolve',
          callId: message.callId,
          value: undefined,
        });
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const firstWorker = MockWorker.instances[0];
    const pendingObservability = runtime.getBackendObservability();
    await waitForCondition(() =>
      firstWorker.messages.some(
        (message) => message.kind === 'get-backend-observability'
      )
    );
    firstWorker.triggerError('worker exploded');
    await assert.rejects(pendingObservability, /worker exploded/);
    assert.equal(firstWorker.terminated, true);

    MockWorker.handlerFactory = () => (worker, message) => {
      if (message.kind === 'init-module') {
        worker.emit({
          kind: 'resolve',
          callId: message.callId,
          value: undefined,
        });
      }
    };

    await runtime.initModule();
    assert.equal(MockWorker.instances.length, 2);
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime streams model chunks through the worker protocol without buffering a whole model', async () => {
  const restoreWorker = installMockWorker();
  try {
    MockWorker.handlerFactory = () => {
      let chunkCount = 0;
      return (worker, message) => {
        switch (message.kind) {
          case 'init-module':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: undefined,
            });
            break;
          case 'load-model-stream-start':
            break;
          case 'load-model-stream-chunk':
            chunkCount += 1;
            worker.emit({
              kind: 'load-progress',
              callId: message.callId,
              progressPct: chunkCount === 1 ? 50 : 100,
            });
            worker.emit({
              kind: 'load-stream-ack',
              callId: message.callId,
            });
            break;
          case 'load-model-stream-end': {
            const result: WorkerLoadModelResult = {
              modelPath: '/models/model.gguf',
              modelLoadInfo: {
                sourceKind: 'buffer',
                reuseMode: 'buffer',
                modelPath: '/models/model.gguf',
                fileName: 'model.gguf',
                byteLength: 3,
                persistentCacheEnabled: false,
                persistentCacheKey: null,
                persistentCacheHit: false,
                persistentCacheStored: false,
              },
              transportObservability: {
                executionMode: 'worker',
                workerBacked: true,
                enabled: false,
                bufferedTokenLimit: 0,
                flushIntervalMs: 0,
                flushCount: 0,
                coalescedTokenCount: 0,
                maxObservedBufferedTokenCount: 0,
              },
            };
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: result,
            });
            break;
          }
          default:
            throw new Error(`Unexpected worker message: ${message.kind}`);
        }
      };
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const progressUpdates: number[] = [];
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(Uint8Array.from([1, 2]));
        controller.enqueue(Uint8Array.from([3]));
        controller.close();
      },
    });

    const modelPath = await runtime.loadModelFromReadableStream(stream, 'model.gguf', {
      expectedBytes: 3,
      onProgress: (pct) => {
        progressUpdates.push(pct);
      },
    });

    assert.equal(modelPath, '/models/model.gguf');
    assert.deepEqual(progressUpdates, [50, 100]);

    const workerMessages = MockWorker.instances[0].messages
      .map((message) => message.kind)
      .filter((kind) => kind !== 'init-module');
    assert.deepEqual(workerMessages, [
      'load-model-stream-start',
      'load-model-stream-chunk',
      'load-model-stream-chunk',
      'load-model-stream-end',
    ]);
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime sends cancel-model-load for aborted streamed loads and rejects with AbortError', async () => {
  const restoreWorker = installMockWorker();
  try {
    MockWorker.handlerFactory = () => (worker, message) => {
      switch (message.kind) {
        case 'init-module':
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: undefined,
          });
          break;
        case 'load-model-stream-start':
          break;
        case 'load-model-stream-chunk':
          worker.emit({
            kind: 'load-stream-ack',
            callId: message.callId,
          });
          break;
        case 'cancel-model-load':
          worker.emit({
            kind: 'reject',
            callId: message.callId,
            message: 'Model load aborted.',
            errorName: 'AbortError',
          });
          break;
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

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

    await waitForCondition(() =>
      MockWorker.instances[0].messages.some(
        (message) => message.kind === 'load-model-stream-chunk'
      )
    );
    abortController.abort();

    await assert.rejects(
      loadPromise,
      (error: unknown) => error instanceof Error && error.name === 'AbortError'
    );
    assert.ok(
      MockWorker.instances[0].messages.some(
        (message) => message.kind === 'cancel-model-load'
      )
    );

    releaseSecondRead();
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime releases queued callback state when cancelling before execution', async () => {
  const restoreWorker = installMockWorker();
  try {
    MockWorker.handlerFactory = () => {
      let nextRequestId = 11;
      return (worker, message) => {
        switch (message.kind) {
          case 'init-module':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: undefined,
            });
            break;
          case 'queue-prompt':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: nextRequestId++,
            });
            break;
          case 'cancel-request':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: true,
            });
            break;
          default:
            throw new Error(`Unexpected worker message: ${message.kind}`);
        }
      };
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const abortController = new AbortController();
    const requestId = await runtime.queuePrompt('ctx', 'prompt', {
      nTokens: 16,
      signal: abortController.signal,
      onToken: () => {},
    });

    const cancelled = await runtime.cancelQueuedRequest(requestId);

    assert.equal(cancelled, true);
    assert.deepEqual(getWorkerQueuedState(runtime), {
      queuedCallbacks: 0,
      pendingCallbacks: 0,
      queuedErrors: 0,
      queuedSignals: 0,
    });
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime queue/cancel churn leaves no queued state residue and still supports a smoke prompt', async () => {
  const restoreWorker = installMockWorker();
  try {
    MockWorker.handlerFactory = () => {
      let nextRequestId = 21;
      return (worker, message) => {
        switch (message.kind) {
          case 'init-module':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: undefined,
            });
            break;
          case 'queue-prompt':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: nextRequestId++,
            });
            break;
          case 'cancel-request':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: true,
            });
            break;
          case 'run-queued-request':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: {
                response: {
                  requestId: message.requestId,
                  completed: true,
                  failed: false,
                  cancelled: false,
                  outputText: 'worker smoke output',
                  errorMessage: null,
                  runtimeObservability: null,
                },
                runtimeObservability: null,
                transportObservability: {
                  executionMode: 'worker',
                  workerBacked: true,
                  enabled: false,
                  bufferedTokenLimit: 0,
                  flushIntervalMs: 0,
                  flushCount: 0,
                  coalescedTokenCount: 0,
                  maxObservedBufferedTokenCount: 0,
                },
              },
            });
            break;
          default:
            throw new Error(`Unexpected worker message: ${message.kind}`);
        }
      };
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const churnCount = 5;
    for (let index = 0; index < churnCount; index += 1) {
      const abortController = new AbortController();
      const requestId = await runtime.queuePrompt(`ctx-${index}`, 'prompt', {
        nTokens: 16,
        signal: abortController.signal,
        onToken: () => {},
      });
      const cancelled = await runtime.cancelQueuedRequest(requestId);
      assert.equal(cancelled, true);
    }

    const smokeOutput = await runtime.submitPrompt('ctx-smoke', 'prompt', 8);
    assert.equal(smokeOutput, 'worker smoke output');

    assert.deepEqual(getWorkerQueuedState(runtime), {
      queuedCallbacks: 0,
      pendingCallbacks: 0,
      queuedErrors: 0,
      queuedSignals: 0,
    });
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime supports split model file shards in worker mode', async () => {
  const restoreWorker = installMockWorker();
  try {
    MockWorker.handlerFactory = () => (worker, message) => {
      if (message.kind === 'init-module') {
        worker.emit({
          kind: 'resolve',
          callId: message.callId,
          value: undefined,
        });
        return;
      }
      throw new Error(`Unexpected worker message: ${message.kind}`);
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const modelPath = await runtime.loadModelFromFileShards([
      new File([Uint8Array.from([1])], 'model-00001-of-00002.gguf'),
      new File([Uint8Array.from([2])], 'model-00002-of-00002.gguf'),
    ]);

    assert.equal(modelPath, '/models/model-00001-of-00002.gguf');
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime supports split model URL loading in worker mode', async () => {
  const restoreWorker = installMockWorker();
  try {
    MockWorker.handlerFactory = () => (worker, message) => {
      if (message.kind === 'init-module') {
        worker.emit({
          kind: 'resolve',
          callId: message.callId,
          value: undefined,
        });
        return;
      }
      throw new Error(`Unexpected worker message: ${message.kind}`);
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const modelPath = await runtime.loadModelFromUrls([
      'https://example.com/model-00001-of-00002.gguf',
      'https://example.com/model-00002-of-00002.gguf',
    ]);

    assert.equal(modelPath, '/models/model-00001-of-00002.gguf');
  } finally {
    restoreWorker();
  }
});
