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
  public readonly transferables: Transferable[][] = [];

  private readonly handler: WorkerMessageHandler;

  constructor(_url: string, _options?: WorkerOptions) {
    this.handler = MockWorker.handlerFactory?.() ?? (() => {});
    MockWorker.instances.push(this);
  }

  public postMessage(message: WorkerRequestMessage, transferables: Transferable[] = []): void {
    this.messages.push(message);
    this.transferables.push(transferables);
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

function createWorkerTransportObservability() {
  return {
    executionMode: 'worker' as const,
    workerBacked: true,
    enabled: false,
    bufferedTokenLimit: 0,
    flushIntervalMs: 0,
    flushCount: 0,
    coalescedTokenCount: 0,
    maxObservedBufferedTokenCount: 0,
  };
}

function createWorkerRuntimeMetadata() {
  return {
    chatTemplate: 'llama3',
    mediaMarker: '<__media__>',
    bosText: '<|begin_of_text|>',
    eosText: '<|end_of_text|>',
  };
}

function createWorkerQueuedResponse(
  requestId: number,
  response: {
    completed: boolean;
    failed: boolean;
    cancelled: boolean;
    outputText: string;
    errorMessage: string | null;
  }
) {
  return {
    response: {
      requestId,
      ...response,
      runtimeObservability: null,
    },
    runtimeAggregateObservability: null,
    transportObservability: createWorkerTransportObservability(),
  };
}

function getWorkerQueuedState(
  runtime: WorkerEngineRuntime
): {
  queuedCallbacks: number;
  pendingCallbacks: number;
  queuedErrors: number;
  queuedSignals: number;
  completions: number;
  activeRuns: number;
} {
  const runtimeState = runtime as unknown as {
    queuedTokenCallbacks: Map<number, unknown>;
    pendingQueuedTokenCallbacks: Map<number, unknown>;
    queuedTokenErrors: Map<number, unknown>;
    tracker: {
      signals: Map<number, unknown>;
      completions: Map<number, unknown>;
      activeRuns: Set<number>;
    };
  };

  return {
    queuedCallbacks: runtimeState.queuedTokenCallbacks.size,
    pendingCallbacks: runtimeState.pendingQueuedTokenCallbacks.size,
    queuedErrors: runtimeState.queuedTokenErrors.size,
    queuedSignals: runtimeState.tracker.signals.size,
    completions: runtimeState.tracker.completions.size,
    activeRuns: runtimeState.tracker.activeRuns.size,
  };
}

type MediaPromptOptions = import('../types.js').PromptOptions & {
  media?: Uint8Array[];
};

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

test('WorkerEngineRuntime only serializes persistentModelCache.enabled to the worker', async () => {
  const restoreWorker = installMockWorker();
  try {
    let initConfig: { persistentModelCache?: { enabled?: boolean } } | null = null;
    MockWorker.handlerFactory = () => (worker, message) => {
      if (message.kind === 'init-module') {
        initConfig = message.config;
        worker.emit({
          kind: 'resolve',
          callId: message.callId,
          value: undefined,
        });
      }
    };

    const runtime = new WorkerEngineRuntime({
      persistentModelCache: {
        enabled: true,
        namespace: 'leak-me',
        cacheLocalFiles: true,
        maxEntryBytes: 123,
      } as never,
    } as never);

    await runtime.initModule();

    if (initConfig == null) {
      throw new Error('Expected init-module config to be captured.');
    }
    const capturedConfig = initConfig as { persistentModelCache?: { enabled?: boolean } };
    assert.deepEqual(capturedConfig.persistentModelCache, {
      enabled: true,
    });
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
            worker.emit({
              kind: 'request-complete',
              requestId: message.requestId,
              result: createWorkerQueuedResponse(message.requestId, {
                completed: false,
                failed: false,
                cancelled: true,
                outputText: '',
                errorMessage: 'Queued request cancelled.',
              }),
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
    await waitForCondition(() => {
      const state = getWorkerQueuedState(runtime);
      return (
        state.queuedCallbacks === 0 &&
        state.pendingCallbacks === 0 &&
        state.queuedErrors === 0 &&
        state.queuedSignals === 0 &&
        state.completions === 0 &&
        state.activeRuns === 0
      );
    });
    assert.deepEqual(getWorkerQueuedState(runtime), {
      queuedCallbacks: 0,
      pendingCallbacks: 0,
      queuedErrors: 0,
      queuedSignals: 0,
      completions: 0,
      activeRuns: 0,
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
            const requestId = nextRequestId++;
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: requestId,
            });
            if (message.contextKey === 'ctx-smoke') {
              queueMicrotask(() => {
                worker.emit({
                  kind: 'request-complete',
                  requestId,
                  result: createWorkerQueuedResponse(requestId, {
                    completed: true,
                    failed: false,
                    cancelled: false,
                    outputText: 'worker smoke output',
                    errorMessage: null,
                  }),
                });
              });
            }
            break;
          case 'cancel-request':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: true,
            });
            worker.emit({
              kind: 'request-complete',
              requestId: message.requestId,
              result: createWorkerQueuedResponse(message.requestId, {
                completed: false,
                failed: false,
                cancelled: true,
                outputText: '',
                errorMessage: 'Queued request cancelled.',
              }),
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
      completions: 0,
      activeRuns: 0,
    });
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime close resets stale lifecycle state and cached metadata', async () => {
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

    const runtimeState = runtime as unknown as {
      queuedTokenCallbacks: Map<number, unknown>;
      pendingQueuedTokenCallbacks: Map<number, unknown>;
      queuedTokenErrors: Map<number, unknown>;
      tracker: {
        track: (requestId: number) => unknown;
        signals: Map<number, unknown>;
      };
      runtimeAggregateObservability: object | null;
      lastModelLoadInfo: object | null;
      transportObservability: {
        enabled: boolean;
        flushCount: number;
        coalescedTokenCount: number;
      };
    };
    runtimeState.queuedTokenCallbacks.set(11, () => {});
    runtimeState.pendingQueuedTokenCallbacks.set(12, () => {});
    runtimeState.queuedTokenErrors.set(11, new Error('token failure'));
    runtimeState.tracker.track(11);
    runtimeState.tracker.signals.set(11, new AbortController().signal);
    runtimeState.runtimeAggregateObservability = { outputTokenCount: 2 };
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
    runtimeState.transportObservability.flushCount = 4;
    runtimeState.transportObservability.coalescedTokenCount = 7;

    runtime.close();

    assert.equal(MockWorker.instances[0].terminated, true);
    assert.deepEqual(getWorkerQueuedState(runtime), {
      queuedCallbacks: 0,
      pendingCallbacks: 0,
      queuedErrors: 0,
      queuedSignals: 0,
      completions: 0,
      activeRuns: 0,
    });
    assert.equal(runtime.getRuntimeAggregateObservability(), null);
    assert.equal(runtime.getRuntimeObservability(), null);
    assert.equal(runtime.getLastModelLoadInfo(), null);
    assert.deepEqual(runtime.getTransportObservability(), {
      executionMode: 'worker',
      workerBacked: true,
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
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime clears local queued lifecycle state before initEngine reinit', async () => {
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
        case 'init-engine':
          worker.emit({
            kind: 'reject',
            callId: message.callId,
            message: 'init failed',
            errorName: 'Error',
          });
          break;
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();
    const runtimeState = runtime as unknown as {
      queuedTokenCallbacks: Map<number, unknown>;
      pendingQueuedTokenCallbacks: Map<number, unknown>;
      queuedTokenErrors: Map<number, unknown>;
      tracker: {
        track: (requestId: number) => unknown;
        signals: Map<number, unknown>;
      };
      runtimeAggregateObservability: object | null;
    };
    runtimeState.queuedTokenCallbacks.set(21, () => {});
    runtimeState.pendingQueuedTokenCallbacks.set(22, () => {});
    runtimeState.queuedTokenErrors.set(21, new Error('token failure'));
    runtimeState.tracker.track(21);
    runtimeState.tracker.signals.set(21, new AbortController().signal);
    runtimeState.runtimeAggregateObservability = { outputTokenCount: 2 };

    await assert.rejects(runtime.initEngine('/models/failing-model.gguf'), /init failed/);

    assert.deepEqual(getWorkerQueuedState(runtime), {
      queuedCallbacks: 0,
      pendingCallbacks: 0,
      queuedErrors: 0,
      queuedSignals: 0,
      completions: 0,
      activeRuns: 0,
    });
    assert.equal(runtime.getRuntimeAggregateObservability(), null);
    assert.equal(runtime.getRuntimeObservability(), null);
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime caches runtime metadata from initEngine and clears it on close', async () => {
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
          return;
        case 'init-engine':
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: createWorkerRuntimeMetadata(),
          });
          return;
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();
    assert.equal(runtime.getChatTemplate(), null);
    assert.equal(runtime.getMediaMarker(), null);
    assert.equal(runtime.getEosText(), '');

    await runtime.initEngine('/models/model.gguf');

    assert.equal(runtime.getChatTemplate(), 'llama3');
    assert.equal(runtime.getMediaMarker(), '<__media__>');
    assert.equal(runtime.getEosText(), '<|end_of_text|>');
    assert.equal(MockWorker.instances[0].messages.some((message) => message.kind === 'init-engine'), true);

    runtime.close();

    assert.equal(runtime.getChatTemplate(), null);
    assert.equal(runtime.getMediaMarker(), null);
    assert.equal(runtime.getEosText(), '');
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime proxies applyChatTemplate through the worker', async () => {
  const restoreWorker = installMockWorker();
  try {
    MockWorker.handlerFactory = () => (worker, message) => {
      switch (message.kind) {
        case 'init-module':
          worker.emit({ kind: 'resolve', callId: message.callId, value: undefined });
          return;
        case 'apply-chat-template':
          worker.emit({ kind: 'resolve', callId: message.callId, value: 'templated:ok' });
          return;
        default:
          worker.emit({ kind: 'resolve', callId: message.callId, value: createWorkerRuntimeMetadata() });
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();
    await runtime.initEngine('/models/model.gguf');

    const rendered = await runtime.applyChatTemplate(
      [
        { role: 'system', content: 'sys' },
        { role: 'user', content: 'hi' },
      ],
      true
    );

    assert.equal(rendered, 'templated:ok');
    assert.equal(
      MockWorker.instances[0].messages.some((message) => message.kind === 'apply-chat-template'),
      true
    );
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime forwards decode sampling config through init-engine', async () => {
  const restoreWorker = installMockWorker();
  try {
    let capturedConfig: WorkerRequestMessage | null = null;
    MockWorker.handlerFactory = () => (worker, message) => {
      switch (message.kind) {
        case 'init-module':
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: undefined,
          });
          return;
        case 'init-engine':
          capturedConfig = message;
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: createWorkerRuntimeMetadata(),
          });
          return;
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();
    await runtime.initEngine('/models/model.gguf', {
      multimodalUseGpu: false,
      sampling: {
        repeatLastN: 96,
        temperature: 0.55,
        topP: 0.92,
        seed: 1337,
      },
    });

    const initMessage =
      capturedConfig as Extract<WorkerRequestMessage, { kind: 'init-engine' }> | null;
    assert.ok(initMessage != null);
    assert.equal(initMessage.kind, 'init-engine');
    assert.equal(initMessage.config?.multimodalUseGpu, false);
    assert.deepEqual(
      initMessage.config?.sampling,
      {
        repeatLastN: 96,
        temperature: 0.55,
        topP: 0.92,
        seed: 1337,
      }
    );
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime sends transferable media buffers for media prompts', async () => {
  const restoreWorker = installMockWorker();
  try {
    MockWorker.handlerFactory = () => {
      let nextRequestId = 201;
      return (worker, message) => {
        switch (message.kind) {
          case 'init-module':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: undefined,
            });
            return;
          case 'init-engine':
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: createWorkerRuntimeMetadata(),
            });
            return;
          case 'queue-prompt-with-media':
            assert.equal(message.options.promptFormat, undefined);
            assert.ok(message.options.media != null);
            assert.equal(message.options.media.length, 2);
            assert.equal(message.promptText, 'first <__media__> second <__media__>');
            worker.emit({
              kind: 'resolve',
              callId: message.callId,
              value: nextRequestId++,
            });
            return;
          default:
            throw new Error(`Unexpected worker message: ${message.kind}`);
        }
      };
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();
    await runtime.initEngine('/models/model.gguf');

    const requestId = await runtime.queuePrompt(
      'ctx-media',
      'first <__media__> second <__media__>',
      {
        nTokens: 16,
        media: [Uint8Array.from([1, 2, 3]), Uint8Array.from([4, 5])],
      } as MediaPromptOptions
    );

    const worker = MockWorker.instances[0];
    const mediaMessageIndex = worker.messages.findIndex(
      (message) => message.kind === 'queue-prompt-with-media'
    );

    assert.equal(requestId, 201);
    assert.ok(mediaMessageIndex >= 0);
    assert.equal(worker.transferables[mediaMessageIndex].length, 2);
    assert.equal(worker.transferables[mediaMessageIndex][0] instanceof ArrayBuffer, true);
    assert.equal(worker.transferables[mediaMessageIndex][1] instanceof ArrayBuffer, true);
    assert.equal((worker.transferables[mediaMessageIndex][0] as ArrayBuffer).byteLength, 3);
    assert.equal((worker.transferables[mediaMessageIndex][1] as ArrayBuffer).byteLength, 2);
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime rejects media prompts when the marker count does not match', async () => {
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
          return;
        case 'init-engine':
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: createWorkerRuntimeMetadata(),
          });
          return;
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();
    await runtime.initEngine('/models/model.gguf');

    await assert.rejects(
      runtime.queuePrompt('ctx-media-mismatch', 'missing marker', {
        nTokens: 16,
        media: [Uint8Array.from([1, 2, 3])],
      } as MediaPromptOptions),
      /media marker\(s\).*media attachment\(s\)/i
    );

    assert.equal(
      MockWorker.instances[0].messages.some(
        (message) => message.kind === 'queue-prompt-with-media'
      ),
      false
    );
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime supports split model file shards in worker mode', async () => {
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
          return;
        case 'load-model-file-shards': {
          assert.deepEqual(
            message.files.map((file) => file.name),
            [
              'model-00001-of-00002.gguf',
              'model-00002-of-00002.gguf',
            ]
          );
          const result: WorkerLoadModelResult = {
            modelPath: '/models/model-00001-of-00002.gguf',
            modelLoadInfo: {
              sourceKind: 'file',
              reuseMode: 'file-read',
              modelPath: '/models/model-00001-of-00002.gguf',
              fileName: 'model-00001-of-00002.gguf',
              byteLength: 2,
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
          return;
        }
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
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
      switch (message.kind) {
        case 'init-module':
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: undefined,
          });
          return;
        case 'load-model-urls': {
          assert.deepEqual(message.urls, [
            'https://example.com/model-00001-of-00002.gguf',
            'https://example.com/model-00002-of-00002.gguf',
          ]);
          const result: WorkerLoadModelResult = {
            modelPath: '/models/model-00001-of-00002.gguf',
            modelLoadInfo: {
              sourceKind: 'url',
              reuseMode: 'network',
              modelPath: '/models/model-00001-of-00002.gguf',
              fileName: 'model-00001-of-00002.gguf',
              byteLength: 2,
              persistentCacheEnabled: true,
              persistentCacheKey: 'cache-key-1,cache-key-2',
              persistentCacheHit: false,
              persistentCacheStored: true,
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
          return;
        }
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
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

test('WorkerEngineRuntime prepares model bundles through the worker protocol and caches model load info', async () => {
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
          return;
        case 'prepare-model-bundle':
          assert.deepEqual(message.descriptor, {
            kind: 'url',
            url: 'https://example.com/Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
          });
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: {
              bundle: {
                sourceKind: 'url',
                modelPath: '/models/model.gguf',
                multimodalProjectorPath: '/models/mmproj.gguf',
                isVisionModel: true,
                projectorStatus: 'discovered',
                modelName: 'Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
                detectionMethod: 'url',
                modelType: null,
                modelArchitecture: null,
                modelLoadInfo: {
                  sourceKind: 'url',
                  reuseMode: 'network',
                  modelPath: '/models/model.gguf',
                  fileName: 'model.gguf',
                  byteLength: 4,
                  persistentCacheEnabled: false,
                  persistentCacheKey: null,
                  persistentCacheHit: false,
                  persistentCacheStored: false,
                },
                projectorLoadInfo: {
                  sourceKind: 'url',
                  reuseMode: 'network',
                  modelPath: '/models/mmproj.gguf',
                  fileName: 'mmproj.gguf',
                  byteLength: 4,
                  persistentCacheEnabled: false,
                  persistentCacheKey: null,
                  persistentCacheHit: false,
                  persistentCacheStored: false,
                },
              },
              transportObservability: createWorkerTransportObservability(),
            },
          });
          return;
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const bundle = await runtime.prepareModelBundle({
      kind: 'url',
      url: 'https://example.com/Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
    });

    assert.equal(bundle.multimodalProjectorPath, '/models/mmproj.gguf');
    assert.equal(bundle.projectorStatus, 'discovered');
    assert.equal(runtime.getLastModelLoadInfo()?.modelPath, '/models/model.gguf');
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime initEngine(bundle) forwards bundle projectors unless explicitly overridden', async () => {
  const restoreWorker = installMockWorker();
  try {
    const initMessages: Array<Extract<WorkerRequestMessage, { kind: 'init-engine' }>> = [];
    MockWorker.handlerFactory = () => (worker, message) => {
      switch (message.kind) {
        case 'init-module':
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: undefined,
          });
          return;
        case 'init-engine':
          initMessages.push(message);
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: createWorkerRuntimeMetadata(),
          });
          return;
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const preparedBundle = {
      sourceKind: 'url' as const,
      modelPath: '/models/model.gguf',
      multimodalProjectorPath: '/models/mmproj.gguf',
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
    await runtime.initEngine(preparedBundle, {
      multimodalProjectorPath: '/override/mmproj.gguf',
    });
    await runtime.initEngine({
      ...preparedBundle,
      multimodalProjectorPath: null,
      projectorStatus: 'missing',
    });

    assert.deepEqual(
      initMessages.map((message) => ({
        modelPath: message.modelPath,
        projectorPath: message.config?.multimodalProjectorPath ?? null,
      })),
      [
        {
          modelPath: '/models/model.gguf',
          projectorPath: '/models/mmproj.gguf',
        },
        {
          modelPath: '/models/model.gguf',
          projectorPath: '/override/mmproj.gguf',
        },
        {
          modelPath: '/models/model.gguf',
          projectorPath: null,
        },
      ]
    );
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime shares one pushed worker completion across concurrent runQueuedRequest() waiters', async () => {
  const restoreWorker = installMockWorker();
  try {
    let requestCompleteCount = 0;
    MockWorker.handlerFactory = () => (worker, message) => {
      switch (message.kind) {
        case 'init-module':
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: undefined,
          });
          return;
        case 'queue-prompt':
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: 77,
          });
          queueMicrotask(() => {
            requestCompleteCount += 1;
            worker.emit({
              kind: 'request-complete',
              requestId: 77,
              result: createWorkerQueuedResponse(77, {
                completed: true,
                failed: false,
                cancelled: false,
                outputText: 'shared worker response',
                errorMessage: null,
              }),
            });
          });
          return;
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const requestId = await runtime.queuePrompt('ctx-shared-waiters', 'prompt', 8);
    const firstPromise = runtime.runQueuedRequest(requestId);
    const secondPromise = runtime.runQueuedRequest(requestId);

    const [firstResponse, secondResponse] = await Promise.all([
      withTimeout(firstPromise, 25, 'Timed out waiting for the first worker waiter.'),
      withTimeout(secondPromise, 25, 'Timed out waiting for the second worker waiter.'),
    ]);

    assert.equal(firstResponse.outputText, 'shared worker response');
    assert.equal(secondResponse.outputText, 'shared worker response');
    assert.equal(requestCompleteCount, 1);
    assert.equal(
      MockWorker.instances[0].messages.filter((message) => message.kind === 'queue-prompt').length,
      1
    );
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime rejects outstanding queued waiters when close() is called', async () => {
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
          return;
        case 'queue-prompt':
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: 88,
          });
          return;
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const requestId = await runtime.queuePrompt('ctx-close-waiter', 'prompt', 8);
    const waiter = runtime.runQueuedRequest(requestId);

    runtime.close();

    await assert.rejects(
      withTimeout(waiter, 25, 'Timed out waiting for worker waiter rejection after close().'),
      /closed/i
    );
  } finally {
    restoreWorker();
  }
});

test('WorkerEngineRuntime rejects outstanding queued waiters when the worker crashes mid-request', async () => {
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
          return;
        case 'queue-prompt':
          worker.emit({
            kind: 'resolve',
            callId: message.callId,
            value: 99,
          });
          return;
        default:
          throw new Error(`Unexpected worker message: ${message.kind}`);
      }
    };

    const runtime = new WorkerEngineRuntime({});
    await runtime.initModule();

    const requestId = await runtime.queuePrompt('ctx-worker-crash', 'prompt', 8);
    const waiter = runtime.runQueuedRequest(requestId);

    MockWorker.instances[0].triggerError('worker exploded during queued execution');

    await assert.rejects(
      withTimeout(waiter, 25, 'Timed out waiting for worker waiter rejection after crash.'),
      /worker exploded during queued execution/i
    );
  } finally {
    restoreWorker();
  }
});
