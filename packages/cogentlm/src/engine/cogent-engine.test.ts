import test from 'node:test';
import assert from 'node:assert/strict';
import * as publicApi from '../index.js';
import { CogentEngine } from './cogent-engine.js';
import { QueryError } from '../models/types.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
} from '../worker/model-service-protocol.js';

import { StreamingRingWriter } from '../runtime/streaming-ring.js';

class FakeWorker {
  public static lastInstance: FakeWorker | null = null;
  public onmessage: ((event: MessageEvent<WorkerResponseMessage>) => void) | null = null;
  public onerror: ((event: ErrorEvent) => void) | null = null;
  public onmessageerror: (() => void) | null = null;
  public readonly messages: WorkerRequestMessage[] = [];
  public terminated = false;
  private ringWriter: StreamingRingWriter | null = null;

  constructor(
    public readonly url: string | URL,
    public readonly options?: WorkerOptions
  ) {
    FakeWorker.lastInstance = this;
  }

  public postMessage(message: WorkerRequestMessage): void {
    this.messages.push(message);

    if (message.kind === 'streaming-init') {
      this.ringWriter = message.ringBuffer ? new StreamingRingWriter(message.ringBuffer) : null;
      return;
    }

    if ('callId' in message) {
      if (message.kind === 'chat' || message.kind === 'query') {
        const text = message.kind === 'chat' ? 'Hello' : 'Hello</assistant>\n<user>ignored';
        if (this.ringWriter) {
          const requestId = 123;
          this.ringWriter.tryWriteString(requestId, text);
          this.onmessage?.({
            data: { kind: 'streaming-claim', callId: message.callId, nativeRequestId: requestId }
          } as MessageEvent<WorkerResponseMessage>);
        }
      }
      queueMicrotask(() => {
        const response: WorkerResponseMessage = {
          kind: 'resolve',
          callId: message.callId,
          value:
            message.kind === 'models-list'
              ? []
              : message.kind === 'models-remove'
                ? null
                : message.kind === 'models-load'
                  ? {
                    id: 'model-fake',
                    name: 'fake.gguf',
                    modality: 'text',
                    status: 'ready',
                    source: 'local',
                    bytes: 1,
                    loaded: true,
                    chatTemplate: 'fake-template',
                    bosText: '<s>',
                    eosText: '</s>',
                    mediaMarker: null,
                    createdAt: new Date(0).toISOString(),
                    updatedAt: new Date(0).toISOString(),
                  }
                  : message.kind === 'query'
                    ? requestResult('Hello</assistant>\n<user>ignored')
                    : message.kind === 'chat'
                      ? requestResult('Hello')
                      : undefined,
        };
        this.onmessage?.({ data: response } as MessageEvent<WorkerResponseMessage>);
      });
    }
  }

  public terminate(): void {
    this.terminated = true;
  }
}

function requestResult(text: string) {
  return {
    id: '123',
    text,
    finishReason: 'stop',
    stats: {
      inputTokens: 1,
      outputTokens: 1,
      cacheHits: 0,
      prefillMs: 0,
      decodeMs: 0,
    },
  };
}

async function withGlobalWorker<T>(worker: typeof Worker, callback: () => Promise<T>): Promise<T> {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  Object.defineProperty(globalThis, 'Worker', {
    configurable: true,
    value: worker,
  });

  try {
    return await callback();
  } finally {
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'Worker');
    } else {
      Object.defineProperty(globalThis, 'Worker', descriptor);
    }
    FakeWorker.lastInstance = null;
  }
}

test('CogentEngine exposes the minimal root API', async () => {
  const engine = await CogentEngine.create({
    moduleUrl: 'https://example.test/runtime.js',
    wasmUrl: 'https://example.test/runtime.wasm',
    executionMode: 'main-thread',
  });

  assert.equal(typeof engine.models.load, 'function');
  assert.equal(typeof engine.models.current, 'function');
  assert.equal(typeof engine.models.list, 'function');
  assert.equal(typeof engine.models.remove, 'function');
  assert.equal(typeof engine.observability.current, 'function');
  assert.equal(typeof engine.observability.subscribe, 'function');
  assert.equal(typeof engine.query, 'function');
  assert.equal(typeof engine.chat, 'function');
  assert.equal(typeof (engine as any).applyChatTemplate, 'undefined');
  assert.equal(typeof (engine as any).getChatTemplate, 'undefined');
  assert.equal(typeof engine.close, 'function');
  assert.deepEqual(Object.keys(engine), ['models', 'observability']);
  assert.deepEqual(Object.keys(publicApi).sort(), ['CogentEngine', 'QueryError']);

  const events: string[] = [];
  engine.observability.subscribe((event) => {
    events.push(event.type);
  });
  await engine.close();
  assert.deepEqual(events, ['close']);
  assert.throws(
    () => engine.models.current(),
    (error) => error instanceof QueryError && error.code === 'ENGINE_CLOSED'
  );
});

test('CogentEngine.create uses bundled runtime URLs internally by default', async () => {
  const engine = await CogentEngine.create({ executionMode: 'main-thread' });
  assert.deepEqual(Object.keys(engine), ['models', 'observability']);
  assert.equal(engine.models.current(), null);
  await engine.close();
});

test('worker-hosted runtime reports worker execution in observability transport', () => {
  const runtime = new MainThreadEngineRuntime({
    moduleUrl: 'https://example.test/runtime.js',
    wasmUrl: 'https://example.test/runtime.wasm',
    executionMode: 'worker',
  });
  const transport = runtime.getTransportObservability();

  assert.equal(runtime.getExecutionMode(), 'worker');
  assert.equal(transport.executionMode, 'worker');
  assert.equal(transport.workerBacked, true);
});

test('worker mode lists models without requiring explicit runtime URLs', async () => {
  await withGlobalWorker(FakeWorker as unknown as typeof Worker, async () => {
    const engine = await CogentEngine.create({
      executionMode: 'worker',
    });

    const models = await engine.models.list();
    const worker = FakeWorker.lastInstance;

    assert.deepEqual(models, []);
    assert.ok(worker != null);
    assert.match(String(worker.url), /model-service-entry\.js$/);
    assert.equal(worker.options?.type, 'module');
    const modelsRequest = worker.messages.find(m => m.kind === 'models-list');
    assert.equal(modelsRequest?.kind, 'models-list');
    assert.equal((modelsRequest as any)?.config?.moduleUrl, undefined);
    assert.equal((modelsRequest as any)?.config?.wasmUrl, undefined);

    await engine.close();
    assert.equal(worker.terminated, true);
  });
});

test('chat() renders messages through the worker service and sanitizes assistant boundaries', async () => {
  await withGlobalWorker(FakeWorker as unknown as typeof Worker, async () => {
    const engine = await CogentEngine.create({
      executionMode: 'worker',
    });

    await engine.models.load('model-fake');
    const chunks: string[] = [];
    const output = await engine.chat([{ role: 'user', content: 'hello' }], {
      onTokens: (batch) => chunks.push(batch.text),
    });
    const worker = FakeWorker.lastInstance;
    const chat = worker?.messages.find((message) => message.kind === 'chat');

    assert.equal(output.text, 'Hello');
    await new Promise((resolve) => setTimeout(resolve, 50));
    assert.deepEqual(chunks, ['Hello']);
    assert.ok(chat != null && chat.kind === 'chat');
    const messages = Array.isArray(chat.input) ? chat.input : chat.input.messages;
    assert.deepEqual(messages, [{ role: 'user', content: 'hello' }]);

    await engine.close();
  });
});

test('worker mode resolves explicit relative runtime overrides on the main thread', async () => {
  const previousLocation = globalThis.location;
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: new URL('https://app.test/page/index.html'),
  });
  await withGlobalWorker(FakeWorker as unknown as typeof Worker, async () => {
    try {
      const engine = await CogentEngine.create({
        executionMode: 'worker',
        moduleUrl: './runtime/custom-module.js',
        wasmUrl: './runtime/custom-module.wasm',
      });

      await engine.models.list();
      const request = FakeWorker.lastInstance?.messages.find(m => m.kind === 'models-list');
      assert.equal(request?.kind, 'models-list');
      assert.equal(request?.config.moduleUrl, 'https://app.test/page/runtime/custom-module.js');
      assert.equal(request?.config.wasmUrl, 'https://app.test/page/runtime/custom-module.wasm');

      await engine.close();
    } finally {
      Object.defineProperty(globalThis, 'location', {
        configurable: true,
        value: previousLocation,
      });
    }
  });
});
