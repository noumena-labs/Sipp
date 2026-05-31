import test from 'node:test';
import assert from 'node:assert/strict';
import * as publicApi from '../index.js';
import { CogentClient } from './browser-client.js';
import { QueryError } from '../models/types.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
} from '../worker/model-service-protocol.js';

class FakeWorker {
  public static lastInstance: FakeWorker | null = null;
  public onmessage: ((event: MessageEvent<WorkerResponseMessage>) => void) | null = null;
  public onerror: ((event: ErrorEvent) => void) | null = null;
  public onmessageerror: (() => void) | null = null;
  public readonly messages: WorkerRequestMessage[] = [];
  public terminated = false;

  constructor(
    public readonly url: string | URL,
    public readonly options?: WorkerOptions
  ) {
    FakeWorker.lastInstance = this;
  }

  public postMessage(message: WorkerRequestMessage): void {
    this.messages.push(message);

    if ('callId' in message) {
      if (message.kind === 'chat' || message.kind === 'query') {
        const text = message.kind === 'chat' ? 'Hello' : 'Hello</assistant>\n<user>ignored';
        if (message.options.emitTokens) {
          this.onmessage?.({
            data: { kind: 'token-batch', callId: message.callId, batch: tokenBatch(text) },
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
                    ? generationResult('Hello</assistant>\n<user>ignored')
                    : message.kind === 'chat'
                      ? generationResult('Hello')
                      : message.kind === 'embed'
                        ? embeddingResult(message.options.normalize === false)
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

function tokenBatch(text: string) {
  return {
    requestId: '123',
    streamId: 123,
    sequenceStart: 0,
    text,
    frameCount: 1,
    byteCount: new TextEncoder().encode(text).byteLength,
    stats: {
      framesSent: 1,
      bytesSent: new TextEncoder().encode(text).byteLength,
      batchesSent: 1,
    },
  };
}

function generationResult(text: string) {
  return {
    id: '123',
    text,
    finishReason: 'stop',
    stats: {
      inputTokens: 1,
      outputTokens: 1,
      cacheHits: 0,
      ttftMs: null,
      interTokenMs: null,
      e2eMs: null,
      decodeTokensPerSecond: null,
      e2eTokensPerSecond: null,
      prefillMs: 0,
      decodeMs: 0,
    },
  };
}

function embeddingResult(raw: boolean) {
  return {
    id: '124',
    values: raw ? [3, 4] : [0.6, 0.8],
    pooling: 'mean',
    normalized: !raw,
    stats: {
      inputTokens: 2,
      outputTokens: 0,
      cacheHits: 0,
      ttftMs: null,
      interTokenMs: null,
      e2eMs: null,
      decodeTokensPerSecond: null,
      e2eTokensPerSecond: null,
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

test('CogentClient exposes the minimal browser API', async () => {
  const client = new CogentClient({
    moduleUrl: 'https://example.test/runtime.js',
    wasmUrl: 'https://example.test/runtime.wasm',
    executionMode: 'main-thread',
  });

  assert.equal(typeof client.addLocal, 'function');
  assert.equal(typeof client.currentLocal, 'function');
  assert.equal(typeof client.listLocal, 'function');
  assert.equal(typeof client.removeLocal, 'function');
  assert.equal(typeof client.observability.current, 'function');
  assert.equal(typeof client.observability.subscribe, 'function');
  assert.equal(typeof client.query, 'function');
  assert.equal(typeof client.chat, 'function');
  assert.equal(typeof client.embed, 'function');
  assert.equal(typeof client.close, 'function');
  assert.deepEqual(Object.keys(client), ['observability']);
  assert.deepEqual(Object.keys(publicApi).sort(), [
    'CogentClient',
    'QueryError',
  ]);

  const events: string[] = [];
  client.observability.subscribe((event) => {
    events.push(event.type);
  });
  await client.close();
  assert.deepEqual(events, ['close']);
  assert.throws(
    () => client.currentLocal(),
    (error) => error instanceof QueryError && error.code === 'ENGINE_CLOSED'
  );
});

test('CogentClient uses bundled runtime URLs internally by default', async () => {
  const client = new CogentClient({ executionMode: 'main-thread' });
  assert.deepEqual(Object.keys(client), ['observability']);
  assert.equal(client.currentLocal(), null);
  await client.close();
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
    const client = new CogentClient({
      executionMode: 'worker',
    });

    const models = await client.listLocal();
    const worker = FakeWorker.lastInstance;

    assert.deepEqual(models, []);
    assert.ok(worker != null);
    assert.match(String(worker.url), /model-service-entry\.js$/);
    assert.equal(worker.options?.type, 'module');
    const modelsRequest = worker.messages.find((message) => message.kind === 'models-list');
    assert.equal(modelsRequest?.kind, 'models-list');
    assert.equal(modelsRequest?.config.moduleUrl, undefined);
    assert.equal(modelsRequest?.config.wasmUrl, undefined);

    await client.close();
    assert.equal(worker.terminated, true);
  });
});

test('chat() renders messages through the worker service and sanitizes assistant boundaries', async () => {
  await withGlobalWorker(FakeWorker as unknown as typeof Worker, async () => {
    const client = new CogentClient({
      executionMode: 'worker',
    });

    await client.addLocal('model-fake');
    const chunks: string[] = [];
    const run = client.chat([{ role: 'user', content: 'hello' }], {
      emitTokens: true,
    });
    const output = await run.response;
    for await (const batch of run.tokens) {
      chunks.push(batch.text);
    }
    const worker = FakeWorker.lastInstance;
    const chat = worker?.messages.find((message) => message.kind === 'chat');

    assert.equal(output.text, 'Hello');
    await new Promise((resolve) => setTimeout(resolve, 50));
    assert.deepEqual(chunks, ['Hello']);
    assert.ok(chat != null && chat.kind === 'chat');
    const messages = Array.isArray(chat.input) ? chat.input : chat.input.messages;
    assert.deepEqual(messages, [{ role: 'user', content: 'hello' }]);

    await client.close();
  });
});

test('embed() routes through the worker service', async () => {
  await withGlobalWorker(FakeWorker as unknown as typeof Worker, async () => {
    const client = new CogentClient({
      executionMode: 'worker',
    });

    await client.addLocal('model-fake');
    const output = await client.embed('hello', { normalize: false, contextKey: 'embeddings' }).response;
    const worker = FakeWorker.lastInstance;
    const embed = worker?.messages.find((message) => message.kind === 'embed');

    assert.deepEqual(output.values, [3, 4]);
    assert.equal(output.normalized, false);
    assert.ok(embed != null && embed.kind === 'embed');
    assert.equal(embed.input, 'hello');
    assert.deepEqual(embed.options, { normalize: false, contextKey: 'embeddings' });

    await client.close();
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
      const client = new CogentClient({
        executionMode: 'worker',
        moduleUrl: './runtime/custom-module.js',
        wasmUrl: './runtime/custom-module.wasm',
      });

      await client.listLocal();
      const request = FakeWorker.lastInstance?.messages.find(m => m.kind === 'models-list');
      assert.equal(request?.kind, 'models-list');
      assert.equal(request?.config.moduleUrl, 'https://app.test/page/runtime/custom-module.js');
      assert.equal(request?.config.wasmUrl, 'https://app.test/page/runtime/custom-module.wasm');

      await client.close();
    } finally {
      Object.defineProperty(globalThis, 'location', {
        configurable: true,
        value: previousLocation,
      });
    }
  });
});
