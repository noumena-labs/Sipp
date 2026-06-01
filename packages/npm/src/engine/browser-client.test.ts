import test from 'node:test';
import assert from 'node:assert/strict';
import * as publicApi from '../index.js';
import { CogentClient } from './browser-client.js';
import { QueryError, type TokenBatch } from '../models/types.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
} from '../worker/model-service-protocol.js';

const TOKEN_RING_HEADER_INTS = 8;
const TOKEN_RING_HEADER_BYTES = TOKEN_RING_HEADER_INTS * 4;
const TOKEN_RING_WRITE_INDEX = 0;
const TOKEN_RING_CAPACITY = 2;
const TOKEN_RECORD_HEADER_BYTES = 16;

class FakeWorker {
  public static lastInstance: FakeWorker | null = null;
  public static tokenRingOrder: 'normal' | 'record-before-claim' = 'normal';
  public static flushAnimationFrame: (() => void) | null = null;
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
          if (message.config.wasmThreading === 'pthread') {
            const ring = createTokenRing();
            this.onmessage?.({
              data: { kind: 'token-ring-ready', descriptor: ring.descriptor },
            } as MessageEvent<WorkerResponseMessage>);
            if (FakeWorker.tokenRingOrder === 'record-before-claim') {
              writeTokenRecord(ring, message.callId, text);
              FakeWorker.flushAnimationFrame?.();
            }
            this.onmessage?.({
              data: {
                kind: 'token-ring-claim',
                callId: message.callId,
                nativeRequestId: message.callId,
              },
            } as MessageEvent<WorkerResponseMessage>);
            if (FakeWorker.tokenRingOrder === 'normal') {
              writeTokenRecord(ring, message.callId, text);
            }
          } else {
            this.onmessage?.({
              data: {
                kind: 'token-batch',
                callId: message.callId,
                batch: tokenBatch(message.callId, text),
              },
            } as MessageEvent<WorkerResponseMessage>);
          }
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

function createTokenRing() {
  const capacity = 1024;
  const buffer = new SharedArrayBuffer(TOKEN_RING_HEADER_BYTES + capacity);
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

function writeTokenRecord(
  ring: ReturnType<typeof createTokenRing>,
  requestId: number,
  text: string
): void {
  const bytes = new TextEncoder().encode(text);
  const offset = ring.header[TOKEN_RING_WRITE_INDEX];
  writeU32(ring.body, offset, requestId);
  writeU32(ring.body, offset + 4, 0);
  writeU32(ring.body, offset + 8, 1);
  writeU32(ring.body, offset + 12, bytes.byteLength);
  ring.body.set(bytes, offset + TOKEN_RECORD_HEADER_BYTES);
  Atomics.store(
    ring.header,
    TOKEN_RING_WRITE_INDEX,
    offset + TOKEN_RECORD_HEADER_BYTES + bytes.byteLength
  );
}

function writeU32(body: Uint8Array, offset: number, value: number): void {
  body[offset] = value & 0xff;
  body[offset + 1] = (value >>> 8) & 0xff;
  body[offset + 2] = (value >>> 16) & 0xff;
  body[offset + 3] = (value >>> 24) & 0xff;
}

function tokenBatch(requestId: number, text: string): TokenBatch {
  const byteCount = new TextEncoder().encode(text).byteLength;
  return {
    requestId: String(requestId),
    streamId: requestId,
    sequenceStart: 0,
    text,
    frameCount: 1,
    byteCount,
    stats: {
      framesSent: 1,
      bytesSent: byteCount,
      batchesSent: 1,
      drainMs: 0,
      drainCalls: 1,
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
      cacheMode: null,
      cacheSource: null,
      cacheHits: 0,
      prefillTokens: null,
      ttftMs: null,
      interTokenMs: null,
      e2eMs: null,
      decodeTokensPerSecond: null,
      e2eTokensPerSecond: null,
      prefillTokensPerSecond: null,
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
      cacheMode: null,
      cacheSource: null,
      cacheHits: 0,
      prefillTokens: null,
      ttftMs: null,
      interTokenMs: null,
      e2eMs: null,
      decodeTokensPerSecond: null,
      e2eTokensPerSecond: null,
      prefillTokensPerSecond: null,
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
    FakeWorker.tokenRingOrder = 'normal';
    FakeWorker.flushAnimationFrame = null;
  }
}

async function withCrossOriginIsolated<T>(callback: () => Promise<T>): Promise<T> {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crossOriginIsolated');
  Object.defineProperty(globalThis, 'crossOriginIsolated', {
    configurable: true,
    value: true,
  });

  try {
    return await callback();
  } finally {
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'crossOriginIsolated');
    } else {
      Object.defineProperty(globalThis, 'crossOriginIsolated', descriptor);
    }
  }
}

async function withManualAnimationFrame<T>(callback: () => Promise<T>): Promise<T> {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'requestAnimationFrame');
  let pendingFrame: FrameRequestCallback | null = null;
  Object.defineProperty(globalThis, 'requestAnimationFrame', {
    configurable: true,
    value: (frame: FrameRequestCallback) => {
      pendingFrame = frame;
      return 1;
    },
  });
  FakeWorker.flushAnimationFrame = () => {
    const frame = pendingFrame;
    pendingFrame = null;
    frame?.(performance.now());
  };

  try {
    return await callback();
  } finally {
    FakeWorker.flushAnimationFrame = null;
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'requestAnimationFrame');
    } else {
      Object.defineProperty(globalThis, 'requestAnimationFrame', descriptor);
    }
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

test('worker mode defaults to pthread wasm when shared memory is available', async () => {
  await withGlobalWorker(FakeWorker as unknown as typeof Worker, async () => {
    await withCrossOriginIsolated(async () => {
      const client = new CogentClient({
        executionMode: 'worker',
      });

      await client.listLocal();
      const worker = FakeWorker.lastInstance;
      const modelsRequest = worker?.messages.find((message) => message.kind === 'models-list');

      assert.equal(modelsRequest?.kind, 'models-list');
      assert.equal(modelsRequest?.config.wasmThreading, 'pthread');

      await client.close();
    });
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

test('worker shared token ring preserves records drained before native request claim', async () => {
  await withGlobalWorker(FakeWorker as unknown as typeof Worker, async () => {
    await withCrossOriginIsolated(async () => {
      await withManualAnimationFrame(async () => {
        FakeWorker.tokenRingOrder = 'record-before-claim';
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

        assert.equal(output.text, 'Hello');
        assert.deepEqual(chunks, ['Hello']);

        await client.close();
      });
    });
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
