import test from 'node:test';
import assert from 'node:assert/strict';
import * as publicApi from '../../src/index.js';
import { CogentClient } from '../../src/engine/browser-client.js';
import {
  QueryError,
  type ChatInput,
  type GatewayOptions,
  type QueryInput,
  type RemoteGatewayConfig,
  type TokenBatch,
} from '../../src/models/types.js';
import { MainThreadEngineRuntime } from '../../src/runtime/main-thread/engine-runtime.js';
import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
} from '../../src/worker/model-service-protocol.js';

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

async function withGlobalFetch<T>(
  fetchImpl: typeof globalThis.fetch,
  callback: () => Promise<T>
): Promise<T> {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'fetch');
  Object.defineProperty(globalThis, 'fetch', {
    configurable: true,
    value: fetchImpl,
  });

  try {
    return await callback();
  } finally {
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'fetch');
    } else {
      Object.defineProperty(globalThis, 'fetch', descriptor);
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
  assert.equal(typeof client.addRemote, 'function');
  assert.equal(typeof client.updateRemote, 'function');
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

test('query() routes explicit remote endpoint through gateway fetch', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response(
        JSON.stringify({
          id: 'gw_1',
          model: 'pro-chat',
          text: 'remote answer',
          finish_reason: 'stop',
          usage: {
            input_tokens: 2,
            output_tokens: 3,
            total_tokens: 5,
          },
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      const output = await client
        .query('hello', {
          endpoint,
          maxTokens: 7,
          temperature: 0.25,
          topP: 0.75,
          stop: ['END'],
        })
        .response;
      const call = calls[0];
      const body = JSON.parse(String(call.init?.body)) as Record<string, unknown>;
      const headers = call.init?.headers as Record<string, string>;

      assert.equal(output.text, 'remote answer');
      assert.equal(output.stats.inputTokens, 2);
      assert.equal(output.stats.outputTokens, 3);
      assert.equal(call.url, 'https://gateway.example.test/v1/query');
      assert.equal(call.init?.method, 'POST');
      assert.equal(call.init?.credentials, 'omit');
      assert.equal(call.init?.mode, 'cors');
      assert.equal(call.init?.redirect, 'error');
      assert.equal(headers.Authorization, 'Bearer secret-token');
      assert.deepEqual(body, {
        model: 'pro-chat',
        prompt: 'hello',
        max_tokens: 7,
        temperature: 0.25,
        top_p: 0.75,
        stop: ['END'],
        stream: false,
      });

      await client.close();
    }
  );
});

test('remote gateway text responses require text and finishReason fields', async () => {
  const cases = [
    {
      id: 'missing-model',
      body: {
        id: 'gw_missing_model',
        text: 'remote answer',
        finish_reason: 'stop',
      },
      message: 'remote gateway response missing model',
    },
    {
      id: 'missing-text',
      body: {
        id: 'gw_missing_text',
        model: 'pro-chat',
        finish_reason: 'stop',
      },
      message: 'remote gateway response missing text',
    },
    {
      id: 'missing-finish',
      body: {
        id: 'gw_missing_finish',
        model: 'pro-chat',
        text: 'remote answer',
      },
      message: 'remote gateway response missing finish_reason',
    },
  ] as const;

  for (const item of cases) {
    await withGlobalFetch(
      async () =>
        new Response(JSON.stringify(item.body), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      async () => {
        const client = new CogentClient({ executionMode: 'main-thread' });
        const endpoint = client.addRemote(item.id, {
          alias: 'pro-chat',
          baseUrl: 'https://gateway.example.test',
          token: 'secret-token',
        });

        await assert.rejects(
          client.query('hello', { endpoint }).response,
          (error) =>
            error instanceof QueryError &&
            error.code === 'QUERY_FAILED' &&
            error.message === item.message
        );

        await client.close();
      }
    );
  }
});

test('remote gateway text responses reject malformed usage fields', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        JSON.stringify({
          id: 'gw_bad_usage',
          model: 'pro-chat',
          text: 'remote answer',
          finish_reason: 'stop',
          usage: {
            input_tokens: 'secret-token',
          },
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-bad-text-usage', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'usage field is not a number: input_tokens' &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway text responses require usage to be a JSON object', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        JSON.stringify({
          id: 'gw_array_usage',
          model: 'pro-chat',
          text: 'remote answer',
          finish_reason: 'stop',
          usage: [],
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-array-text-usage', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'usage must be a JSON object' &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('chat() routes explicit remote endpoint through gateway fetch', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response(
        JSON.stringify({
          id: 'gw_chat',
          model: 'pro-chat',
          text: 'remote chat',
          finish_reason: 'stop',
          usage: {
            input_tokens: 4,
            output_tokens: 2,
            total_tokens: 6,
          },
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-chat', {
        alias: 'team-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      const output = await client
        .chat([{ role: 'user', content: 'hello' }], {
          endpoint,
          maxTokens: 5,
          temperature: 0.5,
          topP: 0.9,
          stop: [],
          gatewayOptions: { seed: 9 },
        })
        .response;
      const call = calls[0];
      const body = JSON.parse(String(call.init?.body)) as Record<string, unknown>;
      const headers = call.init?.headers as Record<string, string>;

      assert.equal(output.id, 'gw_chat');
      assert.equal(output.text, 'remote chat');
      assert.equal(call.url, 'https://gateway.example.test/v1/chat');
      assert.equal(headers.Authorization, 'Bearer secret-token');
      assert.deepEqual(body, {
        model: 'team-chat',
        messages: [{ role: 'user', content: 'hello' }],
        max_tokens: 5,
        temperature: 0.5,
        top_p: 0.9,
        stop: [],
        stream: false,
        seed: 9,
      });

      await client.close();
    }
  );
});

test('embed() routes explicit remote endpoint through gateway fetch', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response(
        JSON.stringify({
          id: 'gw_embed',
          model: 'team-embed',
          embedding: [0.25, -0.5],
          usage: {
            input_tokens: 3,
            total_tokens: 3,
          },
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-embed', {
        alias: 'team-embed',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      const output = await client
        .embed('hello', {
          endpoint,
          gatewayOptions: { input_type: 'query' },
        })
        .response;
      const call = calls[0];
      const body = JSON.parse(String(call.init?.body)) as Record<string, unknown>;
      const headers = call.init?.headers as Record<string, string>;

      assert.deepEqual(output.values, [0.25, -0.5]);
      assert.equal(output.stats.inputTokens, 3);
      assert.equal(call.url, 'https://gateway.example.test/v1/embed');
      assert.equal(headers.Authorization, 'Bearer secret-token');
      assert.deepEqual(body, {
        model: 'team-embed',
        input: 'hello',
        input_type: 'query',
      });

      await client.close();
    }
  );
});

test('remote gateway embedding responses require model and embedding fields', async () => {
  const cases = [
    {
      id: 'missing-model',
      body: {
        id: 'gw_embed_missing_model',
        embedding: [0.25, -0.5],
      },
      message: 'remote gateway response missing model',
    },
    {
      id: 'missing-embedding',
      body: {
        id: 'gw_embed_missing_vector',
        model: 'team-embed',
      },
      message: 'remote gateway response missing embedding',
    },
  ] as const;

  for (const item of cases) {
    await withGlobalFetch(
      async () =>
        new Response(JSON.stringify(item.body), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      async () => {
        const client = new CogentClient({ executionMode: 'main-thread' });
        const endpoint = client.addRemote(`pro-embed-${item.id}`, {
          alias: 'team-embed',
          baseUrl: 'https://gateway.example.test',
          token: 'secret-token',
        });

        await assert.rejects(
          client.embed('hello', { endpoint }).response,
          (error) =>
            error instanceof QueryError &&
            error.code === 'QUERY_FAILED' &&
            error.message === item.message
        );

        await client.close();
      }
    );
  }
});

test('remote gateway embedding responses require a JSON object body', async () => {
  for (const [id, body] of [
    ['null', null],
    ['array', []],
  ] as const) {
    await withGlobalFetch(
      async () =>
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      async () => {
        const client = new CogentClient({ executionMode: 'main-thread' });
        const endpoint = client.addRemote(`pro-${id}-embed`, {
          alias: 'team-embed',
          baseUrl: 'https://gateway.example.test',
          token: 'secret-token',
        });

        await assert.rejects(
          client.embed('hello', { endpoint }).response,
          (error) =>
            error instanceof QueryError &&
            error.code === 'QUERY_FAILED' &&
            error.message === 'remote gateway response must be a JSON object' &&
            !error.message.includes('secret-token')
        );

        await client.close();
      }
    );
  }
});

test('remote gateway embedding responses reject malformed usage fields', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        JSON.stringify({
          id: 'gw_bad_embed_usage',
          model: 'team-embed',
          embedding: [0.25, -0.5],
          usage: {
            total_tokens: 4294967296,
          },
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-bad-embed-usage', {
        alias: 'team-embed',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.embed('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'usage field exceeds u32: total_tokens'
      );

      await client.close();
    }
  );
});

test('remote gateway embedding responses reject values outside f32 range', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        JSON.stringify({
          id: 'gw_embed_out_of_range',
          model: 'team-embed',
          embedding: [3.5e38],
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-embed-out-of-range', {
        alias: 'team-embed',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.embed('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'remote gateway embedding contains value outside f32 range'
      );

      await client.close();
    }
  );
});

test('remote browser calls reject local-only options', async () => {
  const client = new CogentClient({ executionMode: 'main-thread' });
  const endpoint = client.addRemote('pro', {
    alias: 'team-model',
    baseUrl: 'https://gateway.example.test',
    token: 'secret-token',
  });

  await assert.rejects(
    client.query('hello', { endpoint, grammar: 'root ::= "ok"' }).response,
    (error) =>
      error instanceof QueryError &&
      error.code === 'UNSUPPORTED_OPERATION' &&
      error.message === 'local text options are not valid for remote endpoints'
  );

  await assert.rejects(
    client.chat([{ role: 'user', content: 'hello' }], {
      endpoint,
      session: 'local-session',
    }).response,
    (error) =>
      error instanceof QueryError &&
      error.code === 'UNSUPPORTED_OPERATION' &&
      error.message === 'local text options are not valid for remote endpoints'
  );

  await assert.rejects(
    client.embed('hello', { endpoint, normalize: false }).response,
    (error) =>
      error instanceof QueryError &&
      error.code === 'UNSUPPORTED_OPERATION' &&
      error.message === 'local embed options are not valid for remote endpoints'
  );

  await client.close();
});

test('remote query rejects blank prompts before fetch', async () => {
  let calls = 0;
  await withGlobalFetch(
    (async () => {
      calls += 1;
      return new Response('{}', { status: 200 });
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-blank-query', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query(' \t ', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'prompt must not be empty'
      );
      assert.equal(calls, 0);

      await client.close();
    }
  );
});

test('remote chat rejects empty messages before fetch', async () => {
  let calls = 0;
  await withGlobalFetch(
    (async () => {
      calls += 1;
      return new Response('{}', { status: 200 });
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-empty-chat', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.chat([], { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'messages must not be empty'
      );
      assert.equal(calls, 0);

      await client.close();
    }
  );
});

test('remote browser text inputs validate shape before fetch', async () => {
  let calls = 0;
  await withGlobalFetch(
    (async () => {
      calls += 1;
      return new Response(
        JSON.stringify({
          id: 'gw_unexpected_input',
          model: 'pro-chat',
          text: 'unexpected',
          finish_reason: 'stop',
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-invalid-input', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query({ prompt: 7 } as unknown as QueryInput, { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'prompt must be a string'
      );
      await assert.rejects(
        client.chat({ messages: 'secret-token' } as unknown as ChatInput, { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'messages must be an array' &&
          !error.message.includes('secret-token')
      );
      await assert.rejects(
        client.chat([{ role: 'user', content: 7 as unknown as string }], { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'message content must be a string'
      );
      assert.equal(calls, 0);

      await client.close();
    }
  );
});

test('remote embed rejects blank input before fetch', async () => {
  let calls = 0;
  await withGlobalFetch(
    (async () => {
      calls += 1;
      return new Response('{}', { status: 200 });
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-blank-embed', {
        alias: 'pro-embed',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.embed(' \t ', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'input must not be empty'
      );
      assert.equal(calls, 0);

      await client.close();
    }
  );
});

test('remote browser embed input validates shape before fetch', async () => {
  let calls = 0;
  await withGlobalFetch(
    (async () => {
      calls += 1;
      return new Response('{}', { status: 200 });
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-invalid-embed-input', {
        alias: 'pro-embed',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.embed(7 as unknown as string, { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'input must be a string'
      );
      assert.equal(calls, 0);

      await client.close();
    }
  );
});

test('remote browser gatewayOptions reject local-only field names', async () => {
  const client = new CogentClient({ executionMode: 'main-thread' });
  const endpoint = client.addRemote('pro', {
    alias: 'team-model',
    baseUrl: 'https://gateway.example.test',
    token: 'secret-token',
  });

  await assert.rejects(
    client.query('hello', {
      endpoint,
      gatewayOptions: { grammar: 'root ::= "ok"' },
    }).response,
    (error) =>
      error instanceof QueryError &&
      error.code === 'QUERY_FAILED' &&
      error.message === 'gatewayOptions cannot contain local-only field: grammar'
  );

  await assert.rejects(
    client.embed('hello', { endpoint, gatewayOptions: { normalize: true } }).response,
    (error) =>
      error instanceof QueryError &&
      error.code === 'QUERY_FAILED' &&
      error.message === 'gatewayOptions cannot contain local-only field: normalize'
  );

  await client.close();
});

test('remote browser gatewayOptions must be JSON objects', async () => {
  let calls = 0;
  await withGlobalFetch(
    (async () => {
      calls += 1;
      return new Response(
        JSON.stringify({
          id: 'gw_unexpected',
          model: 'pro-chat',
          text: 'unexpected',
          finish_reason: 'stop',
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-invalid-gateway-options', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      const invalidGatewayOptions = 'secret-token' as unknown as GatewayOptions;
      await assert.rejects(
        client.query('hello', { endpoint, gatewayOptions: invalidGatewayOptions }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'gatewayOptions must be a JSON object' &&
          !error.message.includes('secret-token')
      );
      await assert.rejects(
        client.chat([{ role: 'user', content: 'hello' }], {
          endpoint,
          gatewayOptions: [] as unknown as GatewayOptions,
        }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'gatewayOptions must be a JSON object'
      );
      await assert.rejects(
        client.embed('hello', {
          endpoint,
          gatewayOptions: invalidGatewayOptions,
        }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'gatewayOptions must be a JSON object' &&
          !error.message.includes('secret-token')
      );
      assert.equal(calls, 0);

      await client.close();
    }
  );
});

test('remote browser gatewayOptions reject non-JSON values before fetch', async () => {
  let calls = 0;
  await withGlobalFetch(
    (async () => {
      calls += 1;
      return new Response(
        JSON.stringify({
          id: 'gw_unexpected_non_json_options',
          model: 'pro-chat',
          text: 'unexpected',
          finish_reason: 'stop',
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-non-json-gateway-options', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      for (const gatewayOptions of [
        { callback: () => 'provider-secret' },
        { nested: [undefined] },
        { opaque: new Map([['provider-secret', 'value']]) },
      ] as unknown as GatewayOptions[]) {
        await assert.rejects(
          client.query('hello', { endpoint, gatewayOptions }).response,
          (error) =>
            error instanceof QueryError &&
            error.code === 'QUERY_FAILED' &&
            error.message === 'gatewayOptions must contain JSON-compatible values' &&
            !error.message.includes('provider-secret')
        );
      }

      await assert.rejects(
        client.query('hello', {
          endpoint,
          gatewayOptions: { score: Number.NaN } as GatewayOptions,
        }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'gatewayOptions cannot contain non-finite numbers'
      );
      assert.equal(calls, 0);

      await client.close();
    }
  );
});

test('remote browser gatewayOptions snapshot nested values before token provider resolves', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response(
        JSON.stringify({
          id: 'gw_snapshot',
          model: 'pro-chat',
          text: 'snapshot',
          finish_reason: 'stop',
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    },
    async () => {
      const route = { provider: 'initial' };
      const stops = ['first'];
      const gatewayOptions: GatewayOptions = { route, stops };
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-snapshot-gateway-options', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        tokenProvider: () => {
          route.provider = 'mutated';
          stops.push('second');
          return 'secret-token';
        },
      });

      const output = await client.query('hello', { endpoint, gatewayOptions }).response;
      const body = JSON.parse(String(calls[0].init?.body)) as Record<string, unknown>;

      assert.equal(output.text, 'snapshot');
      assert.deepEqual(body.route, { provider: 'initial' });
      assert.deepEqual(body.stops, ['first']);

      await client.close();
    }
  );
});

test('remote browser text options validate numeric fields before fetch', async () => {
  let calls = 0;
  await withGlobalFetch(
    (async () => {
      calls += 1;
      return new Response(
        JSON.stringify({
          id: 'gw_unexpected_options',
          model: 'pro-chat',
          text: 'unexpected',
          finish_reason: 'stop',
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-invalid-text-options', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', { endpoint, maxTokens: 0 }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'max_tokens must be greater than zero'
      );
      await assert.rejects(
        client.query('hello', { endpoint, temperature: Number.POSITIVE_INFINITY }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'temperature must be finite'
      );
      await assert.rejects(
        client.query('hello', { endpoint, temperature: -0.1 }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'temperature must be greater than or equal to zero'
      );
      await assert.rejects(
        client.chat([{ role: 'user', content: 'hello' }], {
          endpoint,
          topP: Number.NaN,
        }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'top_p must be finite'
      );
      await assert.rejects(
        client.chat([{ role: 'user', content: 'hello' }], {
          endpoint,
          topP: 1.1,
        }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'top_p must be between 0 and 1'
      );
      assert.equal(calls, 0);

      await client.close();
    }
  );
});

test('remote browser text options validate stop before fetch', async () => {
  let calls = 0;
  await withGlobalFetch(
    (async () => {
      calls += 1;
      return new Response(
        JSON.stringify({
          id: 'gw_unexpected_stop',
          model: 'pro-chat',
          text: 'unexpected',
          finish_reason: 'stop',
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-invalid-stop', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', {
          endpoint,
          stop: 'secret-token' as unknown as readonly string[],
        }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'stop must be an array of strings' &&
          !error.message.includes('secret-token')
      );
      await assert.rejects(
        client.chat([{ role: 'user', content: 'hello' }], {
          endpoint,
          stop: ['END', 7 as unknown as string],
        }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'stop must be an array of strings'
      );
      assert.equal(calls, 0);

      await client.close();
    }
  );
});

test('updateRemote rotates gateway token without changing endpoint id', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response(
        JSON.stringify({
          id: 'gw_2',
          model: 'pro-chat',
          text: 'updated token',
          finish_reason: 'stop',
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'first-token',
      });

      assert.deepEqual(
        client.updateRemote('pro', {
          alias: 'pro-chat',
          baseUrl: 'https://gateway.example.test',
          token: 'second-token',
        }),
        endpoint
      );

      const output = await client.query('hello', { endpoint }).response;
      const headers = calls[0].init?.headers as Record<string, string>;

      assert.equal(output.text, 'updated token');
      assert.equal(headers.Authorization, 'Bearer second-token');

      await client.close();
    }
  );
});

test('query() streams remote gateway SSE events through token batches', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response(
        [
          'event: token',
          'data: {"text":"he","sequence":7}',
          '',
          'event: usage',
          'data: {"input_tokens":2,"output_tokens":3,"total_tokens":5}',
          '',
          'event: token',
          'data: {"text":"llo"}',
          '',
          'event: done',
          'data: {"finish_reason":"length"}',
          '',
          'data: [DONE]',
          '',
        ].join('\n'),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'x-request-id': 'req-browser-stream',
          },
        }
      );
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-stream', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        tokenProvider: () => 'rotated-token',
      });

      const batches: TokenBatch[] = [];
      const run = client.query('hello', { endpoint, emitTokens: true });
      const tokens = (async () => {
        for await (const batch of run.tokens) {
          batches.push(batch);
        }
      })();
      const output = await run.response;
      await tokens;
      const body = JSON.parse(String(calls[0].init?.body)) as Record<string, unknown>;
      const headers = calls[0].init?.headers as Record<string, string>;

      assert.equal(calls[0].url, 'https://gateway.example.test/v1/query');
      assert.equal(headers.Authorization, 'Bearer rotated-token');
      assert.equal(body.stream, true);
      assert.equal(output.id, 'req-browser-stream');
      assert.equal(output.text, 'hello');
      assert.equal(output.finishReason, 'length');
      assert.equal(output.stats.inputTokens, 2);
      assert.equal(output.stats.outputTokens, 3);
      assert.deepEqual(
        batches.map((batch) => ({
          requestId: batch.requestId,
          sequenceStart: batch.sequenceStart,
          text: batch.text,
        })),
        [
          { requestId: 'req-browser-stream', sequenceStart: 7, text: 'he' },
          { requestId: 'req-browser-stream', sequenceStart: 8, text: 'llo' },
        ]
      );

      await client.close();
    }
  );
});

test('remote gateway streams reject malformed usage events', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        [
          'event: usage',
          'data: {"output_tokens":-1}',
          '',
          'event: done',
          'data: {"finish_reason":"stop"}',
          '',
        ].join('\n'),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'x-request-id': 'req-bad-stream-usage',
          },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-bad-stream-usage', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'usage field is not a number: output_tokens' &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway streams require JSON object payloads', async () => {
  await withGlobalFetch(
    async () =>
      new Response(['event: token', 'data: []', ''].join('\n'), {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'x-request-id': 'req-array-stream-payload',
        },
      }),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-array-stream-payload', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'gateway stream payload must be a JSON object' &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway streams require a terminal done event', async () => {
  await withGlobalFetch(
    async () =>
      new Response(['event: token', 'data: {"text":"partial","sequence":0}', ''].join('\n'), {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'x-request-id': 'req-truncated-stream',
        },
      }),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-truncated-stream', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });
      const iterator = run.tokens[Symbol.asyncIterator]();

      const first = await iterator.next();
      assert.equal(first.done, false);
      assert.equal(first.value.text, 'partial');

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'gateway stream ended before done event' &&
          !error.message.includes('secret-token')
      );
      assert.deepEqual(await iterator.next(), { done: true, value: undefined });

      await client.close();
    }
  );
});

test('remote gateway streams require finishReason in done events', async () => {
  await withGlobalFetch(
    async () =>
      new Response(['event: done', 'data: {}', ''].join('\n'), {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'x-request-id': 'req-missing-finish',
        },
      }),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-missing-stream-finish', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'gateway stream done event missing finish_reason'
      );

      await client.close();
    }
  );
});

test('remote gateway streams reject events after done', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        [
          'event: done',
          'data: {"finish_reason":"stop"}',
          '',
          'event: token',
          'data: {"text":"late secret-token","sequence":0}',
          '',
        ].join('\n'),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'x-request-id': 'req-late-stream-event',
          },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-late-stream-event', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'gateway stream event received after done event' &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway streams respect abort signals after response headers', async () => {
  const encoder = new TextEncoder();
  let fetchSignal: AbortSignal | undefined;

  await withGlobalFetch(
    (async (_input, init) => {
      fetchSignal = init?.signal as AbortSignal | undefined;
      let pullCount = 0;
      return new Response(
        new ReadableStream<Uint8Array>({
          pull(controller) {
            if (fetchSignal?.aborted) {
              controller.error(new Error('stream aborted secret-token'));
              return;
            }

            pullCount += 1;
            if (pullCount === 1) {
              controller.enqueue(
                encoder.encode('event: token\ndata: {"text":"partial","sequence":0}\n\n')
              );
              return;
            }

            return new Promise<void>((resolve) => {
              setTimeout(() => {
                if (fetchSignal?.aborted) {
                  controller.error(new Error('stream aborted secret-token'));
                } else {
                  controller.enqueue(
                    encoder.encode('event: done\ndata: {"finish_reason":"stop"}\n\n')
                  );
                  controller.close();
                }
                resolve();
              }, 20);
            });
          },
        }),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'x-request-id': 'req-stream-abort',
          },
        }
      );
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-stream-abort', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const controller = new AbortController();
      const run = client.query('hello', {
        endpoint,
        emitTokens: true,
        signal: controller.signal,
      });

      const iterator = run.tokens[Symbol.asyncIterator]();
      const first = await iterator.next();
      assert.equal(first.done, false);
      assert.equal(first.value.text, 'partial');

      controller.abort();

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'remote gateway request aborted' &&
          !error.message.includes('secret-token')
      );
      assert.equal(fetchSignal?.aborted, true);
      assert.deepEqual(await iterator.next(), { done: true, value: undefined });

      await client.close();
    }
  );
});

test('remote gateway stream timeout is an idle deadline', async () => {
  const encoder = new TextEncoder();
  let fetchSignal: AbortSignal | undefined;

  await withGlobalFetch(
    (async (_input, init) => {
      fetchSignal = init?.signal as AbortSignal | undefined;
      let pullCount = 0;
      return new Response(
        new ReadableStream<Uint8Array>({
          pull(controller) {
            pullCount += 1;
            if (fetchSignal?.aborted) {
              controller.error(new Error('stream timed out secret-token'));
              return;
            }
            if (pullCount === 1) {
              controller.enqueue(
                encoder.encode('event: token\ndata: {"text":"a","sequence":0}\n\n')
              );
              return;
            }

            return new Promise<void>((resolve) => {
              setTimeout(() => {
                if (fetchSignal?.aborted) {
                  controller.error(new Error('stream timed out secret-token'));
                } else if (pullCount === 2) {
                  controller.enqueue(
                    encoder.encode('event: token\ndata: {"text":"b","sequence":1}\n\n')
                  );
                } else {
                  controller.enqueue(
                    encoder.encode('event: done\ndata: {"finish_reason":"stop"}\n\n')
                  );
                  controller.close();
                }
                resolve();
              }, 35);
            });
          },
        }),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'x-request-id': 'req-stream-idle-timeout',
          },
        }
      );
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-stream-idle-timeout', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
        timeoutMs: 50,
      });
      const run = client.query('hello', { endpoint, emitTokens: true });
      const output = await run.response;

      assert.equal(output.text, 'ab');
      assert.equal(output.finishReason, 'stop');
      assert.equal(fetchSignal?.aborted, false);

      await client.close();
    }
  );
});

test('query() maps remote gateway stream error events', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        ['event: error', 'data: {"error":{"message":"not allowed","code":"permission_error"}}', ''].join(
          '\n'
        ),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'x-request-id': 'req-browser-error',
          },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-error', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'not allowed' &&
          error.gatewayCode === 'permission_error' &&
          error.requestId === 'req-browser-error'
      );

      await client.close();
    }
  );
});

test('query() maps remote gateway stream error event types', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        ['event: error', 'data: {"error":{"message":"slow down","type":"rate_limited"}}', ''].join(
          '\n'
        ),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'x-request-id': 'req-browser-type-error',
          },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-type-error', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'slow down' &&
          error.gatewayCode === 'rate_limited' &&
          error.requestId === 'req-browser-type-error'
      );

      await client.close();
    }
  );
});

test('remote gateway stream parser rejects oversized SSE events', async () => {
  const oversizedText = 'x'.repeat(1_048_577);
  await withGlobalFetch(
    async () =>
      new Response(['event: token', `data: {"text":"${oversizedText}"}`].join('\n'), {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'x-request-id': 'req-browser-oversized-stream',
        },
      }),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-oversized-stream', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'gateway stream event exceeded buffer limit' &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway stream error events redact bearer token echoes', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        [
          'event: error',
          'data: {"error":{"message":"provider echoed secret-token","code":"authentication"}}',
          '',
        ].join('\n'),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'x-request-id': 'req-secret-token',
          },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-stream-redaction', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'provider echoed [redacted]' &&
          error.gatewayCode === 'authentication' &&
          error.requestId === 'req-[redacted]' &&
          !String(error.requestId).includes('secret-token') &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway stream protocol errors redact bearer token echoes', async () => {
  await withGlobalFetch(
    async () =>
      new Response(['event: secret-token', 'data: not-json-secret-token', ''].join('\n'), {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'x-request-id': 'req-stream-protocol',
        },
      }),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-stream-protocol', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'unsupported gateway stream event: [redacted]' &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway stream invalid JSON does not keep secret-bearing causes', async () => {
  await withGlobalFetch(
    async () =>
      new Response(['event: token', 'data: not-json-secret-token', ''].join('\n'), {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'x-request-id': 'req-stream-invalid-json',
        },
      }),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-stream-invalid-json', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const run = client.query('hello', { endpoint, emitTokens: true });

      await assert.rejects(
        run.response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'invalid gateway stream JSON payload' &&
          error.cause == null &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway fetch failures do not expose bearer tokens', async () => {
  await withGlobalFetch(
    async (_input, init) => {
      const headers = init?.headers as Record<string, string>;
      throw new Error(`transport failed with ${headers.Authorization}`);
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-fetch-failure', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'remote gateway request failed' &&
          !error.message.includes('secret-token') &&
          error.cause == null
      );

      await client.close();
    }
  );
});

test('remote gateway calls respect pre-aborted signals before fetch', async () => {
  let calls = 0;
  await withGlobalFetch(
    (async () => {
      calls += 1;
      return new Response(
        JSON.stringify({
          id: 'resp',
          model: 'pro-chat',
          text: 'should not run',
          finish_reason: 'stop',
        }),
        { status: 200 }
      );
    }) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-aborted', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });
      const controller = new AbortController();
      controller.abort();

      await assert.rejects(
        client.query('hello', { endpoint, signal: controller.signal }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'remote gateway request aborted'
      );
      assert.equal(calls, 0);

      await client.close();
    }
  );
});

test('remote gateway timeout aborts fetch without exposing bearer tokens', async () => {
  let aborted = false;
  await withGlobalFetch(
    (async (_input, init) =>
      new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal as AbortSignal | undefined;
        signal?.addEventListener(
          'abort',
          () => {
            aborted = true;
            reject(new Error('fetch aborted secret-token'));
          },
          { once: true }
        );
      })) as typeof fetch,
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-timeout', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
        timeoutMs: 1,
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'remote gateway request timed out' &&
          !error.message.includes('secret-token')
      );
      assert.equal(aborted, true);

      await client.close();
    }
  );
});

test('remote gateway token provider failures do not expose bearer tokens', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response('{}', { status: 200 });
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-token-failure', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        tokenProvider: () => {
          throw new Error('failed to mint secret-token');
        },
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'remote gateway token provider failed' &&
          !error.message.includes('secret-token') &&
          error.cause == null
      );
      assert.equal(calls.length, 0);

      await client.close();
    }
  );
});

test('remote gateway token provider is bounded by request timeout', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response('{}', { status: 200 });
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      try {
        const endpoint = client.addRemote('pro-token-timeout', {
          alias: 'pro-chat',
          baseUrl: 'https://gateway.example.test',
          tokenProvider: () => new Promise<string>(() => {}),
          timeoutMs: 1,
        });

        const result = await Promise.race([
          client.query('hello', { endpoint }).response.then(
            () => 'resolved',
            (error) =>
              error instanceof QueryError &&
              error.code === 'QUERY_FAILED' &&
              error.message === 'remote gateway request timed out'
                ? 'timed-out'
                : 'wrong-error'
          ),
          new Promise<'still-pending'>((resolve) =>
            setTimeout(() => resolve('still-pending'), 30)
          ),
        ]);

        assert.equal(result, 'timed-out');
        assert.equal(calls.length, 0);
      } finally {
        await client.close();
      }
    }
  );
});

test('remote gateway token provider respects abort signals before fetch', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response('{}', { status: 200 });
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      try {
        const endpoint = client.addRemote('pro-token-abort', {
          alias: 'pro-chat',
          baseUrl: 'https://gateway.example.test',
          tokenProvider: () => new Promise<string>(() => {}),
        });
        const controller = new AbortController();
        const response = client.query('hello', { endpoint, signal: controller.signal }).response;

        controller.abort();

        await assert.rejects(
          response,
          (error) =>
            error instanceof QueryError &&
            error.code === 'QUERY_FAILED' &&
            error.message === 'remote gateway request aborted'
        );
        assert.equal(calls.length, 0);
      } finally {
        await client.close();
      }
    }
  );
});

test('remote gateway token provider must return a string token', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response('{}', { status: 200 });
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-token-type', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        tokenProvider: () => 7,
      } as unknown as RemoteGatewayConfig);

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'remote gateway token must be a string'
      );
      assert.equal(calls.length, 0);

      await client.close();
    }
  );
});

test('remote gateway token provider rejects whitespace tokens before fetch', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response('{}', { status: 200 });
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-token-whitespace', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        tokenProvider: () => 'secret token',
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'remote gateway token must not contain whitespace' &&
          !error.message.includes('secret token')
      );
      assert.equal(calls.length, 0);

      await client.close();
    }
  );
});

test('remote gateway HTTP errors redact bearer token echoes', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        JSON.stringify({
          error: {
            code: 'authentication',
            message: 'invalid bearer secret-token',
          },
        }),
        {
          status: 401,
          headers: {
            'Content-Type': 'application/json',
            'x-request-id': 'req-secret-token',
          },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-http-error', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'invalid bearer [redacted]' &&
          error.gatewayCode === 'authentication' &&
          error.requestId === 'req-[redacted]' &&
          !String(error.requestId).includes('secret-token') &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway body errors redact bearer token echoes', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        JSON.stringify({
          error: {
            code: 'authentication_secret-token',
            message: 'invalid bearer secret-token',
            details: ['secret-token'],
          },
        }),
        {
          status: 200,
          headers: {
            'Content-Type': 'application/json',
            'x-request-id': 'req-secret-token',
          },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-body-error', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.message === 'invalid bearer [redacted]' &&
          error.gatewayCode === 'authentication_[redacted]' &&
          error.requestId === 'req-[redacted]' &&
          !String(error.requestId).includes('secret-token') &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway HTTP error bodies are capped', async () => {
  await withGlobalFetch(
    async () =>
      new Response('x'.repeat((1 << 20) + 1), {
        status: 500,
        statusText: 'Internal Server Error',
        headers: {
          'x-request-id': 'req-huge-error',
        },
      }),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-huge-error', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.status === 500 &&
          error.message === 'remote gateway error response exceeded body limit' &&
          error.requestId === 'req-huge-error' &&
          !error.message.includes('secret-token')
      );

      await client.close();
    }
  );
});

test('remote gateway HTTP errors expose structured gateway metadata', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        JSON.stringify({
          error: {
            code: 'rate_limited',
            message: 'slow down',
          },
        }),
        {
          status: 429,
          headers: {
            'Content-Type': 'application/json',
            'retry-after-ms': '1500',
            'x-request-id': 'req-browser-rate-limit',
          },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-metadata-error', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.status === 429 &&
          error.gatewayCode === 'rate_limited' &&
          error.requestId === 'req-browser-rate-limit' &&
          error.retryAfterMs === 1500 &&
          error.message === 'slow down'
      );

      await client.close();
    }
  );
});

test('remote gateway HTTP errors expose error type metadata', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        JSON.stringify({
          error: {
            type: 'rate_limited',
            message: 'slow down',
          },
        }),
        {
          status: 429,
          headers: {
            'Content-Type': 'application/json',
            'retry-after-ms': '1500',
            'x-request-id': 'req-browser-type-rate-limit',
          },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-type-metadata-error', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.status === 429 &&
          error.gatewayCode === 'rate_limited' &&
          error.requestId === 'req-browser-type-rate-limit' &&
          error.retryAfterMs === 1500 &&
          error.message === 'slow down'
      );

      await client.close();
    }
  );
});

test('remote gateway HTTP errors parse retry-after seconds', async () => {
  await withGlobalFetch(
    async () =>
      new Response(
        JSON.stringify({
          error: {
            code: 'rate_limited',
            message: 'slow down',
          },
        }),
        {
          status: 429,
          headers: {
            'Content-Type': 'application/json',
            'retry-after': '2',
          },
        }
      ),
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = client.addRemote('pro-retry-seconds', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        token: 'secret-token',
      });

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.code === 'QUERY_FAILED' &&
          error.retryAfterMs === 2000
      );

      await client.close();
    }
  );
});

test('addRemote requires HTTPS except loopback gateway URLs', async () => {
  const client = new CogentClient({ executionMode: 'main-thread' });

  assert.throws(
    () =>
      client.addRemote('public-http', {
        alias: 'pro-chat',
        baseUrl: 'http://gateway.example.test',
        token: 'secret-token',
      }),
    (error) =>
      error instanceof QueryError &&
      error.code === 'QUERY_FAILED' &&
      error.message.includes('must use HTTPS')
  );
  assert.throws(
    () =>
      client.addRemote('relative', {
        alias: 'pro-chat',
        baseUrl: '/gateway',
        token: 'secret-token',
      }),
    (error) =>
      error instanceof QueryError &&
      error.code === 'QUERY_FAILED' &&
      error.message.includes('baseUrl is invalid')
  );
  assert.throws(
    () =>
      client.addRemote('malformed-secret-url', {
        alias: 'pro-chat',
        baseUrl: 'https://gateway-secret invalid',
        token: 'secret-token',
      }),
    (error) =>
      error instanceof QueryError &&
      error.code === 'QUERY_FAILED' &&
      error.message === 'remote gateway baseUrl is invalid' &&
      error.cause == null &&
      !error.message.includes('gateway-secret')
  );
  assert.throws(
    () =>
      client.addRemote('loopback-prefix-hostname', {
        alias: 'pro-chat',
        baseUrl: 'http://127.evil.example',
        token: 'secret-token',
      }),
    (error) =>
      error instanceof QueryError &&
      error.code === 'QUERY_FAILED' &&
      error.message.includes('must use HTTPS')
  );
  assert.throws(
    () =>
      client.addRemote('userinfo', {
        alias: 'pro-chat',
        baseUrl: 'https://user:gateway-secret@gateway.example.test',
        token: 'secret-token',
      }),
    (error) =>
      error instanceof QueryError &&
      error.code === 'QUERY_FAILED' &&
      error.message === 'remote gateway baseUrl must not include userinfo' &&
      !error.message.includes('gateway-secret')
  );
  for (const [id, baseUrl] of [
    ['query', 'https://gateway.example.test/v1?token=gateway-secret'],
    ['fragment', 'https://gateway.example.test/v1#gateway-secret'],
  ] as const) {
    assert.throws(
      () =>
        client.addRemote(id, {
          alias: 'pro-chat',
          baseUrl,
          token: 'secret-token',
        }),
      (error) =>
        error instanceof QueryError &&
        error.code === 'QUERY_FAILED' &&
        error.message === 'remote gateway baseUrl must not include query or fragment' &&
        !error.message.includes('gateway-secret')
    );
  }

  assert.deepEqual(
    client.addRemote('localhost', {
      alias: 'local-gateway',
      baseUrl: 'http://localhost:8080',
      token: 'secret-token',
    }),
    { kind: 'remote', id: 'localhost' }
  );
  assert.deepEqual(
    client.addRemote('ipv4-loopback', {
      alias: 'local-gateway',
      baseUrl: 'http://127.10.0.1:8080',
      token: 'secret-token',
    }),
    { kind: 'remote', id: 'ipv4-loopback' }
  );
  assert.deepEqual(
    client.addRemote('ipv6-loopback', {
      alias: 'local-gateway',
      baseUrl: 'http://[::1]:8080',
      token: 'secret-token',
    }),
    { kind: 'remote', id: 'ipv6-loopback' }
  );

  await client.close();
});

test('addRemote validates browser remote config runtime field types', async () => {
  const client = new CogentClient({ executionMode: 'main-thread' });

  const valid = {
    alias: 'pro-chat',
    baseUrl: 'https://gateway.example.test',
    token: 'secret-token',
  };
  const cases: Array<{
    readonly id: string;
    readonly config: unknown;
    readonly message: string;
  }> = [
    {
      id: 'null-config',
      config: null,
      message: 'remote gateway config must be an object',
    },
    {
      id: 'bad-alias',
      config: { ...valid, alias: 7 },
      message: 'remote alias must be a string',
    },
    {
      id: 'blank-alias',
      config: { ...valid, alias: '   ' },
      message: 'remote alias must not be empty',
    },
    {
      id: 'whitespace-alias',
      config: { ...valid, alias: ' pro-chat ' },
      message: 'remote alias must not contain surrounding whitespace',
    },
    {
      id: 'bad-base-url',
      config: { ...valid, baseUrl: 7 },
      message: 'remote gateway baseUrl must be a string',
    },
    {
      id: 'blank-base-url',
      config: { ...valid, baseUrl: '   ' },
      message: 'remote gateway baseUrl must not be empty',
    },
    {
      id: 'whitespace-base-url',
      config: { ...valid, baseUrl: ' https://gateway.example.test ' },
      message: 'remote gateway baseUrl must not contain surrounding whitespace',
    },
    {
      id: 'bad-token',
      config: { ...valid, token: 7 },
      message: 'remote gateway token must be a string',
    },
    {
      id: 'empty-token',
      config: { ...valid, token: '' },
      message: 'remote gateway token must not be empty',
    },
    {
      id: 'blank-token',
      config: { ...valid, token: '   ' },
      message: 'remote gateway token must not be empty',
    },
    {
      id: 'whitespace-token',
      config: { ...valid, token: 'secret token' },
      message: 'remote gateway token must not contain whitespace',
    },
    {
      id: 'bad-token-provider',
      config: {
        alias: 'pro-chat',
        baseUrl: 'https://gateway.example.test',
        tokenProvider: 'secret-token',
      },
      message: 'remote gateway tokenProvider must be a function',
    },
    {
      id: 'bad-timeout',
      config: { ...valid, timeoutMs: '1000' },
      message: 'remote gateway timeoutMs must be positive',
    },
  ];

  assert.throws(
    () => client.addRemote(7 as unknown as string, valid),
    (error) =>
      error instanceof QueryError &&
      error.code === 'QUERY_FAILED' &&
      error.message === 'remote id must be a string'
  );
  assert.throws(
    () => client.addRemote(' pro ', valid),
    (error) =>
      error instanceof QueryError &&
      error.code === 'QUERY_FAILED' &&
      error.message === 'remote id must not contain surrounding whitespace'
  );

  for (const item of cases) {
    assert.throws(
      () => client.addRemote(item.id, item.config as RemoteGatewayConfig),
      (error) =>
        error instanceof QueryError &&
        error.code === 'QUERY_FAILED' &&
        error.message === item.message &&
        !error.message.includes('secret-token')
    );
  }

  await client.close();
});

test('addRemote rejects direct-provider fields in browser config', async () => {
  const client = new CogentClient({ executionMode: 'main-thread' });
  const blockedFields: Array<{ readonly field: string; readonly value: unknown }> = [
    { field: 'apiKey', value: 'provider-secret' },
    { field: 'providerApiKey', value: 'provider-secret' },
    { field: 'providerBaseUrl', value: 'https://api.provider.example' },
    { field: 'headers', value: { Authorization: 'Bearer provider-secret' } },
    { field: 'authorization', value: 'Bearer provider-secret' },
  ];

  for (const { field, value } of blockedFields) {
    const config = {
      alias: 'pro-chat',
      baseUrl: 'https://gateway.example.test',
      token: 'secret-token',
      [field]: value,
    } as unknown as RemoteGatewayConfig;

    assert.throws(
      () => client.addRemote(`blocked-${field}`, config),
      (error) =>
        error instanceof QueryError &&
        error.code === 'QUERY_FAILED' &&
        error.message === `unsupported remote gateway config field: ${field}` &&
        !error.message.includes('provider-secret')
    );
  }

  await client.close();
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
      temperature: 0.2,
      topP: 0.8,
      stop: ['END'],
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
    assert.equal(chat.options.temperature, 0.2);
    assert.equal(chat.options.topP, 0.8);
    assert.deepEqual(chat.options.stop, ['END']);

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
