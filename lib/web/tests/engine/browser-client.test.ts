import assert from 'node:assert/strict';
import test from 'node:test';

import {
  Endpoint,
  QueryError,
  SippClient,
} from '../../src/index.js';
import type {
  GatewayEndpointOptions,
  ModelInfo,
  TokenBatch,
} from '../../src/index.js';
import {
  FakeModelWorker,
  waitForModelWorker,
  waitForWorkerMessage,
  withFakeModelWorker,
} from '../support/fake-model-worker.js';

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

function textResponse(text: string): Response {
  return Response.json({
    id: 'response-1',
    model: 'developer-model',
    text,
    finish_reason: 'stop',
    usage: {
      input_tokens: 2,
      output_tokens: 3,
      total_tokens: 5,
    },
  });
}

function gateway(overrides: Partial<GatewayEndpointOptions> = {}): Endpoint {
  return Endpoint.gateway({
    target: 'developer-model',
    baseUrl: 'https://inference.example.test',
    authentication: { kind: 'bearer', value: 'endpoint-secret' },
    ...overrides,
  });
}

function localModel(id: string) {
  return {
    id,
    name: `${id}.gguf`,
    bytes: 1,
    modality: 'text' as const,
    status: 'ready' as const,
  };
}

function loadedModel(id: string): ModelInfo {
  return {
    ...localModel(id),
    source: 'remote',
    assetFingerprint: `asset-${id}`,
    createdAt: '2026-01-01T00:00:00.000Z',
    updatedAt: '2026-01-01T00:00:00.000Z',
    loaded: true,
    chatTemplate: null,
    bosText: '',
    eosText: '',
    mediaMarker: null,
    capabilities: null,
  };
}

function workerClient(): SippClient {
  return new SippClient({
    workerUrl: '/worker.js',
    wasmThreading: 'single-thread',
    moduleUrl: 'https://example.test/runtime.js',
    wasmUrl: 'https://example.test/runtime.wasm',
  });
}

test('SippClient exposes typed inference and endpoint registration', async () => {
  assert.deepEqual(Object.keys(Endpoint), ['local', 'gateway', 'provider']);
  const client = new SippClient();

  assert.equal(typeof client.add, 'function');
  assert.equal(typeof client.remove, 'function');
  assert.equal(typeof client.query, 'function');
  assert.equal(typeof client.chat, 'function');
  assert.equal(typeof client.embed, 'function');
  assert.equal(typeof client.listen, 'function');
  assert.equal(typeof client.speak, 'function');
  assert.equal(typeof client.models.add, 'function');
  assert.equal(typeof client.models.list, 'function');
  assert.equal(typeof client.models.remove, 'function');

  await client.close();
});

test('model sources reject empty, mixed, and unsupported inputs before storage access', async () => {
  const client = new SippClient();

  await assert.rejects(client.models.add([]), { code: 'INVALID_MODEL_SOURCE' });
  await assert.rejects(
    client.models.add([new File(['model'], 'model.gguf'), 'https://models.test/model.gguf']),
    { code: 'INVALID_MODEL_SOURCE' }
  );
  await assert.rejects(client.models.add(['ftp://models.test/model.gguf']), {
    code: 'INVALID_MODEL_SOURCE',
  });

  await client.close();
});

test('gateway query uses custom routes, authentication, headers, and extra fields', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return textResponse('custom route response');
    },
    async () => {
      const client = new SippClient();
      const endpoint = await client.add(
        'custom-http',
        gateway({
          routes: {
            query: '/generate',
            chat: '/conversation',
            embed: '/vectorize',
          },
          staticHeaders: { 'x-tenant': 'developer' },
          protocolOptions: { profile: 'default', region: 'east' },
        })
      );

      const response = await client.query('hello', {
        endpoint,
        maxTokens: 12,
        extra: { profile: 'request', seed: 7 },
      }).response;

      assert.deepEqual(Object.keys(endpoint), []);
      assert.equal(response.text, 'custom route response');
      assert.equal(response.stats.inputTokens, 2);
      assert.equal(response.stats.outputTokens, 3);
      assert.equal(calls[0].url, 'https://inference.example.test/generate');
      const headers = calls[0].init?.headers as Record<string, string>;
      assert.equal(headers.Authorization, 'Bearer endpoint-secret');
      assert.equal(headers['x-tenant'], 'developer');
      assert.deepEqual(JSON.parse(String(calls[0].init?.body)), {
        model: 'developer-model',
        prompt: 'hello',
        max_tokens: 12,
        stream: false,
        profile: 'request',
        region: 'east',
        seed: 7,
      });

      await client.close();
    }
  );
});

test('gateway chat and embed preserve typed capabilities', async () => {
  const calls: Array<{ url: string; body: Record<string, unknown> }> = [];
  await withGlobalFetch(
    async (input, init) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      calls.push({ url: String(input), body });
      if (String(input).endsWith('/embed-custom')) {
        return Response.json({
          id: 'embedding-1',
          model: 'developer-model',
          embedding: [0.25, 0.75],
        });
      }
      return textResponse('chat response');
    },
    async () => {
      const client = new SippClient();
      const endpoint = await client.add(
        'typed-http',
        gateway({
          routes: {
            query: '/query-custom',
            chat: '/chat-custom',
            embed: '/embed-custom',
          },
          authentication: { kind: 'none' },
        })
      );

      const chat = await client.chat(
        [{ role: 'user', content: 'hello' }],
        { endpoint, extra: { response_style: 'brief' } }
      ).response;
      const embed = await client.embed('vector input', {
        endpoint,
        extra: { input_type: 'query' },
      }).response;

      assert.equal(chat.text, 'chat response');
      assert.deepEqual(embed.values, [0.25, 0.75]);
      assert.equal(calls[0].url, 'https://inference.example.test/chat-custom');
      assert.deepEqual(calls[0].body, {
        model: 'developer-model',
        messages: [{ role: 'user', content: 'hello' }],
        stream: false,
        response_style: 'brief',
      });
      assert.equal(calls[1].url, 'https://inference.example.test/embed-custom');
      assert.deepEqual(calls[1].body, {
        model: 'developer-model',
        input: 'vector input',
        input_type: 'query',
      });

      await client.close();
    }
  );
});

test('gateway streaming exposes token batches and terminal response', async () => {
  const streamBody = [
    'event: token',
    'data: {"text":"hello ","sequence":0}',
    '',
    'event: token',
    'data: {"text":"world","sequence":6}',
    '',
    'event: usage',
    'data: {"input_tokens":1,"output_tokens":2,"total_tokens":3}',
    '',
    'event: done',
    'data: {"finish_reason":"stop"}',
    '',
    '',
  ].join('\n');

  await withGlobalFetch(
    async () =>
      new Response(streamBody, {
        status: 200,
        headers: {
          'content-type': 'text/event-stream',
          'x-request-id': 'stream-request',
        },
      }),
    async () => {
      const client = new SippClient();
      const endpoint = await client.add(
        'stream-http',
        gateway({ authentication: { kind: 'none' } })
      );
      const run = client.query('hello', { endpoint, emitTokens: true });
      const batches: TokenBatch[] = [];
      for await (const batch of run.tokens) {
        batches.push(batch);
      }
      const response = await run.response;

      assert.deepEqual(
        batches.map((batch) => batch.text),
        ['hello ', 'world']
      );
      assert.equal(response.text, 'hello world');
      assert.equal(response.stats.inputTokens, 1);
      assert.equal(response.stats.outputTokens, 2);

      await client.close();
    }
  );
});

test('gateway supports custom authentication headers from async providers', async () => {
  let authorization = '';
  await withGlobalFetch(
    async (_input, init) => {
      authorization = (init?.headers as Record<string, string>)['x-api-key'];
      return textResponse('authenticated');
    },
    async () => {
      const client = new SippClient();
      const endpoint = await client.add(
        'header-http',
        gateway({
          authentication: {
            kind: 'header',
            headerName: 'x-api-key',
            valueProvider: async () => 'rotated-secret',
          },
        })
      );

      await client.query('hello', { endpoint }).response;
      assert.equal(authorization, 'rotated-secret');

      await client.close();
    }
  );
});

test('gateway errors expose protocol metadata without leaking secrets', async () => {
  await withGlobalFetch(
    async () =>
      Response.json(
        {
          error: {
            code: 'admission',
            message: 'rejected endpoint-secret',
          },
        },
        {
          status: 429,
          headers: {
            'retry-after-ms': '250',
            'x-request-id': 'request-endpoint-secret',
          },
        }
      ),
    async () => {
      const client = new SippClient();
      const endpoint = await client.add('error-http', gateway());

      await assert.rejects(
        client.query('hello', { endpoint }).response,
        (error) =>
          error instanceof QueryError &&
          error.status === 429 &&
          error.protocolCode === 'admission' &&
          error.retryAfterMs === 250 &&
          error.requestId === 'request-[redacted]' &&
          error.message === 'rejected [redacted]'
      );

      await client.close();
    }
  );
});

test('gateway configuration rejects invalid and unknown fields', async () => {
  const client = new SippClient();

  await assert.rejects(
    client.add(
      'invalid-url',
      gateway({ baseUrl: 'http://public.example.test' })
    ),
    (error) =>
      error instanceof QueryError &&
      error.message ===
        'gateway endpoint baseUrl must use HTTPS unless it targets loopback'
  );
  await assert.rejects(
    client.add(
      'unknown-field',
      Endpoint.gateway({
        target: 'developer-model',
        baseUrl: 'https://inference.example.test',
        authentication: { kind: 'none' },
        policy: 'application-owned',
      } as GatewayEndpointOptions)
    ),
    (error) =>
      error instanceof QueryError &&
      error.message === 'unsupported gateway endpoint field: policy'
  );

  await client.close();
});

test('local endpoints retain their managed model identity opaquely', () => {
  const endpoint = Endpoint.local({
    id: 'model-a',
    name: 'Model A',
    bytes: 1,
    modality: 'text',
    status: 'ready',
  }, { observability: 'runtime' });

  assert.deepEqual(Object.keys(endpoint), []);
});

test('local endpoint publication waits for activation and failed replacement leaves no route', async () => {
  await withFakeModelWorker(async () => {
    const client = workerClient();
    const firstAdd = client.add('local', Endpoint.local(localModel('first')));
    const firstWorker = await waitForModelWorker(0);
    const firstLoad = await waitForWorkerMessage(firstWorker, 'models-load');
    assert.throws(() => client.query('before publish'), { code: 'MODEL_NOT_FOUND' });
    firstWorker.respond({
      kind: 'resolve',
      callId: firstLoad.callId,
      value: loadedModel('first'),
    });
    const endpoint = await firstAdd;

    const replacement = client.add('local', Endpoint.local(localModel('broken')));
    const shutdown = await waitForWorkerMessage(firstWorker, 'shutdown');
    assert.throws(() => client.query('during replacement', { endpoint }), {
      code: 'MODEL_NOT_FOUND',
    });
    assert.equal(FakeModelWorker.instances.length, 1);

    firstWorker.respond({ kind: 'resolve', callId: shutdown.callId });
    const replacementWorker = await waitForModelWorker(1);
    const replacementLoad = await waitForWorkerMessage(replacementWorker, 'models-load');
    replacementWorker.respond({
      kind: 'reject',
      callId: replacementLoad.callId,
      message: 'native activation failed',
    });
    await assert.rejects(replacement, /native activation failed/u);
    assert.equal(replacementWorker.terminated, true);
    assert.throws(() => client.query('after failed replacement', { endpoint }), {
      code: 'MODEL_NOT_FOUND',
    });
    await client.close();
  });
});

test('cross-kind replacement unpublishes local before worker shutdown completes', async () => {
  await withFakeModelWorker(async () => {
    const client = workerClient();
    const localAdd = client.add('shared', Endpoint.local(localModel('first')));
    const worker = await waitForModelWorker(0);
    const load = await waitForWorkerMessage(worker, 'models-load');
    worker.respond({ kind: 'resolve', callId: load.callId, value: loadedModel('first') });
    const local = await localAdd;

    const gatewayAdd = client.add(
      'shared',
      gateway({ authentication: { kind: 'none' } })
    );
    const shutdown = await waitForWorkerMessage(worker, 'shutdown');
    assert.throws(() => client.query('during replacement', { endpoint: local }), {
      code: 'MODEL_NOT_FOUND',
    });
    worker.respond({ kind: 'resolve', callId: shutdown.callId });

    const remote = await gatewayAdd;
    await withGlobalFetch(
      async () => textResponse('remote response'),
      async () => {
        assert.equal((await client.query('remote', { endpoint: remote }).response).text, 'remote response');
      }
    );
    await client.close();
  });
});

test('gateway endpoints reject local-only inference options', async () => {
  const client = new SippClient();
  const endpoint = await client.add(
    'gateway-options',
    gateway({ authentication: { kind: 'none' } })
  );

  await assert.rejects(
    client.query('hello', { endpoint, grammar: 'root ::= "ok"' }).response,
    (error) =>
      error instanceof QueryError &&
      error.code === 'UNSUPPORTED_OPERATION' &&
      error.message === 'local text options are not valid for gateway endpoints'
  );
  await assert.rejects(
    client.chat([{ role: 'user', content: 'hello' }], { endpoint, contextKey: 'local' }).response,
    (error) =>
      error instanceof QueryError &&
      error.code === 'UNSUPPORTED_OPERATION' &&
      error.message === 'local text options are not valid for gateway endpoints'
  );
  await assert.rejects(
    client.embed('hello', { endpoint, normalize: true }).response,
    (error) =>
      error instanceof QueryError &&
      error.code === 'UNSUPPORTED_OPERATION' &&
      error.message === 'local embed options are not valid for gateway endpoints'
  );

  await client.close();
});
