import assert from 'node:assert/strict';
import test from 'node:test';

import { CogentClient, QueryError } from '../../src/index.js';
import type {
  EndpointDescriptor,
  GatewayEndpointDescriptor,
  TokenBatch,
} from '../../src/index.js';

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

function endpointConfig(
  overrides: Partial<GatewayEndpointDescriptor> = {}
): GatewayEndpointDescriptor {
  return {
    kind: 'gateway',
    target: 'developer-model',
    baseUrl: 'https://inference.example.test',
    authentication: { kind: 'bearer', value: 'endpoint-secret' },
    ...overrides,
  };
}

test('CogentClient exposes typed inference and endpoint registration', async () => {
  const client = new CogentClient({ executionMode: 'main-thread' });

  assert.equal(typeof client.add, 'function');
  assert.equal(typeof client.query, 'function');
  assert.equal(typeof client.chat, 'function');
  assert.equal(typeof client.embed, 'function');
  assert.equal(typeof client.currentLocal, 'function');
  assert.equal(typeof client.listLocal, 'function');
  assert.equal(typeof client.removeLocal, 'function');

  await client.close();
});

test('gateway query uses custom routes, authentication, headers, and endpoint options', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  await withGlobalFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return textResponse('custom route response');
    },
    async () => {
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = await client.add(
        'custom-http',
        endpointConfig({
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
        endpointOptions: { profile: 'request', seed: 7 },
      }).response;

      assert.deepEqual(endpoint, { kind: 'gateway', id: 'custom-http' });
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
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = await client.add(
        'typed-http',
        endpointConfig({
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
        { endpoint, endpointOptions: { response_style: 'brief' } }
      ).response;
      const embed = await client.embed('vector input', {
        endpoint,
        endpointOptions: { input_type: 'query' },
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
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = await client.add(
        'stream-http',
        endpointConfig({ authentication: { kind: 'none' } })
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
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = await client.add(
        'header-http',
        endpointConfig({
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
      const client = new CogentClient({ executionMode: 'main-thread' });
      const endpoint = await client.add('error-http', endpointConfig());

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
  const client = new CogentClient({ executionMode: 'main-thread' });

  await assert.rejects(
    client.add(
      'invalid-url',
      endpointConfig({ baseUrl: 'http://public.example.test' })
    ),
    (error) =>
      error instanceof QueryError &&
      error.message ===
        'gateway endpoint baseUrl must use HTTPS unless it targets loopback'
  );
  await assert.rejects(
    client.add(
      'unknown-field',
      {
        ...endpointConfig(),
        policy: 'application-owned',
      } as unknown as EndpointDescriptor
    ),
    (error) =>
      error instanceof QueryError &&
      error.message === 'unsupported gateway endpoint field: policy'
  );

  await client.close();
});

test('gateway endpoints reject local-only inference options', async () => {
  const client = new CogentClient({ executionMode: 'main-thread' });
  const endpoint = await client.add(
    'gateway-options',
    endpointConfig({ authentication: { kind: 'none' } })
  );

  await assert.rejects(
    client.query('hello', { endpoint, grammar: 'root ::= "ok"' }).response,
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
