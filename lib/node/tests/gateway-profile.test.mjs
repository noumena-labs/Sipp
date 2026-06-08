import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  GatewayProfileError,
  decodeGatewayChatBody,
  decodeGatewayEmbedBody,
  decodeGatewayQueryBody,
  gatewayEmbeddingResponseBody,
  gatewayErrorResponse,
  gatewayTextResponseBody,
  gatewayTextStreamResponse,
  isGatewayProfileError,
} = require('../gateway-profile.js');

test('gateway profile decodes query bodies and endpoint options', () => {
  const decoded = decodeGatewayQueryBody({
    model: 'developer-model',
    prompt: 'hello',
    max_tokens: 12,
    temperature: 0.5,
    top_p: 0.9,
    stop: ['END'],
    stream: true,
    seed: 7,
    metadata: { tenant: 'docs' },
  });

  assert.deepEqual(decoded, {
    target: 'developer-model',
    stream: true,
    request: {
      prompt: 'hello',
      emitTokens: true,
      options: {
        maxTokens: 12,
        temperature: 0.5,
        topP: 0.9,
        stop: ['END'],
      },
      endpointOptions: {
        seed: 7,
        metadata: { tenant: 'docs' },
      },
    },
  });
});

test('gateway profile decodes chat and embedding bodies', () => {
  const chat = decodeGatewayChatBody({
    model: 'developer-model',
    messages: [
      { role: 'system', content: 'Answer briefly.' },
      { role: 'user', content: 'hello' },
    ],
    response_style: 'brief',
  });
  const embed = decodeGatewayEmbedBody({
    model: 'developer-model',
    input: 'vector input',
    input_type: 'query',
  });

  assert.deepEqual(chat, {
    target: 'developer-model',
    stream: false,
    request: {
      messages: [
        { role: 'system', content: 'Answer briefly.' },
        { role: 'user', content: 'hello' },
      ],
      emitTokens: false,
      endpointOptions: { response_style: 'brief' },
    },
  });
  assert.deepEqual(embed, {
    target: 'developer-model',
    stream: false,
    request: {
      input: 'vector input',
      endpointOptions: { input_type: 'query' },
    },
  });
});

test('gateway profile formats text, embedding, and error responses', () => {
  assert.deepEqual(
    gatewayTextResponseBody('developer-model', {
      text: 'done',
      finishReason: 'stop',
      usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
      metadata: { upstreamResponseId: 'text-1' },
    }),
    {
      id: 'text-1',
      model: 'developer-model',
      text: 'done',
      finish_reason: 'stop',
      usage: {
        input_tokens: 1,
        output_tokens: 2,
        total_tokens: 3,
      },
    }
  );
  assert.deepEqual(
    gatewayEmbeddingResponseBody('developer-model', {
      values: [0.25, 0.75],
      metadata: { upstreamResponseId: 'embedding-1' },
    }),
    {
      id: 'embedding-1',
      model: 'developer-model',
      embedding: [0.25, 0.75],
    }
  );

  const profileError = new GatewayProfileError('invalid_request', 'bad body', {
    status: 422,
  });
  assert.equal(isGatewayProfileError(profileError), true);
  assert.deepEqual(gatewayErrorResponse(profileError), {
    body: {
      error: {
        code: 'invalid_request',
        message: 'bad body',
      },
    },
    init: {
      status: 422,
    },
  });
  assert.deepEqual(gatewayErrorResponse(new Error('boom')), {
    body: {
      error: {
        code: 'internal',
        message: 'boom',
      },
    },
    init: {
      status: 500,
    },
  });
});

test('gateway profile formats text runs as SSE responses', async () => {
  const run = {
    tokens: {
      async *[Symbol.asyncIterator]() {
        yield { text: 'hello ', sequenceStart: 0 };
        yield { text: 'world', sequenceStart: 1 };
      },
    },
    response: Promise.resolve({
      text: 'hello world',
      finishReason: 'stop',
      usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
      metadata: {},
    }),
    cancel() {},
  };

  const response = gatewayTextStreamResponse(run);
  const body = await response.text();

  assert.equal(response.headers.get('content-type'), 'text/event-stream');
  assert.match(body, /event: token\ndata: \{"text":"hello ","sequence":0\}/);
  assert.match(body, /event: token\ndata: \{"text":"world","sequence":1\}/);
  assert.match(
    body,
    /event: usage\ndata: \{"input_tokens":1,"output_tokens":2,"total_tokens":3\}/
  );
  assert.match(body, /event: done\ndata: \{"finish_reason":"stop"\}/);
});
