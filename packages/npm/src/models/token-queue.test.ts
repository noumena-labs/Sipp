import assert from 'node:assert/strict';
import test from 'node:test';

import type { GenerationResult, TokenBatch } from './types.js';
import { createBrowserTextRun } from './token-queue.js';

function tokenBatch(text: string): TokenBatch {
  return {
    requestId: 'test',
    streamId: 1,
    sequenceStart: 0,
    text,
    frameCount: 1,
    byteCount: new TextEncoder().encode(text).byteLength,
    stats: {
      framesSent: 1,
      bytesSent: new TextEncoder().encode(text).byteLength,
      framesDropped: 0,
      batchesSent: 1,
    },
  };
}

function generationResult(text: string): GenerationResult {
  return {
    id: 'test',
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

test('BrowserTokenBatches.subscribe consumes batches queued before subscription', async () => {
  const run = createBrowserTextRun({ tokenDelivery: 'batch' }, async (tokenSink) => {
    tokenSink?.(tokenBatch('a'));
    return generationResult('a');
  });
  const chunks: string[] = [];

  run.tokens.subscribe((batch) => {
    chunks.push(batch.text);
  });
  await run.response;

  assert.deepEqual(chunks, ['a']);
});

test('BrowserTokenBatches.subscribe receives live batches synchronously', async () => {
  let tokenSink!: (batch: TokenBatch) => void;
  let finish!: () => void;
  const done = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const run = createBrowserTextRun({ tokenDelivery: 'batch' }, async (sink) => {
    tokenSink = sink!;
    await done;
    return generationResult('b');
  });
  const chunks: string[] = [];

  run.tokens.subscribe((batch) => {
    chunks.push(batch.text);
  });
  tokenSink(tokenBatch('b'));

  assert.deepEqual(chunks, ['b']);
  finish();
  await run.response;
});
