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

test('BrowserTokenBatches yields batches queued before iteration', async () => {
  const run = createBrowserTextRun({ emitTokens: true }, async (tokenBatchSink) => {
    tokenBatchSink?.(tokenBatch('a'));
    return generationResult('a');
  });
  const chunks: string[] = [];

  for await (const batch of run.tokens) {
    chunks.push(batch.text);
  }
  await run.response;

  assert.deepEqual(chunks, ['a']);
});

test('BrowserTokenBatches yields live batches', async () => {
  let tokenBatchSink!: (batch: TokenBatch) => void;
  let finish!: () => void;
  const done = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const run = createBrowserTextRun({ emitTokens: true }, async (sink) => {
    tokenBatchSink = sink!;
    await done;
    return generationResult('b');
  });
  const iterator = run.tokens[Symbol.asyncIterator]();

  tokenBatchSink(tokenBatch('b'));

  assert.deepEqual(await iterator.next(), { done: false, value: tokenBatch('b') });
  finish();
  await run.response;
  assert.deepEqual(await iterator.next(), { done: true, value: undefined });
});

test('BrowserTokenBatches coalesces when the queue is full', async () => {
  const run = createBrowserTextRun({ emitTokens: true }, async (tokenBatchSink) => {
    for (let index = 0; index < 300; index += 1) {
      tokenBatchSink?.({
        ...tokenBatch(String(index % 10)),
        sequenceStart: index,
        stats: {
          framesSent: index + 1,
          bytesSent: index + 1,
          batchesSent: index + 1,
        },
      });
    }
    return generationResult('');
  });

  const batches: TokenBatch[] = [];
  for await (const batch of run.tokens) {
    batches.push(batch);
  }
  await run.response;

  assert.equal(batches.length, 256);
  assert.equal(batches.map((batch) => batch.text).join('').length, 300);
  assert.equal(batches.at(-1)?.frameCount, 45);
});
