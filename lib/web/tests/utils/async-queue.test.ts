import assert from 'node:assert/strict';
import test from 'node:test';

import { AsyncSerialQueue } from '../../src/utils/async-queue.js';

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

test('AsyncSerialQueue runs operations in submission order', async () => {
  const queue = new AsyncSerialQueue();
  const order: string[] = [];
  const first = deferred();

  const a = queue.run(async () => {
    order.push('a:start');
    await first.promise;
    order.push('a:end');
    return 'a';
  });
  const b = queue.run(() => {
    order.push('b');
    return 'b';
  });

  assert.deepEqual(order, []);
  first.resolve();
  assert.deepEqual(await Promise.all([a, b]), ['a', 'b']);
  assert.deepEqual(order, ['a:start', 'a:end', 'b']);
});

test('AsyncSerialQueue keeps running after an operation rejects', async () => {
  const queue = new AsyncSerialQueue();
  const failure = queue.run(() => {
    throw new Error('boom');
  });

  await assert.rejects(failure, /boom/);
  assert.equal(await queue.run(() => 'next'), 'next');
});

test('AsyncSerialQueue idle resolves after queued work settles', async () => {
  const queue = new AsyncSerialQueue();
  const gate = deferred();
  let finished = false;

  const running = queue.run(async () => {
    await gate.promise;
    finished = true;
  });

  const idle = queue.idle().then(() => finished);
  gate.resolve();
  await running;
  assert.equal(await idle, true);
});

test('AsyncSerialQueue idle ignores a rejected operation', async () => {
  const queue = new AsyncSerialQueue();
  const failure = queue.run(() => Promise.reject(new Error('boom')));

  await assert.rejects(failure, /boom/);
  await queue.idle();
});
