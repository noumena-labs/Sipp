import test from 'node:test';
import assert from 'node:assert/strict';
import { batchTokensByFrame, type BatchedTokens } from './token-batching.js';

// All tests use a manual scheduler so they're deterministic and don't depend
// on requestAnimationFrame existing in the node test runtime.

function manualScheduler() {
  const pending: Array<() => void> = [];
  return {
    schedule: (cb: () => void) => {
      pending.push(cb);
      return pending.length - 1;
    },
    cancel: (_handle: unknown) => {
      // Cancelled callbacks just stay in `pending`; `runNext` filters them.
      // The helper only cancels when flushing, so this is rare and we don't
      // bother bookkeeping handles.
    },
    runNext: (): boolean => {
      const cb = pending.shift();
      if (cb == null) return false;
      cb();
      return true;
    },
    pendingCount: () => pending.length,
  };
}

test('batchTokensByFrame coalesces multiple tokens into one render call', () => {
  const sched = manualScheduler();
  const renders: BatchedTokens[] = [];
  const stream = batchTokensByFrame((batch) => renders.push(batch), {
    schedule: sched.schedule,
    cancel: sched.cancel,
  });

  stream.onToken('hel');
  stream.onToken('lo ');
  stream.onToken('world');

  // No render until the scheduler fires.
  assert.equal(renders.length, 0);

  sched.runNext();
  assert.equal(renders.length, 1);
  assert.equal(renders[0].accumulated, 'hello world');
  assert.equal(renders[0].delta, 'hello world');
});

test('batchTokensByFrame fires once per scheduler tick with the latest delta', () => {
  const sched = manualScheduler();
  const renders: BatchedTokens[] = [];
  const stream = batchTokensByFrame((batch) => renders.push(batch), {
    schedule: sched.schedule,
    cancel: sched.cancel,
  });

  stream.onToken('one');
  sched.runNext();
  stream.onToken(' two');
  stream.onToken(' three');
  sched.runNext();

  assert.equal(renders.length, 2);
  assert.equal(renders[0].delta, 'one');
  assert.equal(renders[0].accumulated, 'one');
  assert.equal(renders[1].delta, ' two three');
  assert.equal(renders[1].accumulated, 'one two three');
});

test('batchTokensByFrame flush() drains pending tokens immediately', () => {
  const sched = manualScheduler();
  const renders: BatchedTokens[] = [];
  const stream = batchTokensByFrame((batch) => renders.push(batch), {
    schedule: sched.schedule,
    cancel: sched.cancel,
  });

  stream.onToken('tail');
  assert.equal(renders.length, 0);

  stream.flush();
  assert.equal(renders.length, 1);
  assert.equal(renders[0].accumulated, 'tail');
  assert.equal(renders[0].delta, 'tail');
});

test('batchTokensByFrame flush() is a no-op when nothing is buffered', () => {
  const sched = manualScheduler();
  const renders: BatchedTokens[] = [];
  const stream = batchTokensByFrame((batch) => renders.push(batch), {
    schedule: sched.schedule,
    cancel: sched.cancel,
  });

  stream.flush();
  assert.equal(renders.length, 0);

  stream.onToken('a');
  sched.runNext();
  assert.equal(renders.length, 1);
  // A second flush after the scheduler already drained the buffer must not
  // re-render with stale data.
  stream.flush();
  assert.equal(renders.length, 1);
});

test('batchTokensByFrame intervalMs uses setTimeout-based throttling', async () => {
  const renders: BatchedTokens[] = [];
  const stream = batchTokensByFrame((batch) => renders.push(batch), {
    intervalMs: 5,
  });

  stream.onToken('a');
  stream.onToken('b');
  // setTimeout(5) hasn't fired yet.
  assert.equal(renders.length, 0);

  await new Promise((r) => setTimeout(r, 20));
  assert.equal(renders.length, 1);
  assert.equal(renders[0].accumulated, 'ab');

  stream.onToken('c');
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(renders.length, 2);
  assert.equal(renders[1].accumulated, 'abc');
  assert.equal(renders[1].delta, 'c');
});

test('batchTokensByFrame ignores empty tokens', () => {
  const sched = manualScheduler();
  const renders: BatchedTokens[] = [];
  const stream = batchTokensByFrame((batch) => renders.push(batch), {
    schedule: sched.schedule,
    cancel: sched.cancel,
  });

  stream.onToken('');
  stream.onToken('');
  assert.equal(sched.pendingCount(), 0);

  stream.onToken('x');
  assert.equal(sched.pendingCount(), 1);
  sched.runNext();
  assert.equal(renders[0].accumulated, 'x');
});
