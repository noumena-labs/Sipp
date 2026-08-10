import test from 'node:test';
import assert from 'node:assert/strict';
import { RequestTracker } from '../../src/runtime/request-tracker.js';

test('RequestTracker keys equivalent handles by generation and request ID', async () => {
  const tracker = new RequestTracker<string>();
  tracker.track({ generation: 7, requestId: 11 });

  assert.notEqual(tracker.get({ generation: 7, requestId: 11 }), undefined);
  tracker.resolve({ generation: 7, requestId: 11 }, 'done');

  assert.equal(
    await tracker.beginWait({ generation: 7, requestId: 11 }),
    'done'
  );
  tracker.endWait({ generation: 7, requestId: 11 });
});

test('RequestTracker keeps reused native IDs isolated by generation', () => {
  const tracker = new RequestTracker<string>();
  tracker.track({ generation: 1, requestId: 3 });
  tracker.track({ generation: 2, requestId: 3 });

  tracker.resolve({ generation: 1, requestId: 3 }, 'old');

  assert.equal(tracker.get({ generation: 1, requestId: 3 })?.settled, true);
  assert.equal(tracker.get({ generation: 2, requestId: 3 })?.settled, false);
});

test('RequestTracker reports structural request identities in errors', () => {
  const tracker = new RequestTracker<string>();

  assert.throws(
    () => tracker.beginWait({ generation: 4, requestId: 9 }),
    /request 4:9 is not tracked/
  );
});

test('RequestTracker counts a request as active until it is finalized', () => {
  const tracker = new RequestTracker<string>();
  const request = { generation: 1, requestId: 1 };
  tracker.track(request);

  assert.equal(tracker.activeCount, 1);
  assert.equal(tracker.get(request)?.active, true);

  tracker.resolve(request, 'done');
  assert.equal(tracker.activeCount, 1);

  tracker.finalize(request);
  assert.equal(tracker.activeCount, 0);
  assert.equal(tracker.get(request)?.active, false);
});

test('RequestTracker re-tracking a finalized request does not reactivate it', () => {
  const tracker = new RequestTracker<string>();
  const request = { generation: 1, requestId: 1 };
  tracker.track(request);
  tracker.resolve(request, 'done');
  tracker.finalize(request);

  tracker.track(request);

  assert.equal(tracker.activeCount, 0);
  assert.equal(tracker.get(request)?.active, false);
});

test('RequestTracker never drops a record that is still active', () => {
  const tracker = new RequestTracker<string>();
  const request = { generation: 1, requestId: 1 };
  tracker.track(request);
  tracker.resolve(request, 'done');

  // Consume the settled result without finalizing: the record is still active,
  // so it must survive so the active count keeps pointing at a live record.
  void tracker.beginWait(request);
  tracker.endWait(request);

  assert.equal(tracker.activeCount, 1);
  assert.equal(tracker.records().length, 1);

  tracker.finalize(request);
  assert.equal(tracker.activeCount, 0);
  assert.equal(tracker.records().length, 0);
});

test('RequestTracker records() snapshots so callers can finalize while iterating', () => {
  const tracker = new RequestTracker<string>();
  tracker.track({ generation: 1, requestId: 1 });
  tracker.track({ generation: 1, requestId: 2 });

  const seen: number[] = [];
  for (const tracked of tracker.records()) {
    seen.push(tracked.request.requestId);
    tracker.finalize(tracked.request, { deleteCompletion: true });
  }

  assert.deepEqual(seen, [1, 2]);
  assert.equal(tracker.activeCount, 0);
  assert.equal(tracker.records().length, 0);
});

test('RequestTracker rejectAll settles pending requests and resets the active count', async () => {
  const tracker = new RequestTracker<string>();
  const request = { generation: 1, requestId: 1 };
  const tracked = tracker.track(request);

  tracker.rejectAll(new Error('closed'));

  await assert.rejects(tracked.promise, /closed/);
  assert.equal(tracker.activeCount, 0);
  assert.equal(tracker.get(request), undefined);
});

test('RequestTracker clear detaches abort listeners and resets the active count', () => {
  const tracker = new RequestTracker<string>();
  const request = { generation: 1, requestId: 1 };
  const controller = new AbortController();
  let aborts = 0;
  tracker.track(request);
  tracker.attachSignal(request, controller.signal, () => {
    aborts += 1;
  });

  tracker.clear();
  controller.abort();

  assert.equal(aborts, 0);
  assert.equal(tracker.activeCount, 0);
});

test('RequestTracker attaches one cancellation per request and signal', () => {
  const tracker = new RequestTracker<string>();
  const request = { generation: 1, requestId: 1 };
  const controller = new AbortController();
  let aborts = 0;
  tracker.track(request);
  tracker.attachSignal(request, controller.signal, () => {
    aborts += 1;
  });
  tracker.attachSignal(request, controller.signal, () => {
    aborts += 1;
  });

  controller.abort();

  assert.equal(aborts, 1);
});

test('RequestTracker deduplicates an already-aborted signal', () => {
  const tracker = new RequestTracker<string>();
  const request = { generation: 1, requestId: 1 };
  const controller = new AbortController();
  let aborts = 0;
  tracker.track(request);
  controller.abort();
  tracker.attachSignal(request, controller.signal, () => {
    aborts += 1;
  });
  tracker.attachSignal(request, controller.signal, () => {
    aborts += 1;
  });

  assert.equal(aborts, 1);
});

test('RequestTracker keeps a shared abort listener until every owner detaches', () => {
  const tracker = new RequestTracker<string>();
  const request = { generation: 1, requestId: 1 };
  const controller = new AbortController();
  let aborts = 0;
  tracker.track(request);
  const detachFirst = tracker.attachSignal(request, controller.signal, () => {
    aborts += 1;
  });
  tracker.attachSignal(request, controller.signal, () => {
    aborts += 1;
  });

  detachFirst();
  controller.abort();

  assert.equal(aborts, 1);
});

test('RequestTracker preserves undefined as a token-sink failure', () => {
  const tracker = new RequestTracker<string>();
  const request = { generation: 1, requestId: 1 };
  tracker.track(request);

  tracker.setTokenBatchSinkError(request, undefined);

  assert.equal(tracker.get(request)?.tokenBatchSinkFailed, true);
  assert.equal(tracker.get(request)?.tokenBatchSinkError, undefined);
});
