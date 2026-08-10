import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getOptimizedDefaultWorkerUrl,
  WorkerModelServiceClient,
} from '../../src/worker/model-service-client.js';
import type { WorkerRuntimeConfig } from '../../src/worker/model-service-protocol.js';
import type { ModelInfo } from '../../src/models/types.js';
import type { WorkerModelServiceClientInternals } from '../../src/worker/model-service-client.js';
import {
  FakeModelWorker,
  waitForModelWorker,
  waitForWorkerMessage,
  withFakeModelWorker,
} from '../support/fake-model-worker.js';
import { FakeTaskScheduler } from '../support/fake-task-scheduler.js';

/** Must match WORKER_SHUTDOWN_BUDGET_MS in the client under test. */
const WORKER_SHUTDOWN_BUDGET_MS = 1_000;

function readWorkerConfig(client: WorkerModelServiceClient): WorkerRuntimeConfig {
  return (client as unknown as { getWorkerConfig(): WorkerRuntimeConfig }).getWorkerConfig();
}

const loadedModel = (id: string): ModelInfo => ({
  id,
  name: `${id}.gguf`,
  modality: 'text',
  status: 'ready',
  source: 'remote',
  bytes: 1,
  assetFingerprint: `asset-${id}`,
  createdAt: '2026-01-01T00:00:00.000Z',
  updatedAt: '2026-01-01T00:00:00.000Z',
  loaded: true,
  chatTemplate: null,
  bosText: '',
  eosText: '',
  mediaMarker: null,
  capabilities: null,
});

async function closeClient(
  client: WorkerModelServiceClient,
  worker: FakeModelWorker
): Promise<void> {
  const close = client.close();
  const shutdownMessage = await waitForWorkerMessage(worker, 'shutdown');
  worker.respond({ kind: 'resolve', callId: shutdownMessage.callId });
  await close;
}

function createFakeWorkerClient(
  internals: WorkerModelServiceClientInternals = {}
): WorkerModelServiceClient {
  return new WorkerModelServiceClient(
    {
      workerUrl: '/worker.js',
      wasmThreading: 'single-thread',
      moduleUrl: 'https://example.test/runtime.js',
      wasmUrl: 'https://example.test/runtime.wasm',
    },
    internals
  );
}

/** Lets queued microtasks settle after the virtual clock moves. */
async function flushMicrotasks(): Promise<void> {
  for (let turn = 0; turn < 10; turn += 1) {
    await Promise.resolve();
  }
}

function tokenRingWithRecord(streamId: number, text: string): {
  readonly buffer: ArrayBuffer;
  readonly headerOffset: number;
  readonly bodyOffset: number;
  readonly bodyCapacity: number;
} {
  const headerOffset = 0;
  const bodyOffset = 32;
  const bodyCapacity = 64;
  const buffer = new ArrayBuffer(bodyOffset + bodyCapacity);
  const header = new Int32Array(buffer, headerOffset, 8);
  const body = new Uint8Array(buffer, bodyOffset, bodyCapacity);
  const payload = new TextEncoder().encode(text);
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  view.setUint32(0, streamId, true);
  view.setUint32(4, 0, true);
  view.setUint32(8, 1, true);
  view.setUint32(12, payload.byteLength, true);
  body.set(payload, 16);
  header[0] = 16 + payload.byteLength;
  header[2] = bodyCapacity;
  return { buffer, headerOffset, bodyOffset, bodyCapacity };
}

test('getOptimizedDefaultWorkerUrl returns null for normal module imports', () => {
  assert.equal(
    getOptimizedDefaultWorkerUrl(
      'https://app.test/node_modules/@noumena-labs/sipp/dist/esm/worker/model-service-client.js'
    ),
    null
  );
});

test('getOptimizedDefaultWorkerUrl maps Vite optimized deps back to the package worker entry', () => {
  assert.equal(
    getOptimizedDefaultWorkerUrl(
      'https://app.test/node_modules/.vite/deps/@noumena-labs_sipp.js?v=123'
    ),
    'https://app.test/node_modules/@noumena-labs/sipp/dist/esm/worker/model-service-entry.js'
  );
});

test('getOptimizedDefaultWorkerUrl maps public Vite optimized deps back to the worker entry', () => {
  assert.equal(
    getOptimizedDefaultWorkerUrl('https://app.test/node_modules/.vite/deps/@sipphq_sipp.js?v=123'),
    'https://app.test/node_modules/@sipphq/sipp/dist/esm/worker/model-service-entry.js'
  );
});

test('WorkerModelServiceClient leaves bundled runtime selection to the Worker', () => {
  const client = new WorkerModelServiceClient();
  const workerConfig = readWorkerConfig(client);

  assert.equal(workerConfig.moduleUrl, undefined);
  assert.equal(workerConfig.wasmUrl, undefined);
  assert.equal(workerConfig.wasmThreading, undefined);
});

test('WorkerModelServiceClient retires the catalog worker before creating an activation worker', async () => {
  await withFakeModelWorker(async () => {
    const client = createFakeWorkerClient();
    const install = client.add({ kind: 'remote', urls: ['https://example.test/model.gguf'] });
    const catalogWorker = await waitForModelWorker(0);
    const installMessage = await waitForWorkerMessage(catalogWorker, 'models-install');
    catalogWorker.respond({
      kind: 'resolve',
      callId: installMessage.callId,
      value: loadedModel('model'),
    });
    await install;

    const load = client.load('model');
    const shutdownMessage = await waitForWorkerMessage(catalogWorker, 'shutdown');
    assert.equal(FakeModelWorker.instances.length, 1);
    assert.equal(catalogWorker.terminated, false);

    catalogWorker.respond({ kind: 'resolve', callId: shutdownMessage.callId });
    while (FakeModelWorker.instances.length < 2) {
      await Promise.resolve();
    }
    const activationWorker = FakeModelWorker.instances[1];
    assert.equal(catalogWorker.terminated, true);
    const loadMessage = await waitForWorkerMessage(activationWorker, 'models-load');
    activationWorker.respond({
      kind: 'resolve',
      callId: loadMessage.callId,
      value: loadedModel('model'),
    });

    assert.equal((await load).id, 'model');
    await closeClient(client, activationWorker);
  });
});

test('WorkerModelServiceClient blocks inference while replacing the active worker', async () => {
  await withFakeModelWorker(async () => {
    const client = createFakeWorkerClient();
    const firstLoad = client.load('first');
    const firstWorker = await waitForModelWorker(0);
    const firstLoadMessage = await waitForWorkerMessage(firstWorker, 'models-load');
    firstWorker.respond({
      kind: 'resolve',
      callId: firstLoadMessage.callId,
      value: loadedModel('first'),
    });
    await firstLoad;

    const replacement = client.load('second');
    const shutdownMessage = await waitForWorkerMessage(firstWorker, 'shutdown');
    await assert.rejects(
      client.runQuery('stale run', {}),
      (error: unknown) => error instanceof Error && error.message === 'No local model is active.'
    );
    assert.equal(firstWorker.posted.some((message) => message.kind === 'query'), false);

    firstWorker.respond({ kind: 'resolve', callId: shutdownMessage.callId });
    while (FakeModelWorker.instances.length < 2) {
      await Promise.resolve();
    }
    const secondWorker = FakeModelWorker.instances[1];
    const secondLoadMessage = await waitForWorkerMessage(secondWorker, 'models-load');
    secondWorker.respond({
      kind: 'resolve',
      callId: secondLoadMessage.callId,
      value: loadedModel('second'),
    });
    await replacement;
    await closeClient(client, secondWorker);
  });
});

test('WorkerModelServiceClient terminates an unresponsive worker and advances lifecycle work', async () => {
  await withFakeModelWorker(async () => {
    const tasks = new FakeTaskScheduler();
    const client = createFakeWorkerClient({ tasks });
    const firstLoad = client.load('first');
    const firstWorker = await waitForModelWorker(0);
    const firstLoadMessage = await waitForWorkerMessage(firstWorker, 'models-load');
    firstWorker.respond({
      kind: 'resolve',
      callId: firstLoadMessage.callId,
      value: loadedModel('first'),
    });
    await firstLoad;

    const replacement = client.load('second');
    await waitForWorkerMessage(firstWorker, 'shutdown');

    // The worker never acknowledges shutdown, so only the budget can retire it.
    tasks.advance(WORKER_SHUTDOWN_BUDGET_MS - 1);
    await flushMicrotasks();
    assert.equal(firstWorker.terminated, false);
    assert.equal(FakeModelWorker.instances.length, 1);

    tasks.advance(1);
    await flushMicrotasks();
    const secondWorker = await waitForModelWorker(1);
    assert.equal(firstWorker.terminated, true);
    const secondLoadMessage = await waitForWorkerMessage(secondWorker, 'models-load');
    secondWorker.respond({
      kind: 'resolve',
      callId: secondLoadMessage.callId,
      value: loadedModel('second'),
    });
    await replacement;

    const list = client.list();
    const listMessage = await waitForWorkerMessage(secondWorker, 'models-list');
    secondWorker.respond({
      kind: 'resolve',
      callId: listMessage.callId,
      value: [loadedModel('second')],
    });
    assert.equal((await list)[0]?.id, 'second');
    await closeClient(client, secondWorker);
    assert.equal(secondWorker.terminated, true);
  });
});

test('WorkerModelServiceClient cancels a queued token-ring drain when the runtime is retired', async () => {
  await withFakeModelWorker(async () => {
    const tasks = new FakeTaskScheduler();
    const client = createFakeWorkerClient({ tasks });
    const load = client.load('first');
    const worker = await waitForModelWorker(0);
    const loadMessage = await waitForWorkerMessage(worker, 'models-load');
    worker.respond({
      kind: 'resolve',
      callId: loadMessage.callId,
      value: loadedModel('first'),
    });
    await load;

    const query = client.runQuery('hi', { tokenBatchSink: () => {} });
    await waitForWorkerMessage(worker, 'query');
    worker.respond({
      kind: 'token-ring-ready',
      descriptor: {
        buffer: new ArrayBuffer(64),
        headerOffset: 0,
        bodyOffset: 32,
        bodyCapacity: 32,
      },
    });
    assert.equal(tasks.pendingFrameCount, 1);

    const retirement = client.close();
    const shutdown = await waitForWorkerMessage(worker, 'shutdown');
    worker.respond({ kind: 'resolve', callId: shutdown.callId });
    await retirement;
    await assert.rejects(query);

    // The frame queued before retirement must not run against the dead ring.
    assert.equal(tasks.pendingFrameCount, 0);
    tasks.runFrames();
    assert.equal(tasks.pendingFrameCount, 0);
  });
});

test('WorkerModelServiceClient terminates a failed activation worker without restoring it', async () => {
  await withFakeModelWorker(async () => {
    const client = createFakeWorkerClient();
    const load = client.load('broken');
    const worker = await waitForModelWorker(0);
    const loadMessage = await waitForWorkerMessage(worker, 'models-load');
    worker.respond({
      kind: 'reject',
      callId: loadMessage.callId,
      message: 'native activation failed',
    });

    await assert.rejects(load, /native activation failed/u);
    assert.equal(worker.terminated, true);
    assert.equal(client.current(), null);
    await assert.rejects(client.runQuery('no runtime', {}), /No local model is active/u);
    await client.close();
  });
});

test('WorkerModelServiceClient ignores presentation events from a retired incarnation', async () => {
  await withFakeModelWorker(async () => {
    const client = createFakeWorkerClient();
    const firstLoad = client.load('first');
    const firstWorker = await waitForModelWorker(0);
    const firstLoadMessage = await waitForWorkerMessage(firstWorker, 'models-load');
    firstWorker.respond({
      kind: 'resolve',
      callId: firstLoadMessage.callId,
      value: loadedModel('first'),
    });
    await firstLoad;
    const retiredHandler = firstWorker.onmessage;

    const replacement = client.load('second');
    const shutdownMessage = await waitForWorkerMessage(firstWorker, 'shutdown');
    firstWorker.respond({ kind: 'resolve', callId: shutdownMessage.callId });
    const secondWorker = await waitForModelWorker(1);
    const secondLoadMessage = await waitForWorkerMessage(secondWorker, 'models-load');
    secondWorker.respond({
      kind: 'resolve',
      callId: secondLoadMessage.callId,
      value: loadedModel('second'),
    });
    await replacement;

    retiredHandler?.({
      data: {
        kind: 'observability-event',
        event: {
          type: 'load-complete',
          snapshot: {
            mode: 'off',
            state: 'ready',
            updatedAt: '2026-01-01T00:00:00.000Z',
            model: loadedModel('first'),
            query: null,
          },
        },
      },
    } as MessageEvent);
    assert.equal(client.current()?.id, 'second');
    await closeClient(client, secondWorker);
  });
});

test('WorkerModelServiceClient configures a new worker before its first request', async () => {
  await withFakeModelWorker(async () => {
    const client = createFakeWorkerClient();
    const listing = client.list();
    const worker = await waitForModelWorker(0);
    const listMessage = await waitForWorkerMessage(worker, 'models-list');

    const initialize = worker.posted[0];
    assert.equal(initialize.kind, 'initialize');
    assert.deepEqual(
      initialize.kind === 'initialize' ? initialize.config : null,
      readWorkerConfig(client)
    );
    // Operational requests no longer carry runtime configuration.
    assert.equal('config' in listMessage, false);

    worker.respond({ kind: 'resolve', callId: listMessage.callId, value: [] });
    assert.deepEqual(await listing, []);
    await closeClient(client, worker);
  });
});

test('WorkerModelServiceClient configures each replacement worker exactly once', async () => {
  await withFakeModelWorker(async () => {
    const client = createFakeWorkerClient();
    const listing = client.list();
    const catalogWorker = await waitForModelWorker(0);
    const listMessage = await waitForWorkerMessage(catalogWorker, 'models-list');
    workerRespondList(catalogWorker, listMessage.callId);
    await listing;

    const load = client.load('model-1');
    const shutdown = await waitForWorkerMessage(catalogWorker, 'shutdown');
    catalogWorker.respond({ kind: 'resolve', callId: shutdown.callId });
    const activationWorker = await waitForModelWorker(1);
    const loadMessage = await waitForWorkerMessage(activationWorker, 'models-load');

    assert.equal(activationWorker.posted[0].kind, 'initialize');
    assert.equal(
      activationWorker.posted.filter((message) => message.kind === 'initialize').length,
      1
    );

    activationWorker.respond({
      kind: 'resolve',
      callId: loadMessage.callId,
      value: loadedModel('model-1'),
    });
    await load;
    await closeClient(client, activationWorker);
  });
});

function workerRespondList(worker: FakeModelWorker, callId: number): void {
  worker.respond({ kind: 'resolve', callId, value: [] });
}

test('WorkerModelServiceClient contains a throwing token sink and settles the request', async () => {
  await withFakeModelWorker(async () => {
    const tasks = new FakeTaskScheduler();
    const client = createFakeWorkerClient({ tasks });
    const load = client.load('first');
    const worker = await waitForModelWorker(0);
    const loadMessage = await waitForWorkerMessage(worker, 'models-load');
    worker.respond({ kind: 'resolve', callId: loadMessage.callId, value: loadedModel('first') });
    await load;

    const sinkFailure = new Error('sink exploded');
    const query = client.runQuery('hi', {
      tokenBatchSink: () => {
        throw sinkFailure;
      },
    });
    const queryMessage = await waitForWorkerMessage(worker, 'query');

    // Message-transport batches go through the same guarded sink path.
    worker.respond({
      kind: 'token-batch',
      callId: queryMessage.callId,
      batch: {
        requestId: '1',
        streamId: 1,
        sequenceStart: 0,
        text: 'a',
        frameCount: 1,
        byteCount: 1,
        stats: { framesSent: 1, bytesSent: 1, batchesSent: 1 },
      },
    });

    // The client cancels the request whose sink failed.
    const cancel = worker.posted.find((message) => message.kind === 'cancel');
    assert.notEqual(cancel, undefined);

    // The caller sees its own failure, not a cancellation, and the promise settles.
    worker.respond({ kind: 'resolve', callId: queryMessage.callId, value: { text: 'a' } });
    await assert.rejects(query, /sink exploded/);

    // A second request on the same worker still works.
    const followUp = client.runQuery('again', {});
    const followUpMessage = worker.posted.filter((message) => message.kind === 'query')[1];
    assert.notEqual(followUpMessage, undefined);
    worker.respond({
      kind: 'resolve',
      callId: (followUpMessage as { callId: number }).callId,
      value: { text: 'ok' },
    });
    await followUp;
    await closeClient(client, worker);
  });
});

test('WorkerModelServiceClient preserves an undefined sink failure from the final ring drain', async () => {
  await withFakeModelWorker(async () => {
    const client = createFakeWorkerClient();
    const load = client.load('first');
    const worker = await waitForModelWorker(0);
    const loadMessage = await waitForWorkerMessage(worker, 'models-load');
    worker.respond({ kind: 'resolve', callId: loadMessage.callId, value: loadedModel('first') });
    await load;

    const query = client.runQuery('hi', {
      tokenBatchSink: () => {
        throw undefined;
      },
    });
    const queryMessage = await waitForWorkerMessage(worker, 'query');
    worker.respond({
      kind: 'token-ring-ready',
      descriptor: tokenRingWithRecord(7, 'a'),
    });
    worker.respond({
      kind: 'token-ring-claim',
      callId: queryMessage.callId,
      nativeRequestId: 7,
    });

    // The record is still in the ring when the response arrives, so finalize
    // performs the first sink invocation and must observe the failure afterward.
    worker.respond({ kind: 'resolve', callId: queryMessage.callId, value: { text: 'a' } });
    const outcome = await query.then(
      () => ({ resolved: true, error: null }),
      (error: unknown) => ({ resolved: false, error })
    );
    assert.equal(outcome.resolved, false);
    assert.equal(outcome.error, undefined);
    assert.equal(worker.posted.some((message) => message.kind === 'cancel'), true);
    await closeClient(client, worker);
  });
});

test('WorkerModelServiceClient presents no tokens once retirement begins', async () => {
  await withFakeModelWorker(async () => {
    const tasks = new FakeTaskScheduler();
    const client = createFakeWorkerClient({ tasks });
    const load = client.load('first');
    const worker = await waitForModelWorker(0);
    const loadMessage = await waitForWorkerMessage(worker, 'models-load');
    worker.respond({ kind: 'resolve', callId: loadMessage.callId, value: loadedModel('first') });
    await load;

    const batches: unknown[] = [];
    const query = client.runQuery('hi', { tokenBatchSink: (batch) => batches.push(batch) });
    const queryMessage = await waitForWorkerMessage(worker, 'query');
    const queryOutcome = query.then(
      () => 'resolved' as const,
      () => 'rejected' as const
    );
    let querySettled = false;
    void queryOutcome.then(() => {
      querySettled = true;
    });
    worker.respond({
      kind: 'token-ring-ready',
      descriptor: {
        buffer: new ArrayBuffer(64),
        headerOffset: 0,
        bodyOffset: 32,
        bodyCapacity: 32,
      },
    });
    assert.equal(tasks.pendingFrameCount, 1);

    // Retire without acknowledging shutdown: the budget is still running.
    const replacement = client.load('second');
    await waitForWorkerMessage(worker, 'shutdown');

    // Any frame queued before retirement must not present during the budget.
    tasks.runFrames();
    await flushMicrotasks();
    assert.deepEqual(batches, []);

    // A terminal response from the retiring runtime is presentation too. It is
    // ignored; only the shutdown acknowledgement may settle during the budget.
    worker.respond({
      kind: 'resolve',
      callId: queryMessage.callId,
      value: { text: 'stale' },
    });
    await flushMicrotasks();
    assert.equal(querySettled, false);

    tasks.advance(WORKER_SHUTDOWN_BUDGET_MS);
    await flushMicrotasks();
    assert.equal(await queryOutcome, 'rejected');
    assert.deepEqual(batches, []);

    const secondWorker = await waitForModelWorker(1);
    const secondLoad = await waitForWorkerMessage(secondWorker, 'models-load');
    secondWorker.respond({
      kind: 'resolve',
      callId: secondLoad.callId,
      value: loadedModel('second'),
    });
    await replacement;
    await closeClient(client, secondWorker);
  });
});

test('WorkerModelServiceClient revives cleanupFailures attached by the Worker', async () => {
  await withFakeModelWorker(async () => {
    const client = createFakeWorkerClient();
    const listing = client.list();
    const worker = await waitForModelWorker(0);
    const listMessage = await waitForWorkerMessage(worker, 'models-list');
    worker.respond({
      kind: 'reject',
      callId: listMessage.callId,
      message: 'activation failed',
      errorName: 'Error',
      cleanupFailures: ['unmount /sah_model: busy'],
    });

    const error = await listing.then(
      () => null,
      (caught: unknown) => caught
    );
    const failures = (error as { cleanupFailures?: AggregateError }).cleanupFailures;
    assert.equal((error as Error).message, 'activation failed');
    assert.equal(failures?.errors.length, 1);
    assert.equal((failures?.errors[0] as Error).message, 'unmount /sah_model: busy');
    await closeClient(client, worker);
  });
});
