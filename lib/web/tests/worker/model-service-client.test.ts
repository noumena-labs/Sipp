import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getOptimizedDefaultWorkerUrl,
  WorkerModelServiceClient,
} from '../../src/worker/model-service-client.js';
import type { WorkerRuntimeConfig } from '../../src/worker/model-service-protocol.js';
import {
  withNavigatorUserAgent,
  withWasmPthreadSupport,
} from '../support/browser-env.js';

function readWorkerConfig(client: WorkerModelServiceClient): WorkerRuntimeConfig {
  return (client as unknown as { readonly workerConfig: WorkerRuntimeConfig }).workerConfig;
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

test('WorkerModelServiceClient carries Firefox CPU backend override into worker config', () => {
  withNavigatorUserAgent('Mozilla/5.0 Firefox/152.0.2', () => {
    withWasmPthreadSupport(() => {
      const client = new WorkerModelServiceClient();
      const workerConfig = readWorkerConfig(client);

      assert.equal(workerConfig.wasmThreading, 'pthread');
      assert.equal(workerConfig.defaultBackendOverride, 'cpu');
    });
  });
});
