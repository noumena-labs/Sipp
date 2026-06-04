import test from 'node:test';
import assert from 'node:assert/strict';
import { getOptimizedDefaultWorkerUrl } from '../../src/worker/model-service-client.js';

test('getOptimizedDefaultWorkerUrl returns null for normal module imports', () => {
  assert.equal(
    getOptimizedDefaultWorkerUrl(
      'https://app.test/node_modules/@noumena-labs/cogentlm/dist/esm/worker/model-service-client.js'
    ),
    null
  );
});

test('getOptimizedDefaultWorkerUrl maps Vite optimized deps back to the package worker entry', () => {
  assert.equal(
    getOptimizedDefaultWorkerUrl(
      'https://app.test/node_modules/.vite/deps/@noumena-labs_cogentlm.js?v=123'
    ),
    'https://app.test/node_modules/@noumena-labs/cogentlm/dist/esm/worker/model-service-entry.js'
  );
});

test('getOptimizedDefaultWorkerUrl maps public Vite optimized deps back to the worker entry', () => {
  assert.equal(
    getOptimizedDefaultWorkerUrl('https://app.test/node_modules/.vite/deps/cogentlm.js?v=123'),
    'https://app.test/node_modules/cogentlm/dist/esm/worker/model-service-entry.js'
  );
});
