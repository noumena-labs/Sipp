import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveOptimizedPackageAssetUrl } from './package-assets.js';

test('resolveOptimizedPackageAssetUrl returns null for normal module URLs', () => {
  assert.equal(
    resolveOptimizedPackageAssetUrl(
      'dist/esm/worker/model-service-entry.js',
      'https://app.test/node_modules/@noumena-labs/cogentlm-browser/dist/esm/worker/model-service-client.js'
    ),
    null
  );
});

test('resolveOptimizedPackageAssetUrl maps Vite optimized deps back to package files', () => {
  assert.equal(
    resolveOptimizedPackageAssetUrl(
      'dist/esm/worker/model-service-entry.js',
      'https://app.test/node_modules/.vite/deps/@noumena-labs_cogentlm-browser.js?v=123'
    ),
    'https://app.test/node_modules/@noumena-labs/cogentlm-browser/dist/esm/worker/model-service-entry.js'
  );
});

test('resolveOptimizedPackageAssetUrl preserves a Vite dev base path', () => {
  assert.equal(
    resolveOptimizedPackageAssetUrl(
      '/dist/wasm/cogentlm-wasm.js',
      'https://app.test/subapp/node_modules/.vite/deps/@noumena-labs_cogentlm-browser.js?v=123'
    ),
    'https://app.test/subapp/node_modules/@noumena-labs/cogentlm-browser/dist/wasm/cogentlm-wasm.js'
  );
});
