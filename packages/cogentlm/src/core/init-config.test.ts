import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeInitConfig } from './init-config.js';

test('normalizeInitConfig preserves auto GPU layer offload by default', () => {
  assert.equal(normalizeInitConfig(undefined).nGpuLayers, -1);
});

test('normalizeInitConfig accepts explicit auto and CPU-only GPU layer modes', () => {
  assert.equal(normalizeInitConfig({ nGpuLayers: -1 }).nGpuLayers, -1);
  assert.equal(normalizeInitConfig({ nGpuLayers: 0 }).nGpuLayers, 0);
  assert.equal(normalizeInitConfig({ nGpuLayers: 99 }).nGpuLayers, 99);
});

test('normalizeInitConfig rejects unsupported GPU layer values', () => {
  assert.throws(
    () => normalizeInitConfig({ nGpuLayers: -2 }),
    /"nGpuLayers" must be an integer >= -1/
  );
});
