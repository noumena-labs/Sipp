import test from 'node:test';
import assert from 'node:assert/strict';
import {
  observabilitySnapshotToEngineState,
  toBackendProfileObservation,
} from '../../src/models/observability-controller.js';
import type { BackendObservability } from '../../src/engine/inference-types.js';
import type { ResolvedBrowserBackend } from '../../src/engine/runtime-assets.js';
import type { BackendInfo, BackendProfileObservation } from '../../src/models/types.js';

// A WebGPU device is registered, as the default runtime does even for CPU loads.
const registeredWebGpu: BackendObservability = {
  profilingEnabled: false,
  webgpuCompiled: true,
  webgpuRegistered: true,
  webgpuDeviceCount: 1,
  gpuOffloadSupported: true,
  engineInitialized: true,
  availableBackends: [
    { name: 'WebGPU', deviceCount: 1 },
    { name: 'CPU', deviceCount: 1 },
    { name: 'cpu', deviceCount: 1 },
  ],
  devices: [],
};

function backendInfo(profile: BackendProfileObservation | undefined): BackendInfo {
  return observabilitySnapshotToEngineState({
    mode: 'off',
    state: 'ready',
    updatedAt: '2026-01-01T00:00:00.000Z',
    model: null,
    query: null,
    profile,
  }).backend;
}

function profileFor(backend: ResolvedBrowserBackend['backend']): BackendProfileObservation | undefined {
  return toBackendProfileObservation(registeredWebGpu, { backend, webgpuAdapter: null });
}

test('backend state reports the resolved backend, not the registered devices', () => {
  assert.equal(backendInfo(profileFor('cpu')).selected, 'cpu');
  assert.equal(backendInfo(profileFor('webgpu')).selected, 'webgpu');
});

test('backend state reports canonical, deduplicated available backends', () => {
  const backend = backendInfo(profileFor('cpu'));

  assert.deepEqual(backend.available, ['webgpu', 'cpu']);
  assert.ok(backend.available.includes(backend.selected));
});

test('backend state reports unknown before the first load', () => {
  assert.equal(backendInfo(undefined).selected, 'unknown');
});
