import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getDefaultRuntimeUrls,
  resolveOptimizedPackageAssetUrl,
  resolveRuntimeThreadingMode,
  resolveRuntimeUrls,
  supportsWasmPthreads,
} from '../../src/engine/runtime-assets.js';

interface LocationStub {
  href: string;
  origin: string;
}

function withLocation<T>(href: string | undefined, callback: () => T): T {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'location');

  if (href == null) {
    Reflect.deleteProperty(globalThis, 'location');
  } else {
    const location: LocationStub = {
      href,
      origin: new URL(href).origin,
    };
    Object.defineProperty(globalThis, 'location', {
      configurable: true,
      value: location,
    });
  }

  try {
    return callback();
  } finally {
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'location');
    } else {
      Object.defineProperty(globalThis, 'location', descriptor);
    }
  }
}

function withWasmPthreadSupport<T>(callback: () => T): T {
  const crossOriginIsolatedDescriptor = Object.getOwnPropertyDescriptor(
    globalThis,
    'crossOriginIsolated'
  );
  const workerDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  Object.defineProperty(globalThis, 'crossOriginIsolated', {
    configurable: true,
    value: true,
  });
  Object.defineProperty(globalThis, 'Worker', {
    configurable: true,
    value: class FakeWorker {},
  });
  try {
    return callback();
  } finally {
    if (crossOriginIsolatedDescriptor == null) {
      Reflect.deleteProperty(globalThis, 'crossOriginIsolated');
    } else {
      Object.defineProperty(globalThis, 'crossOriginIsolated', crossOriginIsolatedDescriptor);
    }
    if (workerDescriptor == null) {
      Reflect.deleteProperty(globalThis, 'Worker');
    } else {
      Object.defineProperty(globalThis, 'Worker', workerDescriptor);
    }
  }
}

test('resolveRuntimeUrls uses bundled runtime assets when no overrides are provided', () => {
  const resolved = withLocation(undefined, () => resolveRuntimeUrls({}));
  assert.deepEqual(resolved, getDefaultRuntimeUrls());
});

test('getDefaultRuntimeUrls maps Vite optimized deps back to package wasm assets', () => {
  assert.deepEqual(
    getDefaultRuntimeUrls('https://app.test/node_modules/.vite/deps/@noumena-labs_sipp.js?v=123'),
    {
      moduleUrl: 'https://app.test/node_modules/@noumena-labs/sipp/dist/wasm/sipp-wasm.js',
      wasmUrl: 'https://app.test/node_modules/@noumena-labs/sipp/dist/wasm/sipp-wasm.wasm',
      threading: 'single-thread',
    }
  );
});

test('getDefaultRuntimeUrls maps public Vite optimized deps back to package wasm assets', () => {
  assert.deepEqual(
    getDefaultRuntimeUrls('https://app.test/node_modules/.vite/deps/@sipp_sipp.js?v=123'),
    {
      moduleUrl: 'https://app.test/node_modules/@sipp/sipp/dist/wasm/sipp-wasm.js',
      wasmUrl: 'https://app.test/node_modules/@sipp/sipp/dist/wasm/sipp-wasm.wasm',
      threading: 'single-thread',
    }
  );
});

test('resolveOptimizedPackageAssetUrl returns null for normal module URLs', () => {
  assert.equal(
    resolveOptimizedPackageAssetUrl(
      'dist/esm/worker/model-service-entry.js',
      'https://app.test/node_modules/@noumena-labs/sipp/dist/esm/worker/model-service-client.js'
    ),
    null
  );
});

test('resolveOptimizedPackageAssetUrl maps Vite optimized deps back to package files', () => {
  assert.equal(
    resolveOptimizedPackageAssetUrl(
      'dist/esm/worker/model-service-entry.js',
      'https://app.test/node_modules/.vite/deps/@noumena-labs_sipp.js?v=123'
    ),
    'https://app.test/node_modules/@noumena-labs/sipp/dist/esm/worker/model-service-entry.js'
  );
});

test('resolveOptimizedPackageAssetUrl preserves a Vite dev base path', () => {
  assert.equal(
    resolveOptimizedPackageAssetUrl(
      '/dist/wasm/sipp-wasm.js',
      'https://app.test/subapp/node_modules/.vite/deps/@noumena-labs_sipp.js?v=123'
    ),
    'https://app.test/subapp/node_modules/@noumena-labs/sipp/dist/wasm/sipp-wasm.js'
  );
});

test('resolveRuntimeUrls defaults to the single-thread artifact when wasm pthreads are available', () => {
  withWasmPthreadSupport(() => {
    assert.equal(supportsWasmPthreads(), true);
    assert.equal(resolveRuntimeThreadingMode({}), 'single-thread');
    const resolved = resolveRuntimeUrls({});
    assert.match(resolved.moduleUrl, /sipp-wasm\.js$/);
    assert.match(resolved.wasmUrl, /sipp-wasm\.wasm$/);
    assert.equal(resolved.threading, 'single-thread');
  });
});

test('resolveRuntimeUrls selects the pthread artifact when explicitly requested', () => {
  withWasmPthreadSupport(() => {
    assert.equal(resolveRuntimeThreadingMode({ wasmThreading: 'pthread' }), 'pthread');
    const resolved = resolveRuntimeUrls({ wasmThreading: 'pthread' });
    assert.match(resolved.moduleUrl, /sipp-wasm-pthread\.js$/);
    assert.match(resolved.wasmUrl, /sipp-wasm-pthread\.wasm$/);
    assert.equal(resolved.threading, 'pthread');
  });
});

test('resolveRuntimeUrls honors the single-thread runtime preference', () => {
  withWasmPthreadSupport(() => {
    const resolved = resolveRuntimeUrls({ wasmThreading: 'single-thread' });
    assert.match(resolved.moduleUrl, /sipp-wasm\.js$/);
    assert.match(resolved.wasmUrl, /sipp-wasm\.wasm$/);
    assert.equal(resolved.threading, 'single-thread');
  });
});

test('resolveRuntimeUrls uses the current window-like location for relative overrides', () => {
  const resolved = withLocation('https://app.test/ui/index.html', () =>
    resolveRuntimeUrls({
      moduleUrl: './assets/runtime.js',
      wasmUrl: './assets/runtime.wasm',
    })
  );

  assert.deepEqual(resolved, {
    moduleUrl: 'https://app.test/ui/assets/runtime.js',
    wasmUrl: 'https://app.test/ui/assets/runtime.wasm',
    threading: 'single-thread',
  });
});

test('resolveRuntimeUrls uses the current worker-like location for relative overrides', () => {
  const resolved = withLocation('https://app.test/pkg/worker/model-service-entry.js', () =>
    resolveRuntimeUrls({
      moduleUrl: '../wasm/custom-runtime.js',
      wasmUrl: '../wasm/custom-runtime.wasm',
    })
  );

  assert.deepEqual(resolved, {
    moduleUrl: 'https://app.test/pkg/wasm/custom-runtime.js',
    wasmUrl: 'https://app.test/pkg/wasm/custom-runtime.wasm',
    threading: 'single-thread',
  });
});

test('resolveRuntimeUrls blocks cross-origin overrides when trustedOrigins are not expanded', () => {
  withLocation('https://app.test/ui/index.html', () => {
    assert.throws(
      () =>
        resolveRuntimeUrls({
          moduleUrl: 'https://cdn.test/runtime.js',
          wasmUrl: 'https://cdn.test/runtime.wasm',
        }),
      /Blocked moduleUrl origin "https:\/\/cdn\.test"/
    );
  });
});
