import test from 'node:test';
import assert from 'node:assert/strict';
import {
  memoizeWasmJspiProbe,
  resolveOptimizedPackageAssetUrl,
  resolveRuntimeAssetSelection,
  resolveRuntimeThreadingMode,
  resolveRuntimeUrls,
  supportsWasmPthreads,
} from '../../src/engine/runtime-assets.js';
import { withWasmPthreadSupport } from '../support/browser-env.js';

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

  const restore = () => {
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'location');
    } else {
      Object.defineProperty(globalThis, 'location', descriptor);
    }
  };
  try {
    const result = callback();
    if (result instanceof Promise) {
      return result.finally(restore) as T;
    }
    restore();
    return result;
  } catch (error) {
    restore();
    throw error;
  }
}

test('resolveRuntimeAssetSelection uses bundled runtime assets after probing', async () => {
  await withWasmPthreadSupport(async () => {
    const resolved = await withLocation(undefined, () =>
      resolveRuntimeAssetSelection({}, { probeWasmJspi: async () => true })
    );
    assert.match(resolved.moduleUrl, /sipp-wasm-pthread\.js$/);
    assert.equal(resolved.backendConstraint, null);
  });
});

test('resolveRuntimeUrls accepts root-relative bundled runtime assets', () => {
  withWasmPthreadSupport(() => {
    const resolved = withLocation('https://app.test/_next/static/chunks/model-service-entry.js', () =>
      resolveRuntimeUrls(
        {},
        {
          bundledRuntimeUrls: () => ({
            moduleUrl: '/_next/static/media/sipp-wasm-pthread.41xk03exto5xs.js',
            wasmUrl: '/_next/static/media/sipp-wasm-pthread.0vprqr1jq_rfe.wasm',
            threading: 'pthread',
          }),
        }
      )
    );

    assert.deepEqual(resolved, {
      moduleUrl: 'https://app.test/_next/static/media/sipp-wasm-pthread.41xk03exto5xs.js',
      wasmUrl: 'https://app.test/_next/static/media/sipp-wasm-pthread.0vprqr1jq_rfe.wasm',
      threading: 'pthread',
    });
  });
});

test('bundled selection maps Vite optimized deps back to package wasm assets', async () => {
  await withWasmPthreadSupport(async () => {
    assert.deepEqual(
      await resolveRuntimeAssetSelection({}, {
        importerUrl: 'https://app.test/node_modules/.vite/deps/@noumena-labs_sipp.js?v=123',
        probeWasmJspi: async () => true,
      }),
      {
        moduleUrl: 'https://app.test/node_modules/@noumena-labs/sipp/dist/wasm/sipp-wasm-pthread.js',
        wasmUrl: 'https://app.test/node_modules/@noumena-labs/sipp/dist/wasm/sipp-wasm-pthread.wasm',
        threading: 'pthread',
        backendConstraint: null,
      }
    );
  });
});

test('bundled selection maps public Vite optimized deps back to package wasm assets', async () => {
  await withWasmPthreadSupport(async () => {
    assert.deepEqual(
      await resolveRuntimeAssetSelection({}, {
        importerUrl: 'https://app.test/node_modules/.vite/deps/@sipphq_sipp.js?v=123',
        probeWasmJspi: async () => true,
      }),
      {
        moduleUrl: 'https://app.test/node_modules/@sipphq/sipp/dist/wasm/sipp-wasm-pthread.js',
        wasmUrl: 'https://app.test/node_modules/@sipphq/sipp/dist/wasm/sipp-wasm-pthread.wasm',
        threading: 'pthread',
        backendConstraint: null,
      }
    );
  });
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

test('resolveRuntimeAssetSelection defaults to the pthread artifact when JSPI works', async () => {
  await withWasmPthreadSupport(async () => {
    assert.equal(supportsWasmPthreads(), true);
    assert.equal(resolveRuntimeThreadingMode({}), 'pthread');
    const resolved = await resolveRuntimeAssetSelection(
      {},
      { probeWasmJspi: async () => true }
    );
    assert.match(resolved.moduleUrl, /sipp-wasm-pthread\.js$/);
    assert.match(resolved.wasmUrl, /sipp-wasm-pthread\.wasm$/);
    assert.equal(resolved.threading, 'pthread');
  });
});

test('resolveRuntimeAssetSelection rejects bundled runtimes without wasm pthread support', async () => {
  await assert.rejects(
    resolveRuntimeAssetSelection({}, { probeWasmJspi: async () => true }),
    /requires SharedArrayBuffer and cross-origin isolation/
  );
});

test('resolveRuntimeAssetSelection selects the pthread artifact when explicitly requested', async () => {
  await withWasmPthreadSupport(async () => {
    assert.equal(resolveRuntimeThreadingMode({ wasmThreading: 'pthread' }), 'pthread');
    const resolved = await resolveRuntimeAssetSelection(
      { wasmThreading: 'pthread' },
      { probeWasmJspi: async () => true }
    );
    assert.match(resolved.moduleUrl, /sipp-wasm-pthread\.js$/);
    assert.match(resolved.wasmUrl, /sipp-wasm-pthread\.wasm$/);
    assert.equal(resolved.threading, 'pthread');
  });
});

test('resolveRuntimeAssetSelection falls back to CPU when functional JSPI fails', async () => {
  await withWasmPthreadSupport(async () => {
    const resolved = await resolveRuntimeAssetSelection(
      {},
      { probeWasmJspi: async () => false }
    );
    assert.match(resolved.moduleUrl, /sipp-wasm-pthread-cpu-nojspi\.js$/);
    assert.match(resolved.wasmUrl, /sipp-wasm-pthread-cpu-nojspi\.wasm$/);
    assert.equal(resolved.threading, 'pthread');
    assert.equal(resolved.backendConstraint, 'cpu-only');
  });
});

test('resolveRuntimeAssetSelection treats a throwing JSPI probe as unsupported', async () => {
  await withWasmPthreadSupport(async () => {
    const resolved = await resolveRuntimeAssetSelection(
      {},
      { probeWasmJspi: async () => { throw new Error('probe failed'); } }
    );

    assert.match(resolved.moduleUrl, /sipp-wasm-pthread-cpu-nojspi\.js$/);
    assert.equal(resolved.backendConstraint, 'cpu-only');
  });
});

test('memoizeWasmJspiProbe executes its functional probe once', async () => {
  let calls = 0;
  const probe = memoizeWasmJspiProbe(async () => {
    calls += 1;
    return true;
  });

  assert.deepEqual(await Promise.all([probe(), probe(), probe()]), [true, true, true]);
  assert.equal(calls, 1);
});

test('resolveRuntimeAssetSelection does not force CPU for custom runtime URLs', async () => {
  await withWasmPthreadSupport(async () => {
    assert.deepEqual(
      await resolveRuntimeAssetSelection({
          wasmThreading: 'pthread',
          moduleUrl: 'https://app.test/custom.js',
          wasmUrl: 'https://app.test/custom.wasm',
        }),
        {
          moduleUrl: 'https://app.test/custom.js',
          wasmUrl: 'https://app.test/custom.wasm',
          threading: 'pthread',
          backendConstraint: null,
        }
    );
  });
});

test('resolveRuntimeUrls rejects bundled single-thread runtime preference', () => {
  withWasmPthreadSupport(() => {
    assert.throws(
      () => resolveRuntimeUrls({ wasmThreading: 'single-thread' }),
      /bundled Sipp browser runtime is pthread-only/
    );
  });
});

test('resolveRuntimeUrls uses the current window-like location for relative overrides', () => {
  withWasmPthreadSupport(() => {
    const resolved = withLocation('https://app.test/ui/index.html', () =>
      resolveRuntimeUrls({
        moduleUrl: './assets/runtime.js',
        wasmUrl: './assets/runtime.wasm',
      })
    );

    assert.deepEqual(resolved, {
      moduleUrl: 'https://app.test/ui/assets/runtime.js',
      wasmUrl: 'https://app.test/ui/assets/runtime.wasm',
      threading: 'pthread',
    });
  });
});

test('resolveRuntimeUrls uses the current worker-like location for relative overrides', () => {
  withWasmPthreadSupport(() => {
    const resolved = withLocation('https://app.test/pkg/worker/model-service-entry.js', () =>
      resolveRuntimeUrls({
        moduleUrl: '../wasm/custom-runtime.js',
        wasmUrl: '../wasm/custom-runtime.wasm',
      })
    );

    assert.deepEqual(resolved, {
      moduleUrl: 'https://app.test/pkg/wasm/custom-runtime.js',
      wasmUrl: 'https://app.test/pkg/wasm/custom-runtime.wasm',
      threading: 'pthread',
    });
  });
});

test('resolveRuntimeUrls uses moduleUrl and wasmUrl for custom single-thread runtime when selected', () => {
  const resolved = withLocation('https://app.test/ui/index.html', () =>
    resolveRuntimeUrls({
      wasmThreading: 'single-thread',
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

test('resolveRuntimeUrls blocks cross-origin overrides when trustedOrigins are not expanded', () => {
  withLocation('https://app.test/ui/index.html', () => {
    assert.throws(
      () =>
        resolveRuntimeUrls({
          wasmThreading: 'single-thread',
          moduleUrl: 'https://cdn.test/runtime.js',
          wasmUrl: 'https://cdn.test/runtime.wasm',
        }),
      /Blocked moduleUrl origin "https:\/\/cdn\.test"/
    );
  });
});
