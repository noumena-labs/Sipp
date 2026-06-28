import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getDefaultRuntimeUrls,
  resolveOptimizedPackageAssetUrl,
  resolveRuntimeBackendOverride,
  resolveRuntimeThreadingMode,
  resolveRuntimeUrls,
  supportsWasmPthreads,
} from '../../src/engine/runtime-assets.js';
import {
  withNavigatorUserAgent,
  withoutWasmJspiSupport,
  withWasmPthreadSupport,
} from '../support/browser-env.js';

const SAFARI_USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 Safari/605.1.15';

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

test('resolveRuntimeUrls uses bundled runtime assets when no overrides are provided', () => {
  withWasmPthreadSupport(() => {
    const resolved = withLocation(undefined, () => resolveRuntimeUrls({}));
    assert.deepEqual(resolved, getDefaultRuntimeUrls());
  });
});

test('getDefaultRuntimeUrls maps Vite optimized deps back to package wasm assets', () => {
  withWasmPthreadSupport(() => {
    assert.deepEqual(
      getDefaultRuntimeUrls('https://app.test/node_modules/.vite/deps/@noumena-labs_sipp.js?v=123'),
      {
        moduleUrl: 'https://app.test/node_modules/@noumena-labs/sipp/dist/wasm/sipp-wasm-pthread.js',
        wasmUrl: 'https://app.test/node_modules/@noumena-labs/sipp/dist/wasm/sipp-wasm-pthread.wasm',
        threading: 'pthread',
      }
    );
  });
});

test('getDefaultRuntimeUrls maps public Vite optimized deps back to package wasm assets', () => {
  withWasmPthreadSupport(() => {
    assert.deepEqual(
      getDefaultRuntimeUrls('https://app.test/node_modules/.vite/deps/@sipphq_sipp.js?v=123'),
      {
        moduleUrl: 'https://app.test/node_modules/@sipphq/sipp/dist/wasm/sipp-wasm-pthread.js',
        wasmUrl: 'https://app.test/node_modules/@sipphq/sipp/dist/wasm/sipp-wasm-pthread.wasm',
        threading: 'pthread',
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

test('resolveRuntimeUrls defaults to the pthread artifact when wasm pthreads are available', () => {
  withWasmPthreadSupport(() => {
    assert.equal(supportsWasmPthreads(), true);
    assert.equal(resolveRuntimeThreadingMode({}), 'pthread');
    const resolved = resolveRuntimeUrls({});
    assert.match(resolved.moduleUrl, /sipp-wasm-pthread\.js$/);
    assert.match(resolved.wasmUrl, /sipp-wasm-pthread\.wasm$/);
    assert.equal(resolved.threading, 'pthread');
  });
});

test('resolveRuntimeUrls rejects bundled runtimes without wasm pthread support', () => {
  assert.throws(
    () => resolveRuntimeUrls({}),
    /requires SharedArrayBuffer and cross-origin isolation/
  );
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

test('resolveRuntimeUrls auto-selects CPU non-JSPI on Firefox', () => {
  withNavigatorUserAgent('Mozilla/5.0 Firefox/127.0', () => {
    withWasmPthreadSupport(() => {
      assert.equal(resolveRuntimeThreadingMode({}), 'pthread');
      const resolved = resolveRuntimeUrls({});
      assert.match(resolved.moduleUrl, /sipp-wasm-pthread-cpu-nojspi\.js$/);
      assert.match(resolved.wasmUrl, /sipp-wasm-pthread-cpu-nojspi\.wasm$/);
      assert.equal(resolved.threading, 'pthread');
    });
  });
});

test('resolveRuntimeBackendOverride forces CPU for bundled Firefox pthread runtime', () => {
  withNavigatorUserAgent('Mozilla/5.0 Firefox/152.0.2', () => {
    withWasmPthreadSupport(() => {
      assert.equal(resolveRuntimeBackendOverride({ wasmThreading: 'pthread' }), 'cpu');
    });
  });
});

test('resolveRuntimeUrls auto-selects CPU non-JSPI when JSPI is unavailable (e.g. Safari)', () => {
  withNavigatorUserAgent(SAFARI_USER_AGENT, () => {
    withoutWasmJspiSupport(() => {
      withWasmPthreadSupport(() => {
        const resolved = resolveRuntimeUrls({});
        assert.match(resolved.moduleUrl, /sipp-wasm-pthread-cpu-nojspi\.js$/);
        assert.match(resolved.wasmUrl, /sipp-wasm-pthread-cpu-nojspi\.wasm$/);
        assert.equal(resolved.threading, 'pthread');
      });
    });
  });
});

test('resolveRuntimeUrls keeps the WebGPU+JSPI artifact on Safari once JSPI is exposed', () => {
  withNavigatorUserAgent(SAFARI_USER_AGENT, () => {
    withWasmPthreadSupport(() => {
      const resolved = resolveRuntimeUrls({});
      assert.match(resolved.moduleUrl, /sipp-wasm-pthread\.js$/);
      assert.match(resolved.wasmUrl, /sipp-wasm-pthread\.wasm$/);
    });
  });
});

test('resolveRuntimeBackendOverride forces CPU when JSPI is unavailable', () => {
  withoutWasmJspiSupport(() => {
    withWasmPthreadSupport(() => {
      assert.equal(resolveRuntimeBackendOverride({ wasmThreading: 'pthread' }), 'cpu');
    });
  });
});

test('resolveRuntimeBackendOverride does not force CPU for custom runtime URLs', () => {
  withNavigatorUserAgent('Mozilla/5.0 Firefox/152.0.2', () => {
    withWasmPthreadSupport(() => {
      assert.equal(
        resolveRuntimeBackendOverride({
          wasmThreading: 'pthread',
          moduleUrl: '/custom.js',
          wasmUrl: '/custom.wasm',
        }),
        null
      );
    });
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

test('resolveRuntimeUrls accepts legacy pthread runtime aliases', () => {
  withWasmPthreadSupport(() => {
    const resolved = withLocation('https://app.test/ui/index.html', () =>
      resolveRuntimeUrls({
        pthreadModuleUrl: './assets/runtime.js',
        pthreadWasmUrl: './assets/runtime.wasm',
      })
    );

    assert.deepEqual(resolved, {
      moduleUrl: 'https://app.test/ui/assets/runtime.js',
      wasmUrl: 'https://app.test/ui/assets/runtime.wasm',
      threading: 'pthread',
    });
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
