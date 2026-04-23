import test from 'node:test';
import assert from 'node:assert/strict';
import { getDefaultRuntimeUrls, resolveRuntimeUrls } from './runtime-assets.js';

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
  const resolved = withLocation(undefined, () => resolveRuntimeUrls({}));
  assert.deepEqual(resolved, getDefaultRuntimeUrls());
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
