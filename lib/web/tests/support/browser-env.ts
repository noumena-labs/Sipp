interface NavigatorUserAgentStub {
  readonly userAgent: string;
}

export function withNavigatorUserAgent<T>(userAgent: string, callback: () => T): T {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const navigatorStub: NavigatorUserAgentStub = { userAgent };
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: navigatorStub,
  });

  try {
    return callback();
  } finally {
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'navigator');
    } else {
      Object.defineProperty(globalThis, 'navigator', descriptor);
    }
  }
}

export function withoutWasmJspiSupport<T>(callback: () => T): T {
  const target = WebAssembly as object;
  const descriptor = Object.getOwnPropertyDescriptor(target, 'Suspending');
  Reflect.deleteProperty(target, 'Suspending');
  try {
    return callback();
  } finally {
    if (descriptor != null) {
      Object.defineProperty(target, 'Suspending', descriptor);
    }
  }
}

export function withWasmPthreadSupport<T>(callback: () => T): T {
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
