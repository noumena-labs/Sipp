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
  const restore = () => {
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
