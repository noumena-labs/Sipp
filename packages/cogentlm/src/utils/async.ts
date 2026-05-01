export function createDeferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

export async function mapWithConcurrency<T, TResult>(
  items: readonly T[],
  concurrency: number,
  mapper: (item: T, index: number) => Promise<TResult>,
  onError?: (error: unknown) => void
): Promise<TResult[]> {
  if (items.length === 0) {
    return [];
  }

  const results = new Array<TResult>(items.length);
  const workerCount = Math.min(Math.max(1, concurrency), items.length);
  let nextIndex = 0;
  let firstError: unknown = null;

  const workers = Array.from({ length: workerCount }, async () => {
    while (true) {
      if (firstError != null) {
        return;
      }

      const currentIndex = nextIndex;
      nextIndex += 1;
      if (currentIndex >= items.length) {
        return;
      }

      try {
        results[currentIndex] = await mapper(items[currentIndex], currentIndex);
      } catch (error) {
        if (firstError == null) {
          firstError = error;
          onError?.(error);
        }
        throw error;
      }
    }
  });

  await Promise.allSettled(workers);
  if (firstError != null) {
    throw firstError;
  }
  return results;
}
