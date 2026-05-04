export function createAbortError(message = 'The operation was aborted.'): Error {
  if (typeof DOMException === 'function') {
    return new DOMException(message, 'AbortError');
  }
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

export function isAbortError(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'name' in error &&
    error.name === 'AbortError'
  );
}

export function createLinkedAbortController(signal?: AbortSignal): {
  controller: AbortController;
  signal: AbortSignal;
  dispose: () => void;
} {
  const controller = new AbortController();
  if (signal?.aborted) {
    controller.abort();
    return {
      controller,
      signal: controller.signal,
      dispose: () => {},
    };
  }

  if (signal == null) {
    return {
      controller,
      signal: controller.signal,
      dispose: () => {},
    };
  }

  const abortListener = () => {
    controller.abort();
  };
  signal.addEventListener('abort', abortListener, { once: true });

  return {
    controller,
    signal: controller.signal,
    dispose: () => {
      signal.removeEventListener('abort', abortListener);
    },
  };
}

export function createTimedAbortController(
  signal?: AbortSignal,
  timeoutMs?: number
): {
  controller: AbortController;
  signal: AbortSignal;
  timedOut: () => boolean;
  dispose: () => void;
} {
  const linked = createLinkedAbortController(signal);
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let didTimeOut = false;

  if (timeoutMs != null && Number.isFinite(timeoutMs) && timeoutMs >= 0) {
    timeoutId = setTimeout(() => {
      didTimeOut = true;
      linked.controller.abort();
    }, timeoutMs);
  }

  return {
    controller: linked.controller,
    signal: linked.signal,
    timedOut: () => didTimeOut,
    dispose: () => {
      if (timeoutId != null) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      linked.dispose();
    },
  };
}
