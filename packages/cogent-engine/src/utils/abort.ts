export function createAbortError(message = 'The operation was aborted.'): Error {
  if (typeof DOMException === 'function') {
    return new DOMException(message, 'AbortError');
  }
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
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

  const abortListener =
    signal == null
      ? null
      : () => {
          controller.abort();
        };
  const linkedSignal = signal;
  if (abortListener != null && linkedSignal != null) {
    linkedSignal.addEventListener('abort', abortListener, { once: true });
  }

  return {
    controller,
    signal: controller.signal,
    dispose: () => {
      if (abortListener != null && linkedSignal != null) {
        linkedSignal.removeEventListener('abort', abortListener);
      }
    },
  };
}
