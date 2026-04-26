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

export function waitForAbort(
  signal: AbortSignal,
  options: { timedOut?: () => boolean; timeoutMessage?: string; abortMessage?: string } = {}
): Promise<never> {
  const timeoutMessage = options.timeoutMessage ?? 'The operation timed out.';
  const abortMessage = options.abortMessage ?? 'The operation was aborted.';
  if (signal.aborted) {
    return Promise.reject(
      createAbortError(options.timedOut?.() ? timeoutMessage : abortMessage)
    );
  }
  return new Promise((_, reject) => {
    const onAbort = (): void => {
      signal.removeEventListener('abort', onAbort);
      reject(createAbortError(options.timedOut?.() ? timeoutMessage : abortMessage));
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
}
