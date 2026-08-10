import type { GenerateRequestHandle } from '../engine/inference-types.js';

function createDeferred<T>(): {
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

export interface RequestHandle<TResult> {
  readonly promise: Promise<TResult>;
}

/**
 * Read-only bookkeeping for one tracked request. All mutations stay inside the
 * tracker; `finalize` is the only operation that clears `active`.
 */
export interface TrackedRequest<TResult> extends RequestHandle<TResult> {
  readonly request: GenerateRequestHandle;
  readonly active: boolean;
  readonly settled: boolean;
  readonly consumed: boolean;
  readonly waiterCount: number;
  readonly tokenBatchSinkError: unknown;
  readonly tokenBatchSinkFailed: boolean;
  readonly cancelRequested: boolean;
}

interface TrackerRecord<TResult> {
  readonly request: GenerateRequestHandle;
  readonly promise: Promise<TResult>;
  readonly resolve: (value: TResult) => void;
  readonly reject: (error: unknown) => void;
  active: boolean;
  settled: boolean;
  consumed: boolean;
  waiterCount: number;
  tokenBatchSinkError: unknown;
  tokenBatchSinkFailed: boolean;
  cancelRequested: boolean;
}

function requestKey(request: GenerateRequestHandle): string {
  return `${request.generation}:${request.requestId}`;
}

interface AbortRegistration {
  readonly listener: () => void;
  referenceCount: number;
  fired: boolean;
}

/**
 * Tracks pending requests: their promise, settlement state, abort signal, and
 * cleanup. Generic over the result type so both native completions and worker
 * call responses can share the same bookkeeping.
 *
 * There is exactly one record per `generation:requestId`. A record is active
 * from `track` until `finalize`, and it can only leave the tracker once it is
 * inactive, so the active set can never outlive the record it points at.
 */
export class RequestTracker<TResult> {
  private readonly requests = new Map<string, TrackerRecord<TResult>>();
  private readonly abortRegistrations = new Map<
    string,
    Map<AbortSignal, AbortRegistration>
  >();
  private activeRequestCount = 0;

  get activeCount(): number {
    return this.activeRequestCount;
  }

  /** The record for `request`, or undefined when it is not tracked. */
  get(request: GenerateRequestHandle): TrackedRequest<TResult> | undefined {
    return this.requests.get(requestKey(request));
  }

  /**
   * A snapshot of every tracked record. Snapshotting lets callers settle or
   * finalize requests while iterating.
   */
  records(): TrackedRequest<TResult>[] {
    return Array.from(this.requests.values());
  }

  /**
   * Starts tracking a request and marks it active. An already-tracked request
   * is returned unchanged: re-tracking never reactivates a finalized request,
   * because the scheduler would then pump a request that already settled.
   */
  track(request: GenerateRequestHandle): RequestHandle<TResult> {
    const key = requestKey(request);
    const existing = this.requests.get(key);
    if (existing != null) {
      return existing;
    }

    const deferred = createDeferred<TResult>();
    const record: TrackerRecord<TResult> = {
      request,
      promise: deferred.promise,
      resolve: deferred.resolve,
      reject: deferred.reject,
      active: true,
      settled: false,
      consumed: false,
      waiterCount: 0,
      tokenBatchSinkError: undefined,
      tokenBatchSinkFailed: false,
      cancelRequested: false,
    };
    // Prevent unhandled rejection warnings for unconsumed requests.
    void record.promise.catch(() => {});
    this.requests.set(key, record);
    this.activeRequestCount += 1;
    return record;
  }

  beginWait(request: GenerateRequestHandle): Promise<TResult> {
    const record = this.requests.get(requestKey(request));
    if (record == null) {
      throw new Error(`request ${requestKey(request)} is not tracked.`);
    }
    record.consumed = true;
    record.waiterCount += 1;
    return record.promise;
  }

  endWait(request: GenerateRequestHandle): void {
    const key = requestKey(request);
    const record = this.requests.get(key);
    if (record == null) {
      return;
    }
    record.waiterCount = Math.max(0, record.waiterCount - 1);
    this.removeIfFullyConsumed(key, record);
  }

  /** Resolves a tracked request. No-op if already settled. */
  resolve(request: GenerateRequestHandle, result: TResult): void {
    const record = this.requests.get(requestKey(request));
    if (record == null || record.settled) {
      return;
    }
    record.settled = true;
    record.resolve(result);
  }

  /** Rejects a tracked request. No-op if already settled. */
  reject(request: GenerateRequestHandle, error: unknown): void {
    const record = this.requests.get(requestKey(request));
    if (record == null || record.settled) {
      return;
    }
    record.settled = true;
    record.reject(error);
  }

  /** Marks a tracked request as cancel-requested. No-op if it is not tracked. */
  requestCancel(request: GenerateRequestHandle): void {
    const record = this.requests.get(requestKey(request));
    if (record != null) {
      record.cancelRequested = true;
    }
  }

  /** Records a token-sink failure. No-op if the request is not tracked. */
  setTokenBatchSinkError(request: GenerateRequestHandle, error: unknown): void {
    const record = this.requests.get(requestKey(request));
    if (record != null && !record.tokenBatchSinkFailed) {
      record.tokenBatchSinkFailed = true;
      record.tokenBatchSinkError = error;
    }
  }

  /** Rejects every unsettled request and clears all tracker state. */
  rejectAll(error: unknown): void {
    for (const record of this.requests.values()) {
      if (!record.settled) {
        record.settled = true;
        record.reject(error);
      }
    }
    this.clear();
  }

  /**
   * Attaches an AbortSignal to a request. When the signal fires, `onAbort` is
   * called (typically to issue a cancellation to the engine).
   */
  attachSignal(
    request: GenerateRequestHandle,
    signal: AbortSignal,
    onAbort: () => void
  ): () => void {
    const key = requestKey(request);
    let registrations = this.abortRegistrations.get(key);
    if (registrations == null) {
      registrations = new Map();
      this.abortRegistrations.set(key, registrations);
    }
    const existing = registrations.get(signal);
    if (existing != null) {
      existing.referenceCount += 1;
      return () => {
        this.releaseAbortRegistration(request, signal, existing);
      };
    }
    // Enqueue and awaitQuery can share one signal. Keep one listener and count
    // both owners so either owner may detach without disabling the other.
    const registration: AbortRegistration = {
      listener: () => {
        if (registration.fired) {
          return;
        }
        registration.fired = true;
        onAbort();
      },
      referenceCount: 1,
      fired: false,
    };
    registrations.set(signal, registration);
    if (signal.aborted) {
      registration.listener();
      return () => {
        this.releaseAbortRegistration(request, signal, registration);
      };
    }
    signal.addEventListener('abort', registration.listener, { once: true });
    return () => {
      this.releaseAbortRegistration(request, signal, registration);
    };
  }

  /** Detaches every AbortSignal listener for a request. */
  releaseSignal(request: GenerateRequestHandle): void {
    const key = requestKey(request);
    const registrations = this.abortRegistrations.get(key);
    if (registrations == null) {
      return;
    }
    for (const [signal, registration] of registrations) {
      signal.removeEventListener('abort', registration.listener);
    }
    this.abortRegistrations.delete(key);
  }

  private releaseAbortRegistration(
    request: GenerateRequestHandle,
    signal: AbortSignal,
    registration: AbortRegistration
  ): void {
    const key = requestKey(request);
    const registrations = this.abortRegistrations.get(key);
    if (registrations?.get(signal) !== registration) {
      return;
    }
    registration.referenceCount = Math.max(0, registration.referenceCount - 1);
    if (registration.referenceCount > 0 || registration.fired) {
      return;
    }
    signal.removeEventListener('abort', registration.listener);
    registrations.delete(signal);
    if (registrations.size === 0) {
      this.abortRegistrations.delete(key);
    }
  }

  /**
   * Retires a finished request: releases its signal, drops it from the active
   * set, and either deletes its record outright or leaves it for a waiter to
   * consume. This is the only operation that clears `active`.
   */
  finalize(
    request: GenerateRequestHandle,
    options: { deleteCompletion?: boolean } = {}
  ): void {
    this.releaseSignal(request);
    const key = requestKey(request);
    const record = this.requests.get(key);
    if (record == null) {
      return;
    }
    if (record.active) {
      record.active = false;
      this.activeRequestCount -= 1;
    }
    if (options.deleteCompletion) {
      this.requests.delete(key);
      return;
    }
    this.removeIfFullyConsumed(key, record);
  }

  /** Clears all state, dropping every record and abort listener. */
  clear(): void {
    for (const registrations of this.abortRegistrations.values()) {
      for (const [signal, registration] of registrations) {
        signal.removeEventListener('abort', registration.listener);
      }
    }
    this.abortRegistrations.clear();
    this.requests.clear();
    this.activeRequestCount = 0;
  }

  /**
   * Drops a record that has settled, been consumed, has no waiter left, and is
   * no longer active. The `active` term is what keeps the active count and the
   * record map from disagreeing.
   */
  private removeIfFullyConsumed(key: string, record: TrackerRecord<TResult>): void {
    if (
      record.active ||
      !record.settled ||
      !record.consumed ||
      record.waiterCount > 0
    ) {
      return;
    }
    this.requests.delete(key);
  }
}
