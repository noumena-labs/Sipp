import { GenerateRequestId } from '../types.js';
import { createDeferred } from '../utils/async.js';

/**
 * Tracks the lifecycle of a pending request: its promise, settlement state,
 * abort signal, and cleanup. Generic over the result type so both the
 * main-thread runtime (GenerateResponse) and the worker runtime
 * (WorkerRunQueuedRequestResult) can share the same bookkeeping.
 */

export interface TrackedRequest<TResult> {
  promise: Promise<TResult>;
  resolve: (value: TResult) => void;
  reject: (error: unknown) => void;
  settled: boolean;
  consumed: boolean;
  waiterCount: number;
  callbackError: unknown;
  cancelRequested: boolean;
}

interface AbortRegistration {
  signal: AbortSignal;
  listener: () => void;
}

export class RequestTracker<TResult> {
  private readonly completions = new Map<GenerateRequestId, TrackedRequest<TResult>>();
  private readonly abortRegistrations = new Map<GenerateRequestId, Set<AbortRegistration>>();
  private readonly activeRuns = new Set<GenerateRequestId>();

  // ── Query ──────────────────────────────────────────────────────────

  get activeCount(): number {
    return this.activeRuns.size;
  }

  hasActive(requestId: GenerateRequestId): boolean {
    return this.activeRuns.has(requestId);
  }

  get(requestId: GenerateRequestId): TrackedRequest<TResult> | undefined {
    return this.completions.get(requestId);
  }

  /** All request IDs that appear in any internal collection. */
  allTrackedIds(): GenerateRequestId[] {
    const ids = new Set<GenerateRequestId>();
    for (const id of this.completions.keys()) ids.add(id);
    for (const id of this.abortRegistrations.keys()) ids.add(id);
    for (const id of this.activeRuns) ids.add(id);
    return Array.from(ids);
  }

  // ── Lifecycle ──────────────────────────────────────────────────────

  /**
   * Start tracking a request. Returns the existing tracker if already tracked,
   * or creates a new one. Marks the request as active.
   */
  track(requestId: GenerateRequestId): TrackedRequest<TResult> {
    const existing = this.completions.get(requestId);
    if (existing != null) {
      return existing;
    }

    const deferred = createDeferred<TResult>();
    const tracked: TrackedRequest<TResult> = {
      promise: deferred.promise,
      resolve: deferred.resolve,
      reject: deferred.reject,
      settled: false,
      consumed: false,
      waiterCount: 0,
      callbackError: undefined,
      cancelRequested: false,
    };
    // Prevent unhandled rejection warnings for unconsumed requests.
    void tracked.promise.catch(() => {});
    this.completions.set(requestId, tracked);
    this.activeRuns.add(requestId);
    return tracked;
  }

  /**
   * Resolve a tracked request with a result.
   * No-op if already settled.
   */
  resolve(requestId: GenerateRequestId, result: TResult): void {
    const tracked = this.completions.get(requestId);
    if (tracked == null || tracked.settled) {
      return;
    }
    tracked.settled = true;
    tracked.resolve(result);
  }

  /**
   * Reject a tracked request with an error.
   * No-op if already settled.
   */
  reject(requestId: GenerateRequestId, error: unknown): void {
    const tracked = this.completions.get(requestId);
    if (tracked == null || tracked.settled) {
      return;
    }
    tracked.settled = true;
    tracked.reject(error);
  }

  /** Reject all unsettled requests and clear everything. */
  rejectAll(error: unknown): void {
    for (const requestId of this.allTrackedIds()) {
      const tracked = this.completions.get(requestId);
      if (tracked != null && !tracked.settled) {
        tracked.settled = true;
        tracked.reject(error);
      }
      this.releaseSignal(requestId);
    }
    this.completions.clear();
    this.activeRuns.clear();
  }

  // ── Signal management ──────────────────────────────────────────────

  /**
   * Attach an AbortSignal to a request. When the signal fires, `onAbort`
   * is called (typically to issue a cancellation to the engine).
   */
  attachSignal(
    requestId: GenerateRequestId,
    signal: AbortSignal,
    onAbort: () => void
  ): () => void {
    if (signal.aborted) {
      onAbort();
      return () => {};
    }

    const listener = () => onAbort();
    const registration = { signal, listener };
    let registrations = this.abortRegistrations.get(requestId);
    if (registrations == null) {
      registrations = new Set();
      this.abortRegistrations.set(requestId, registrations);
    }
    registrations.add(registration);
    signal.addEventListener('abort', listener, { once: true });
    return () => {
      this.releaseAbortRegistration(requestId, registration);
    };
  }

  /** Detach and clean up the AbortSignal listener for a request. */
  releaseSignal(requestId: GenerateRequestId): void {
    const registrations = this.abortRegistrations.get(requestId);
    if (registrations == null) {
      return;
    }
    for (const registration of registrations) {
      registration.signal.removeEventListener('abort', registration.listener);
    }
    this.abortRegistrations.delete(requestId);
  }

  private releaseAbortRegistration(
    requestId: GenerateRequestId,
    registration: AbortRegistration
  ): void {
    const registrations = this.abortRegistrations.get(requestId);
    if (registrations == null || !registrations.delete(registration)) {
      return;
    }
    registration.signal.removeEventListener('abort', registration.listener);
    if (registrations.size === 0) {
      this.abortRegistrations.delete(requestId);
    }
  }

  // ── Cleanup ────────────────────────────────────────────────────────

  /**
   * Full cleanup for a finished request: release signal, remove from active,
   * and optionally delete its completion entry entirely.
   */
  finalize(
    requestId: GenerateRequestId,
    options: { deleteCompletion?: boolean } = {}
  ): void {
    this.releaseSignal(requestId);
    this.activeRuns.delete(requestId);
    if (options.deleteCompletion) {
      this.completions.delete(requestId);
      return;
    }
    this.cleanupIfConsumed(requestId);
  }

  /**
   * If a request is settled, consumed, and has no active waiters,
   * delete it from the map to free memory.
   */
  cleanupIfConsumed(requestId: GenerateRequestId): void {
    const tracked = this.completions.get(requestId);
    if (
      tracked != null &&
      tracked.settled &&
      tracked.consumed &&
      tracked.waiterCount === 0
    ) {
      this.completions.delete(requestId);
    }
  }

  /** Clear all state (used during runtime reset). */
  clear(): void {
    for (const requestId of Array.from(this.abortRegistrations.keys())) {
      this.releaseSignal(requestId);
    }
    this.completions.clear();
    this.activeRuns.clear();
  }
}
