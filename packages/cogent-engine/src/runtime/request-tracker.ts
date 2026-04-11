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
  settlementState: 'pending' | 'resolved' | 'rejected';
  settledResult: TResult | undefined;
  settledError: unknown;
  consumed: boolean;
  waiterCount: number;
  callbackError: unknown;
  cancelRequested: boolean;
}

export class RequestTracker<TResult> {
  private readonly completions = new Map<GenerateRequestId, TrackedRequest<TResult>>();
  private readonly signals = new Map<GenerateRequestId, AbortSignal>();
  private readonly abortListeners = new Map<GenerateRequestId, () => void>();
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
    for (const id of this.signals.keys()) ids.add(id);
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
      settlementState: 'pending',
      settledResult: undefined,
      settledError: undefined,
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
    tracked.settlementState = 'resolved';
    tracked.settledResult = result;
    tracked.settledError = undefined;
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
    tracked.settlementState = 'rejected';
    tracked.settledResult = undefined;
    tracked.settledError = error;
    tracked.reject(error);
  }

  /** Reject all unsettled requests and clear everything. */
  rejectAll(error: unknown): void {
    for (const requestId of this.allTrackedIds()) {
      const tracked = this.completions.get(requestId);
      if (tracked != null && !tracked.settled) {
        tracked.settled = true;
        tracked.settlementState = 'rejected';
        tracked.settledResult = undefined;
        tracked.settledError = error;
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
  ): void {
    const listener = () => onAbort();
    this.signals.set(requestId, signal);
    this.abortListeners.set(requestId, listener);
    signal.addEventListener('abort', listener, { once: true });
  }

  /** Detach and clean up the AbortSignal listener for a request. */
  releaseSignal(requestId: GenerateRequestId): void {
    const signal = this.signals.get(requestId);
    const listener = this.abortListeners.get(requestId);
    if (signal != null && listener != null) {
      signal.removeEventListener('abort', listener);
    }
    this.signals.delete(requestId);
    this.abortListeners.delete(requestId);
  }

  // ── Cleanup ────────────────────────────────────────────────────────

  /** Remove a request from active tracking. */
  deactivate(requestId: GenerateRequestId): void {
    this.activeRuns.delete(requestId);
  }

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
    for (const requestId of this.signals.keys()) {
      this.releaseSignal(requestId);
    }
    this.completions.clear();
    this.activeRuns.clear();
  }
}
