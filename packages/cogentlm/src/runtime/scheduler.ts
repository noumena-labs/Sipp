import {
  GenerateRequestId,
  GenerateResponse,
  TransportObservability,
} from '../types.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  REQUEST_STEP_RESULT_FATAL_NO_PROGRESS,
  REQUEST_STEP_RESULT_INVALID,
} from './main-thread-runtime-constants.js';
import { RequestTracker } from './request-tracker.js';
import { WasmBridge } from '../wasm/wasm-bridge.js';
import type { StreamingRingWriter } from './streaming-ring.js';

// Native owns the scheduling policy; JS drives the loop and handles UI 
// responsiveness through internal native yielding (JSPI).
const CONTINUOUS_LOOP_TICK_LIMIT = 1024;
const CONTINUOUS_LOOP_TOKEN_LIMIT = 512;

type SchedulerFinalizeOptions = {
  consumeCompletedResponse?: boolean;
  deleteCompletion?: boolean;
};

type QueuedRequestSchedulerOptions = {
  tracker: RequestTracker<GenerateResponse>;
  queuedPromptCallbacks: Map<
    GenerateRequestId,
    ((token: string) => void) | undefined
  >;
  queuedPromptCallbackErrors: Map<GenerateRequestId, unknown>;
  getTransportObservability: () => TransportObservability;
  getBridge: () => WasmBridge;
  finalizeRequest: (
    bridge: WasmBridge,
    requestId: GenerateRequestId,
    options?: SchedulerFinalizeOptions
  ) => void;
  cancelQuery: (requestId: GenerateRequestId) => Promise<boolean>;
  // Worker-side SAB ring writer.  When set, the scheduler installs
  // `_ce_yield_drain` to copy the native streaming buffer into the ring
  // once per yield.  Null on the main-thread engine.
  getStreamingRingWriter?: () => StreamingRingWriter | null;
};

export class QueuedRequestScheduler {
  private schedulerPumpPromise: Promise<void> | null = null;
  private schedulerPumpGeneration = 0;

  public constructor(private readonly options: QueuedRequestSchedulerOptions) { }

  public reset(): void {
    this.schedulerPumpGeneration += 1;
    this.schedulerPumpPromise = null;
    this.cachedDrainBridge = null;
    this.cachedBufferByteAddr = 0;
    this.cachedUsedHeap32Index = 0;
    this.cachedDropCountHeap32Index = 0;
    this.lastSeenDropCount = 0;
  }

  public track(requestId: GenerateRequestId) {
    const tracked = this.options.tracker.track(requestId);
    this.ensureRunning();
    return tracked;
  }

  public ensureRunning(): void {
    if (
      this.schedulerPumpPromise != null ||
      this.options.tracker.activeCount === 0
    ) {
      return;
    }

    const generation = this.schedulerPumpGeneration;
    const pumpPromise = this.runSchedulerPump(generation);
    this.schedulerPumpPromise = pumpPromise;
    void pumpPromise.finally(() => {
      if (this.schedulerPumpPromise === pumpPromise) {
        this.schedulerPumpPromise = null;
        if (
          generation === this.schedulerPumpGeneration &&
          this.options.tracker.activeCount > 0
        ) {
          this.ensureRunning();
        }
      }
    });
  }


  private requestCancellationForCallbackErrors(): void {
    for (const requestId of this.options.tracker.allTrackedIds()) {
      const tracked = this.options.tracker.get(requestId);
      if (tracked == null || tracked.settled || tracked.cancelRequested) {
        continue;
      }

      const callbackError =
        this.options.queuedPromptCallbackErrors.get(requestId);
      if (callbackError == null) {
        continue;
      }

      tracked.callbackError = callbackError;
      tracked.cancelRequested = true;
      void this.options.cancelQuery(requestId);
    }
  }

  public settleCompletedRequestIfPresent(
    bridge: WasmBridge,
    requestId: GenerateRequestId
  ): boolean {
    const tracked = this.options.tracker.get(requestId);
    if (tracked == null || tracked.settled) {
      return false;
    }

    const status = bridge.getCompletedRequestStatus(requestId);
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      return false;
    }

    try {
      const response = bridge.takeCompletedResponse(requestId);
      tracked.callbackError =
        this.options.queuedPromptCallbackErrors.get(requestId);
      this.options.tracker.resolve(requestId, response);
      this.options.finalizeRequest(bridge, requestId, {
        deleteCompletion:
          (response.cancelled || tracked.cancelRequested) && !tracked.consumed,
      });
    } catch (error) {
      this.options.tracker.reject(requestId, error);
      this.options.finalizeRequest(bridge, requestId);
    }
    return true;
  }

  private settleCompletedQueuedRequestsByIds(
    bridge: WasmBridge,
    requestIds: Iterable<GenerateRequestId>
  ): boolean {
    let settledAny = false;
    for (const requestId of requestIds) {
      if (this.options.tracker.get(requestId) == null) {
        continue;
      }
      settledAny =
        this.settleCompletedRequestIfPresent(bridge, requestId) || settledAny;
    }
    return settledAny;
  }

  private settleCompletedTrackedRequests(bridge: WasmBridge): boolean {
    let settledAny = false;
    for (const requestId of this.options.tracker.allTrackedIds()) {
      settledAny =
        this.settleCompletedRequestIfPresent(bridge, requestId) || settledAny;
    }
    return settledAny;
  }

  private drainRuntimeEvents(bridge: WasmBridge) {
    // The native runtime-event queue can hold the full token stream (one
    // event per emitted token) for an entire query before JS gets a chance
    // to drain.  Cap the per-call drain at the reusable buffer's event
    // capacity but loop until the native queue reports it returned fewer
    // events than asked, so we never drop the tail of a long response.
    const perCallEventCap = Math.max(
      256,
      this.options.tracker.activeCount * 64
    );
    const transport = this.options.getTransportObservability();
    const drainStart = performance.now();
    let drained = bridge.drainRuntimeEvents(perCallEventCap);
    const aggregatedTokenEvents: typeof drained.tokenEvents = [];
    const aggregatedTerminalIds: typeof drained.terminalRequestIds = [];
    let aggregatedTextBytes = 0;
    let totalEventCount = 0;
    let callbackMs = 0;
    let callbackCount = 0;
    while (true) {
      for (const tokenEvent of drained.tokenEvents) {
        const callback = this.options.queuedPromptCallbacks.get(tokenEvent.requestId);
        if (
          !callback ||
          this.options.queuedPromptCallbackErrors.has(tokenEvent.requestId)
        ) {
          continue;
        }
        const cbStart = performance.now();
        try {
          callback(tokenEvent.token);
        } catch (error) {
          this.options.queuedPromptCallbackErrors.set(tokenEvent.requestId, error);
        }
        callbackMs += performance.now() - cbStart;
        callbackCount += 1;
      }
      aggregatedTokenEvents.push(...drained.tokenEvents);
      aggregatedTerminalIds.push(...drained.terminalRequestIds);
      aggregatedTextBytes += drained.textBytes;
      totalEventCount += drained.eventCount;
      if (drained.eventCount < perCallEventCap) {
        break;
      }
      drained = bridge.drainRuntimeEvents(perCallEventCap);
    }
    // The drain *includes* user-callback time; record both separately so
    // consumers can see callback contribution to bridge jitter.
    transport.runtimeEventDrainMs =
      (transport.runtimeEventDrainMs ?? 0) + (performance.now() - drainStart);
    transport.runtimeEventDrainCount =
      (transport.runtimeEventDrainCount ?? 0) + 1;
    if (callbackCount > 0) {
      transport.tokenCallbackMs = (transport.tokenCallbackMs ?? 0) + callbackMs;
      transport.tokenCallbackCount =
        (transport.tokenCallbackCount ?? 0) + callbackCount;
    }

    return {
      eventCount: totalEventCount,
      tokenEvents: aggregatedTokenEvents,
      terminalRequestIds: aggregatedTerminalIds,
      textBytes: aggregatedTextBytes,
    };
  }

  private rejectPendingQueuedRequests(
    bridge: WasmBridge,
    error: unknown
  ): void {
    for (const requestId of this.options.tracker.allTrackedIds()) {
      const tracked = this.options.tracker.get(requestId);
      if (tracked == null || tracked.settled) {
        continue;
      }
      this.options.tracker.reject(requestId, error);
      this.options.finalizeRequest(bridge, requestId, {
        deleteCompletion: true,
      });
    }
  }

  private async runSchedulerPump(generation: number): Promise<void> {
    const bridge = this.options.getBridge();
    const uninstallYieldDrain = this.installStreamingDrainHook(bridge, generation);

    try {
      const transport = this.options.getTransportObservability();
      while (
        generation === this.schedulerPumpGeneration &&
        this.options.tracker.activeCount > 0
      ) {
        try {
          const loopStart = performance.now();
          const loopResult = await bridge.runInferenceLoop(
            CONTINUOUS_LOOP_TICK_LIMIT,
            this.options.tracker.activeCount,
            CONTINUOUS_LOOP_TOKEN_LIMIT
          );
          transport.schedulerProgressMs =
            (transport.schedulerProgressMs ?? 0) +
            (performance.now() - loopStart);
          transport.schedulerProgressCount =
            (transport.schedulerProgressCount ?? 0) + 1;
          const drainedEvents = this.drainRuntimeEvents(bridge);
          this.requestCancellationForCallbackErrors();
          this.settleCompletedQueuedRequestsByIds(bridge, drainedEvents.terminalRequestIds);
          if (loopResult.completedResponseCount > drainedEvents.terminalRequestIds.length) {
            this.settleCompletedTrackedRequests(bridge);
          }
          if (loopResult.stepResult === REQUEST_STEP_RESULT_INVALID) {
            this.rejectPendingQueuedRequests(bridge, new Error('Inference loop became invalid.'));
            break;
          }
          if (loopResult.stepResult === REQUEST_STEP_RESULT_FATAL_NO_PROGRESS) {
            this.rejectPendingQueuedRequests(bridge, new Error('Inference loop failed to make progress.'));
            break;
          }
        } catch (error) {
          if (generation === this.schedulerPumpGeneration) {
            this.rejectPendingQueuedRequests(bridge, error);
          }
          break;
        }
      }
    } finally {
      uninstallYieldDrain();
      try { this.drainStreamingBufferToRing(bridge); } catch { /* cleanup */ }
      try { this.drainRuntimeEvents(bridge); } catch { /* cleanup */ }
    }
  }

  // Installs `Module._ce_yield_drain` to copy the native streaming buffer
  // into the worker SAB ring once per yield.  No-op when no ring writer.
  private installStreamingDrainHook(
    bridge: WasmBridge,
    generation: number
  ): () => void {
    const ringWriter = this.options.getStreamingRingWriter?.() ?? null;
    if (ringWriter == null) {
      return () => {};
    }
    const moduleAny = bridge.module as any;
    const drain = () => {
      if (generation !== this.schedulerPumpGeneration) {
        return;
      }
      const transport = this.options.getTransportObservability();
      const start = performance.now();
      try {
        this.drainStreamingBufferToRing(bridge);
      } catch (error) {
        // Drain runs inside the wasm yield body; throwing here aborts the
        // scheduler via a JSPI rejection.  Record + swallow instead.
        for (const requestId of this.options.tracker.allTrackedIds()) {
          this.options.queuedPromptCallbackErrors.set(requestId, error);
        }
      }
      transport.streamingDrainMs =
        (transport.streamingDrainMs ?? 0) + (performance.now() - start);
      transport.streamingDrainCount =
        (transport.streamingDrainCount ?? 0) + 1;
    };
    moduleAny._ce_yield_drain = drain;
    return () => {
      if (moduleAny._ce_yield_drain === drain) {
        moduleAny._ce_yield_drain = undefined;
      }
    };
  }

  // Cached streaming buffer addresses; resolved once per bridge.
  private cachedDrainBridge: WasmBridge | null = null;
  private cachedBufferByteAddr = 0;
  private cachedUsedHeap32Index = 0;
  private cachedDropCountHeap32Index = 0;
  private lastSeenDropCount = 0;

  private ensureStreamingDrainCache(bridge: WasmBridge): boolean {
    if (this.cachedDrainBridge === bridge) {
      return this.cachedBufferByteAddr !== 0;
    }
    const bufferAddr = bridge.getStreamingBufferPointer();
    const usedAddr = bridge.getStreamingBufferUsedAddress();
    const dropAddr = bridge.getStreamingBufferDropCountAddress();
    if (bufferAddr === 0 || usedAddr === 0 || dropAddr === 0) {
      this.cachedDrainBridge = null;
      this.cachedBufferByteAddr = 0;
      return false;
    }
    this.cachedDrainBridge = bridge;
    this.cachedBufferByteAddr = bufferAddr;
    this.cachedUsedHeap32Index = Math.floor(usedAddr / 4);
    this.cachedDropCountHeap32Index = Math.floor(dropAddr / 4);
    this.lastSeenDropCount = 0;
    return true;
  }

  // Zero-ccall drain: reads `used` via HEAP32, parses records via HEAPU8,
  // writes each into the SAB ring, then clears the `used` cell.  Safe
  // because wasm is suspended inside the `ce_native_yield` body.
  private drainStreamingBufferToRing(bridge: WasmBridge): void {
    const ringWriter = this.options.getStreamingRingWriter?.() ?? null;
    if (ringWriter == null) {
      return;
    }
    if (!this.ensureStreamingDrainCache(bridge)) {
      return;
    }
    const heapU8 = bridge.module.HEAPU8;
    const heap32 = bridge.module.HEAP32;
    const totalDrops = heap32[this.cachedDropCountHeap32Index];
    if (totalDrops !== this.lastSeenDropCount) {
      const delta = (totalDrops - this.lastSeenDropCount) | 0;
      this.lastSeenDropCount = totalDrops;
      if (delta > 0 && typeof console !== 'undefined') {
        console.warn(`[cogentlm] dropped ${delta} streaming token record(s).`);
      }
    }
    const used = heap32[this.cachedUsedHeap32Index];
    if (used <= 0) {
      return;
    }
    heap32[this.cachedUsedHeap32Index] = 0;
    let offset = this.cachedBufferByteAddr;
    const end = offset + used;
    while (offset + 8 <= end) {
      const requestId =
        heapU8[offset] |
        (heapU8[offset + 1] << 8) |
        (heapU8[offset + 2] << 16) |
        (heapU8[offset + 3] << 24);
      const textLength =
        heapU8[offset + 4] |
        (heapU8[offset + 5] << 8) |
        (heapU8[offset + 6] << 16) |
        (heapU8[offset + 7] << 24);
      const payloadStart = offset + 8;
      if (payloadStart + textLength > end) {
        break;
      }
      const payload = heapU8.subarray(payloadStart, payloadStart + textLength);
      ringWriter.tryWriteBytes(requestId >>> 0, payload);
      offset = payloadStart + textLength;
    }
  }
}
