import {
  GenerateRequestId,
  GenerateResponse,
  TokenBatch,
  TokenFlushMode,
  TransportObservability,
} from '../core/inference-types.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  WasmBridge,
} from '../wasm/wasm-bridge.js';
import { RequestTracker } from './request-tracker.js';
import type { StreamingRingWriter } from './streaming-ring.js';

// Native owns the scheduling policy; JS drives the outer loop. Token-flush
// streams ask native to yield via ce_native_yield once per emitted token for
// interactive delivery. Batch-flush streams keep the monolithic native loop
// and drain after larger slices, avoiding a JSPI round-trip on the decode hot
// path while still using the same token transport.
const CONTINUOUS_LOOP_TICK_LIMIT = 1024;
const CONTINUOUS_LOOP_TOKEN_LIMIT = 512;
const CONTINUOUS_LOOP_TOKEN_FLUSH_LIMIT = 256;
const REQUEST_STEP_RESULT_INVALID = -1;
const REQUEST_STEP_RESULT_FATAL_NO_PROGRESS = -2;

type SchedulerFinalizeOptions = {
  consumeCompletedResponse?: boolean;
  deleteCompletion?: boolean;
};

type QueuedRequestSchedulerOptions = {
  tracker: RequestTracker<GenerateResponse>;
  queuedPromptCallbacks: Map<
    GenerateRequestId,
    ((batch: TokenBatch) => void) | undefined
  >;
  queuedPromptTokenFlushModes: Map<GenerateRequestId, TokenFlushMode>;
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
  /**
   * Called on the worker thread after successful copy to the SAB ring.
   * Typically used to post a 'streaming-tick' message to the main thread.
   */
  onStreamingTick?: () => void;
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
    this.callbackDecoders.clear();
    this.callbackSequences.clear();
    this.callbackStats.clear();
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


  private requestCancellationForCallbackErrors(): boolean {
    let canceledAny = false;
    for (const requestId of this.options.tracker.allTrackedIds()) {
      if (
        !this.options.tracker.has(requestId) ||
        this.options.tracker.isSettled(requestId) ||
        this.options.tracker.isCancelRequested(requestId)
      ) {
        continue;
      }

      const callbackError =
        this.options.queuedPromptCallbackErrors.get(requestId);
      if (callbackError == null) {
        continue;
      }

      this.options.tracker.setCallbackError(requestId, callbackError);
      this.options.tracker.requestCancel(requestId);
      void this.options.cancelQuery(requestId);
      canceledAny = true;
    }
    return canceledAny;
  }

  public settleCompletedRequestIfPresent(
    bridge: WasmBridge,
    requestId: GenerateRequestId
  ): boolean {
    if (
      !this.options.tracker.has(requestId) ||
      this.options.tracker.isSettled(requestId)
    ) {
      return false;
    }

    const status = bridge.getCompletedRequestStatus(requestId);
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      return false;
    }

    try {
      const response = bridge.takeCompletedResponse(requestId);
      this.options.tracker.setCallbackError(
        requestId,
        this.options.queuedPromptCallbackErrors.get(requestId)
      );
      this.options.tracker.resolve(requestId, response);
      this.options.finalizeRequest(bridge, requestId, {
        deleteCompletion:
          (response.cancelled || this.options.tracker.isCancelRequested(requestId)) &&
          !this.options.tracker.isConsumed(requestId),
      });
      this.forgetCallbackStream(requestId);
    } catch (error) {
      this.options.tracker.reject(requestId, error);
      this.options.finalizeRequest(bridge, requestId);
      this.forgetCallbackStream(requestId);
    }
    return true;
  }

  private settleCompletedTrackedRequests(bridge: WasmBridge): boolean {
    let settledAny = false;
    for (const requestId of this.options.tracker.allTrackedIds()) {
      settledAny =
        this.settleCompletedRequestIfPresent(bridge, requestId) || settledAny;
    }
    return settledAny;
  }

  private rejectPendingQueuedRequests(
    bridge: WasmBridge,
    error: unknown
  ): void {
    for (const requestId of this.options.tracker.allTrackedIds()) {
      if (
        !this.options.tracker.has(requestId) ||
        this.options.tracker.isSettled(requestId)
      ) {
        continue;
      }
      this.options.tracker.reject(requestId, error);
      this.options.finalizeRequest(bridge, requestId, {
        deleteCompletion: true,
      });
      this.forgetCallbackStream(requestId);
    }
  }

  private async runSchedulerPump(generation: number): Promise<void> {
    const bridge = this.options.getBridge();
    const uninstallYieldDrain = this.installStreamingDrainHook(bridge, generation);

    try {
      while (
        generation === this.schedulerPumpGeneration &&
        this.options.tracker.activeCount > 0
      ) {
        try {
          const loopResult = await bridge.runInferenceLoop(
            CONTINUOUS_LOOP_TICK_LIMIT,
            this.options.tracker.activeCount,
            this.loopTokenLimit(),
            { streamingActive: this.hasTokenFlushCall() }
          );
          const ringWritten = this.drainStreamingBuffer(bridge);
          if (ringWritten) {
            this.options.onStreamingTick?.();
          }
          const hadCancellations = this.requestCancellationForCallbackErrors();
          if (hadCancellations) {
            this.options.onStreamingTick?.();
          }
          if (loopResult.completedResponseCount > 0) {
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
      // Final pass to flush tail tokens written between the last yield
      // drain and request settlement.  The main-side tick will pick
      // them up on its next message (or via the drain-on-resolve in the
      // worker client's call finalizer).
      try {
        if (this.drainStreamingBuffer(bridge)) {
          this.options.onStreamingTick?.();
        }
      } catch {
        /* cleanup */
      }
    }
  }

  private loopTokenLimit(): number {
    return this.hasTokenFlushCall()
      ? CONTINUOUS_LOOP_TOKEN_FLUSH_LIMIT
      : CONTINUOUS_LOOP_TOKEN_LIMIT;
  }

  private hasTokenFlushCall(): boolean {
    for (const requestId of this.options.tracker.allTrackedIds()) {
      if (this.options.queuedPromptTokenFlushModes.get(requestId) === 'token') {
        return true;
      }
    }
    return false;
  }

  // Installs `Module._ce_yield_drain` to copy the native streaming buffer
  // into the active token transport once per yield.  No-op when no request
  // asked for token delivery.
  private installStreamingDrainHook(
    bridge: WasmBridge,
    generation: number
  ): () => void {
    const ringWriter = this.options.getStreamingRingWriter?.() ?? null;
    if (ringWriter == null && this.options.queuedPromptCallbacks.size === 0) {
      return () => { };
    }
    const drain = () => {
      if (generation !== this.schedulerPumpGeneration) {
        return;
      }
      const transport = this.options.getTransportObservability();
      const start = performance.now();
      try {
        if (this.drainStreamingBuffer(bridge)) {
          this.options.onStreamingTick?.();
        }
      } catch (error) {
        // Drain runs inside the wasm yield body; throwing here aborts the
        // scheduler via a JSPI rejection.  Record + swallow instead.
        for (const requestId of this.options.tracker.allTrackedIds()) {
          this.options.queuedPromptCallbackErrors.set(requestId, error);
          this.options.tracker.setCallbackError(requestId, error);
        }
      }
      transport.streamingDrainMs =
        (transport.streamingDrainMs ?? 0) + (performance.now() - start);
      transport.streamingDrainCount =
        (transport.streamingDrainCount ?? 0) + 1;
    };
    bridge.module._ce_yield_drain = drain;
    return () => {
      if (bridge.module._ce_yield_drain === drain) {
        bridge.module._ce_yield_drain = undefined;
      }
    };
  }

  // Cached streaming buffer addresses; resolved once per bridge.
  private cachedDrainBridge: WasmBridge | null = null;
  private cachedBufferByteAddr = 0;
  private cachedUsedHeap32Index = 0;
  private cachedDropCountHeap32Index = 0;
  private lastSeenDropCount = 0;
  private readonly callbackDecoders = new Map<number, TextDecoder>();
  private readonly callbackSequences = new Map<number, number>();
  private readonly callbackStats = new Map<
    number,
    {
      framesSent: number;
      bytesSent: number;
      framesDropped: number;
      batchesSent: number;
    }
  >();

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
  // writes them into the SAB ring when present, or decodes them into
  // TokenBatch callbacks when no ring is installed.  Safe because wasm is
  // suspended inside the `ce_native_yield` body.
  // Returns true if any bytes were written to the ring.
  private drainStreamingBuffer(bridge: WasmBridge): boolean {
    const ringWriter = this.options.getStreamingRingWriter?.() ?? null;
    const hasCallbacks = this.options.queuedPromptCallbacks.size > 0;
    if (ringWriter == null && !hasCallbacks) {
      return false;
    }
    const deliverCallbacks = ringWriter == null && hasCallbacks;
    if (!this.ensureStreamingDrainCache(bridge)) {
      return false;
    }
    const heapU8 = bridge.module.HEAPU8;
    const heap32 = bridge.module.HEAP32;
    const totalDrops = heap32[this.cachedDropCountHeap32Index];
    let dropDelta = 0;
    if (totalDrops !== this.lastSeenDropCount) {
      const delta = (totalDrops - this.lastSeenDropCount) | 0;
      dropDelta = delta > 0 ? delta : 0;
      this.lastSeenDropCount = totalDrops;
      if (delta > 0 && typeof console !== 'undefined') {
        console.warn(`[cogentlm] dropped ${delta} streaming token record(s).`);
      }
    }
    const used = heap32[this.cachedUsedHeap32Index];
    if (used <= 0) {
      return false;
    }
    heap32[this.cachedUsedHeap32Index] = 0;
    let offset = this.cachedBufferByteAddr;
    const end = offset + used;
    let ringWritten = false;
    const callbackBatches = new Map<
      number,
      {
        sequenceStart: number;
        text: string[];
        frameCount: number;
        byteCount: number;
        framesDropped: number;
      }
    >();
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
      const streamId = requestId >>> 0;
      if (ringWriter?.tryWriteBytes(streamId, payload)) {
        ringWritten = true;
      }
      if (deliverCallbacks) {
        const callback = this.options.queuedPromptCallbacks.get(streamId);
        if (callback != null) {
          const sequence = this.callbackSequences.get(streamId) ?? 0;
          this.callbackSequences.set(streamId, (sequence + 1) >>> 0);
          let batch = callbackBatches.get(streamId);
          if (batch == null) {
            batch = {
              sequenceStart: sequence,
              text: [],
              frameCount: 0,
              byteCount: 0,
              framesDropped: dropDelta,
            };
            callbackBatches.set(streamId, batch);
          }
          const decoded = this.decoderForCallback(streamId).decode(payload, {
            stream: true,
          });
          if (decoded.length > 0) {
            batch.text.push(decoded);
          }
          batch.frameCount += 1;
          batch.byteCount += payload.byteLength;
        }
      }
      offset = payloadStart + textLength;
    }
    for (const [requestId, batch] of callbackBatches) {
      this.deliverCallbackBatch(requestId, batch);
    }
    return ringWritten;
  }

  private decoderForCallback(requestId: number): TextDecoder {
    let decoder = this.callbackDecoders.get(requestId);
    if (decoder == null) {
      decoder = new TextDecoder('utf-8', { fatal: false });
      this.callbackDecoders.set(requestId, decoder);
    }
    return decoder;
  }

  private forgetCallbackStream(requestId: number): void {
    this.callbackDecoders.delete(requestId);
    this.callbackSequences.delete(requestId);
    this.callbackStats.delete(requestId);
  }

  private deliverCallbackBatch(
    requestId: number,
    batch: {
      sequenceStart: number;
      text: string[];
      frameCount: number;
      byteCount: number;
      framesDropped: number;
    }
  ): void {
    const callback = this.options.queuedPromptCallbacks.get(requestId);
    if (callback == null || batch.frameCount === 0) {
      return;
    }
    const stats = this.callbackStats.get(requestId) ?? {
      framesSent: 0,
      bytesSent: 0,
      framesDropped: 0,
      batchesSent: 0,
    };
    stats.framesSent += batch.frameCount;
    stats.bytesSent += batch.byteCount;
    stats.framesDropped += batch.framesDropped;
    stats.batchesSent += 1;
    this.callbackStats.set(requestId, stats);
    try {
      callback({
        requestId: String(requestId),
        streamId: requestId,
        sequenceStart: batch.sequenceStart,
        text: batch.text.join(''),
        frameCount: batch.frameCount,
        byteCount: batch.byteCount,
        stats: { ...stats },
      });
    } catch (error) {
      this.options.queuedPromptCallbackErrors.set(requestId, error);
    }
  }
}
