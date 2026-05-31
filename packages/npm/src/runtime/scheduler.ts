import {
  GenerateRequestId,
  GenerateResponse,
  TokenBatch,
  TokenDeliveryMode,
  TransportObservability,
} from '../engine/inference-types.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  WasmBridge,
} from '../wasm/wasm-bridge.js';
import { RequestTracker } from './request-tracker.js';

// Native owns the scheduling policy; JS drives the outer loop. Interactive
// token delivery asks native to yield via ce_native_yield once per emitted
// token. Batch delivery keeps the monolithic native loop and drains after
// larger slices, avoiding a JSPI round-trip on the decode hot path.
const CONTINUOUS_LOOP_TICK_LIMIT = 1024;
const CONTINUOUS_LOOP_TOKEN_LIMIT = 512;
const CONTINUOUS_LOOP_INTERACTIVE_TOKEN_LIMIT = 256;
const REQUEST_STEP_RESULT_INVALID = -1;
const REQUEST_STEP_RESULT_FATAL_NO_PROGRESS = -2;

type SchedulerFinalizeOptions = {
  consumeCompletedResponse?: boolean;
  deleteCompletion?: boolean;
};

type QueuedRequestSchedulerOptions = {
  tracker: RequestTracker<GenerateResponse>;
  queuedPromptTokenSinks: Map<
    GenerateRequestId,
    (batch: TokenBatch) => void
  >;
  queuedPromptTokenDeliveryModes: Map<GenerateRequestId, TokenDeliveryMode>;
  queuedPromptTokenSinkErrors: Map<GenerateRequestId, unknown>;
  getTransportObservability: () => TransportObservability;
  getBridge: () => WasmBridge;
  finalizeRequest: (
    bridge: WasmBridge,
    requestId: GenerateRequestId,
    options?: SchedulerFinalizeOptions
  ) => void;
  cancelQuery: (requestId: GenerateRequestId) => Promise<boolean>;
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
    this.tokenSinkDecoders.clear();
    this.tokenSinkSequences.clear();
    this.tokenSinkStats.clear();
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

  private requestCancellationForTokenSinkErrors(): void {
    for (const requestId of this.options.tracker.allTrackedIds()) {
      if (
        !this.options.tracker.has(requestId) ||
        this.options.tracker.isSettled(requestId) ||
        this.options.tracker.isCancelRequested(requestId)
      ) {
        continue;
      }

      const tokenSinkError =
        this.options.queuedPromptTokenSinkErrors.get(requestId);
      if (tokenSinkError == null) {
        continue;
      }

      this.options.tracker.setTokenSinkError(requestId, tokenSinkError);
      this.options.tracker.requestCancel(requestId);
      void this.options.cancelQuery(requestId);
    }
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
      this.options.tracker.setTokenSinkError(
        requestId,
        this.options.queuedPromptTokenSinkErrors.get(requestId)
      );
      this.options.tracker.resolve(requestId, response);
      this.options.finalizeRequest(bridge, requestId, {
        deleteCompletion:
          (response.cancelled || this.options.tracker.isCancelRequested(requestId)) &&
          !this.options.tracker.isConsumed(requestId),
      });
      this.forgetTokenSinkStream(requestId);
    } catch (error) {
      this.options.tracker.reject(requestId, error);
      this.options.finalizeRequest(bridge, requestId);
      this.forgetTokenSinkStream(requestId);
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
      this.forgetTokenSinkStream(requestId);
    }
  }

  private async runSchedulerPump(generation: number): Promise<void> {
    const bridge = this.options.getBridge();
    const uninstallYieldDrain = this.installTokenDrainHook(bridge, generation);

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
            { interactiveTokenDelivery: this.hasInteractiveTokenCall() }
          );
          this.drainTokenBufferObserved(bridge);
          this.requestCancellationForTokenSinkErrors();
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
      // drain and request settlement.
      try {
        this.drainTokenBufferObserved(bridge);
      } catch {
        /* cleanup */
      }
    }
  }

  private loopTokenLimit(): number {
    return this.hasInteractiveTokenCall()
      ? CONTINUOUS_LOOP_INTERACTIVE_TOKEN_LIMIT
      : CONTINUOUS_LOOP_TOKEN_LIMIT;
  }

  private hasInteractiveTokenCall(): boolean {
    for (const requestId of this.options.tracker.allTrackedIds()) {
      if (this.options.queuedPromptTokenDeliveryModes.get(requestId) === 'interactive') {
        return true;
      }
    }
    return false;
  }

  // Installs `Module._ce_yield_drain` so interactive delivery can drain the
  // native token buffer on each scheduler yield.
  private installTokenDrainHook(
    bridge: WasmBridge,
    generation: number
  ): () => void {
    if (this.options.queuedPromptTokenSinks.size === 0) {
      return () => { };
    }
    const drain = () => {
      if (generation !== this.schedulerPumpGeneration) {
        return;
      }
      try {
        this.drainTokenBufferObserved(bridge);
      } catch (error) {
        // Drain runs inside the wasm yield body; throwing here aborts the
        // scheduler via a JSPI rejection.  Record + swallow instead.
        for (const requestId of this.options.tracker.allTrackedIds()) {
          this.options.queuedPromptTokenSinkErrors.set(requestId, error);
          this.options.tracker.setTokenSinkError(requestId, error);
        }
      }
    };
    bridge.module._ce_yield_drain = drain;
    return () => {
      if (bridge.module._ce_yield_drain === drain) {
        bridge.module._ce_yield_drain = undefined;
      }
    };
  }

  // Cached token buffer addresses; resolved once per bridge.
  private cachedDrainBridge: WasmBridge | null = null;
  private cachedBufferByteAddr = 0;
  private cachedUsedHeap32Index = 0;
  private cachedDropCountHeap32Index = 0;
  private lastSeenDropCount = 0;
  private readonly tokenSinkDecoders = new Map<number, TextDecoder>();
  private readonly tokenSinkSequences = new Map<number, number>();
  private readonly tokenSinkStats = new Map<
    number,
    {
      framesSent: number;
      bytesSent: number;
      framesDropped: number;
      batchesSent: number;
    }
  >();

  private ensureTokenDrainCache(bridge: WasmBridge): boolean {
    if (this.cachedDrainBridge === bridge) {
      return this.cachedBufferByteAddr !== 0;
    }
    const bufferAddr = bridge.getTokenBufferPointer();
    const usedAddr = bridge.getTokenBufferUsedAddress();
    const dropAddr = bridge.getTokenBufferDropCountAddress();
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

  private drainTokenBufferObserved(bridge: WasmBridge): boolean {
    if (this.options.queuedPromptTokenSinks.size === 0) {
      return false;
    }
    const transport = this.options.getTransportObservability();
    const start = performance.now();
    try {
      return this.drainTokenBuffer(bridge);
    } finally {
      transport.tokenDrainMs =
        (transport.tokenDrainMs ?? 0) + (performance.now() - start);
      transport.tokenDrainCount =
        (transport.tokenDrainCount ?? 0) + 1;
    }
  }

  // Zero-ccall drain: reads `used` via HEAP32, parses records via HEAPU8,
  // and decodes them into TokenBatch sinks. Safe because wasm is suspended
  // inside the `ce_native_yield` body when called from the yield hook.
  private drainTokenBuffer(bridge: WasmBridge): boolean {
    if (this.options.queuedPromptTokenSinks.size === 0) {
      return false;
    }
    if (!this.ensureTokenDrainCache(bridge)) {
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
        console.warn(`[cogentlm] dropped ${delta} token record(s).`);
      }
    }
    const used = heap32[this.cachedUsedHeap32Index];
    if (used <= 0) {
      return false;
    }
    heap32[this.cachedUsedHeap32Index] = 0;
    let offset = this.cachedBufferByteAddr;
    const end = offset + used;
    const tokenSinkBatches = new Map<
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
      const tokenSink = this.options.queuedPromptTokenSinks.get(streamId);
      if (tokenSink != null) {
        const sequence = this.tokenSinkSequences.get(streamId) ?? 0;
        this.tokenSinkSequences.set(streamId, (sequence + 1) >>> 0);
        let batch = tokenSinkBatches.get(streamId);
        if (batch == null) {
          batch = {
            sequenceStart: sequence,
            text: [],
            frameCount: 0,
            byteCount: 0,
            framesDropped: dropDelta,
          };
          tokenSinkBatches.set(streamId, batch);
        }
        const decoded = this.decoderForTokenSink(streamId).decode(payload, {
          stream: true,
        });
        if (decoded.length > 0) {
          batch.text.push(decoded);
        }
        batch.frameCount += 1;
        batch.byteCount += payload.byteLength;
      }
      offset = payloadStart + textLength;
    }
    for (const [requestId, batch] of tokenSinkBatches) {
      this.deliverTokenSinkBatch(requestId, batch);
    }
    return tokenSinkBatches.size > 0;
  }

  private decoderForTokenSink(requestId: number): TextDecoder {
    let decoder = this.tokenSinkDecoders.get(requestId);
    if (decoder == null) {
      decoder = new TextDecoder('utf-8', { fatal: false });
      this.tokenSinkDecoders.set(requestId, decoder);
    }
    return decoder;
  }

  private forgetTokenSinkStream(requestId: number): void {
    this.tokenSinkDecoders.delete(requestId);
    this.tokenSinkSequences.delete(requestId);
    this.tokenSinkStats.delete(requestId);
  }

  private deliverTokenSinkBatch(
    requestId: number,
    batch: {
      sequenceStart: number;
      text: string[];
      frameCount: number;
      byteCount: number;
      framesDropped: number;
    }
  ): void {
    const tokenSink = this.options.queuedPromptTokenSinks.get(requestId);
    if (tokenSink == null || batch.frameCount === 0) {
      return;
    }
    const stats = this.tokenSinkStats.get(requestId) ?? {
      framesSent: 0,
      bytesSent: 0,
      framesDropped: 0,
      batchesSent: 0,
    };
    stats.framesSent += batch.frameCount;
    stats.bytesSent += batch.byteCount;
    stats.framesDropped += batch.framesDropped;
    stats.batchesSent += 1;
    this.tokenSinkStats.set(requestId, stats);
    try {
      tokenSink({
        requestId: String(requestId),
        streamId: requestId,
        sequenceStart: batch.sequenceStart,
        text: batch.text.join(''),
        frameCount: batch.frameCount,
        byteCount: batch.byteCount,
        stats: { ...stats },
      });
    } catch (error) {
      this.options.queuedPromptTokenSinkErrors.set(requestId, error);
    }
  }
}
