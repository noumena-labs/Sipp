import {
  GenerateRequestId,
  GenerateResponse,
  TokenBatch,
  TransportObservability,
} from '../engine/inference-types.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  WasmBridge,
} from '../wasm/wasm-bridge.js';
import { RequestTracker } from './request-tracker.js';

// Native owns model scheduling; JS owns host pump cadence. Token emission
// stays on bulk native loops, but emitted batches must return to JS often
// enough to be visible while the request is still running.
const CONTINUOUS_LOOP_TICK_LIMIT = 1024;
const CONTINUOUS_LOOP_TOKEN_LIMIT = 512;
const MAIN_THREAD_TOKEN_SLICE_US = 8_000;
const WORKER_TOKEN_SLICE_US = 16_000;
const REQUEST_STEP_RESULT_INVALID = -1;
const REQUEST_STEP_RESULT_FATAL_NO_PROGRESS = -2;

type SchedulerFinalizeOptions = {
  consumeCompletedResponse?: boolean;
  deleteCompletion?: boolean;
};

type QueuedRequestSchedulerOptions = {
  tracker: RequestTracker<GenerateResponse>;
  queuedPromptTokenBatchSinks: Map<
    GenerateRequestId,
    (batch: TokenBatch) => void
  >;
  queuedPromptTokenBatchSinkErrors: Map<GenerateRequestId, unknown>;
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
    this.cachedUsedHeap32Index = -1;
    this.tokenBatchSinkStats.clear();
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

  private requestCancellationForTokenBatchSinkErrors(): void {
    for (const requestId of this.options.tracker.allTrackedIds()) {
      if (
        !this.options.tracker.has(requestId) ||
        this.options.tracker.isSettled(requestId) ||
        this.options.tracker.isCancelRequested(requestId)
      ) {
        continue;
      }

      const tokenBatchSinkError =
        this.options.queuedPromptTokenBatchSinkErrors.get(requestId);
      if (tokenBatchSinkError == null) {
        continue;
      }

      this.options.tracker.setTokenBatchSinkError(requestId, tokenBatchSinkError);
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
      this.options.tracker.setTokenBatchSinkError(
        requestId,
        this.options.queuedPromptTokenBatchSinkErrors.get(requestId)
      );
      this.options.tracker.resolve(requestId, response);
      this.options.finalizeRequest(bridge, requestId, {
        deleteCompletion:
          (response.cancelled || this.options.tracker.isCancelRequested(requestId)) &&
          !this.options.tracker.isConsumed(requestId),
      });
      this.forgetTokenBatchSinkStream(requestId);
    } catch (error) {
      this.options.tracker.reject(requestId, error);
      this.options.finalizeRequest(bridge, requestId);
      this.forgetTokenBatchSinkStream(requestId);
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
      this.forgetTokenBatchSinkStream(requestId);
    }
  }

  private async runSchedulerPump(generation: number): Promise<void> {
    const bridge = this.options.getBridge();

    try {
      while (
        generation === this.schedulerPumpGeneration &&
        this.options.tracker.activeCount > 0
      ) {
        try {
          const loopResult = await bridge.runInferenceLoop(
            CONTINUOUS_LOOP_TICK_LIMIT,
            this.options.tracker.activeCount,
            CONTINUOUS_LOOP_TOKEN_LIMIT,
            { maxDurationUs: this.loopDurationUs() }
          );
          this.drainTokenBufferObserved(bridge);
          this.requestCancellationForTokenBatchSinkErrors();
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
      // Final pass to flush tail tokens written before request settlement.
      try {
        this.drainTokenBufferObserved(bridge);
      } catch {
        /* cleanup */
      }
    }
  }

  private loopDurationUs(): number {
    if (this.options.getTransportObservability().executionMode === 'main-thread') {
      return MAIN_THREAD_TOKEN_SLICE_US;
    }
    return this.options.queuedPromptTokenBatchSinks.size > 0
      ? WORKER_TOKEN_SLICE_US
      : 0;
  }

  // Cached token buffer control cell; payload pointer may move if wasm grows it.
  private cachedDrainBridge: WasmBridge | null = null;
  private cachedUsedHeap32Index = -1;
  private readonly tokenBatchDecoder = new TextDecoder('utf-8', { fatal: false });
  private readonly tokenBatchSinkStats = new Map<
    number,
    {
      framesSent: number;
      bytesSent: number;
      batchesSent: number;
    }
  >();

  private ensureTokenDrainCache(bridge: WasmBridge): boolean {
    if (this.cachedDrainBridge === bridge) {
      return this.cachedUsedHeap32Index >= 0;
    }
    const usedAddr = bridge.getTokenBufferUsedAddress();
    if (usedAddr === 0) {
      this.cachedDrainBridge = null;
      this.cachedUsedHeap32Index = -1;
      return false;
    }
    this.cachedDrainBridge = bridge;
    this.cachedUsedHeap32Index = Math.floor(usedAddr / 4);
    return true;
  }

  private drainTokenBufferObserved(bridge: WasmBridge): boolean {
    if (this.options.queuedPromptTokenBatchSinks.size === 0) {
      return false;
    }
    const transport = this.options.getTransportObservability();
    if (!transport.enabled) {
      return this.drainTokenBuffer(bridge);
    }
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

  // Zero-ccall drain: reads `used` via HEAP32, parses batched records via
  // HEAPU8, and decodes one TokenBatch per native batch record.
  private drainTokenBuffer(bridge: WasmBridge): boolean {
    if (this.options.queuedPromptTokenBatchSinks.size === 0) {
      return false;
    }
    if (!this.ensureTokenDrainCache(bridge)) {
      return false;
    }
    const heapU8 = bridge.module.HEAPU8;
    const heap32 = bridge.module.HEAP32;
    const used = heap32[this.cachedUsedHeap32Index];
    if (used <= 0) {
      return false;
    }
    heap32[this.cachedUsedHeap32Index] = 0;
    let offset = bridge.getTokenBufferPointer();
    if (offset === 0) {
      return false;
    }
    const end = offset + used;
    let delivered = false;
    while (offset + 16 <= end) {
      const requestId =
        heapU8[offset] |
        (heapU8[offset + 1] << 8) |
        (heapU8[offset + 2] << 16) |
        (heapU8[offset + 3] << 24);
      const sequenceStart =
        heapU8[offset + 4] |
        (heapU8[offset + 5] << 8) |
        (heapU8[offset + 6] << 16) |
        (heapU8[offset + 7] << 24);
      const frameCount =
        heapU8[offset + 8] |
        (heapU8[offset + 9] << 8) |
        (heapU8[offset + 10] << 16) |
        (heapU8[offset + 11] << 24);
      const textLength =
        heapU8[offset + 12] |
        (heapU8[offset + 13] << 8) |
        (heapU8[offset + 14] << 16) |
        (heapU8[offset + 15] << 24);
      const payloadStart = offset + 16;
      if (payloadStart + textLength > end) {
        break;
      }
      const streamId = requestId >>> 0;
      const tokenBatchSink = this.options.queuedPromptTokenBatchSinks.get(streamId);
      if (tokenBatchSink != null) {
        const payload = heapU8.subarray(payloadStart, payloadStart + textLength);
        this.deliverTokenBatchSinkBatch(streamId, {
          sequenceStart: sequenceStart >>> 0,
          text: this.tokenBatchDecoder.decode(payload),
          frameCount: frameCount >>> 0,
          byteCount: textLength >>> 0,
        });
        delivered = true;
      }
      offset = payloadStart + textLength;
    }
    return delivered;
  }

  private forgetTokenBatchSinkStream(requestId: number): void {
    this.tokenBatchSinkStats.delete(requestId);
  }

  private deliverTokenBatchSinkBatch(
    requestId: number,
    batch: {
      sequenceStart: number;
      text: string;
      frameCount: number;
      byteCount: number;
    }
  ): void {
    const tokenBatchSink = this.options.queuedPromptTokenBatchSinks.get(requestId);
    if (tokenBatchSink == null || batch.frameCount === 0) {
      return;
    }
    const stats = this.tokenBatchSinkStats.get(requestId) ?? {
      framesSent: 0,
      bytesSent: 0,
      batchesSent: 0,
    };
    stats.framesSent += batch.frameCount;
    stats.bytesSent += batch.byteCount;
    stats.batchesSent += 1;
    this.tokenBatchSinkStats.set(requestId, stats);
    try {
      tokenBatchSink({
        requestId: String(requestId),
        streamId: requestId,
        sequenceStart: batch.sequenceStart,
        text: batch.text,
        frameCount: batch.frameCount,
        byteCount: batch.byteCount,
        stats: { ...stats },
      });
    } catch (error) {
      this.options.queuedPromptTokenBatchSinkErrors.set(requestId, error);
    }
  }
}
