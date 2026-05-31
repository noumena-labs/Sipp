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

// Native owns the scheduling policy; JS drives the outer loop. Token emission
// stays on the bulk native loop and drains in chunks so visible tokens do not
// force a per-token JSPI round trip.
const CONTINUOUS_LOOP_TICK_LIMIT = 1024;
const CONTINUOUS_LOOP_TOKEN_LIMIT = 512;
const MAIN_THREAD_LOOP_DURATION_US = 8_000;
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
    this.tokenBatchSinkDecoders.clear();
    this.tokenBatchSinkSequences.clear();
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
    return this.options.getTransportObservability().executionMode === 'main-thread'
      ? MAIN_THREAD_LOOP_DURATION_US
      : 0;
  }

  // Cached token buffer control cell; payload pointer may move if wasm grows it.
  private cachedDrainBridge: WasmBridge | null = null;
  private cachedUsedHeap32Index = -1;
  private readonly tokenBatchSinkDecoders = new Map<number, TextDecoder>();
  private readonly tokenBatchSinkSequences = new Map<number, number>();
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
  // and decodes them into TokenBatch sinks after each native loop chunk.
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
    const tokenBatchSinkBatches = new Map<
      number,
      {
        sequenceStart: number;
        text: string[];
        frameCount: number;
        byteCount: number;
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
      const tokenBatchSink = this.options.queuedPromptTokenBatchSinks.get(streamId);
      if (tokenBatchSink != null) {
        const sequence = this.tokenBatchSinkSequences.get(streamId) ?? 0;
        this.tokenBatchSinkSequences.set(streamId, (sequence + 1) >>> 0);
        let batch = tokenBatchSinkBatches.get(streamId);
        if (batch == null) {
          batch = {
            sequenceStart: sequence,
            text: [],
            frameCount: 0,
            byteCount: 0,
          };
          tokenBatchSinkBatches.set(streamId, batch);
        }
        const decoded = this.decoderForTokenBatchSink(streamId).decode(payload, {
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
    for (const [requestId, batch] of tokenBatchSinkBatches) {
      this.deliverTokenBatchSinkBatch(requestId, batch);
    }
    return tokenBatchSinkBatches.size > 0;
  }

  private decoderForTokenBatchSink(requestId: number): TextDecoder {
    let decoder = this.tokenBatchSinkDecoders.get(requestId);
    if (decoder == null) {
      decoder = new TextDecoder('utf-8', { fatal: false });
      this.tokenBatchSinkDecoders.set(requestId, decoder);
    }
    return decoder;
  }

  private forgetTokenBatchSinkStream(requestId: number): void {
    this.tokenBatchSinkDecoders.delete(requestId);
    this.tokenBatchSinkSequences.delete(requestId);
    this.tokenBatchSinkStats.delete(requestId);
  }

  private deliverTokenBatchSinkBatch(
    requestId: number,
    batch: {
      sequenceStart: number;
      text: string[];
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
        text: batch.text.join(''),
        frameCount: batch.frameCount,
        byteCount: batch.byteCount,
        stats: { ...stats },
      });
    } catch (error) {
      this.options.queuedPromptTokenBatchSinkErrors.set(requestId, error);
    }
  }
}
