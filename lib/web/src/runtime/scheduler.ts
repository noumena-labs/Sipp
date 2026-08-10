import {
  GenerateRequestHandle,
  GenerateRequestId,
  GenerateResponse,
  TokenBatch,
  TokenEmissionStats,
  TransportObservability,
} from '../engine/inference-types.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  WasmBridge,
} from '../wasm/wasm-bridge.js';
import { RequestTracker } from './request-tracker.js';
import { SharedTokenRingReader } from './shared-token-ring.js';

// Native owns model scheduling; JS only drains the shared token ring after a
// host loop returns. Worker-mode token presentation is pulled by the main
// thread from the same ring and does not run through this scheduler.
const CONTINUOUS_LOOP_TICK_LIMIT = 1024;
const CONTINUOUS_LOOP_TOKEN_LIMIT = 512;
const STREAMING_LOOP_TOKEN_LIMIT = 1;
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
  getTransportObservability: () => TransportObservability;
  getRuntimeGeneration: () => number;
  finalizeRequest: (
    bridge: WasmBridge | null,
    request: GenerateRequestHandle,
    options?: SchedulerFinalizeOptions
  ) => void;
  cancelQuery: (request: GenerateRequestHandle) => Promise<boolean>;
  /** Runs `operation` inside the runtime's serialized Wasm-bridge queue. */
  withWasmBridge: <T>(
    operation: (bridge: WasmBridge) => T | Promise<T>
  ) => Promise<T>;
};

export class QueuedRequestScheduler {
  private schedulerPumpPromise: Promise<void> | null = null;
  private schedulerPumpTimer: ReturnType<typeof setTimeout> | null = null;
  private schedulerPumpGeneration = 0;

  public constructor(private readonly options: QueuedRequestSchedulerOptions) { }

  public reset(): void {
    this.schedulerPumpGeneration += 1;
    this.schedulerPumpPromise = null;
    if (this.schedulerPumpTimer != null) {
      clearTimeout(this.schedulerPumpTimer);
      this.schedulerPumpTimer = null;
    }
    this.tokenRingBridge = null;
    this.tokenRingReader = null;
    this.tokenBatchSinkStats.clear();
  }

  public track(request: GenerateRequestHandle) {
    const tracked = this.options.tracker.track(request);
    this.scheduleRunning();
    return tracked;
  }

  public ensureRunning(): void {
    this.scheduleRunning();
  }

  private scheduleRunning(): void {
    if (
      this.schedulerPumpPromise != null ||
      this.schedulerPumpTimer != null ||
      this.options.tracker.activeCount === 0
    ) {
      return;
    }

    const generation = this.schedulerPumpGeneration;
    this.schedulerPumpTimer = setTimeout(() => {
      this.schedulerPumpTimer = null;
      this.startPump(generation);
    }, 0);
  }

  private startPump(generation: number): void {
    if (
      this.schedulerPumpPromise != null ||
      generation !== this.schedulerPumpGeneration ||
      this.options.tracker.activeCount === 0
    ) {
      return;
    }

    const pumpPromise = this.runSchedulerPump(generation);
    this.schedulerPumpPromise = pumpPromise;
    void pumpPromise
      .catch((error: unknown) => {
        // Bridge acquisition happens before the pump callback. Reject current
        // requests before the finalizer decides whether another pump is needed.
        if (generation === this.schedulerPumpGeneration) {
          this.rejectPendingQueuedRequests(null, error);
        }
      })
      .finally(() => {
        if (this.schedulerPumpPromise === pumpPromise) {
          this.schedulerPumpPromise = null;
          if (
            generation === this.schedulerPumpGeneration &&
            this.options.tracker.activeCount > 0
          ) {
            this.scheduleRunning();
          }
        }
      });
  }

  private requestCancellationForTokenBatchSinkErrors(): void {
    for (const tracked of this.options.tracker.records()) {
      if (
        tracked.settled ||
        tracked.cancelRequested ||
        !tracked.tokenBatchSinkFailed
      ) {
        continue;
      }
      this.options.tracker.requestCancel(tracked.request);
      // Best effort: the request still settles through the tracker.
      void this.options.cancelQuery(tracked.request).catch(() => {});
    }
  }

  public settleCompletedRequestIfPresent(
    bridge: WasmBridge,
    requestId: GenerateRequestHandle
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
      this.options.tracker.resolve(requestId, response);
      this.options.finalizeRequest(bridge, requestId, {
        deleteCompletion:
          (response.cancelled || tracked.cancelRequested) && !tracked.consumed,
      });
    } catch (error) {
      this.options.tracker.reject(requestId, error);
      this.options.finalizeRequest(bridge, requestId);
    }
    this.forgetTokenBatchSinkStream(requestId);
    return true;
  }

  private settleCompletedTrackedRequests(bridge: WasmBridge): boolean {
    let settledAny = false;
    for (const tracked of this.options.tracker.records()) {
      settledAny =
        this.settleCompletedRequestIfPresent(bridge, tracked.request) || settledAny;
    }
    return settledAny;
  }

  private rejectPendingQueuedRequests(
    bridge: WasmBridge | null,
    error: unknown
  ): void {
    for (const tracked of this.options.tracker.records()) {
      if (tracked.settled) {
        continue;
      }
      this.options.tracker.reject(tracked.request, error);
      this.options.finalizeRequest(bridge, tracked.request, {
        deleteCompletion: true,
      });
      this.forgetTokenBatchSinkStream(tracked.request);
    }
  }

  private async runSchedulerPump(generation: number): Promise<void> {
    await this.options.withWasmBridge(async (bridge) => {
      try {
        if (
          generation !== this.schedulerPumpGeneration ||
          this.options.tracker.activeCount === 0
        ) {
          return;
        }

        const generatedTokenLimit =
          this.options.queuedPromptTokenBatchSinks.size > 0
            ? STREAMING_LOOP_TOKEN_LIMIT
            : CONTINUOUS_LOOP_TOKEN_LIMIT;
        const runtimeGeneration = this.options.getRuntimeGeneration();
        const loopResult = await this.runInferenceLoopObserved(
          bridge,
          runtimeGeneration,
          CONTINUOUS_LOOP_TICK_LIMIT,
          this.options.tracker.activeCount,
          generatedTokenLimit
        );
        this.drainTokenRingObserved(bridge);
        this.requestCancellationForTokenBatchSinkErrors();
        this.settleCompletedTrackedRequests(bridge);
        if (loopResult.stepResult === REQUEST_STEP_RESULT_INVALID) {
          this.rejectPendingQueuedRequests(bridge, new Error('Inference loop became invalid.'));
        }
        if (loopResult.stepResult === REQUEST_STEP_RESULT_FATAL_NO_PROGRESS) {
          this.rejectPendingQueuedRequests(bridge, new Error('Inference loop failed to make progress.'));
        }
      } catch (error) {
        if (generation === this.schedulerPumpGeneration) {
          this.rejectPendingQueuedRequests(bridge, error);
        }
      } finally {
        // Final pass to flush tail tokens written before request settlement.
        try {
          this.drainTokenRingObserved(bridge);
        } catch {
          /* cleanup */
        }
      }
    });
  }

  private async runInferenceLoopObserved(
    bridge: WasmBridge,
    generation: number,
    maxTicks: number,
    maxCompletedResponses: number,
    maxGeneratedTokens: number
  ): Promise<Awaited<ReturnType<WasmBridge['runInferenceLoop']>>> {
    const transport = this.options.getTransportObservability();
    if (!transport.enabled) {
      return await bridge.runInferenceLoop(
        generation,
        maxTicks,
        maxCompletedResponses,
        maxGeneratedTokens
      );
    }

    const start = performance.now();
    try {
      return await bridge.runInferenceLoop(
        generation,
        maxTicks,
        maxCompletedResponses,
        maxGeneratedTokens
      );
    } finally {
      transport.wasmRunLoopCalls += 1;
      transport.wasmRunLoopMs += performance.now() - start;
    }
  }

  private tokenRingBridge: WasmBridge | null = null;
  private tokenRingReader: SharedTokenRingReader | null = null;
  private readonly tokenBatchSinkStats = new Map<number, TokenEmissionStats>();

  private sharedTokenRingReader(bridge: WasmBridge): SharedTokenRingReader {
    if (this.tokenRingBridge !== bridge || this.tokenRingReader == null) {
      this.tokenRingBridge = bridge;
      this.tokenRingReader = new SharedTokenRingReader(
        bridge.getSharedTokenRingDescriptor()
      );
    }
    return this.tokenRingReader;
  }

  private drainTokenRingObserved(bridge: WasmBridge): boolean {
    if (this.options.queuedPromptTokenBatchSinks.size === 0) {
      return false;
    }
    const transport = this.options.getTransportObservability();
    if (!transport.enabled) {
      return this.drainTokenRing(bridge);
    }
    const start = performance.now();
    try {
      return this.drainTokenRing(bridge);
    } finally {
      transport.tokenDrainMs =
        (transport.tokenDrainMs ?? 0) + (performance.now() - start);
      transport.tokenDrainCalls =
        (transport.tokenDrainCalls ?? 0) + 1;
    }
  }

  private drainTokenRing(bridge: WasmBridge): boolean {
    let delivered = false;
    this.sharedTokenRingReader(bridge).drain(
      (recordStreamId, sequenceStart, frameCount, byteCount, text) => {
        const streamId = recordStreamId >>> 0;
        const tokenBatchSink = this.options.queuedPromptTokenBatchSinks.get(streamId);
        if (tokenBatchSink == null) {
          return;
        }
        this.deliverTokenBatchSinkBatch(
          tokenBatchSink,
          { generation: this.options.getRuntimeGeneration(), requestId: streamId },
          sequenceStart,
          text,
          frameCount,
          byteCount
        );
        delivered = true;
      }
    );
    return delivered;
  }

  private forgetTokenBatchSinkStream(request: GenerateRequestHandle): void {
    this.tokenBatchSinkStats.delete(request.requestId);
  }

  private deliverTokenBatchSinkBatch(
    tokenBatchSink: (batch: TokenBatch) => void,
    request: GenerateRequestHandle,
    sequenceStart: number,
    text: string,
    frameCount: number,
    byteCount: number
  ): void {
    if (frameCount === 0) {
      return;
    }
    const requestId = request.requestId;
    const stats = this.tokenBatchSinkStats.get(requestId) ?? {
      framesSent: 0,
      bytesSent: 0,
      batchesSent: 0,
    };
    stats.framesSent += frameCount;
    stats.bytesSent += byteCount;
    stats.batchesSent += 1;
    this.tokenBatchSinkStats.set(requestId, stats);
    try {
      tokenBatchSink({
        requestId: `${request.generation}:${requestId}`,
        streamId: requestId,
        sequenceStart,
        text,
        frameCount,
        byteCount,
        stats: { ...stats },
      });
    } catch (error) {
      this.options.tracker.setTokenBatchSinkError(request, error);
      // Stop invoking a sink after its first failure. Cancellation is requested
      // after the drain completes so the ring reader itself remains consistent.
      this.options.queuedPromptTokenBatchSinks.delete(requestId);
    }
  }
}
