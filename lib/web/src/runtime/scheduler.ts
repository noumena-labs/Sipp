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
import { SharedTokenRingReader } from './shared-token-ring.js';

// Native owns model scheduling; JS only drains the shared token ring after a
// host loop returns. Worker-mode token presentation is pulled by the main
// thread from the same ring and does not run through this scheduler.
const CONTINUOUS_LOOP_TICK_LIMIT = 1024;
const CONTINUOUS_LOOP_TOKEN_LIMIT = 512;
const MAIN_THREAD_TOKEN_SLICE_US = 8_000;
const WORKER_TOKEN_SLICE_US = 0;
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
  withWasmBridge?: <T>(
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

  public track(requestId: GenerateRequestId) {
    const tracked = this.options.tracker.track(requestId);
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
    void pumpPromise.finally(() => {
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
    await this.withWasmBridge(async (bridge) => {
      try {
        if (
          generation !== this.schedulerPumpGeneration ||
          this.options.tracker.activeCount === 0
        ) {
          return;
        }

        const loopResult = await bridge.runInferenceLoop(
          CONTINUOUS_LOOP_TICK_LIMIT,
          this.options.tracker.activeCount,
          CONTINUOUS_LOOP_TOKEN_LIMIT,
          { maxDurationUs: this.loopDurationUs() }
        );
        this.drainTokenRingObserved(bridge);
        this.requestCancellationForTokenBatchSinkErrors();
        if (loopResult.completedResponseCount > 0) {
          this.settleCompletedTrackedRequests(bridge);
        }
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

  private withWasmBridge<T>(
    operation: (bridge: WasmBridge) => T | Promise<T>
  ): Promise<T> {
    if (this.options.withWasmBridge != null) {
      return this.options.withWasmBridge(operation);
    }
    return this.runWithCurrentBridge(operation);
  }

  private async runWithCurrentBridge<T>(
    operation: (bridge: WasmBridge) => T | Promise<T>
  ): Promise<T> {
    return await operation(this.options.getBridge());
  }

  private loopDurationUs(): number {
    if (this.options.getTransportObservability().executionMode === 'main-thread') {
      return MAIN_THREAD_TOKEN_SLICE_US;
    }
    return WORKER_TOKEN_SLICE_US;
  }

  private tokenRingBridge: WasmBridge | null = null;
  private tokenRingReader: SharedTokenRingReader | null = null;
  private readonly tokenBatchSinkStats = new Map<
    number,
    {
      framesSent: number;
      bytesSent: number;
      batchesSent: number;
      drainMs: number;
      drainCalls: number;
    }
  >();

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
    if (this.options.queuedPromptTokenBatchSinks.size === 0) {
      return false;
    }
    let delivered = false;
    this.sharedTokenRingReader(bridge).drain(
      (recordStreamId, sequenceStart, frameCount, byteCount, text) => {
        const streamId = recordStreamId >>> 0;
        const tokenBatchSink = this.options.queuedPromptTokenBatchSinks.get(streamId);
        if (tokenBatchSink != null) {
          const recordDrainStart = performance.now();
          this.deliverTokenBatchSinkBatch(
            streamId,
            sequenceStart,
            text,
            frameCount,
            byteCount,
            performance.now() - recordDrainStart
          );
          delivered = true;
        }
      }
    );
    return delivered;
  }

  private forgetTokenBatchSinkStream(requestId: number): void {
    this.tokenBatchSinkStats.delete(requestId);
  }

  private deliverTokenBatchSinkBatch(
    requestId: number,
    sequenceStart: number,
    text: string,
    frameCount: number,
    byteCount: number,
    drainMs: number
  ): void {
    const tokenBatchSink = this.options.queuedPromptTokenBatchSinks.get(requestId);
    if (tokenBatchSink == null || frameCount === 0) {
      return;
    }
    const stats = this.tokenBatchSinkStats.get(requestId) ?? {
      framesSent: 0,
      bytesSent: 0,
      batchesSent: 0,
      drainMs: 0,
      drainCalls: 0,
    };
    stats.framesSent += frameCount;
    stats.bytesSent += byteCount;
    stats.batchesSent += 1;
    stats.drainMs += drainMs;
    stats.drainCalls += 1;
    this.tokenBatchSinkStats.set(requestId, stats);
    try {
      tokenBatchSink({
        requestId: String(requestId),
        streamId: requestId,
        sequenceStart,
        text,
        frameCount,
        byteCount,
        stats: { ...stats },
      });
    } catch (error) {
      this.options.queuedPromptTokenBatchSinkErrors.set(requestId, error);
    }
  }
}
