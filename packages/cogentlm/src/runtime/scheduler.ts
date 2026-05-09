import {
  GenerateRequestId,
  GenerateResponse,
  TransportObservability,
} from '../types.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  REQUEST_STEP_RESULT_FATAL_NO_PROGRESS,
  REQUEST_STEP_RESULT_INVALID,
  REQUEST_STEP_RESULT_PROGRESSED,
  REQUEST_STEP_RESULT_TERMINAL,
  REQUEST_STEP_RESULT_WAITING,
} from './main-thread-runtime-constants.js';
import {
  DEFAULT_QUEUED_REQUEST_PUMP_SYNC_BURST_LIMIT,
  QueuedRequestPumpStepResult,
  runRequestPumpLoop,
} from './queued-request-pump.js';
import { RequestTracker } from './request-tracker.js';
import { WasmBridge } from '../wasm/wasm-bridge.js';

// Use bounded native bursts; native owns scheduling policy while this loop
// drives browser event-loop progress and observability delivery.
const SCHEDULER_PUMP_NATIVE_BURST_TICK_LIMIT = 64;
const SCHEDULER_PUMP_NATIVE_BURST_EMITTED_TOKEN_LIMIT = 32;
const SCHEDULER_PUMP_INTERACTIVE_FIRST_TOKEN_TICK_LIMIT = 8;
const SCHEDULER_PUMP_INTERACTIVE_FIRST_TOKEN_EMITTED_TOKEN_LIMIT = 1;
// Streaming bursts return after a single emitted token so the runtime-event
// drain delivers each token to the JS layer at decode cadence rather than
// in clumps at the end of an 80ms burst.  Without this, multiple tokens
// produced inside a single WASM burst all surface to the browser at the
// same wall-clock instant when the burst yields, which collapses ITL to
// near-zero within a clump and inflates p99 ITL between clumps – the user
// perceives stutter even though native decode is uniform.  TICK_LIMIT and
// DURATION are kept as upstream backstops for stalls or non-streaming
// contributions inside the same burst.
const SCHEDULER_PUMP_INTERACTIVE_STREAMING_TICK_LIMIT = 16;
const SCHEDULER_PUMP_INTERACTIVE_STREAMING_EMITTED_TOKEN_LIMIT = 1;
const SCHEDULER_PUMP_INTERACTIVE_STREAMING_DURATION_US = 80_000;

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

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
  queuedPromptTokenBuffers: Map<GenerateRequestId, string[]>;
  queuedPromptCallbackErrors: Map<GenerateRequestId, unknown>;
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
  private readonly requestsAwaitingFirstToken = new Set<GenerateRequestId>();
  private readonly interactiveStreamingRequests = new Set<GenerateRequestId>();

  public constructor(private readonly options: QueuedRequestSchedulerOptions) { }

  public reset(): void {
    this.schedulerPumpGeneration += 1;
    this.schedulerPumpPromise = null;
    this.requestsAwaitingFirstToken.clear();
    this.interactiveStreamingRequests.clear();
  }

  public track(requestId: GenerateRequestId) {
    const tracked = this.options.tracker.track(requestId);
    if (this.options.queuedPromptCallbacks.get(requestId) != null) {
      this.requestsAwaitingFirstToken.add(requestId);
      this.interactiveStreamingRequests.add(requestId);
    }
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

  public settleCompletedRequestIfPresent(
    bridge: WasmBridge,
    requestId: GenerateRequestId
  ): boolean {
    return this.settleCompletedQueuedRequest(bridge, requestId);
  }

  private get transportObservability(): TransportObservability {
    return this.options.getTransportObservability();
  }

  private bufferQueuedTokenPiece(
    requestId: GenerateRequestId,
    token: string
  ): void {
    const buffered = this.options.queuedPromptTokenBuffers.get(requestId);
    if (buffered != null) {
      buffered.push(token);
      return;
    }
    this.options.queuedPromptTokenBuffers.set(requestId, [token]);
  }

  private flushQueuedTokenPieces(requestId: GenerateRequestId): void {
    const onToken = this.options.queuedPromptCallbacks.get(requestId);
    const bufferedPieces =
      this.options.queuedPromptTokenBuffers.get(requestId);
    if (onToken == null || bufferedPieces == null || bufferedPieces.length === 0) {
      if (bufferedPieces != null) {
        bufferedPieces.length = 0;
      }
      return;
    }

    let readIndex = 0;
    while (readIndex < bufferedPieces.length) {
      const piece = bufferedPieces[readIndex];
      const measureCallback = this.transportObservability.enabled;
      const callbackStart = measureCallback ? nowMs() : 0;
      try {
        onToken(piece);
      } catch (error) {
        this.options.queuedPromptCallbackErrors.set(requestId, error);
        this.options.queuedPromptCallbacks.delete(requestId);
        bufferedPieces.length = 0;
        break;
      } finally {
        if (measureCallback) {
          this.transportObservability.tokenCallbackCount =
            (this.transportObservability.tokenCallbackCount ?? 0) + 1;
          this.transportObservability.tokenCallbackMs =
            (this.transportObservability.tokenCallbackMs ?? 0) +
            Math.max(0, nowMs() - callbackStart);
        }
      }
      readIndex += 1;
    }

    if (readIndex >= bufferedPieces.length) {
      bufferedPieces.length = 0;
    }

    if (readIndex > 0) {
      this.requestsAwaitingFirstToken.delete(requestId);
    }
  }

  private flushAllQueuedTokenPieces(): void {
    for (const requestId of this.options.queuedPromptTokenBuffers.keys()) {
      this.flushQueuedTokenPieces(requestId);
    }
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

  private async runSchedulerProgress(
    bridge: WasmBridge
  ): Promise<{ stepResult: number; completedResponseCount: number }> {
    const usingFirstTokenBurst = this.requestsAwaitingFirstToken.size > 0;
    const usingInteractiveStreamingBurst =
      !usingFirstTokenBurst && this.interactiveStreamingRequests.size > 0;
    const measureProgress = this.transportObservability.enabled;
    const progressStart = measureProgress ? nowMs() : 0;
    try {
      return await bridge.runSchedulerProgress(
        usingFirstTokenBurst
          ? SCHEDULER_PUMP_INTERACTIVE_FIRST_TOKEN_TICK_LIMIT
          : usingInteractiveStreamingBurst
            ? SCHEDULER_PUMP_INTERACTIVE_STREAMING_TICK_LIMIT
            : SCHEDULER_PUMP_NATIVE_BURST_TICK_LIMIT,
        Math.max(1, this.options.tracker.activeCount),
        usingFirstTokenBurst
          ? SCHEDULER_PUMP_INTERACTIVE_FIRST_TOKEN_EMITTED_TOKEN_LIMIT
          : usingInteractiveStreamingBurst
            ? SCHEDULER_PUMP_INTERACTIVE_STREAMING_EMITTED_TOKEN_LIMIT
            : SCHEDULER_PUMP_NATIVE_BURST_EMITTED_TOKEN_LIMIT,
        usingInteractiveStreamingBurst
          ? { maxDurationUs: SCHEDULER_PUMP_INTERACTIVE_STREAMING_DURATION_US }
          : undefined
      );
    } finally {
      if (measureProgress) {
        this.transportObservability.schedulerProgressCount =
          (this.transportObservability.schedulerProgressCount ?? 0) + 1;
        this.transportObservability.schedulerProgressMs =
          (this.transportObservability.schedulerProgressMs ?? 0) +
          Math.max(0, nowMs() - progressStart);
      }
    }
  }

  private settleCompletedQueuedRequest(
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
      this.requestsAwaitingFirstToken.delete(requestId);
      this.interactiveStreamingRequests.delete(requestId);
      this.options.finalizeRequest(bridge, requestId, {
        deleteCompletion:
          (response.cancelled || tracked.cancelRequested) && !tracked.consumed,
      });
    } catch (error) {
      this.requestsAwaitingFirstToken.delete(requestId);
      this.interactiveStreamingRequests.delete(requestId);
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
        this.settleCompletedQueuedRequest(bridge, requestId) || settledAny;
    }
    return settledAny;
  }

  private settleCompletedTrackedRequests(bridge: WasmBridge): boolean {
    let settledAny = false;
    for (const requestId of this.options.tracker.allTrackedIds()) {
      settledAny =
        this.settleCompletedQueuedRequest(bridge, requestId) || settledAny;
    }
    return settledAny;
  }

  private drainRuntimeEvents(
    bridge: WasmBridge
  ): {
    terminalRequestIds: GenerateRequestId[];
    tokenEventCount: number;
    tokenRequestIds: GenerateRequestId[];
  } {
    const measureDrain = this.transportObservability.enabled;
    const drainStart = measureDrain ? nowMs() : 0;
    const drained = bridge.drainRuntimeEvents(
      Math.max(8, this.options.tracker.activeCount * 2)
    );

    this.transportObservability.runtimeEventDrainCount =
      (this.transportObservability.runtimeEventDrainCount ?? 0) + 1;
    if (measureDrain) {
      this.transportObservability.runtimeEventDrainMs =
        (this.transportObservability.runtimeEventDrainMs ?? 0) +
        Math.max(0, nowMs() - drainStart);
    }
    const tokenRequestIds: GenerateRequestId[] = [];
    for (const tokenEvent of drained.tokenEvents) {
      if (
        !this.options.queuedPromptCallbacks.has(tokenEvent.requestId) ||
        this.options.queuedPromptCallbackErrors.has(tokenEvent.requestId)
      ) {
        continue;
      }
      this.bufferQueuedTokenPiece(tokenEvent.requestId, tokenEvent.token);
      tokenRequestIds.push(tokenEvent.requestId);
    }
    this.transportObservability.runtimeEventTextBytes =
      (this.transportObservability.runtimeEventTextBytes ?? 0) + drained.textBytes;
    this.transportObservability.runtimeEventTokenCount =
      (this.transportObservability.runtimeEventTokenCount ?? 0) +
      drained.tokenEvents.length;
    this.transportObservability.runtimeEventTerminalCount =
      (this.transportObservability.runtimeEventTerminalCount ?? 0) +
      drained.terminalRequestIds.length;

    return {
      terminalRequestIds: drained.terminalRequestIds,
      tokenEventCount: drained.tokenEvents.length,
      tokenRequestIds,
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
      this.requestsAwaitingFirstToken.delete(requestId);
      this.interactiveStreamingRequests.delete(requestId);
      this.options.tracker.reject(requestId, error);
      this.options.finalizeRequest(bridge, requestId, {
        deleteCompletion: true,
      });
    }
  }

  private async pumpQueuedRequestsStep(
    bridge: WasmBridge
  ): Promise<QueuedRequestPumpStepResult> {
    if (this.options.tracker.activeCount === 0) {
      return {
        hasActiveRequests: false,
        stepResult: null,
        settledAny: false,
      };
    }

    const measurePumpStep = this.transportObservability.enabled;
    const pumpStepStart = measurePumpStep ? nowMs() : 0;
    try {
      const schedulerProgress = await this.runSchedulerProgress(bridge);
      const stepResult = schedulerProgress.stepResult;
      const drainedEvents = this.drainRuntimeEvents(bridge);
      for (const requestId of drainedEvents.tokenRequestIds) {
        this.requestsAwaitingFirstToken.delete(requestId);
      }
      this.flushAllQueuedTokenPieces();
      this.requestCancellationForCallbackErrors();
      const settledFromEvents = this.settleCompletedQueuedRequestsByIds(
        bridge,
        drainedEvents.terminalRequestIds
      );
      const needsCompletionScan =
        schedulerProgress.completedResponseCount > drainedEvents.terminalRequestIds.length;
      const settledFromScan = needsCompletionScan
        ? this.settleCompletedTrackedRequests(bridge)
        : false;
      const settledAny = settledFromEvents || settledFromScan;
      if (this.options.tracker.activeCount === 0) {
        return {
          hasActiveRequests: false,
          stepResult,
          settledAny,
        };
      }

      if (stepResult === REQUEST_STEP_RESULT_INVALID) {
        this.rejectPendingQueuedRequests(
          bridge,
          new Error('Queued scheduler tick became invalid.')
        );
        return {
          hasActiveRequests: false,
          stepResult,
          settledAny,
        };
      }

      if (stepResult === REQUEST_STEP_RESULT_FATAL_NO_PROGRESS) {
        this.rejectPendingQueuedRequests(
          bridge,
          new Error('Queued request execution failed to make progress.')
        );
        return {
          hasActiveRequests: false,
          stepResult,
          settledAny,
        };
      }

      if (
        stepResult !== REQUEST_STEP_RESULT_WAITING &&
        stepResult !== REQUEST_STEP_RESULT_PROGRESSED &&
        stepResult !== REQUEST_STEP_RESULT_TERMINAL
      ) {
        this.rejectPendingQueuedRequests(
          bridge,
          new Error(`Queued scheduler returned unknown step result ${stepResult}.`)
        );
        return {
          hasActiveRequests: false,
          stepResult,
          settledAny,
        };
      }

      return {
        hasActiveRequests: this.options.tracker.activeCount > 0,
        stepResult,
        settledAny,
      };
    } finally {
      if (measurePumpStep) {
        this.transportObservability.pumpStepCount =
          (this.transportObservability.pumpStepCount ?? 0) + 1;
        this.transportObservability.pumpStepMs =
          (this.transportObservability.pumpStepMs ?? 0) +
          Math.max(0, nowMs() - pumpStepStart);
      }
    }
  }

  private async runSchedulerPump(generation: number): Promise<void> {
    const bridge = this.options.getBridge();
    await runRequestPumpLoop({
      isCurrentGeneration: () => generation === this.schedulerPumpGeneration,
      waitingStepResult: REQUEST_STEP_RESULT_WAITING,
      shouldYieldForResponsiveness: (burstTickCount) =>
        this.shouldYieldForResponsiveness(burstTickCount),
      runStep: async () => {
        try {
          return await this.pumpQueuedRequestsStep(bridge);
        } catch (error) {
          if (generation !== this.schedulerPumpGeneration) {
            return {
              hasActiveRequests: false,
              stepResult: null,
              settledAny: false,
            };
          }
          this.rejectPendingQueuedRequests(bridge, error);
          return {
            hasActiveRequests: false,
            stepResult: null,
            settledAny: false,
          };
        }
      },
      waitForNextSchedulerStep: () => this.waitForNextSchedulerStep(),
    });
  }

  private async waitForNextSchedulerStep(): Promise<void> {
    const measureYield = this.transportObservability.enabled;
    const yieldStart = measureYield ? nowMs() : 0;
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    if (measureYield) {
      this.transportObservability.schedulerYieldCount =
        (this.transportObservability.schedulerYieldCount ?? 0) + 1;
      this.transportObservability.schedulerYieldMs =
        (this.transportObservability.schedulerYieldMs ?? 0) +
        Math.max(0, nowMs() - yieldStart);
    }
  }

  private shouldYieldForResponsiveness(burstTickCount: number): boolean {
    return (
      typeof window !== 'undefined' &&
      typeof document !== 'undefined' &&
      burstTickCount >= DEFAULT_QUEUED_REQUEST_PUMP_SYNC_BURST_LIMIT
    );
  }
}
