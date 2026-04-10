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
  runQueuedRequestPumpLoop,
} from './queued-request-pump.js';
import { RequestTracker } from './request-tracker.js';
import { WasmBridge } from '../wasm/wasm-bridge.js';

const SCHEDULER_PUMP_NATIVE_BURST_TICK_LIMIT = 16;
const SCHEDULER_PUMP_NATIVE_BURST_EMITTED_TOKEN_LIMIT = 1;

export type QueuedRequestPumpMode = 'internal' | 'external';

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
  cancelQueuedRequest: (requestId: GenerateRequestId) => Promise<boolean>;
};

export class QueuedRequestScheduler {
  private queuedRequestPumpMode: QueuedRequestPumpMode = 'internal';
  private schedulerPumpPromise: Promise<void> | null = null;
  private schedulerPumpGeneration = 0;

  public constructor(private readonly options: QueuedRequestSchedulerOptions) {}

  public reset(): void {
    this.schedulerPumpGeneration += 1;
    this.schedulerPumpPromise = null;
  }

  public setPumpMode(mode: QueuedRequestPumpMode): void {
    if (this.queuedRequestPumpMode === mode) {
      return;
    }
    this.queuedRequestPumpMode = mode;
    this.reset();
  }

  public hasActiveRequests(): boolean {
    return this.options.tracker.activeCount > 0;
  }

  public track(requestId: GenerateRequestId) {
    const tracked = this.options.tracker.track(requestId);
    if (this.queuedRequestPumpMode === 'internal') {
      this.ensureRunning();
    }
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

  public async pumpOnce(): Promise<QueuedRequestPumpStepResult> {
    return this.pumpQueuedRequestsStep(this.options.getBridge());
  }

  public bufferTokenPiece(
    requestId: GenerateRequestId,
    token: string
  ): void {
    this.bufferQueuedTokenPiece(requestId, token);
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
      try {
        onToken(piece);
      } catch (error) {
        this.options.queuedPromptCallbackErrors.set(requestId, error);
        const remainingStart = readIndex + 1;
        const remainingCount = bufferedPieces.length - remainingStart;
        if (remainingCount > 0) {
          bufferedPieces.copyWithin(0, remainingStart);
        }
        bufferedPieces.length = Math.max(remainingCount, 0);
        break;
      }
      readIndex += 1;
    }

    if (readIndex >= bufferedPieces.length) {
      bufferedPieces.length = 0;
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
      void this.options.cancelQueuedRequest(requestId);
    }
  }

  private async runSchedulerProgress(
    bridge: WasmBridge
  ): Promise<{ stepResult: number; completedResponseCount: number }> {
    return bridge.runSchedulerProgress(
      SCHEDULER_PUMP_NATIVE_BURST_TICK_LIMIT,
      Math.max(1, this.options.tracker.activeCount),
      SCHEDULER_PUMP_NATIVE_BURST_EMITTED_TOKEN_LIMIT
    );
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

  private drainCompletedQueuedRequestIds(
    bridge: WasmBridge
  ): GenerateRequestId[] | null {
    return bridge.drainCompletedRequestIds(
      Math.max(1, this.options.tracker.activeCount)
    );
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

  private drainRuntimeEvents(
    bridge: WasmBridge
  ): { terminalRequestIds: GenerateRequestId[]; tokenEventCount: number } | null {
    const drained = bridge.drainRuntimeEvents(
      Math.max(8, this.options.tracker.activeCount * 2)
    );
    if (drained == null) {
      return null;
    }

    this.transportObservability.runtimeEventDrainCount =
      (this.transportObservability.runtimeEventDrainCount ?? 0) + 1;
    for (const tokenEvent of drained.tokenEvents) {
      this.bufferQueuedTokenPiece(tokenEvent.requestId, tokenEvent.token);
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
    };
  }

  private settleCompletedQueuedRequests(bridge: WasmBridge): boolean {
    const drainedRequestIds = this.drainCompletedQueuedRequestIds(bridge);
    if (drainedRequestIds != null) {
      return this.settleCompletedQueuedRequestsByIds(bridge, drainedRequestIds);
    }

    let settledAny = false;
    for (const requestId of this.options.tracker.allTrackedIds()) {
      settledAny =
        this.settleCompletedQueuedRequest(bridge, requestId) || settledAny;
    }
    return settledAny;
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

  private async pumpQueuedRequestsStep(
    bridge: WasmBridge
  ): Promise<QueuedRequestPumpStepResult> {
    const pendingEvents = this.drainRuntimeEvents(bridge);
    this.flushAllQueuedTokenPieces();
    this.requestCancellationForCallbackErrors();
    let settledAny =
      pendingEvents != null
        ? this.settleCompletedQueuedRequestsByIds(
            bridge,
            pendingEvents.terminalRequestIds
          )
        : this.settleCompletedQueuedRequests(bridge);
    if (this.options.tracker.activeCount === 0) {
      return {
        hasActiveRequests: false,
        stepResult: null,
        settledAny,
      };
    }

    const schedulerProgress = await this.runSchedulerProgress(bridge);
    const stepResult = schedulerProgress.stepResult;
    const drainedEvents = this.drainRuntimeEvents(bridge);
    this.flushAllQueuedTokenPieces();
    this.requestCancellationForCallbackErrors();
    settledAny =
      (drainedEvents != null
        ? this.settleCompletedQueuedRequestsByIds(
            bridge,
            drainedEvents.terminalRequestIds
          )
        : this.settleCompletedQueuedRequests(bridge)) || settledAny;
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
  }

  private async runSchedulerPump(generation: number): Promise<void> {
    const bridge = this.options.getBridge();
    await runQueuedRequestPumpLoop({
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
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  }

  private shouldYieldForResponsiveness(burstTickCount: number): boolean {
    return (
      typeof window !== 'undefined' &&
      typeof document !== 'undefined' &&
      burstTickCount >= DEFAULT_QUEUED_REQUEST_PUMP_SYNC_BURST_LIMIT
    );
  }
}
