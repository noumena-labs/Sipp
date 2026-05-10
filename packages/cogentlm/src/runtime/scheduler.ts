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
};

export class QueuedRequestScheduler {
  private schedulerPumpPromise: Promise<void> | null = null;
  private schedulerPumpGeneration = 0;

  public constructor(private readonly options: QueuedRequestSchedulerOptions) { }

  public reset(): void {
    this.schedulerPumpGeneration += 1;
    this.schedulerPumpPromise = null;
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
    let drained = bridge.drainRuntimeEvents(perCallEventCap);
    const aggregatedTokenEvents: typeof drained.tokenEvents = [];
    const aggregatedTerminalIds: typeof drained.terminalRequestIds = [];
    let aggregatedTextBytes = 0;
    let totalEventCount = 0;
    while (true) {
      for (const tokenEvent of drained.tokenEvents) {
        const callback = this.options.queuedPromptCallbacks.get(tokenEvent.requestId);
        if (
          !callback ||
          this.options.queuedPromptCallbackErrors.has(tokenEvent.requestId)
        ) {
          continue;
        }
        try {
          callback(tokenEvent.token);
        } catch (error) {
          this.options.queuedPromptCallbackErrors.set(tokenEvent.requestId, error);
        }
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
    
    while (generation === this.schedulerPumpGeneration && this.options.tracker.activeCount > 0) {
      try {
        const loopResult = await bridge.runInferenceLoop(
          CONTINUOUS_LOOP_TICK_LIMIT,
          this.options.tracker.activeCount,
          CONTINUOUS_LOOP_TOKEN_LIMIT
        );

        const drainedEvents = this.drainRuntimeEvents(bridge);
        this.requestCancellationForCallbackErrors();
        
        const settledFromEvents = this.settleCompletedQueuedRequestsByIds(
          bridge,
          drainedEvents.terminalRequestIds
        );
        
        const needsCompletionScan =
          loopResult.completedResponseCount > drainedEvents.terminalRequestIds.length;
        if (needsCompletionScan) {
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

        // JSPI handles yielding at the native layer. We don't need artificial 
        // timeouts here which would introduce 4ms+ stalls per batch.
      } catch (error) {

        if (generation === this.schedulerPumpGeneration) {
          this.rejectPendingQueuedRequests(bridge, error);
        }
        break;
      }
    }
  }
}
