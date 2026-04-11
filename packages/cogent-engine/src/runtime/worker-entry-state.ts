import { CogentConfig } from '../cogent-config.js';
import { MainThreadEngineRuntime } from './engine-runtime-main-thread.js';
import { GenerateRequestId, TransportObservability } from '../types.js';
import { WorkerResponseMessage, WorkerSerializableCogentConfig } from './engine-runtime-worker-protocol.js';
import {
  DEFAULT_QUEUED_REQUEST_PUMP_IDLE_STREAK_BEFORE_YIELD,
  DEFAULT_QUEUED_REQUEST_PUMP_SYNC_BURST_LIMIT,
  runQueuedRequestPumpLoop,
} from './queued-request-pump.js';
import { createAbortError } from '../utils/abort.js';

interface BufferedTokenState {
  text: string;
  tokenCount: number;
  timer: number | null;
}

interface ActiveModelLoadState {
  abortController: AbortController;
  streamController: ReadableStreamDefaultController<Uint8Array> | null;
}

type ExternalQueuedRequestSettlement =
  | {
      response: import('../types.js').GenerateResponse;
      callbackError: unknown;
    }
  | {
      error: unknown;
      callbackError: unknown;
    };

const DEFAULT_MAX_BUFFERED_TOKENS = 8;
const DEFAULT_FLUSH_INTERVAL_MS = 16;
export class WorkerEntryState {
  private engine: MainThreadEngineRuntime | null = null;
  private readonly requestAbortControllers = new Map<GenerateRequestId, AbortController>();
  private readonly bufferedTokens = new Map<GenerateRequestId, BufferedTokenState>();
  private readonly runningRequestIds = new Set<GenerateRequestId>();
  private readonly activeModelLoads = new Map<number, ActiveModelLoadState>();
  private schedulerPumpPromise: Promise<void> | null = null;
  private schedulerPumpGeneration = 0;
  private shouldYieldAfterTokenActivity = false;
  private shouldYieldAfterTokenPost = false;
  private requestSettlementHandler:
    | ((requestId: GenerateRequestId, settlement: ExternalQueuedRequestSettlement) => void)
    | null = null;
  private readonly transportObservability: TransportObservability = {
    executionMode: 'worker',
    workerBacked: true,
    enabled: false,
    bufferedTokenLimit: DEFAULT_MAX_BUFFERED_TOKENS,
    flushIntervalMs: DEFAULT_FLUSH_INTERVAL_MS,
    flushCount: 0,
    coalescedTokenCount: 0,
    maxObservedBufferedTokenCount: 0,
    tokenTransportPreference: 'auto',
    activeTokenTransport: 'none',
    tokenCallbackRegistrationCount: 0,
    nativeCallbackTokenCount: 0,
    runtimeEventDrainCount: 0,
    runtimeEventTokenCount: 0,
    runtimeEventTerminalCount: 0,
    runtimeEventTextBytes: 0,
  };

  public cloneTransportObservability(): TransportObservability {
    return { ...this.transportObservability };
  }

  public toErrorMessage(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  }

  public ensureEngine(): MainThreadEngineRuntime {
    if (this.engine == null) {
      throw new Error('Worker runtime is not initialized.');
    }
    return this.engine;
  }

  public async initModule(config: WorkerSerializableCogentConfig): Promise<void> {
    if (this.engine == null) {
      this.engine = new MainThreadEngineRuntime(this.buildEngineConfig(config));
    }
    this.engine.setQueuedRequestPumpMode('external');
    await this.engine.initModule();
  }

  public abortAllModelLoads(): void {
    for (const callId of this.activeModelLoads.keys()) {
      this.abortModelLoad(callId);
    }
  }

  public releaseAllRequestResources(): void {
    this.stopSchedulerPump();
    for (const requestId of this.bufferedTokens.keys()) {
      this.releaseRequestResources(requestId);
    }
    this.runningRequestIds.clear();
  }

  public ensureSchedulerPumpRunning(): void {
    if (this.schedulerPumpPromise != null || this.engine == null) {
      return;
    }
    const scheduler = this.engine.getQueuedRequestSchedulerForExternalControl();
    if (!scheduler.hasActiveRequests()) {
      return;
    }

    const generation = this.schedulerPumpGeneration;
    const schedulerPumpPromise = this.runSchedulerPump(generation);
    this.schedulerPumpPromise = schedulerPumpPromise;
    void schedulerPumpPromise.finally(() => {
      if (this.schedulerPumpPromise === schedulerPumpPromise) {
        this.schedulerPumpPromise = null;
        if (
          generation === this.schedulerPumpGeneration &&
          this.engine != null &&
          this.engine
            .getQueuedRequestSchedulerForExternalControl()
            .hasActiveRequests()
        ) {
          this.ensureSchedulerPumpRunning();
        }
      }
    });
  }

  public setRuntimeObservabilityEnabled(enabled: boolean): void {
    this.transportObservability.enabled = enabled;
  }

  public setRequestSettlementHandler(
    handler:
      | ((requestId: GenerateRequestId, settlement: ExternalQueuedRequestSettlement) => void)
      | null
  ): void {
    this.requestSettlementHandler = handler;
  }

  public beginModelLoad(callId: number): AbortSignal {
    const abortController = new AbortController();
    this.activeModelLoads.set(callId, {
      abortController,
      streamController: null,
    });
    return abortController.signal;
  }

  public beginStreamModelLoad(
    callId: number
  ): {
    signal: AbortSignal;
    stream: ReadableStream<Uint8Array>;
  } {
    const abortController = new AbortController();
    const loadState: ActiveModelLoadState = {
      abortController,
      streamController: null,
    };
    const stream = new ReadableStream<Uint8Array>({
      start: (controller) => {
        loadState.streamController = controller;
      },
    });
    this.activeModelLoads.set(callId, loadState);
    return {
      signal: abortController.signal,
      stream,
    };
  }

  public releaseModelLoad(callId: number): void {
    this.activeModelLoads.delete(callId);
  }

  public abortModelLoad(callId: number): void {
    const loadState = this.activeModelLoads.get(callId);
    if (loadState == null) {
      return;
    }
    loadState.abortController.abort();
    if (loadState.streamController != null) {
      loadState.streamController.error(createAbortError('Model load aborted.'));
      loadState.streamController = null;
    }
  }

  public enqueueStreamChunk(callId: number, chunk: ArrayBuffer): void {
    const loadState = this.activeModelLoads.get(callId);
    if (loadState?.streamController == null) {
      throw new Error(`No active model stream for call ${callId}.`);
    }
    loadState.streamController.enqueue(new Uint8Array(chunk));
    const response: WorkerResponseMessage = {
      kind: 'load-stream-ack',
      callId,
    };
    self.postMessage(response);
  }

  public closeStreamModelLoad(callId: number): void {
    const loadState = this.activeModelLoads.get(callId);
    if (loadState?.streamController == null) {
      return;
    }
    loadState.streamController.close();
    loadState.streamController = null;
  }

  public postLoadProgress(callId: number, progressPct: number): void {
    const progressMessage: WorkerResponseMessage = {
      kind: 'load-progress',
      callId,
      progressPct,
    };
    self.postMessage(progressMessage);
  }

  public bufferTokenPiece(requestId: GenerateRequestId, token: string): void {
    this.shouldYieldAfterTokenActivity = true;
    let state = this.bufferedTokens.get(requestId);
    if (state == null) {
      state = {
        text: '',
        tokenCount: 0,
        timer: null,
      };
      this.bufferedTokens.set(requestId, state);
    }

    state.text += token;
    state.tokenCount += 1;

    if (state.tokenCount >= this.transportObservability.bufferedTokenLimit) {
      this.flushBufferedTokens(requestId);
      return;
    }

    if (state.timer == null) {
      state.timer = self.setTimeout(() => {
        this.flushBufferedTokens(requestId);
      }, this.transportObservability.flushIntervalMs);
    }
  }

  public flushBufferedTokens(requestId: GenerateRequestId): void {
    const state = this.bufferedTokens.get(requestId);
    if (state == null || state.text.length === 0) {
      return;
    }

    if (state.timer != null) {
      clearTimeout(state.timer);
      state.timer = null;
    }

    const payload: WorkerResponseMessage = {
      kind: 'token',
      requestId,
      text: state.text,
      bufferedTokenCount: state.tokenCount,
    };
    self.postMessage(payload);
    this.shouldYieldAfterTokenPost = true;
    if (this.transportObservability.enabled) {
      this.transportObservability.flushCount += 1;
      this.transportObservability.coalescedTokenCount += state.tokenCount;
      this.transportObservability.maxObservedBufferedTokenCount = Math.max(
        this.transportObservability.maxObservedBufferedTokenCount,
        state.tokenCount
      );
    }
    state.text = '';
    state.tokenCount = 0;
  }

  public rememberRequestAbortController(
    requestId: GenerateRequestId,
    abortController: AbortController
  ): void {
    this.requestAbortControllers.set(requestId, abortController);
  }

  public abortQueuedRequest(requestId: GenerateRequestId): void {
    this.requestAbortControllers.get(requestId)?.abort();
  }

  public markRequestRunning(requestId: GenerateRequestId): void {
    this.runningRequestIds.add(requestId);
  }

  public unmarkRequestRunning(requestId: GenerateRequestId): void {
    this.runningRequestIds.delete(requestId);
  }

  public isRequestRunning(requestId: GenerateRequestId): boolean {
    return this.runningRequestIds.has(requestId);
  }

  public releaseRequestResources(requestId: GenerateRequestId): void {
    const state = this.bufferedTokens.get(requestId);
    if (state?.timer != null) {
      clearTimeout(state.timer);
    }
    this.bufferedTokens.delete(requestId);
    this.requestAbortControllers.delete(requestId);
  }

  private stopSchedulerPump(): void {
    this.schedulerPumpGeneration += 1;
    this.schedulerPumpPromise = null;
  }

  private async waitForNextSchedulerStep(): Promise<void> {
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  }

  private async runSchedulerPump(generation: number): Promise<void> {
    const runtime = this.ensureEngine();
    const scheduler = runtime.getQueuedRequestSchedulerForExternalControl();
    await runQueuedRequestPumpLoop({
      isCurrentGeneration: () =>
        generation === this.schedulerPumpGeneration && this.engine != null,
      waitingStepResult: 0,
      syncBurstLimit: DEFAULT_QUEUED_REQUEST_PUMP_SYNC_BURST_LIMIT,
      idleStreakBeforeYield: DEFAULT_QUEUED_REQUEST_PUMP_IDLE_STREAK_BEFORE_YIELD,
      runStep: async () => {
        const pumpStep = await scheduler.pumpOnce();
        this.emitSettledQueuedRequests(runtime);
        return {
          ...pumpStep,
          shouldYieldAfterStep:
            pumpStep.shouldYieldAfterStep === true ||
            this.consumeShouldYieldAfterTokenActivity() ||
            this.consumeShouldYieldAfterTokenPost(),
        };
      },
      waitForNextSchedulerStep: () => this.waitForNextSchedulerStep(),
    });
  }

  private consumeShouldYieldAfterTokenActivity(): boolean {
    const shouldYield = this.shouldYieldAfterTokenActivity;
    this.shouldYieldAfterTokenActivity = false;
    return shouldYield;
  }

  private consumeShouldYieldAfterTokenPost(): boolean {
    const shouldYield = this.shouldYieldAfterTokenPost;
    this.shouldYieldAfterTokenPost = false;
    return shouldYield;
  }

  private emitSettledQueuedRequests(runtime: MainThreadEngineRuntime): void {
    const handler = this.requestSettlementHandler;
    if (handler == null || this.runningRequestIds.size === 0) {
      return;
    }

    for (const requestId of Array.from(this.runningRequestIds)) {
      const settlement =
        runtime.takeSettledQueuedRequestForExternalControl(requestId);
      if (settlement == null) {
        continue;
      }

      this.flushBufferedTokens(requestId);
      handler(requestId, settlement);
      this.releaseRequestResources(requestId);
      this.unmarkRequestRunning(requestId);
    }
  }

  private buildEngineConfig(config: WorkerSerializableCogentConfig): CogentConfig {
    this.transportObservability.bufferedTokenLimit =
      config.workerMaxBufferedTokens ?? DEFAULT_MAX_BUFFERED_TOKENS;
    this.transportObservability.flushIntervalMs =
      config.workerTokenFlushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS;

    return {
      ...config,
      executionMode: 'main-thread',
    };
  }
}
