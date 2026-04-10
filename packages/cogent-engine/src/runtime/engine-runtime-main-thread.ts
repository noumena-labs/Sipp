import { FileSystemStorage } from '../storage/file-system-storage.js';
import {
  BrowserModelCache,
} from '../storage/browser-model-cache.js';
import { CogentConfig, EngineModuleOptions } from '../cogent-config.js';
import { normalizeInitConfig } from '../core/init-config.js';
import { formatPromptText } from '../core/prompt-format.js';
import {
  BackendObservability,
  EngineExecutionMode,
  GenerateRequest,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  ModelLoadInfo,
  PromptOptions,
  RuntimeAggregateObservabilityMetrics,
  TransportObservability,
} from '../types.js';
import { EngineRuntime } from './engine-runtime.js';
import { MainThreadModelLoader } from './main-thread-model-loader.js';
import {
  callModuleNumber,
  callModuleNumberAsync,
  COMPLETED_REQUEST_STATUS_PENDING,
  DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
  DEFAULT_PROMPT_FORMAT,
  EngineModule,
  MAX_PROMPT_TOKENS,
  QueuedRequestCompletionState,
  REQUEST_STEP_RESULT_FATAL_NO_PROGRESS,
  REQUEST_STEP_RESULT_INVALID,
  REQUEST_STEP_RESULT_PROGRESSED,
  REQUEST_STEP_RESULT_TERMINAL,
  REQUEST_STEP_RESULT_WAITING,
} from './main-thread-runtime-shared.js';
import {
  readRuntimeObservabilityFromModule,
  takeCompletedResponse,
} from './main-thread-runtime-observability.js';
import {
  asErrorMessage,
  createAbortError,
  createDeferred,
} from './runtime-shared.js';

const SCHEDULER_PUMP_SYNC_BURST_LIMIT = 128;
const SCHEDULER_PUMP_IDLE_STREAK_BEFORE_YIELD = 4;

export class MainThreadEngineRuntime implements EngineRuntime {
  private module: EngineModule | null = null;
  private initPromise: Promise<void> | null = null;
  private engineInitialized = false;
  private readonly opfs = new FileSystemStorage();
  private readonly browserModelCache = new BrowserModelCache(this.opfs);
  private readonly modelLoader: MainThreadModelLoader;
  private queuedPromptCallbacks = new Map<
    GenerateRequestId,
    ((token: string) => void) | undefined
  >();
  private queuedPromptCallbackPtrs = new Map<GenerateRequestId, number | bigint>();
  private queuedPromptTokenBuffers = new Map<GenerateRequestId, string[]>();
  private queuedPromptCallbackErrors = new Map<GenerateRequestId, unknown>();
  private queuedPromptSignals = new Map<GenerateRequestId, AbortSignal>();
  private queuedPromptSignalAbortListeners = new Map<GenerateRequestId, () => void>();
  private activeQueuedRequestRuns = new Set<GenerateRequestId>();
  private queuedRequestCompletions = new Map<
    GenerateRequestId,
    QueuedRequestCompletionState
  >();
  private schedulerPumpPromise: Promise<void> | null = null;
  private schedulerPumpGeneration = 0;
  private lastModelLoadInfo: ModelLoadInfo | null = null;
  private runtimeObservabilityEnabled = false;
  private backendProfilingEnabled = false;
  private transportObservability: TransportObservability = {
    ...DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
  };
  constructor(private config: CogentConfig = {}) {
    this.modelLoader = new MainThreadModelLoader(
      this.config,
      this.opfs,
      this.browserModelCache,
      this.parseConfiguredUrl.bind(this),
      (info) => {
        this.lastModelLoadInfo = info;
      }
    );
  }

  private resolveWasmUrls(): { moduleUrl: string; wasmUrl: string } {
    const moduleUrl = this.config.moduleUrl?.trim();
    const wasmUrl = this.config.wasmUrl?.trim();

    if (!moduleUrl || !wasmUrl) {
      throw new Error(
        'Both "moduleUrl" and "wasmUrl" must be provided in CogentEngine config. Use getBundledRuntimeUrls() for the package defaults.'
      );
    }

    const module = this.parseConfiguredUrl(moduleUrl, 'moduleUrl');
    const wasm = this.parseConfiguredUrl(wasmUrl, 'wasmUrl');
    const trustedOrigins = this.resolveTrustedOrigins();

    if (trustedOrigins.size > 0) {
      if (!trustedOrigins.has(module.origin)) {
        throw new Error(`Blocked moduleUrl origin "${module.origin}". Add it to trustedOrigins to allow it.`);
      }
      if (!trustedOrigins.has(wasm.origin)) {
        throw new Error(`Blocked wasmUrl origin "${wasm.origin}". Add it to trustedOrigins to allow it.`);
      }
    }

    return { moduleUrl: module.toString(), wasmUrl: wasm.toString() };
  }

  private parseConfiguredUrl(rawUrl: string, fieldName: string): URL {
    try {
      if (typeof window !== 'undefined' && typeof window.location?.href === 'string') {
        return new URL(rawUrl, window.location.href);
      }
      return new URL(rawUrl);
    } catch {
      throw new Error(`Invalid ${fieldName} value "${rawUrl}".`);
    }
  }

  private resolveTrustedOrigins(): Set<string> {
    const configuredOrigins = this.config.trustedOrigins ?? [];
    if (configuredOrigins.length > 0) {
      const allowed = new Set<string>();
      for (const originValue of configuredOrigins) {
        const normalizedOrigin = this.parseConfiguredUrl(originValue, 'trustedOrigins').origin;
        allowed.add(normalizedOrigin);
      }
      return allowed;
    }

    if (typeof window !== 'undefined' && typeof window.location?.origin === 'string') {
      return new Set([window.location.origin]);
    }

    return new Set();
  }

  public getExecutionMode(): EngineExecutionMode {
    return 'main-thread';
  }

  public getLastModelLoadInfo(): ModelLoadInfo | null {
    return this.lastModelLoadInfo;
  }

  public getTransportObservability(): TransportObservability {
    return { ...this.transportObservability };
  }

  private normalizeTokenCount(nTokens: number): number {
    if (!Number.isInteger(nTokens)) {
      throw new Error('nTokens must be an integer.');
    }
    if (nTokens <= 0 || nTokens > MAX_PROMPT_TOKENS) {
      throw new Error(`nTokens must be between 1 and ${MAX_PROMPT_TOKENS}.`);
    }
    return nTokens;
  }

  private resolvePromptTokenCount(
    input: number | PromptOptions | undefined
  ): number {
    if (typeof input === 'number' || input === undefined) {
      return this.normalizeTokenCount(input ?? 128);
    }
    return this.normalizeTokenCount(input.nTokens ?? 128);
  }

  private resolvePromptFormat(
    input: PromptOptions | number | undefined
  ): 'auto-chat' | 'raw' {
    if (typeof input === 'number' || input === undefined) {
      return DEFAULT_PROMPT_FORMAT;
    }
    return input.promptFormat ?? DEFAULT_PROMPT_FORMAT;
  }

  private buildGenerateRequest(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions
  ): GenerateRequest {
    const promptFormat = this.resolvePromptFormat(options);
    return {
      contextKey,
      promptText: formatPromptText(promptText, promptFormat),
      maxOutputTokens: this.resolvePromptTokenCount(options),
      promptFormat,
    };
  }

  private getLoadedModule(): EngineModule {
    if (!this.module) {
      throw new Error('Module is not initialized. Call initModule() first.');
    }
    return this.module;
  }

  private getReadyEngineModule(): EngineModule {
    const module = this.getLoadedModule();
    if (!this.engineInitialized) {
      throw new Error('Engine is not initialized. Call initEngine(modelPath, config?) first.');
    }
    return module;
  }

  private releaseQueuedPromptCallback(module: EngineModule, requestId: GenerateRequestId): void {
    const callbackPtr = this.queuedPromptCallbackPtrs.get(requestId);
    if (callbackPtr != null) {
      module.removeFunction(callbackPtr);
    }
    const signal = this.queuedPromptSignals.get(requestId);
    const abortListener = this.queuedPromptSignalAbortListeners.get(requestId);
    if (signal != null && abortListener != null) {
      signal.removeEventListener('abort', abortListener);
    }
    this.queuedPromptCallbacks.delete(requestId);
    this.queuedPromptTokenBuffers.delete(requestId);
    this.queuedPromptCallbackPtrs.delete(requestId);
    this.queuedPromptCallbackErrors.delete(requestId);
    this.queuedPromptSignals.delete(requestId);
    this.queuedPromptSignalAbortListeners.delete(requestId);
  }

  private consumeCompletedResponseIfPresent(
    module: EngineModule,
    requestId: GenerateRequestId
  ): boolean {
    const status = this.callNumber(module, 'CE_GetCompletedRequestStatus', ['number'], [requestId]);
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      return false;
    }

    const consumed = this.callNumber(module, 'CE_ConsumeCompletedRequest', ['number'], [requestId]);
    if (!consumed) {
      throw new Error('Failed to consume completed queued request response.');
    }
    return true;
  }

  private releaseCancelledQueuedRequestState(
    module: EngineModule,
    requestId: GenerateRequestId
  ): void {
    this.consumeCompletedResponseIfPresent(module, requestId);
    this.releaseQueuedPromptCallback(module, requestId);
  }

  private releaseAllQueuedPromptCallbacks(module: EngineModule): void {
    for (const callbackPtr of this.queuedPromptCallbackPtrs.values()) {
      module.removeFunction(callbackPtr);
    }
    this.queuedPromptCallbacks.clear();
    this.queuedPromptTokenBuffers.clear();
    this.queuedPromptCallbackPtrs.clear();
    this.queuedPromptCallbackErrors.clear();
    for (const [requestId] of this.queuedPromptSignals) {
      this.releaseQueuedPromptCallback(module, requestId);
    }
    this.activeQueuedRequestRuns.clear();
  }

  private rejectQueuedRequestCompletions(error: unknown): void {
    this.schedulerPumpGeneration += 1;
    this.schedulerPumpPromise = null;
    for (const [requestId, completionState] of this.queuedRequestCompletions) {
      if (completionState.settled) {
        continue;
      }
      completionState.settled = true;
      completionState.reject(error);
      this.activeQueuedRequestRuns.delete(requestId);
    }
    this.queuedRequestCompletions.clear();
  }

  private resetRuntimeLifecycleState(error?: unknown): void {
    if (error != null) {
      this.rejectQueuedRequestCompletions(error);
    } else {
      this.schedulerPumpGeneration += 1;
      this.schedulerPumpPromise = null;
      this.queuedRequestCompletions.clear();
    }
    this.activeQueuedRequestRuns.clear();
    this.runtimeObservabilityEnabled = false;
    this.backendProfilingEnabled = false;
    this.transportObservability = {
      ...DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
    };
  }

  private bufferQueuedTokenPiece(requestId: GenerateRequestId, token: string): void {
    const buffered = this.queuedPromptTokenBuffers.get(requestId);
    if (buffered != null) {
      buffered.push(token);
      return;
    }
    this.queuedPromptTokenBuffers.set(requestId, [token]);
  }

  private flushQueuedTokenPieces(requestId: GenerateRequestId): void {
    const onToken = this.queuedPromptCallbacks.get(requestId);
    const bufferedPieces = this.queuedPromptTokenBuffers.get(requestId);
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
        this.queuedPromptCallbackErrors.set(requestId, error);
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

  private waitForNextSchedulerStep(): Promise<void> {
    return new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  }

  private shouldYieldForResponsiveness(
    burstTickCount: number
  ): boolean {
    return (
      typeof window !== 'undefined' &&
      typeof document !== 'undefined' &&
      burstTickCount >= SCHEDULER_PUMP_SYNC_BURST_LIMIT
    );
  }

  private shouldYieldSchedulerPump(
    burstTickCount: number,
    waitingStreak: number
  ): boolean {
    if (this.shouldYieldForResponsiveness(burstTickCount)) {
      return true;
    }
    return waitingStreak >= SCHEDULER_PUMP_IDLE_STREAK_BEFORE_YIELD;
  }

  private cleanupConsumedCompletionState(requestId: GenerateRequestId): void {
    const completion = this.queuedRequestCompletions.get(requestId);
    if (
      completion != null &&
      completion.settled &&
      completion.consumed &&
      completion.waiterCount === 0
    ) {
      this.queuedRequestCompletions.delete(requestId);
    }
  }

  private getOrCreateQueuedRequestCompletion(
    requestId: GenerateRequestId
  ): QueuedRequestCompletionState {
    const existing = this.queuedRequestCompletions.get(requestId);
    if (existing != null) {
      return existing;
    }

    const deferred = createDeferred<GenerateResponse>();
    const completionState: QueuedRequestCompletionState = {
      promise: deferred.promise,
      resolve: deferred.resolve,
      reject: deferred.reject,
      settled: false,
      consumed: false,
      waiterCount: 0,
      callbackError: undefined,
      cancelRequested: false,
    };
    void completionState.promise.catch(() => {});
    this.queuedRequestCompletions.set(requestId, completionState);
    this.activeQueuedRequestRuns.add(requestId);
    this.ensureSchedulerPumpRunning();
    return completionState;
  }

  private settleCompletedQueuedRequest(
    module: EngineModule,
    requestId: GenerateRequestId,
    completionState: QueuedRequestCompletionState
  ): boolean {
    if (completionState.settled) {
      return false;
    }
    const status = this.callNumber(module, 'CE_GetCompletedRequestStatus', ['number'], [requestId]);
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      return false;
    }

    try {
      const response = this.takeCompletedResponse(module, requestId);
      completionState.callbackError = this.queuedPromptCallbackErrors.get(requestId);
      completionState.settled = true;
      completionState.resolve(response);
    } catch (error) {
      completionState.settled = true;
      completionState.reject(error);
    } finally {
      this.releaseQueuedPromptCallback(module, requestId);
      this.activeQueuedRequestRuns.delete(requestId);
      this.cleanupConsumedCompletionState(requestId);
    }
    return true;
  }

  private settleCompletedQueuedRequests(module: EngineModule): boolean {
    let settledAny = false;
    for (const [requestId, completionState] of this.queuedRequestCompletions) {
      settledAny =
        this.settleCompletedQueuedRequest(module, requestId, completionState) || settledAny;
    }
    return settledAny;
  }

  private requestCancellationForCallbackErrors(): void {
    for (const [requestId, completionState] of this.queuedRequestCompletions) {
      if (completionState.settled || completionState.cancelRequested) {
        continue;
      }
      const callbackError = this.queuedPromptCallbackErrors.get(requestId);
      if (callbackError == null) {
        continue;
      }
      completionState.callbackError = callbackError;
      completionState.cancelRequested = true;
      void this.cancelQueuedRequest(requestId);
    }
  }

  private flushAllQueuedTokenPieces(): void {
    for (const requestId of this.queuedPromptTokenBuffers.keys()) {
      this.flushQueuedTokenPieces(requestId);
    }
  }

  private rejectPendingQueuedRequests(
    module: EngineModule,
    error: unknown
  ): void {
    for (const [requestId, completionState] of this.queuedRequestCompletions) {
      if (completionState.settled) {
        continue;
      }
      completionState.settled = true;
      completionState.reject(error);
      this.releaseQueuedPromptCallback(module, requestId);
      this.activeQueuedRequestRuns.delete(requestId);
      this.cleanupConsumedCompletionState(requestId);
    }
  }

  private ensureSchedulerPumpRunning(): void {
    if (this.schedulerPumpPromise != null || this.queuedRequestCompletions.size === 0) {
      return;
    }
    const generation = this.schedulerPumpGeneration;
    const pumpPromise = this.runSchedulerPump(generation);
    this.schedulerPumpPromise = pumpPromise;
    void pumpPromise.finally(() => {
      if (this.schedulerPumpPromise === pumpPromise) {
        this.schedulerPumpPromise = null;
        if (this.engineInitialized && this.activeQueuedRequestRuns.size > 0) {
          this.ensureSchedulerPumpRunning();
        }
      }
    });
  }

  private async runSchedulerPump(generation: number): Promise<void> {
    const module = this.getReadyEngineModule();
    let burstTickCount = 0;
    let waitingStreak = 0;
    while (generation === this.schedulerPumpGeneration) {
      this.settleCompletedQueuedRequests(module);
      if (this.activeQueuedRequestRuns.size === 0) {
        return;
      }

      let stepResult: number;
      try {
        stepResult = await this.callNumberAsync(module, 'CE_RunSchedulerTick');
      } catch (error) {
        if (generation !== this.schedulerPumpGeneration || !this.engineInitialized || this.module == null) {
          return;
        }
        this.rejectPendingQueuedRequests(module, error);
        return;
      }

      if (generation !== this.schedulerPumpGeneration) {
        return;
      }

      this.flushAllQueuedTokenPieces();
      this.requestCancellationForCallbackErrors();
      const settledAfterTick = this.settleCompletedQueuedRequests(module);
      if (this.activeQueuedRequestRuns.size === 0) {
        return;
      }

      if (stepResult === REQUEST_STEP_RESULT_INVALID) {
        this.rejectPendingQueuedRequests(module, new Error('Queued scheduler tick became invalid.'));
        return;
      }
      if (stepResult === REQUEST_STEP_RESULT_FATAL_NO_PROGRESS) {
        this.rejectPendingQueuedRequests(
          module,
          new Error('Queued request execution failed to make progress.')
        );
        return;
      }
      if (
        stepResult !== REQUEST_STEP_RESULT_WAITING &&
        stepResult !== REQUEST_STEP_RESULT_PROGRESSED &&
        stepResult !== REQUEST_STEP_RESULT_TERMINAL
      ) {
        this.rejectPendingQueuedRequests(
          module,
          new Error(`Queued scheduler returned unknown step result ${stepResult}.`)
        );
        return;
      }

      burstTickCount += 1;
      if (stepResult === REQUEST_STEP_RESULT_WAITING && !settledAfterTick) {
        waitingStreak += 1;
      } else {
        waitingStreak = 0;
      }

      if (this.shouldYieldSchedulerPump(burstTickCount, waitingStreak)) {
        burstTickCount = 0;
        waitingStreak = 0;
        await this.waitForNextSchedulerStep();
      }
    }
  }

  private callNumber(
    module: EngineModule,
    ident: string,
    argTypes: string[] = [],
    args: unknown[] = []
  ): number {
    return callModuleNumber(module, ident, argTypes, args);
  }

  private async callNumberAsync(
    module: EngineModule,
    ident: string,
    argTypes: string[] = [],
    args: unknown[] = []
  ): Promise<number> {
    return callModuleNumberAsync(module, ident, argTypes, args);
  }

  private readRuntimeObservabilityFromModule(
    module: EngineModule
  ): RuntimeAggregateObservabilityMetrics | null {
    return readRuntimeObservabilityFromModule(
      module,
      this.callNumber.bind(this)
    );
  }

  private takeCompletedResponse(
    module: EngineModule,
    requestId: GenerateRequestId
  ): GenerateResponse {
    return takeCompletedResponse(module, requestId, this.callNumber.bind(this));
  }

  private async importModuleFactory(moduleUrl: string): Promise<(options: EngineModuleOptions) => Promise<EngineModule>> {
    const importedModule = await import(/* @vite-ignore */ moduleUrl);
    const createModule = importedModule.default;
    if (typeof createModule !== 'function') {
      throw new Error(`Invalid Emscripten module at "${moduleUrl}"`);
    }
    return createModule as (options: EngineModuleOptions) => Promise<EngineModule>;
  }

  private async ensureModule(): Promise<EngineModule> {
    if (this.module) {
      return this.module;
    }
    await this.initModule();
    return this.getLoadedModule();
  }

  /**
   * Initializes the underlying WebAssembly module.
   */
  public async initModule() {
    if (this.module) {
      return;
    }
    if (!this.initPromise) {
      this.initPromise = (async () => {
        const { moduleUrl, wasmUrl } = this.resolveWasmUrls();
        const createModule = await this.importModuleFactory(moduleUrl);
        const moduleConfig: EngineModuleOptions = { ...(this.config.moduleOptions ?? {}) };
        const userLocateFile = moduleConfig.locateFile;

        moduleConfig.locateFile = (path: string, prefix?: string) => {
          if (path.endsWith('.wasm')) {
            return wasmUrl;
          }
          if (userLocateFile) {
            return userLocateFile(path, prefix);
          }
          return prefix ? `${prefix}${path}` : path;
        };

        this.module = await createModule(moduleConfig);
      })().catch((error) => {
        this.initPromise = null;
        this.module = null;
        throw error;
      });
    }
    await this.initPromise;
  }

  public async loadModelFromUrl(
    url: string,
    destFileName: string = 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    const module = await this.ensureModule();
    return this.modelLoader.loadModelFromUrl(
      module,
      url,
      destFileName,
      onProgress,
      signal
    );
  }

  public async loadModelFromReadableStream(
    stream: ReadableStream<Uint8Array>,
    destFileName: string = 'model.gguf',
    options: {
      expectedBytes?: number;
      onProgress?: (pct: number) => void;
      signal?: AbortSignal;
    } = {}
  ): Promise<string> {
    const module = await this.ensureModule();
    return this.modelLoader.loadModelFromReadableStream(
      module,
      stream,
      destFileName,
      options
    );
  }

  public async loadModelFromFile(
    file: File,
    destFileName?: string,
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    const module = await this.ensureModule();
    return this.modelLoader.loadModelFromFile(
      module,
      file,
      destFileName,
      onProgress,
      signal
    );
  }

  public loadModelFromBuffer(buffer: Uint8Array, destFileName: string = 'model.gguf'): string {
    const module = this.getLoadedModule();
    return this.modelLoader.loadModelFromBuffer(module, buffer, destFileName);
  }

  public async loadModelFromFileShards(
    files: File[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    const module = await this.ensureModule();
    return this.modelLoader.loadModelFromFileShards(module, files, onProgress, signal);
  }

  public async loadModelFromUrls(
    urls: string[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    const module = await this.ensureModule();
    return this.modelLoader.loadModelFromUrls(module, urls, onProgress, signal);
  }

  /**
   * Initialize engine state with a model path in MEMFS.
   */
  public async initEngine(
    modelPath: string,
    config?: InferenceInitConfig
  ): Promise<void> {
    const module = await this.ensureModule();
    if (!modelPath || modelPath.trim().length === 0) {
      throw new Error('modelPath must not be empty.');
    }
    if (this.engineInitialized) {
      this.rejectQueuedRequestCompletions(
        new Error('Engine runtime was reset during reinitialization.')
      );
      this.releaseAllQueuedPromptCallbacks(module);
      module.ccall('CE_Close', null, [], []);
      this.engineInitialized = false;
      this.resetRuntimeLifecycleState();
    }

    const normalizedConfig = normalizeInitConfig(config);
    this.runtimeObservabilityEnabled =
      normalizedConfig.enableRuntimeObservability > 0;
    this.backendProfilingEnabled = normalizedConfig.enableBackendProfiling > 0;
    this.transportObservability.enabled = this.runtimeObservabilityEnabled;
    const result = await module.ccall(
      'CE_Init',
      'number',
      [
        'string',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
      ],
      [
        modelPath,
        normalizedConfig.nCtx,
        normalizedConfig.nBatch,
        normalizedConfig.nUbatch,
        normalizedConfig.nSeqMax,
        normalizedConfig.nThreads,
        normalizedConfig.nThreadsBatch,
        normalizedConfig.nGpuLayers,
        normalizedConfig.flashAttention,
        normalizedConfig.kvUnified,
        normalizedConfig.maxCachedSessions,
        normalizedConfig.retainedPrefixTokens,
        normalizedConfig.prefillChunkSize,
        normalizedConfig.prefixCacheIntervalTokens,
        normalizedConfig.maxPrefixCacheEntries,
        normalizedConfig.schedulerPolicy,
        normalizedConfig.decodeTokenReserve,
        normalizedConfig.adaptivePrefillChunking,
        normalizedConfig.enableRuntimeObservability,
        normalizedConfig.enableBackendProfiling,
      ],
      { async: true }
    );
    if (result !== 0) {
      this.engineInitialized = false;
      this.resetRuntimeLifecycleState(
        new Error(`Engine runtime failed to initialize. Code: ${result}`)
      );
      throw new Error(`Failed to initialize engine. Code: ${result}`);
    }
    this.engineInitialized = true;

    this.modelLoader.cleanupAfterEngineInit(module);
  }

  /**
   * Shutdown engine instance.
   */
  public close(): void {
    const module = this.module;
    if (!module) {
      return;
    }
    this.rejectQueuedRequestCompletions(new Error('Engine runtime was closed.'));
    module.ccall('CE_Close', null, [], []);
    this.modelLoader.cleanupAfterClose(module);
    this.releaseAllQueuedPromptCallbacks(module);
    this.engineInitialized = false;
    this.resetRuntimeLifecycleState();
    this.lastModelLoadInfo = null;
    this.module = null;
    this.initPromise = null;
  }

  public async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
    const module = this.getReadyEngineModule();
    if (!Number.isInteger(requestId) || requestId <= 0) {
      return false;
    }

    const result = module.ccall(
      'CE_CancelQueuedRequest',
      'number',
      ['number'],
      [requestId]
    );
    const cancelled = result instanceof Promise ? Boolean(await result) : Boolean(result);
    if (!cancelled) {
      return false;
    }

    const completionState = this.queuedRequestCompletions.get(requestId);
    if (completionState != null) {
      completionState.cancelRequested = true;
      if (this.settleCompletedQueuedRequest(module, requestId, completionState)) {
        return true;
      }
    }

    if (!this.activeQueuedRequestRuns.has(requestId)) {
      this.releaseCancelledQueuedRequestState(module, requestId);
    }
    return true;
  }

  public async queuePrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<GenerateRequestId> {
    const module = this.getReadyEngineModule();
    const request = this.buildGenerateRequest(contextKey, promptText, options);
    const onToken = typeof options === 'object' ? options.onToken : undefined;
    const signal = typeof options === 'object' ? options.signal : undefined;

    if (signal?.aborted) {
      throw createAbortError('Prompt was aborted before it was enqueued.');
    }

    let requestId: GenerateRequestId = 0;
    const callbackPtr =
      onToken == null && signal == null
        ? 0
        : module.addFunction((rawPtr: bigint, length: number) => {
          if (signal?.aborted) {
            return 1;
          }
          if (onToken != null && requestId !== 0) {
            this.bufferQueuedTokenPiece(requestId, module.UTF8ToString(Number(rawPtr), length));
          }
          return signal?.aborted ? 1 : 0;
        }, 'ipi');

    const requestIdResult = module.ccall(
      'CE_EnqueuePrompt',
      'number',
      ['string', 'string', 'number', 'pointer'],
      [request.contextKey, request.promptText, request.maxOutputTokens, Number(callbackPtr)]
    );
    if (requestIdResult instanceof Promise) {
      if (callbackPtr !== 0) {
        module.removeFunction(callbackPtr);
      }
      throw new Error('Unexpected async result while enqueuing a request.');
    }

    requestId = requestIdResult as GenerateRequestId;
    if (!requestId) {
      if (callbackPtr !== 0) {
        module.removeFunction(callbackPtr);
      }
      throw new Error('Failed to enqueue request.');
    }

    if (callbackPtr !== 0) {
      this.queuedPromptCallbacks.set(requestId, onToken);
      this.queuedPromptTokenBuffers.set(requestId, []);
      this.queuedPromptCallbackPtrs.set(requestId, callbackPtr);
    }

    if (signal != null) {
      const abortListener = () => {
        void this.cancelQueuedRequest(requestId);
      };
      this.queuedPromptSignals.set(requestId, signal);
      this.queuedPromptSignalAbortListeners.set(requestId, abortListener);
      signal.addEventListener('abort', abortListener, { once: true });
    }

    this.getOrCreateQueuedRequestCompletion(requestId);

    return requestId;
  }

  public async runQueuedRequest(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    this.getReadyEngineModule();
    if (!Number.isInteger(requestId) || requestId <= 0) {
      throw new Error('requestId must be a positive integer.');
    }
    if (options?.signal?.aborted) {
      await this.cancelQueuedRequest(requestId);
      throw createAbortError('Prompt was aborted before execution started.');
    }

    const completionState = this.getOrCreateQueuedRequestCompletion(requestId);
    const abortListener =
      options?.signal == null
        ? null
        : () => {
            void this.cancelQueuedRequest(requestId);
          };
    if (abortListener != null) {
      options?.signal?.addEventListener('abort', abortListener, { once: true });
    }

    completionState.consumed = true;
    completionState.waiterCount += 1;
    try {
      const response = await completionState.promise;
      const callbackError = completionState.callbackError;
      if (callbackError != null) {
        throw callbackError;
      }
      if (response.cancelled || options?.signal?.aborted) {
        throw createAbortError(response.errorMessage ?? 'Queued request cancelled.');
      }
      return response;
    } finally {
      if (abortListener != null) {
        options?.signal?.removeEventListener('abort', abortListener);
      }
      completionState.waiterCount = Math.max(0, completionState.waiterCount - 1);
      this.cleanupConsumedCompletionState(requestId);
    }
  }

  /**
   * Submit a generation prompt.
   */
  public async submitPrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<string> {
    const request = this.buildGenerateRequest(contextKey, promptText, options);
    const onToken = typeof options === 'object' ? options.onToken : undefined;
    const signal = typeof options === 'object' ? options.signal : undefined;
    const requestId = await this.queuePrompt(request.contextKey, request.promptText, {
      nTokens: request.maxOutputTokens,
      promptFormat: request.promptFormat,
      onToken,
      signal,
    });
    const response = await this.runQueuedRequest(requestId, { signal });
    if (response.cancelled) {
      throw createAbortError(response.errorMessage ?? 'Queued prompt cancelled.');
    }
    if (response.failed) {
      throw new Error(response.errorMessage ?? 'Queued prompt failed.');
    }
    return response.outputText;
  }

  public getRuntimeAggregateObservability(): RuntimeAggregateObservabilityMetrics | null {
    if (!this.runtimeObservabilityEnabled) {
      return null;
    }

    const module = this.getReadyEngineModule();
    return this.readRuntimeObservabilityFromModule(module);
  }

  public getRuntimeObservability(): RuntimeAggregateObservabilityMetrics | null {
    return this.getRuntimeAggregateObservability();
  }

  public async getBackendObservability(): Promise<BackendObservability | null> {
    const module = this.getLoadedModule();
    const rawPtr = await module.ccall('CE_GetBackendObservabilityJson', 'pointer', [], [], {
      async: true,
    });

    // ccall with 'pointer' return type already converts BigInt → Number.
    const ptr = rawPtr as number;

    if (!ptr) {
      return null;
    }

    try {
      const raw = module.UTF8ToString(ptr);
      const parsed = JSON.parse(raw) as BackendObservability;
      parsed.profilingEnabled = this.backendProfilingEnabled;
      return parsed;
    } catch (error) {
      throw new Error(`Failed to parse backend observability: ${asErrorMessage(error)}`);
    } finally {
      module.ccall('CE_FreeString', null, ['pointer'], [ptr]);
    }
  }
}
