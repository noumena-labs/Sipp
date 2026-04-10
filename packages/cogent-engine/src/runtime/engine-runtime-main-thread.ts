import { FileSystemStorage } from '../storage/file-system-storage.js';
import {
  BrowserModelCache,
  BrowserModelCacheIdentity,
  BrowserModelCacheLookupResult,
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
  RequestObservabilityMetrics,
  RuntimeAggregateObservabilityMetrics,
  RuntimeObservabilityMetrics,
  TransportObservability,
} from '../types.js';
import { EngineRuntime } from './engine-runtime.js';

interface EmscriptenFs {
  analyzePath(path: string): { exists: boolean };
  mkdir(path: string): void;
  writeFile(path: string, data: Uint8Array): void;
  unlink(path: string): void;
  mount(type: any, opts: any, mountpoint: string): void;
  unmount(mountpoint: string): void;
}

interface EngineModule {
  FS: EmscriptenFs;
  WORKERFS: any;
  HEAP32: Int32Array;
  HEAPF64: Float64Array;
  _free(ptr: number | bigint): void;
  _malloc(size: number | bigint): number | bigint;
  addFunction(func: (...args: any[]) => any, signature: string): number | bigint;
  ccall(ident: string, returnType: string | null, argTypes: string[], args: any[], opts?: { async?: boolean }): Promise<any> | any;
  removeFunction(ptr: number | bigint): void;
  UTF8ToString(ptr: number | bigint, maxBytesToRead?: number): string;
}

type MountableModelFile = Blob & { name?: string };
type HeaderLookup = { get(name: string): string | null };
type UrlShardMetadata = {
  url: string;
  fileName: string;
  contentLength: number;
  cacheIdentity: BrowserModelCacheIdentity;
};

interface QueuedRequestCompletionState {
  promise: Promise<GenerateResponse>;
  resolve: (value: GenerateResponse) => void;
  reject: (error: unknown) => void;
  settled: boolean;
  consumed: boolean;
  waiterCount: number;
  callbackError: unknown;
  cancelRequested: boolean;
}

const MAX_PROMPT_TOKENS = 2048;
const DEFAULT_MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024;
const DEFAULT_PROMPT_FORMAT = 'auto-chat';
const URL_METADATA_FETCH_CONCURRENCY = 4;
const URL_DOWNLOAD_CONCURRENCY_OPFS = 4;
const URL_DOWNLOAD_CONCURRENCY_MEMORY = 2;
const REQUEST_STEP_RESULT_INVALID = -1;
const REQUEST_STEP_RESULT_FATAL_NO_PROGRESS = -2;
const REQUEST_STEP_RESULT_WAITING = 0;
const REQUEST_STEP_RESULT_PROGRESSED = 1;
const REQUEST_STEP_RESULT_TERMINAL = 2;
const COMPLETED_REQUEST_STATUS_PENDING = 0;
const COMPLETED_REQUEST_STATUS_COMPLETED = 1;
const COMPLETED_REQUEST_STATUS_CANCELLED = 2;
const COMPLETED_REQUEST_STATUS_FAILED = 3;
const RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES = 128;
const RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT = 9;

const DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY: TransportObservability = {
  executionMode: 'main-thread',
  workerBacked: false,
  enabled: false,
  bufferedTokenLimit: 0,
  flushIntervalMs: 0,
  flushCount: 0,
  coalescedTokenCount: 0,
  maxObservedBufferedTokenCount: 0,
};

function createAbortError(message = 'The operation was aborted.'): Error {
  if (typeof DOMException === 'function') {
    return new DOMException(message, 'AbortError');
  }
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function normalizeModelFileName(fileName: string): string {
  const trimmed = fileName.trim();
  if (!trimmed) {
    throw new Error('Model file name must not be empty.');
  }
  if (trimmed.includes('/') || trimmed.includes('\\') || trimmed.includes('..')) {
    throw new Error(`Invalid model file name "${fileName}". Provide a simple file name, not a path.`);
  }
  return trimmed;
}

function asErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function createLinkedAbortController(signal?: AbortSignal): {
  controller: AbortController;
  signal: AbortSignal;
  dispose: () => void;
} {
  const controller = new AbortController();
  if (signal?.aborted) {
    controller.abort();
    return {
      controller,
      signal: controller.signal,
      dispose: () => {},
    };
  }

  const abortListener =
    signal == null
      ? null
      : () => {
          controller.abort();
        };
  if (abortListener != null) {
    signal!.addEventListener('abort', abortListener, { once: true });
  }

  return {
    controller,
    signal: controller.signal,
    dispose: () => {
      if (abortListener != null) {
        signal?.removeEventListener('abort', abortListener);
      }
    },
  };
}

function createDeferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

export class MainThreadEngineRuntime implements EngineRuntime {
  private module: EngineModule | null = null;
  private initPromise: Promise<void> | null = null;
  private engineInitialized = false;
  private loadedModelPaths: string[] = [];
  private workerFsMountPath: string | null = null;
  private readonly opfs = new FileSystemStorage();
  private readonly browserModelCache = new BrowserModelCache(this.opfs);
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
  constructor(private config: CogentConfig = {}) { }

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

  private resolveMaxModelBytes(): number {
    const maxModelBytes = this.config.maxModelBytes ?? DEFAULT_MAX_MODEL_BYTES;
    if (!Number.isInteger(maxModelBytes) || maxModelBytes <= 0) {
      throw new Error('"maxModelBytes" must be a positive integer.');
    }
    return maxModelBytes;
  }

  private setLastModelLoadInfo(info: ModelLoadInfo): void {
    this.lastModelLoadInfo = info;
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
      if (stepResult === REQUEST_STEP_RESULT_WAITING && !settledAfterTick) {
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
    const result = module.ccall(ident, 'number', argTypes, args);
    if (result instanceof Promise) {
      throw new Error(`Unexpected async result while calling ${ident}.`);
    }
    return Number(result);
  }

  private async callNumberAsync(
    module: EngineModule,
    ident: string,
    argTypes: string[] = [],
    args: unknown[] = []
  ): Promise<number> {
    const result = module.ccall(ident, 'number', argTypes, args, {
      async: true,
    });
    return Number(await result);
  }

  private async mapWithConcurrency<T, TResult>(
    items: readonly T[],
    concurrency: number,
    mapper: (item: T, index: number) => Promise<TResult>,
    onError?: (error: unknown) => void
  ): Promise<TResult[]> {
    if (items.length === 0) {
      return [];
    }

    const results = new Array<TResult>(items.length);
    const workerCount = Math.min(Math.max(1, concurrency), items.length);
    let nextIndex = 0;
    let firstError: unknown = null;

    const workers = Array.from({ length: workerCount }, async () => {
      while (true) {
        if (firstError != null) {
          return;
        }

        const currentIndex = nextIndex;
        nextIndex += 1;
        if (currentIndex >= items.length) {
          return;
        }

        try {
          results[currentIndex] = await mapper(items[currentIndex], currentIndex);
        } catch (error) {
          if (firstError == null) {
            firstError = error;
            onError?.(error);
          }
          throw error;
        }
      }
    });

    await Promise.allSettled(workers);
    if (firstError != null) {
      throw firstError;
    }
    return results;
  }

  private readRuntimeObservabilityFromModule(
    module: EngineModule
  ): RuntimeAggregateObservabilityMetrics | null {
    return this.readRuntimeObservabilityViaCall(
      module,
      'CE_GetRuntimeObservability',
      ['pointer'],
      []
    );
  }

  private readRuntimeObservabilityViaCall(
    module: EngineModule,
    ident: string,
    argTypes: string[],
    args: unknown[]
  ): RuntimeObservabilityMetrics | null {
    const metricsPtr = Number(module._malloc(RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES));
    if (!metricsPtr) {
      throw new Error('Failed to allocate runtime observability buffer.');
    }

    try {
      const status = this.callNumber(module, ident, [...argTypes, 'pointer'], [...args, metricsPtr]);
      if (status !== 0) {
        return null;
      }

      // Integer division for typed array index: ptr must be aligned.
      const f64Offset = (metricsPtr / 8) | 0;
      const i32Offset = ((metricsPtr + RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT * 8) / 4) | 0;

      return {
        totalMs: module.HEAPF64[f64Offset],
        promptEvalMs: module.HEAPF64[f64Offset + 1],
        decodeEvalMs: module.HEAPF64[f64Offset + 2],
        sampleMs: module.HEAPF64[f64Offset + 3],
        queueDelayMs: module.HEAPF64[f64Offset + 4],
        ttftMs: module.HEAPF64[f64Offset + 5],
        meanItlMs: module.HEAPF64[f64Offset + 6],
        tailItlMs: module.HEAPF64[f64Offset + 7],
        e2elMs: module.HEAPF64[f64Offset + 8],
        inputTokenCount: module.HEAP32[i32Offset],
        promptEvalTokens: module.HEAP32[i32Offset + 1],
        decodeEvalCount: module.HEAP32[i32Offset + 2],
        sampleCount: module.HEAP32[i32Offset + 3],
        outputTokenCount: module.HEAP32[i32Offset + 4],
        batchParticipationCount: module.HEAP32[i32Offset + 5],
        decodeFirstTickCount: module.HEAP32[i32Offset + 6],
        chunkedPrefillTickCount: module.HEAP32[i32Offset + 7],
        mixedWorkloadTickCount: module.HEAP32[i32Offset + 8],
        lcpReuseTokens: module.HEAP32[i32Offset + 9],
        prefixCacheRestoreTokens: module.HEAP32[i32Offset + 10],
        prefixCacheHitCount: module.HEAP32[i32Offset + 11],
        prefixCacheStoreCount: module.HEAP32[i32Offset + 12],
      };
    } finally {
      module._free(metricsPtr);
    }
  }

  private readCompletedRequestRuntimeObservability(
    module: EngineModule,
    requestId: GenerateRequestId
  ): RequestObservabilityMetrics | null {
    return this.readRuntimeObservabilityViaCall(
      module,
      'CE_GetCompletedRequestRuntimeObservability',
      ['number'],
      [requestId]
    );
  }

  private copyCompletedRequestText(
    module: EngineModule,
    requestId: GenerateRequestId,
    sizeFunction: string,
    copyFunction: string,
    fieldName: string
  ): string {
    const byteLength = this.callNumber(module, sizeFunction, ['number'], [requestId]);
    if (byteLength < 0) {
      throw new Error(`Failed to read queued request ${fieldName} size.`);
    }
    if (byteLength === 0) {
      return '';
    }

    const rawBufferPtr = module._malloc(byteLength + 1);
    if (!rawBufferPtr) {
      throw new Error(`Failed to allocate queued request ${fieldName} buffer.`);
    }
    const bufferPtr = Number(rawBufferPtr);

    try {
      const copied = this.callNumber(module, copyFunction, ['number', 'pointer', 'number'], [
        requestId,
        bufferPtr,
        byteLength + 1,
      ]);
      if (copied !== byteLength) {
        throw new Error(`Failed to copy queued request ${fieldName}.`);
      }
      return module.UTF8ToString(bufferPtr, byteLength);
    } finally {
      module._free(bufferPtr);
    }
  }

  private takeCompletedResponse(
    module: EngineModule,
    requestId: GenerateRequestId
  ): GenerateResponse {
    const status = this.callNumber(module, 'CE_GetCompletedRequestStatus', ['number'], [requestId]);
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      throw new Error('Queued request reached a terminal step without a completed response.');
    }

    const outputText = this.copyCompletedRequestText(
      module,
      requestId,
      'CE_GetCompletedRequestOutputSize',
      'CE_CopyCompletedRequestOutput',
      'output'
    );
    const errorText = this.copyCompletedRequestText(
      module,
      requestId,
      'CE_GetCompletedRequestErrorSize',
      'CE_CopyCompletedRequestError',
      'error'
    );
    const runtimeObservability = this.readCompletedRequestRuntimeObservability(
      module,
      requestId
    );
    const consumed = this.callNumber(module, 'CE_ConsumeCompletedRequest', ['number'], [requestId]);
    if (!consumed) {
      throw new Error('Failed to consume completed queued request response.');
    }

    return {
      requestId,
      completed: status === COMPLETED_REQUEST_STATUS_COMPLETED,
      failed: status === COMPLETED_REQUEST_STATUS_FAILED,
      cancelled: status === COMPLETED_REQUEST_STATUS_CANCELLED,
      outputText,
      errorMessage: errorText.length > 0 ? errorText : null,
      requestObservability: runtimeObservability,
      runtimeObservability,
    };
  }

  private removeFileIfExists(module: EngineModule, path: string): void {
    if (module.FS.analyzePath(path).exists) {
      module.FS.unlink(path);
    }
  }

  private createMountableModelFile(blob: Blob, fileName: string): MountableModelFile {
    const normalizedFileName = normalizeModelFileName(fileName);
    const existingName = (blob as MountableModelFile).name;
    if (existingName === normalizedFileName) {
      return blob as MountableModelFile;
    }

    if (typeof File === 'function') {
      return new File([blob], normalizedFileName, {
        type: blob.type,
      }) as MountableModelFile;
    }

    const copiedBlob = blob.slice(0, blob.size, blob.type) as MountableModelFile;
    Object.defineProperty(copiedBlob, 'name', {
      configurable: true,
      value: normalizedFileName,
      writable: false,
    });
    return copiedBlob;
  }

  private removeAllLoadedModelFiles(module: EngineModule): void {
    for (const p of this.loadedModelPaths) {
      // Don't try to unlink files inside a WORKERFS mount point.
      // They will be "cleaned up" when the mount is unmounted.
      if (this.workerFsMountPath && p.startsWith(this.workerFsMountPath)) {
        continue;
      }
      this.removeFileIfExists(module, p);
    }
    this.loadedModelPaths = [];
  }

  private commitLoadedModelPaths(module: EngineModule, paths: string[]): void {
    // Clean up any previously loaded model files that aren't in the new set
    const newSet = new Set(paths);
    for (const p of this.loadedModelPaths) {
      if (!newSet.has(p)) {
        this.removeFileIfExists(module, p);
      }
    }
    this.loadedModelPaths = [...paths];
  }

  private prepareModelPath(module: EngineModule, destFileName: string): string {
    const safeName = normalizeModelFileName(destFileName);
    const modelPath = `/models/${safeName}`;
    this.ensureModelsDir(module);
    this.removeFileIfExists(module, modelPath);
    return modelPath;
  }

  private async readStreamToMountableModelFile(
    stream: ReadableStream<Uint8Array>,
    fileName: string,
    onProgress?: (bytes: number) => void,
    signal?: AbortSignal
  ): Promise<MountableModelFile> {
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];
    let bytesRead = 0;
    const abortListener =
      signal == null
        ? null
        : () => {
            void reader.cancel(createAbortError('Model load aborted.'));
          };
    if (abortListener != null) {
      signal!.addEventListener('abort', abortListener, { once: true });
    }
    try {
      while (true) {
        if (signal?.aborted) {
          throw createAbortError('Model load aborted.');
        }
        const { done, value } = await reader.read();
        if (done) {
          if (signal?.aborted) {
            throw createAbortError('Model load aborted.');
          }
          break;
        }
        if (value != null) {
          chunks.push(value);
          bytesRead += value.byteLength;
          onProgress?.(bytesRead);
        }
      }
    } catch (error) {
      if (isAbortError(error) || signal?.aborted) {
        throw createAbortError('Model load aborted.');
      }
      throw error;
    } finally {
      if (abortListener != null) {
        signal!.removeEventListener('abort', abortListener);
      }
      reader.releaseLock();
    }

    return this.createMountableModelFile(new Blob(chunks as any), fileName);
  }

  private async resolveUrlShardMetadata(
    urls: string[],
    signal: AbortSignal
  ): Promise<UrlShardMetadata[]> {
    return this.mapWithConcurrency(
      urls,
      URL_METADATA_FETCH_CONCURRENCY,
      async (url) => {
        const parsed = this.parseConfiguredUrl(url, 'modelUrl');
        const canonicalUrl = parsed.toString();
        const fileName = normalizeModelFileName(parsed.pathname.split('/').pop() || 'model.gguf');
        try {
          const headResp = await fetch(url, { method: 'HEAD', signal });
          const cl = Number.parseInt(headResp.headers.get('Content-Length') ?? '0', 10) || 0;
          return {
            url,
            fileName,
            contentLength: cl,
            cacheIdentity: {
              canonicalUrl,
              fileName,
              etag: headResp.headers.get('ETag')?.trim() ?? '',
              lastModified: headResp.headers.get('Last-Modified')?.trim() ?? '',
              contentLength: cl,
            },
          };
        } catch (error) {
          if (isAbortError(error) || signal.aborted) {
            throw createAbortError('Model load aborted.');
          }
          return {
            url,
            fileName,
            contentLength: 0,
            cacheIdentity: {
              canonicalUrl,
              fileName,
              etag: '',
              lastModified: '',
              contentLength: 0,
            },
          };
        }
      }
    );
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

  private ensureModelsDir(module: EngineModule) {
    const modelsPath = '/models';
    if (!module.FS.analyzePath(modelsPath).exists) {
      module.FS.mkdir(modelsPath);
    }
  }


  /**
   * Load a GGUF model from a URL.
   * 
   * This streams the model directly to OPFS (Persistent Storage) if supported,
   * then mounts it into the WASM filesystem via WORKERFS. This is zero-copy
   * and handles files >2GB.
   */
  public async loadModelFromUrl(
    url: string,
    destFileName: string = 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    return this.loadModelFromUrls([url], onProgress, signal);
  }

  /**
   * Load a model from a ReadableStream.
   * 
   * The stream is piped to OPFS then zero-copy mounted.
   */
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
    const opfsEnabled = FileSystemStorage.isSupported() && this.config.persistentModelCache?.enabled !== false;

    let modelFile: Blob;
    if (opfsEnabled) {
      modelFile = await this.opfs.streamToDisk(
        destFileName,
        stream,
        options.onProgress,
        options.signal
      );
    } else {
      modelFile = await this.readStreamToMountableModelFile(
        stream,
        destFileName,
        undefined,
        options.signal
      );
    }

    const modelPath = await this.mountModelFiles(module, [modelFile]);

    this.setLastModelLoadInfo({
      sourceKind: 'buffer',
      reuseMode: 'buffer',
      modelPath,
      fileName: destFileName,
      byteLength: modelFile.size,
      persistentCacheEnabled: opfsEnabled,
      persistentCacheKey: null,
      persistentCacheHit: false,
      persistentCacheStored: opfsEnabled,
    });

    return modelPath;
  }

  /**
   * Internal helper to mount one or more Blob/File objects into the WASM filesystem
   * using the zero-copy WORKERFS driver.
   *
   * @param module The Emscripten module instance.
   * @param files Array of Blob or File objects to mount.
   * @param mountDir The path in the virtual filesystem where files will be mounted.
   * @returns The virtual path to the first file in the set.
   */
  private async mountModelFiles(
    module: EngineModule,
    files: MountableModelFile[],
    mountDir = '/workerfs_model'
  ): Promise<string> {
    const fs = module.FS;

    // Ensure mount directory exists
    if (!fs.analyzePath(mountDir).exists) {
      fs.mkdir(mountDir);
    } else if (this.workerFsMountPath) {
      // If we already have something mounted, unmount first
      try {
        fs.unmount(this.workerFsMountPath);
      } catch (e) { }
    }

    if (!module.WORKERFS) {
      throw new Error(
        'WORKERFS is not available in the Emscripten module. ' +
        'Ensure the module was linked with -lworkerfs.js and WORKERFS is exported.'
      );
    }

    fs.mount(module.WORKERFS, { files }, mountDir);
    this.workerFsMountPath = mountDir;

    // Path in WORKERFS is /mountDir/fileName
    const firstFileName = files[0].name || 'model.gguf';
    const firstModelPath = `${mountDir}/${firstFileName}`;

    this.commitLoadedModelPaths(module, files.map((file) => `${mountDir}/${file.name || 'model.gguf'}`));

    return firstModelPath;
  }

  /**
   * Load a model from a local File object.
   *
   * This uses WORKERFS for zero-copy, low-RAM loading, bypassing any 2GB MEMFS size limits.
   */
  public async loadModelFromFile(
    file: File,
    destFileName?: string,
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    const module = await this.ensureModule();

    if (onProgress) {
      onProgress(100); // WORKERFS mount is instant
    }

    const modelPath = await this.mountModelFiles(module, [file]);

    this.setLastModelLoadInfo({
      sourceKind: 'file',
      reuseMode: 'file-read',
      modelPath,
      fileName: normalizeModelFileName(destFileName || file.name),
      byteLength: file.size,
      persistentCacheEnabled: false,
      persistentCacheKey: null,
      persistentCacheHit: false,
      persistentCacheStored: false,
    });

    return modelPath;
  }

  /**
   * Load a GGUF model from a local buffer into MEMFS.
   */
  public loadModelFromBuffer(buffer: Uint8Array, destFileName: string = 'model.gguf'): string {
    const module = this.getLoadedModule();
    const maxModelBytes = this.resolveMaxModelBytes();
    if (buffer.byteLength === 0) {
      throw new Error('Model buffer is empty.');
    }
    if (buffer.byteLength > maxModelBytes) {
      throw new Error(`Model exceeds configured maxModelBytes (${maxModelBytes} bytes).`);
    }

    const modelPath = this.prepareModelPath(module, destFileName);
    module.FS.writeFile(modelPath, buffer);
    this.commitLoadedModelPaths(module, [modelPath]);
    this.setLastModelLoadInfo({
      sourceKind: 'buffer',
      reuseMode: 'buffer',
      modelPath,
      fileName: normalizeModelFileName(destFileName),
      byteLength: buffer.byteLength,
      persistentCacheEnabled: false,
      persistentCacheKey: null,
      persistentCacheHit: false,
      persistentCacheStored: false,
    });
    return modelPath;
  }

  /**
   * Load a split GGUF model from an array of File objects.
   *
   * Use `gguf-split` to split a large model into shards (<2GB each) so that
   * each shard fits within the MEMFS file-size limit. llama.cpp natively
   * detects the split naming convention (e.g. `model-00001-of-00010.gguf`)
   * and loads all shards automatically when given the first shard's path.
   *
   * @param files Array of File objects, one per shard, in order.
   * @param onProgress Optional progress callback (0–100 across all shards).
   * @param signal Optional AbortSignal.
   * @returns The MEMFS path to the first shard (pass this to initEngine).
   */
  public async loadModelFromFileShards(
    files: File[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    if (!files || files.length === 0) {
      throw new Error('No shard files provided.');
    }
    if (files.length === 1) {
      return this.loadModelFromFile(files[0], files[0].name, onProgress, signal);
    }

    const module = await this.ensureModule();
    const maxModelBytes = this.resolveMaxModelBytes();
    this.ensureModelsDir(module);

    const totalBytes = files.reduce((sum, f) => sum + f.size, 0);
    if (totalBytes <= 0) {
      throw new Error('Model shards are empty.');
    }
    if (totalBytes > maxModelBytes) {
      throw new Error(`Total model size (${totalBytes} bytes) exceeds configured maxModelBytes (${maxModelBytes} bytes).`);
    }

    try {
      // Mount all shards at once as a single WORKERFS directory
      const modelPath = await this.mountModelFiles(module, files);

      this.commitLoadedModelPaths(module, files.map(f => `/workerfs_model/${f.name}`));
      this.setLastModelLoadInfo({
        sourceKind: 'file',
        reuseMode: 'file-read',
        modelPath,
        fileName: normalizeModelFileName(files[0].name),
        byteLength: totalBytes,
        persistentCacheEnabled: false,
        persistentCacheKey: null,
        persistentCacheHit: false,
        persistentCacheStored: false,
      });

      if (onProgress) onProgress(100);
      return modelPath;
    } catch (error) {
      if (isAbortError(error) || signal?.aborted) {
        throw createAbortError('Model load aborted.');
      }
      throw new Error(`Failed while loading model shards: ${asErrorMessage(error)}`);
    }
  }

  /**
   * Load a split GGUF model from an array of URLs.
   */
  public async loadModelFromUrls(
    urls: string[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    if (!urls || urls.length === 0) {
      throw new Error('No shard URLs provided.');
    }

    const module = await this.ensureModule();
    const opfsSupported = FileSystemStorage.isSupported() && this.config.persistentModelCache?.enabled !== false;
    const linkedAbort = createLinkedAbortController(signal);
    const loadSignal = linkedAbort.signal;
    const downloadConcurrency = opfsSupported
      ? URL_DOWNLOAD_CONCURRENCY_OPFS
      : URL_DOWNLOAD_CONCURRENCY_MEMORY;

    try {
      const shardMeta = await this.resolveUrlShardMetadata(urls, loadSignal);
      const totalBytes = shardMeta.reduce((sum, shard) => sum + shard.contentLength, 0);
      const shardLoadedBytes = new Array<number>(shardMeta.length).fill(0);
      let totalLoadedBytes = 0;

      const reportShardProgress = (index: number, loadedBytes: number) => {
        const normalizedBytes = Math.max(0, loadedBytes);
        const previousBytes = shardLoadedBytes[index];
        if (normalizedBytes <= previousBytes) {
          return;
        }
        shardLoadedBytes[index] = normalizedBytes;
        totalLoadedBytes += normalizedBytes - previousBytes;
        if (onProgress != null && totalBytes > 0) {
          onProgress(Math.min(100, Math.round((totalLoadedBytes / totalBytes) * 100)));
        }
      };

      const shardResults = await this.mapWithConcurrency(
        shardMeta,
        downloadConcurrency,
        async (shard, index) => {
          if (loadSignal.aborted) {
            throw createAbortError('Model load aborted.');
          }

          const cachedEntry: BrowserModelCacheLookupResult | null = opfsSupported
            ? await this.browserModelCache.get(shard.cacheIdentity)
            : null;
          if (cachedEntry != null) {
            reportShardProgress(index, cachedEntry.file.size);
            return {
              file: this.createMountableModelFile(cachedEntry.file, shard.fileName),
              cacheKey: cachedEntry.key,
              cacheHit: true,
              cacheStored: false,
            };
          }

          const response = await fetch(shard.url, { signal: loadSignal });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status} for ${shard.fileName}`);
          }

          if (opfsSupported) {
            if (!response.body) {
              throw new Error(`Empty body for ${shard.fileName}`);
            }
            const storedEntry = await this.browserModelCache.storeStream(
              shard.cacheIdentity,
              response.body,
              (written) => {
                reportShardProgress(index, written);
              },
              loadSignal
            );
            reportShardProgress(index, storedEntry.file.size);
            return {
              file: this.createMountableModelFile(storedEntry.file, shard.fileName),
              cacheKey: storedEntry.key,
              cacheHit: false,
              cacheStored: true,
            };
          }

          if (!response.body) {
            const buffer = await response.arrayBuffer();
            reportShardProgress(index, buffer.byteLength);
            return {
              file: this.createMountableModelFile(new Blob([buffer]), shard.fileName),
              cacheKey: null,
              cacheHit: false,
              cacheStored: false,
            };
          }

          return {
            file: await this.readStreamToMountableModelFile(
              response.body,
              shard.fileName,
              (written) => {
                reportShardProgress(index, written);
              },
              loadSignal
            ),
            cacheKey: null,
            cacheHit: false,
            cacheStored: false,
          };
        },
        () => {
          linkedAbort.controller.abort();
        }
      );

      const shardBlobs = shardResults.map((result) => result.file);
      const modelPath = await this.mountModelFiles(module, shardBlobs);
      if (onProgress != null && totalBytes === 0) {
        onProgress(100);
      }

      const cacheKeys = shardResults
        .map((result, index) => result.cacheKey ?? this.browserModelCache.buildEntryKey(shardMeta[index].cacheIdentity));
      const allCacheHits = opfsSupported && shardResults.every((result) => result.cacheHit);
      const anyCacheStored = opfsSupported && shardResults.some((result) => result.cacheStored);

      this.setLastModelLoadInfo({
        sourceKind: 'url',
        reuseMode: allCacheHits ? 'persistent-cache' : 'network',
        modelPath,
        fileName: shardMeta[0].fileName,
        byteLength: shardBlobs.reduce((sum, b) => sum + b.size, 0),
        persistentCacheEnabled: opfsSupported,
        persistentCacheKey: opfsSupported ? cacheKeys.join(',') : null,
        persistentCacheHit: allCacheHits,
        persistentCacheStored: anyCacheStored,
      });

      return modelPath;
    } catch (e) {
      linkedAbort.controller.abort();
      if (isAbortError(e) || signal?.aborted || loadSignal.aborted) throw createAbortError();
      throw new Error(`Model load from URLs failed: ${asErrorMessage(e)}`);
    } finally {
      linkedAbort.dispose();
    }
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

    this.removeAllLoadedModelFiles(module);
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

    if (this.workerFsMountPath) {
      try {
        module.FS.unmount(this.workerFsMountPath);
      } catch (e) {
        // Ignore
      }
      this.workerFsMountPath = null;
    }

    this.releaseAllQueuedPromptCallbacks(module);
    this.engineInitialized = false;
    this.loadedModelPaths = [];
    this.workerFsMountPath = null;
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
