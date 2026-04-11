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
  COMPLETED_REQUEST_STATUS_PENDING,
  DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
  DEFAULT_PROMPT_FORMAT,
  MAX_PROMPT_TOKENS,
} from './main-thread-runtime-constants.js';
import {
  QueuedRequestPumpStepResult,
} from './queued-request-pump.js';
import { RequestTracker } from './request-tracker.js';
import {
  parseBackendObservabilityJson,
  WasmBridge,
} from '../wasm/wasm-bridge.js';
import { EngineModule } from '../wasm/engine-module.js';
import { createAbortError } from '../utils/abort.js';
import { asErrorMessage } from '../utils/error.js';
import {
  QueuedRequestPumpMode,
  QueuedRequestScheduler,
} from './scheduler.js';

export class MainThreadEngineRuntime implements EngineRuntime {
  private module: EngineModule | null = null;
  private wasmBridge: WasmBridge | null = null;
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
  private readonly tracker = new RequestTracker<GenerateResponse>();
  private readonly scheduler: QueuedRequestScheduler;
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
    this.scheduler = new QueuedRequestScheduler({
      tracker: this.tracker,
      queuedPromptCallbacks: this.queuedPromptCallbacks,
      queuedPromptTokenBuffers: this.queuedPromptTokenBuffers,
      queuedPromptCallbackErrors: this.queuedPromptCallbackErrors,
      getTransportObservability: () => this.transportObservability,
      getBridge: () => this.getReadyEngineBridge(),
      finalizeRequest: (bridge, requestId, options) => {
        this.finalizeRequest(bridge, requestId, options);
      },
      cancelQueuedRequest: (requestId) => this.cancelQueuedRequest(requestId),
    });
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

  private getLoadedWasmBridge(): WasmBridge {
    if (this.wasmBridge == null) {
      this.wasmBridge = new WasmBridge(this.getLoadedModule());
    }
    return this.wasmBridge;
  }

  private getReadyEngineBridge(): WasmBridge {
    this.getReadyEngineModule();
    return this.getLoadedWasmBridge();
  }

  private releaseTokenState(requestId: GenerateRequestId): void {
    this.queuedPromptCallbacks.delete(requestId);
    this.queuedPromptTokenBuffers.delete(requestId);
    this.queuedPromptCallbackErrors.delete(requestId);
  }

  private releaseCallbackPtr(
    bridge: WasmBridge,
    requestId: GenerateRequestId
  ): void {
    const callbackPtr = this.queuedPromptCallbackPtrs.get(requestId);
    if (callbackPtr != null) {
      bridge.unregisterCallback(callbackPtr);
    }
    this.queuedPromptCallbackPtrs.delete(requestId);
  }

  private finalizeRequest(
    bridge: WasmBridge | null,
    requestId: GenerateRequestId,
    options: {
      consumeCompletedResponse?: boolean;
      deleteCompletion?: boolean;
    } = {}
  ): void {
    if (options.consumeCompletedResponse && bridge != null) {
      this.consumeCompletedResponseIfPresent(bridge, requestId);
    }
    if (bridge != null) {
      this.releaseCallbackPtr(bridge, requestId);
    }
    this.releaseTokenState(requestId);
    this.tracker.finalize(requestId, options);
  }

  private consumeCompletedResponseIfPresent(
    bridge: WasmBridge,
    requestId: GenerateRequestId
  ): boolean {
    const status = bridge.getCompletedRequestStatus(requestId);
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      return false;
    }
    return bridge.consumeCompletedResponseIfPresent(requestId);
  }

  private resolveTokenTransportPreference(): 'auto' | 'runtime-events' {
    return this.config.debugTokenTransport ?? 'auto';
  }

  private shouldUseNativeRuntimeEvents(
    bridge: WasmBridge,
    onToken: ((token: string) => void) | undefined
  ): boolean {
    const preference = this.resolveTokenTransportPreference();
    this.transportObservability.tokenTransportPreference = preference;

    if (onToken == null) {
      this.transportObservability.activeTokenTransport = 'none';
      return false;
    }

    const runtimeEventsAvailable = bridge.supportsRuntimeEventDrain();
    if (preference === 'runtime-events') {
      if (!runtimeEventsAvailable) {
        throw new Error(
          'debugTokenTransport=runtime-events requires CE_DrainRuntimeEvents support in the loaded runtime module.'
        );
      }
      this.transportObservability.activeTokenTransport = 'runtime-events';
      return true;
    }

    this.transportObservability.activeTokenTransport = runtimeEventsAvailable
      ? 'runtime-events'
      : 'callbacks';
    return runtimeEventsAvailable;
  }

  private rejectAllTrackedRequests(error: unknown, bridge: WasmBridge | null = null): void {
    this.scheduler.reset();
    for (const requestId of this.tracker.allTrackedIds()) {
      if (bridge != null) {
        this.releaseCallbackPtr(bridge, requestId);
      }
      this.releaseTokenState(requestId);
    }
    this.tracker.rejectAll(error);
  }

  private resetRuntimeLifecycleState(error?: unknown): void {
    if (error != null) {
      this.rejectAllTrackedRequests(error);
    } else {
      this.scheduler.reset();
      this.tracker.clear();
    }
    this.runtimeObservabilityEnabled = false;
    this.backendProfilingEnabled = false;
    this.transportObservability = {
      ...DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
    };
  }

  private ensureTracked(requestId: GenerateRequestId) {
    return this.scheduler.track(requestId);
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
        this.wasmBridge = new WasmBridge(this.module);
      })().catch((error) => {
        this.initPromise = null;
        this.module = null;
        this.wasmBridge = null;
        throw error;
      });
    }
    await this.initPromise;
  }

  public setQueuedRequestPumpMode(mode: QueuedRequestPumpMode): void {
    this.scheduler.setPumpMode(mode);
    if (mode === 'internal' && this.engineInitialized) {
      this.scheduler.ensureRunning();
    }
  }

  public hasActiveQueuedRequests(): boolean {
    return this.scheduler.hasActiveRequests();
  }

  public getQueuedRequestSchedulerForExternalControl(): QueuedRequestScheduler {
    return this.scheduler;
  }

  public takeSettledQueuedRequestForExternalControl(
    requestId: GenerateRequestId
  ):
    | {
        response: GenerateResponse;
        callbackError: unknown;
      }
    | {
        error: unknown;
        callbackError: unknown;
      }
    | null {
    const tracked = this.tracker.get(requestId);
    if (tracked == null || !tracked.settled) {
      return null;
    }

    tracked.consumed = true;
    const callbackError = tracked.callbackError;
    const settlement =
      tracked.settlementState === 'resolved'
        ? tracked.settledResult == null
          ? {
              error: new Error(
                `Tracked queued request ${requestId} settled without a response.`
              ),
              callbackError,
            }
          : {
              response: tracked.settledResult,
              callbackError,
            }
        : {
            error:
              tracked.settledError ??
              new Error(`Tracked queued request ${requestId} rejected without an error.`),
            callbackError,
          };

    this.tracker.cleanupIfConsumed(requestId);
    return settlement;
  }

  public async pumpQueuedRequestsOnce(): Promise<QueuedRequestPumpStepResult> {
    return this.scheduler.pumpOnce();
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
    const bridge = this.getLoadedWasmBridge();
    if (!modelPath || modelPath.trim().length === 0) {
      throw new Error('modelPath must not be empty.');
    }
    if (this.engineInitialized) {
      this.rejectAllTrackedRequests(
        new Error('Engine runtime was reset during reinitialization.'),
        bridge
      );
      bridge.close();
      this.engineInitialized = false;
      this.resetRuntimeLifecycleState();
    }

    const normalizedConfig = normalizeInitConfig(config);
    this.runtimeObservabilityEnabled =
      normalizedConfig.enableRuntimeObservability > 0;
    this.backendProfilingEnabled = normalizedConfig.enableBackendProfiling > 0;
    this.transportObservability.enabled = this.runtimeObservabilityEnabled;
    const result = await bridge.initEngine(modelPath, normalizedConfig);
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
    const bridge = this.wasmBridge ?? new WasmBridge(module);
    this.rejectAllTrackedRequests(new Error('Engine runtime was closed.'), bridge);
    bridge.close();
    this.modelLoader.cleanupAfterClose(module);
    this.engineInitialized = false;
    this.resetRuntimeLifecycleState();
    this.lastModelLoadInfo = null;
    this.module = null;
    this.wasmBridge = null;
    this.initPromise = null;
  }

  public async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
    const bridge = this.getReadyEngineBridge();
    if (!Number.isInteger(requestId) || requestId <= 0) {
      return false;
    }

    const cancelled = await bridge.cancelQueuedRequest(requestId);
    if (!cancelled) {
      return false;
    }

    const tracked = this.tracker.get(requestId);
    if (tracked != null) {
      tracked.cancelRequested = true;
      if (this.scheduler.settleCompletedRequestIfPresent(bridge, requestId)) {
        return true;
      }
    }

    if (!this.tracker.hasActive(requestId)) {
      this.finalizeRequest(bridge, requestId, {
        consumeCompletedResponse: true,
        deleteCompletion: true,
      });
    }
    return true;
  }

  public async queuePrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<GenerateRequestId> {
    const bridge = this.getReadyEngineBridge();
    const request = this.buildGenerateRequest(contextKey, promptText, options);
    const onToken = typeof options === 'object' ? options.onToken : undefined;
    const signal = typeof options === 'object' ? options.signal : undefined;

    if (signal?.aborted) {
      throw createAbortError('Prompt was aborted before it was enqueued.');
    }

    const useNativeRuntimeEvents = this.shouldUseNativeRuntimeEvents(bridge, onToken);

    let requestId: GenerateRequestId = 0;
    const callbackPtr =
      useNativeRuntimeEvents || (onToken == null && signal == null)
        ? 0
        : bridge.registerTokenCallback((token: string) => {
            if (signal?.aborted) {
              return 1;
            }
            if (onToken != null && requestId !== 0) {
              this.scheduler.bufferTokenPiece(requestId, token);
              this.transportObservability.nativeCallbackTokenCount =
                (this.transportObservability.nativeCallbackTokenCount ?? 0) + 1;
            }
            return signal?.aborted ? 1 : 0;
          });

    try {
      requestId = bridge.enqueuePrompt(
        request.contextKey,
        request.promptText,
        request.maxOutputTokens,
        Number(callbackPtr)
      );
    } catch (error) {
      if (callbackPtr !== 0) {
        bridge.unregisterCallback(callbackPtr);
      }
      throw error;
    }
    if (!requestId) {
      if (callbackPtr !== 0) {
        bridge.unregisterCallback(callbackPtr);
      }
      throw new Error('Failed to enqueue request.');
    }

    if (callbackPtr !== 0) {
      this.transportObservability.tokenCallbackRegistrationCount =
        (this.transportObservability.tokenCallbackRegistrationCount ?? 0) + 1;
      this.queuedPromptCallbacks.set(requestId, onToken);
      this.queuedPromptTokenBuffers.set(requestId, []);
      this.queuedPromptCallbackPtrs.set(requestId, callbackPtr);
    } else if (onToken != null) {
      this.queuedPromptCallbacks.set(requestId, onToken);
      this.queuedPromptTokenBuffers.set(requestId, []);
    }

    if (signal != null) {
      this.tracker.attachSignal(requestId, signal, () => {
        void this.cancelQueuedRequest(requestId);
      });
    }

    this.ensureTracked(requestId);

    return requestId;
  }

  public async runQueuedRequest(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    this.getReadyEngineBridge();
    if (!Number.isInteger(requestId) || requestId <= 0) {
      throw new Error('requestId must be a positive integer.');
    }
    if (options?.signal?.aborted) {
      await this.cancelQueuedRequest(requestId);
      throw createAbortError('Prompt was aborted before execution started.');
    }

    const tracked = this.ensureTracked(requestId);
    const signal = options?.signal;
    const abortListener =
      signal == null
        ? null
        : () => {
            void this.cancelQueuedRequest(requestId);
          };
    if (abortListener != null) {
      signal?.addEventListener('abort', abortListener, { once: true });
    }

    tracked.consumed = true;
    tracked.waiterCount += 1;
    try {
      const response = await tracked.promise;
      const callbackError = tracked.callbackError;
      if (callbackError != null) {
        throw callbackError;
      }
      if (response.cancelled || signal?.aborted) {
        throw createAbortError(response.errorMessage ?? 'Queued request cancelled.');
      }
      return response;
    } finally {
      if (abortListener != null) {
        signal?.removeEventListener('abort', abortListener);
      }
      tracked.waiterCount = Math.max(0, tracked.waiterCount - 1);
      this.tracker.cleanupIfConsumed(requestId);
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

    return this.getReadyEngineBridge().readRuntimeObservability();
  }

  public getRuntimeObservability(): RuntimeAggregateObservabilityMetrics | null {
    return this.getRuntimeAggregateObservability();
  }

  public async getBackendObservability(): Promise<BackendObservability | null> {
    const raw = await this.getLoadedWasmBridge().getBackendObservabilityJson();
    if (raw == null) {
      return null;
    }

    try {
      const parsed = parseBackendObservabilityJson(raw);
      parsed.profilingEnabled = this.backendProfilingEnabled;
      return parsed;
    } catch (error) {
      throw new Error(`Failed to parse backend observability: ${asErrorMessage(error)}`);
    }
  }
}
