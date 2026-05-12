import { CogentConfig, EngineModuleOptions } from '../cogent-config.js';
import { normalizeInitConfig } from '../core/init-config.js';
import {
  BackendObservability,
  EngineExecutionMode,
  GenerateRequest,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  InternalBundleDescriptor,
  PromptOptions,
  StagedModelBundle,
  StageModelBundleOptions,
  RuntimeAggregateObservabilityMetrics,
  TransportObservability,
} from '../types.js';
import { EngineRuntime } from './engine-runtime.js';
import { MainThreadModelLoader } from './main-thread-model-loader.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
} from './main-thread-runtime-constants.js';
import { RequestTracker } from './request-tracker.js';
import {
  TOKEN_EMISSION_NONE,
  TOKEN_EMISSION_STREAMING_BUFFER,
  type ChatTemplateMessage,
  parseBackendObservabilityJson,
  WasmBridge,
} from '../wasm/wasm-bridge.js';
import type { StreamingRingWriter } from './streaming-ring.js';
import { EngineModule } from '../wasm/engine-module.js';
import { createAbortError } from '../utils/abort.js';
import { asErrorMessage } from '../utils/error.js';
import { QueuedRequestScheduler } from './scheduler.js';
import { resolveRuntimeUrls } from '../runtime-assets.js';

function normalizePromptText(value: string): string {
  return value.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

export class MainThreadEngineRuntime implements EngineRuntime {
  private module: EngineModule | null = null;
  private wasmBridge: WasmBridge | null = null;
  private initPromise: Promise<void> | null = null;
  private moduleGeneration = 0;
  private engineInitialized = false;
  private cachedMediaMarker: string | null = null;
  private cachedChatTemplate: string | null = null;
  private cachedBosText: string = '';
  private cachedEosText: string = '';

  private readonly modelLoader: MainThreadModelLoader;
  private readonly executionMode: EngineExecutionMode;
  private queuedPromptCallbacks = new Map<
    GenerateRequestId,
    ((token: string) => void) | undefined
  >();
  private queuedPromptCallbackErrors = new Map<GenerateRequestId, unknown>();
  private readonly tracker = new RequestTracker<GenerateResponse>();
  private readonly scheduler: QueuedRequestScheduler;
  private runtimeObservabilityEnabled = false;
  private backendProfilingEnabled = false;
  private transportObservability: TransportObservability;
  // Worker-side SAB ring writer.  When set, requests with onToken run in
  // StreamingBuffer emission mode; otherwise streaming is rejected with an
  // explicit error (cross-origin isolation required).
  private streamingRingWriter: StreamingRingWriter | null = null;
  constructor(private config: CogentConfig = {}) {
    this.executionMode = config.executionMode === 'worker' ? 'worker' : 'main-thread';
    this.transportObservability = this.createTransportObservability();
    this.modelLoader = new MainThreadModelLoader(this.config);
    this.scheduler = new QueuedRequestScheduler({
      tracker: this.tracker,
      queuedPromptCallbacks: this.queuedPromptCallbacks,
      queuedPromptCallbackErrors: this.queuedPromptCallbackErrors,
      getTransportObservability: () => this.transportObservability,
      getBridge: () => this.getReadyEngineBridge(),
      finalizeRequest: (bridge, requestId, options) => {
        this.finalizeRequest(bridge, requestId, options);
      },
      cancelQuery: (requestId) => this.cancelQuery(requestId),
      getStreamingRingWriter: () => this.streamingRingWriter,
    });
  }

  // Wires the worker-side SAB streaming ring writer.  Called once by the
  // worker entry after the main thread allocates the ring SAB.
  public setStreamingRingWriter(writer: StreamingRingWriter | null): void {
    this.streamingRingWriter = writer;
  }



  public getExecutionMode(): EngineExecutionMode {
    return this.executionMode;
  }

  public getTransportObservability(): TransportObservability {
    return { ...this.transportObservability };
  }

  private normalizeTokenCount(nTokens: number): number {
    if (!Number.isInteger(nTokens)) {
      throw new Error('nTokens must be an integer.');
    }
    if (nTokens <= 0) {
      throw new Error('nTokens must be a positive integer.');
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

  private resolvePromptMedia(
    input: PromptOptions | number | undefined
  ): Uint8Array[] | undefined {
    if (typeof input === 'number' || input === undefined || input.media == null) {
      return undefined;
    }
    if (!Array.isArray(input.media)) {
      throw new Error('media must be an array of Uint8Array instances.');
    }
    if (input.media.length === 0) {
      return undefined;
    }
    if (input.media.some((image) => !(image instanceof Uint8Array))) {
      throw new Error('media entries must be Uint8Array instances.');
    }
    return input.media;
  }

  private resolvePromptGrammar(
    input: PromptOptions | number | undefined
  ): string | undefined {
    if (typeof input === 'number' || input === undefined) {
      return undefined;
    }
    if (input.grammar == null) {
      return undefined;
    }
    if (typeof input.grammar !== 'string') {
      throw new Error('grammar must be a string when provided.');
    }
    if (input.grammar.length === 0) {
      return undefined;
    }
    return input.grammar;
  }

  private countMarkerOccurrences(promptText: string, marker: string): number {
    const escapedMarker = marker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return (promptText.match(new RegExp(escapedMarker, 'g')) ?? []).length;
  }

  private buildGenerateRequest(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions
  ): GenerateRequest {
    const media = this.resolvePromptMedia(options);
    const normalizedPromptText = normalizePromptText(promptText);
    const request: GenerateRequest = {
      contextKey,
      promptText: normalizedPromptText,
      maxOutputTokens: this.resolvePromptTokenCount(options),
      media,
      grammar: this.resolvePromptGrammar(options),
    };
    return request;
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
      throw new Error('Engine is not initialized. Call loadRuntimeModel(modelPath, config?) first.');
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
    this.queuedPromptCallbackErrors.delete(requestId);
    this.refreshTokenTransportObservability();
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

  private updateTokenTransportObservability(
    onToken: ((token: string) => void) | undefined
  ): void {
    if (onToken == null) {
      this.refreshTokenTransportObservability();
      return;
    }

    this.transportObservability.activeTokenTransport =
      this.streamingRingWriter != null ? 'streaming-buffer' : 'none';
  }

  private refreshTokenTransportObservability(): void {
    this.transportObservability.activeTokenTransport = 'none';
  }

  private rejectAllTrackedRequests(error: unknown, _bridge: WasmBridge | null = null): void {
    this.scheduler.reset();
    for (const requestId of this.tracker.allTrackedIds()) {
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
    this.cachedMediaMarker = null;
    this.cachedChatTemplate = null;
    this.cachedBosText = '';
    this.cachedEosText = '';
    this.transportObservability = this.createTransportObservability();
  }

  private createTransportObservability(): TransportObservability {
    return {
      ...DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
      executionMode: this.executionMode,
      workerBacked: this.executionMode === 'worker',
    };
  }

  private ensureTracked(requestId: GenerateRequestId) {
    return this.scheduler.track(requestId);
  }

  private async importModuleFactory(moduleUrl: string): Promise<(options: EngineModuleOptions) => Promise<EngineModule>> {
    // Dynamic import of an Emscripten glue module resolved at runtime from a URL.
    // Stack ignore comments so every major bundler skips static analysis:
    //   - @vite-ignore       -> Vite / Rollup
    //   - webpackIgnore      -> webpack (>=2)
    //   - turbopackIgnore    -> Turbopack (Next.js)
    // esbuild, Bun, and native ESM ignore unknown comments and pass through.
    const importedModule = await import(
      /* @vite-ignore */
      /* webpackIgnore: true */
      /* turbopackIgnore: true */
      moduleUrl
    );
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
      const generation = this.moduleGeneration;
      this.initPromise = (async () => {
        const { moduleUrl, wasmUrl } = resolveRuntimeUrls(this.config);
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

        const module = await createModule(moduleConfig);
        if (generation !== this.moduleGeneration) {
          new WasmBridge(module).close();
          throw createAbortError('Module initialization was cancelled.');
        }
        this.module = module;
        this.wasmBridge = new WasmBridge(module);
      })().catch((error) => {
        if (generation === this.moduleGeneration) {
          this.initPromise = null;
          this.module = null;
          this.wasmBridge = null;
        }
        throw error;
      });
    }
    await this.initPromise;
  }

  public async stageModelBundle(
    descriptor: InternalBundleDescriptor,
    options?: StageModelBundleOptions
  ): Promise<StagedModelBundle> {
    const module = await this.ensureModule();
    return this.modelLoader.stageModelBundle(module, descriptor, options);
  }

  /**
   * Initialize engine state with a model path in MEMFS.
   */
  public async loadRuntimeModel(
    modelPathOrBundle: string | StagedModelBundle,
    config?: InferenceInitConfig
  ): Promise<void> {
    const module = await this.ensureModule();
    const bridge = this.getLoadedWasmBridge();
    const modelPath =
      typeof modelPathOrBundle === 'string' ? modelPathOrBundle : modelPathOrBundle.modelPath;
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

    const effectiveConfig =
      typeof modelPathOrBundle === 'string' ||
        this.hasExplicitProjectorPath(config) ||
        modelPathOrBundle.multimodalProjectorPath == null
        ? config
        : {
          ...config,
          multimodalProjectorPath: modelPathOrBundle.multimodalProjectorPath,
        };
    const normalizedConfig = normalizeInitConfig(effectiveConfig);
    const expectsMediaSupport = normalizedConfig.multimodalProjectorPath != null;
    this.runtimeObservabilityEnabled =
      normalizedConfig.enableRuntimeObservability > 0;
    this.backendProfilingEnabled = normalizedConfig.enableBackendProfiling > 0;
    this.transportObservability.enabled = this.runtimeObservabilityEnabled;
    try {
      const result = await bridge.loadRuntimeModel(modelPath, normalizedConfig);
      if (result !== 0) {
        throw new Error(`Failed to initialize engine. Code: ${result}`);
      }
      this.engineInitialized = true;
      this.cachedMediaMarker = bridge.readMediaMarker();
      this.cachedChatTemplate = bridge.readNativeChatTemplate();
      this.cachedBosText = bridge.getBosText();
      this.cachedEosText = bridge.getEosText();
      if (expectsMediaSupport && this.cachedMediaMarker == null) {
        const error = new Error(
          'Failed to initialize multimodal runtime: loaded projector did not expose a media marker.'
        );
        bridge.close();
        throw error;
      }
    } catch (error) {
      this.engineInitialized = false;
      this.resetRuntimeLifecycleState(error);
      throw error;
    } finally {
      this.modelLoader.cleanup(module);
    }
  }

  private hasExplicitProjectorPath(config: InferenceInitConfig | undefined): boolean {
    return (
      typeof config?.multimodalProjectorPath === 'string' &&
      config.multimodalProjectorPath.trim().length > 0
    );
  }

  /**
   * Shutdown engine instance.
   */
  public close(): void {
    this.moduleGeneration += 1;
    const module = this.module;
    this.initPromise = null;
    if (!module) {
      this.engineInitialized = false;
      this.resetRuntimeLifecycleState(new Error('Engine runtime was closed.'));
      this.module = null;
      this.wasmBridge = null;
      return;
    }
    const bridge = this.wasmBridge ?? new WasmBridge(module);
    this.rejectAllTrackedRequests(new Error('Engine runtime was closed.'), bridge);
    bridge.close();
    this.modelLoader.cleanup(module);
    this.engineInitialized = false;
    this.resetRuntimeLifecycleState();
    this.module = null;
    this.wasmBridge = null;
    this.initPromise = null;
  }

  public async cancelQuery(requestId: GenerateRequestId): Promise<boolean> {
    if (!Number.isInteger(requestId) || requestId <= 0) {
      return false;
    }
    const bridge = this.getReadyEngineBridge();

    const cancelled = await bridge.cancelQuery(requestId);
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

  public async enqueueQuery(
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

    this.updateTokenTransportObservability(onToken);

    // Streaming requires a worker-side SAB ring writer.  In worker mode the
    // worker entry installs one before enqueueing; on the main thread there
    // is no streaming consumer and onToken with no ring is a usage error.
    if (onToken != null && this.streamingRingWriter == null) {
      throw new Error(
        'onToken streaming requires worker execution mode with cross-origin isolation (SAB).'
      );
    }
    const emissionMode =
      onToken == null ? TOKEN_EMISSION_NONE : TOKEN_EMISSION_STREAMING_BUFFER;

    let requestId: GenerateRequestId = 0;
    if (request.media != null && request.media.length > 0) {
      if (this.cachedMediaMarker == null) {
        throw new Error(
          'Loaded runtime does not expose a media marker for the current model.'
        );
      }
      const markerCount = this.countMarkerOccurrences(
        request.promptText,
        this.cachedMediaMarker
      );
      if (markerCount !== request.media.length) {
        throw new Error(
          `Prompt contains ${markerCount} media marker(s) but ${request.media.length} image(s) were provided. Use "${this.cachedMediaMarker}" in your prompt to place each image.`
        );
      }
      requestId = bridge.startMediaRequest(
        request.contextKey,
        request.promptText,
        request.maxOutputTokens,
        request.media,
        request.grammar,
        emissionMode
      );
    } else {
      requestId = bridge.startTextRequest(
        request.contextKey,
        request.promptText,
        request.maxOutputTokens,
        request.grammar,
        emissionMode
      );
    }
    if (!requestId) {
      throw new Error('Failed to enqueue request.');
    }

    // Worker entry uses this hook to publish a streaming-claim message
    // before inference produces tokens.  Errors are swallowed.
    if (
      typeof options === 'object' &&
      typeof options.__internalRequestStarted === 'function'
    ) {
      try {
        options.__internalRequestStarted(requestId);
      } catch {
        /* internal hook errors must not abort enqueue */
      }
    }

    if (onToken != null) {
      this.queuedPromptCallbacks.set(requestId, onToken);
    }

    if (signal != null) {
      this.tracker.attachSignal(requestId, signal, () => {
        void this.cancelQuery(requestId);
      });
    }

    this.ensureTracked(requestId);

    return requestId;
  }

  public async awaitQuery(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    if (!Number.isInteger(requestId) || requestId <= 0) {
      throw new Error('requestId must be a positive integer.');
    }
    this.getReadyEngineBridge();
    if (options?.signal?.aborted) {
      await this.cancelQuery(requestId);
      throw createAbortError('Prompt was aborted before execution started.');
    }

    const tracked = this.ensureTracked(requestId);
    const signal = options?.signal;
    const detachAbort =
      signal == null
        ? () => {}
        : this.tracker.attachSignal(requestId, signal, () => {
            void this.cancelQuery(requestId);
          });

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
      detachAbort();
      tracked.waiterCount = Math.max(0, tracked.waiterCount - 1);
      this.tracker.cleanupIfConsumed(requestId);
    }
  }

  public getRuntimeObservability(): RuntimeAggregateObservabilityMetrics | null {
    if (!this.runtimeObservabilityEnabled) {
      return null;
    }

    return this.getReadyEngineBridge().readRuntimeObservability();
  }

  public readMediaMarker(): string | null {
    return this.cachedMediaMarker;
  }

  public getChatTemplate(): string | null {
    return this.cachedChatTemplate;
  }

  public getBosText(): string {
    return this.cachedBosText;
  }

  public getEosText(): string {
    return this.cachedEosText;
  }

  public async applyChatTemplate(
    messages: ChatTemplateMessage[],
    addAssistant: boolean
  ): Promise<string> {
    return this.getReadyEngineBridge().applyChatTemplate(messages, addAssistant);
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
