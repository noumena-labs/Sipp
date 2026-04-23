import { CogentConfig, EngineModuleOptions } from '../cogent-config.js';
import { normalizeInitConfig } from '../core/init-config.js';
import {
  buildChatTemplateUserMessage,
  normalizePromptText,
  resolveEffectivePromptFormat,
} from '../core/prompt-format.js';
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
  DEFAULT_PROMPT_FORMAT,
  MAX_PROMPT_TOKENS,
} from './main-thread-runtime-constants.js';
import { RequestTracker } from './request-tracker.js';
import {
  parseBackendObservabilityJson,
  WasmBridge,
} from '../wasm/wasm-bridge.js';
import { EngineModule } from '../wasm/engine-module.js';
import { createAbortError } from '../utils/abort.js';
import { asErrorMessage } from '../utils/error.js';
import { QueuedRequestScheduler } from './scheduler.js';
import { resolveRuntimeUrls } from '../runtime-assets.js';

export class MainThreadEngineRuntime implements EngineRuntime {
  private module: EngineModule | null = null;
  private wasmBridge: WasmBridge | null = null;
  private initPromise: Promise<void> | null = null;
  private engineInitialized = false;
  private cachedMediaMarker: string | null = null;
  private cachedChatTemplate: string | null = null;
  private readonly modelLoader: MainThreadModelLoader;
  private readonly executionMode: EngineExecutionMode;
  private queuedPromptCallbacks = new Map<
    GenerateRequestId,
    ((token: string) => void) | undefined
  >();
  private queuedPromptTokenBuffers = new Map<GenerateRequestId, string[]>();
  private queuedPromptCallbackErrors = new Map<GenerateRequestId, unknown>();
  private readonly tracker = new RequestTracker<GenerateResponse>();
  private readonly scheduler: QueuedRequestScheduler;
  private runtimeObservabilityEnabled = false;
  private backendProfilingEnabled = false;
  private transportObservability: TransportObservability;
  constructor(private config: CogentConfig = {}) {
    this.executionMode = config.executionMode === 'worker' ? 'worker' : 'main-thread';
    this.transportObservability = this.createTransportObservability();
    this.modelLoader = new MainThreadModelLoader(this.config);
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
      cancelQuery: (requestId) => this.cancelQuery(requestId),
    });
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

  private countMarkerOccurrences(promptText: string, marker: string): number {
    const escapedMarker = marker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return (promptText.match(new RegExp(escapedMarker, 'g')) ?? []).length;
  }

  private buildGenerateRequest(
    bridge: WasmBridge,
    contextKey: string,
    promptText: string,
    options: number | PromptOptions
  ): GenerateRequest {
    const media = this.resolvePromptMedia(options);
    const promptFormat = resolveEffectivePromptFormat(
      this.resolvePromptFormat(options),
      Boolean(media && media.length > 0)
    );
    const normalizedPromptText = normalizePromptText(promptText);
    let formattedPromptText = normalizedPromptText;
    if (promptFormat === 'auto-chat' && this.cachedChatTemplate != null) {
      const chatMessage =
        media != null && media.length > 0
          ? buildChatTemplateUserMessage(normalizedPromptText, this.cachedMediaMarker)
          : { role: 'user' as const, content: normalizedPromptText };
      formattedPromptText = bridge.applyChatTemplate(
        [chatMessage],
        true
      );
      if (formattedPromptText.length === 0) {
        throw new Error(
          'Failed to apply the model chat template for this prompt. Use promptFormat="raw" to bypass native template formatting.'
        );
      }
    }
    return {
      contextKey,
      promptText: formattedPromptText,
      maxOutputTokens: this.resolvePromptTokenCount(options),
      promptFormat,
      media,
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
    this.queuedPromptTokenBuffers.delete(requestId);
    this.queuedPromptCallbackErrors.delete(requestId);
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

  private shouldUseNativeRuntimeEvents(
    onToken: ((token: string) => void) | undefined
  ): void {
    if (onToken == null) {
      this.transportObservability.activeTokenTransport = 'none';
      return;
    }

    this.transportObservability.activeTokenTransport = 'runtime-events';
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
    this.runtimeObservabilityEnabled =
      normalizedConfig.enableRuntimeObservability > 0;
    this.backendProfilingEnabled = normalizedConfig.enableBackendProfiling > 0;
    this.transportObservability.enabled = this.runtimeObservabilityEnabled;
    const result = await bridge.loadRuntimeModel(modelPath, normalizedConfig);
    if (result !== 0) {
      this.engineInitialized = false;
      this.resetRuntimeLifecycleState(
        new Error(`Engine runtime failed to initialize. Code: ${result}`)
      );
      throw new Error(`Failed to initialize engine. Code: ${result}`);
    }
    this.engineInitialized = true;
    this.cachedMediaMarker = bridge.readMediaMarker();
    this.cachedChatTemplate = bridge.readNativeChatTemplate();

    this.modelLoader.cleanupAfterEngineInit(module);
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
    this.module = null;
    this.wasmBridge = null;
    this.initPromise = null;
  }

  public async cancelQuery(requestId: GenerateRequestId): Promise<boolean> {
    const bridge = this.getReadyEngineBridge();
    if (!Number.isInteger(requestId) || requestId <= 0) {
      return false;
    }

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
    const request = this.buildGenerateRequest(bridge, contextKey, promptText, options);
    const onToken = typeof options === 'object' ? options.onToken : undefined;
    const signal = typeof options === 'object' ? options.signal : undefined;

    if (signal?.aborted) {
      throw createAbortError('Prompt was aborted before it was enqueued.');
    }

    this.shouldUseNativeRuntimeEvents(onToken);

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
        0
      );
    } else {
      requestId = bridge.startTextRequest(
        request.contextKey,
        request.promptText,
        request.maxOutputTokens,
        0
      );
    }
    if (!requestId) {
      throw new Error('Failed to enqueue request.');
    }

    if (onToken != null) {
      this.queuedPromptCallbacks.set(requestId, onToken);
      this.queuedPromptTokenBuffers.set(requestId, []);
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
    this.getReadyEngineBridge();
    if (!Number.isInteger(requestId) || requestId <= 0) {
      throw new Error('requestId must be a positive integer.');
    }
    if (options?.signal?.aborted) {
      await this.cancelQuery(requestId);
      throw createAbortError('Prompt was aborted before execution started.');
    }

    const tracked = this.ensureTracked(requestId);
    const signal = options?.signal;
    const abortListener =
      signal == null
        ? null
        : () => {
            void this.cancelQuery(requestId);
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

  public getRuntimeObservability(): RuntimeAggregateObservabilityMetrics | null {
    if (!this.runtimeObservabilityEnabled) {
      return null;
    }

    return this.getReadyEngineBridge().readRuntimeObservability();
  }

  public readMediaMarker(): string | null {
    return this.cachedMediaMarker;
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
