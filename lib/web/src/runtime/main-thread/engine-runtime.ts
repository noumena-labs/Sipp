import type { SippClientOptions, EngineModuleOptions } from '../../engine/browser-client.js';
import type {
  BackendObservability,
  ChatMessage,
  EmbedRuntimeOptions,
  EngineExecutionMode,
  GenerateRequest,
  GenerateRequestId,
  GenerateResponse,
  NativeRuntimeConfig,
  PromptOptions,
  SamplingRuntimeOverride,
  RequestObservabilityMetrics,
  TokenBatch,
  TransportObservability,
} from '../../engine/inference-types.js';
import type { SharedTokenRingDescriptor } from '../shared-token-ring.js';
import type {
  ClassifiedAsset,
  InternalBundleDescriptor,
  ModelDetectionResult,
  PairingPlan,
  RegistryManifest,
  RuntimePairingErrorCode,
  StagedModelBundle,
  StageModelBundleOptions,
} from '../../models/types.js';
import type { ChatBoundaryInfo } from '../../engine/chat-boundary-sanitizer.js';
import type { EngineRuntime } from '../engine-runtime.js';
import { MainThreadModelLoader } from './model-loader.js';
import { RequestTracker } from '../request-tracker.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  RustLifecycleBridge,
  parseBackendObservabilityJson,
  WasmBridge,
} from '../../wasm/wasm-bridge.js';
import { EngineModule } from '../../wasm/engine-module.js';
import { createAbortError } from '../../utils/abort.js';
import { QueuedRequestScheduler } from '../scheduler.js';
import { hasSamplingRuntimeOverrideFields } from '../../engine/inference-types.js';
import {
  resolveRuntimeBackendOverride,
  resolveRuntimeThreadingMode,
  resolveRuntimeUrls,
  type RuntimeBackendOverride,
  type WasmThreadingMode,
} from '../../engine/runtime-assets.js';
import { RuntimePairingValidationError } from '../../models/types.js';

function normalizePromptText(value: string): string {
  return value.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

function normalizePairingErrorCode(code: string | undefined): RuntimePairingErrorCode {
  switch (code) {
    case 'INVALID_MODEL_SOURCE':
    case 'INVALID_MODEL_PAIRING':
    case 'MODEL_BROKEN':
      return code;
    default:
      return 'INVALID_MODEL_PAIRING';
  }
}

function resolveRuntimeSiblingUrl(moduleUrl: string, extension: string): string {
  const parsedModuleUrl = new URL(moduleUrl);
  const filename = parsedModuleUrl.pathname.split('/').pop() ?? 'sipp-wasm.js';
  const stem = filename.endsWith('.js') ? filename.slice(0, -'.js'.length) : filename;
  return new URL(`${stem}${extension}`, parsedModuleUrl).toString();
}

const EXPECTED_RUST_BROWSER_ENGINE_ABI_VERSION = 6;
const DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY: TransportObservability = {
  executionMode: 'main-thread',
  workerBacked: false,
  enabled: false,
  activeTokenTransport: 'none',
  activeTokenEmission: false,
  tokenDrainCalls: 0,
  tokenDrainMs: 0,
};

interface MainThreadEngineRuntimeOptions {
  readonly defaultBackendOverride?: RuntimeBackendOverride | null;
}

function asErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function verifyRustBrowserEngineAbi(bridge: WasmBridge): number {
  let abiVersion: number;
  try {
    abiVersion = bridge.rustBrowserEngineAbiVersion();
  } catch (error) {
    throw new Error('Sipp browser runtime ABI check failed.', { cause: error });
  }
  if (abiVersion !== EXPECTED_RUST_BROWSER_ENGINE_ABI_VERSION) {
    throw new Error(
      `Sipp browser runtime ABI mismatch: expected ${EXPECTED_RUST_BROWSER_ENGINE_ABI_VERSION}, got ${abiVersion}. Rebuild the WebAssembly runtime and clear cached browser runtime assets.`
    );
  }
  return abiVersion;
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
  private queuedPromptTokenBatchSinks = new Map<
    GenerateRequestId,
    (batch: TokenBatch) => void
  >();
  private queuedPromptTokenBatchSinkErrors = new Map<GenerateRequestId, unknown>();
  private readonly tracker = new RequestTracker<GenerateResponse>();
  private readonly scheduler: QueuedRequestScheduler;
  private runtimeObservabilityEnabled = false;
  private backendProfilingEnabled = false;
  private readonly defaultBackendOverride: RuntimeBackendOverride | null;
  private transportObservability: TransportObservability;
  private wasmBridgeOperationTail: Promise<void> = Promise.resolve();

  constructor(
    private readonly config: SippClientOptions = {},
    options: MainThreadEngineRuntimeOptions = {}
  ) {
    this.defaultBackendOverride = options.defaultBackendOverride ?? null;
    this.executionMode = config.executionMode === 'worker' ? 'worker' : 'main-thread';
    this.transportObservability = this.createTransportObservability();
    this.modelLoader = new MainThreadModelLoader(this.config);
    this.scheduler = new QueuedRequestScheduler({
      tracker: this.tracker,
      queuedPromptTokenBatchSinks: this.queuedPromptTokenBatchSinks,
      queuedPromptTokenBatchSinkErrors: this.queuedPromptTokenBatchSinkErrors,
      getTransportObservability: () => this.transportObservability,
      getBridge: () => this.getReadyEngineBridge(),
      finalizeRequest: (bridge, requestId, options) => {
        this.finalizeRequest(bridge, requestId, options);
      },
      cancelQuery: (requestId) => this.cancelQuery(requestId),
      withWasmBridge: (operation) => this.withReadyWasmBridge(operation),
    });
  }

  public getExecutionMode(): EngineExecutionMode {
    return this.executionMode;
  }

  public getWasmThreadingMode(): WasmThreadingMode {
    return resolveRuntimeThreadingMode(this.config);
  }

  public getDefaultBackendOverride(): RuntimeBackendOverride | null {
    return this.defaultBackendOverride ?? resolveRuntimeBackendOverride(this.config);
  }

  public getTransportObservability(): TransportObservability {
    return { ...this.transportObservability };
  }

  public getSharedTokenRingDescriptor(): SharedTokenRingDescriptor | null {
    return this.wasmBridge?.getSharedTokenRingDescriptor() ?? null;
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

  private resolvePromptStop(
    input: PromptOptions | number | undefined
  ): readonly string[] | undefined {
    if (typeof input === 'number' || input === undefined || input.stop == null) {
      return undefined;
    }
    return input.stop.length === 0 ? undefined : input.stop;
  }

  private resolvePromptSampling(
    input: PromptOptions | number | undefined
  ): SamplingRuntimeOverride | undefined {
    if (typeof input === 'number' || input === undefined || input.sampling == null) {
      return undefined;
    }
    const sampling = input.sampling;
    return hasSamplingRuntimeOverrideFields(sampling) ? sampling : undefined;
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
      stop: this.resolvePromptStop(options),
      sampling: this.resolvePromptSampling(options),
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

  private async withReadyWasmBridge<T>(
    operation: (bridge: WasmBridge) => T | Promise<T>
  ): Promise<T> {
    const previous = this.wasmBridgeOperationTail;
    let release!: () => void;
    this.wasmBridgeOperationTail = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await operation(this.getReadyEngineBridge());
    } finally {
      release();
    }
  }

  private releaseTokenState(requestId: GenerateRequestId): void {
    this.queuedPromptTokenBatchSinks.delete(requestId);
    this.queuedPromptTokenBatchSinkErrors.delete(requestId);
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

  private refreshTokenTransportObservability(): void {
    const activeTokenEmission = this.queuedPromptTokenBatchSinks.size > 0;
    this.transportObservability.activeTokenEmission = activeTokenEmission;
    this.transportObservability.activeTokenTransport =
      activeTokenEmission ? 'token-stream' : 'none';
  }

  private rejectAllTrackedRequests(error: unknown): void {
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
        const { moduleUrl, wasmUrl, threading } = resolveRuntimeUrls(this.config);
        const createModule = await this.importModuleFactory(moduleUrl);
        const moduleConfig: EngineModuleOptions = { ...(this.config.moduleOptions ?? {}) };
        const userLocateFile = moduleConfig.locateFile;
        moduleConfig.printErr ??= (message: string) => {
          if (typeof message === 'string' && message.startsWith('[sipp/')) {
            console.log(message);
          }
        };
        if (threading === 'pthread') {
          moduleConfig.mainScriptUrlOrBlob ??= moduleUrl;
        }

        moduleConfig.locateFile = (path: string, prefix?: string) => {
          if (path.endsWith('.wasm')) {
            return wasmUrl;
          }
          if (path.endsWith('.worker.js')) {
            return resolveRuntimeSiblingUrl(moduleUrl, '.worker.js');
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
        const bridge = new WasmBridge(module);
        try {
          verifyRustBrowserEngineAbi(bridge);
        } catch (error) {
          try {
            bridge.close();
          } catch {
            /* preserve the ABI failure as the primary error */
          }
          throw error;
        }
        this.module = module;
        this.wasmBridge = bridge;
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

  public async detectModelFromGgufFile(
    file: Blob & { name?: string },
    signal?: AbortSignal
  ): Promise<ModelDetectionResult> {
    await this.ensureModule();
    return this.getLoadedWasmBridge().detectModelFromGgufFile(file, signal);
  }

  public async browserCacheLayout(
    sourceBytes: number,
    sourceBytesKnown: boolean,
    directLoadMaxBytes: number,
    shardMaxBytes: number
  ) {
    await this.ensureModule();
    return this.getLoadedWasmBridge().browserCacheLayout(
      sourceBytes,
      sourceBytesKnown,
      directLoadMaxBytes,
      shardMaxBytes
    );
  }

  public async planGgufSplitCount(
    sourceBytes: number,
    shardMaxBytes: number,
    callbacks: Parameters<WasmBridge['planGgufSplitCount']>[2]
  ): Promise<number> {
    await this.ensureModule();
    return this.getLoadedWasmBridge().planGgufSplitCount(
      sourceBytes,
      shardMaxBytes,
      callbacks
    );
  }

  public async splitGgufStream(
    sourceBytes: number,
    outputPrefix: string,
    shardMaxBytes: number,
    callbacks: Parameters<WasmBridge['splitGgufStream']>[3]
  ): Promise<void> {
    await this.ensureModule();
    this.getLoadedWasmBridge().splitGgufStream(
      sourceBytes,
      outputPrefix,
      shardMaxBytes,
      callbacks
    );
  }

  public async resolvePairing(
    classified: readonly ClassifiedAsset[],
    explicitProjectorId?: string | null
  ): Promise<PairingPlan> {
    await this.ensureModule();
    const response = this.getLoadedWasmBridge().validatePairing(
      classified,
      explicitProjectorId
    );
    if (response.ok && response.plan != null) {
      return response.plan;
    }
    const code = normalizePairingErrorCode(response.error?.code);
    const message = response.error?.message ?? 'Model pairing validation failed.';
    throw new RuntimePairingValidationError(code, message);
  }

  public async createRustLifecycleBridge(
    manifest: RegistryManifest
  ): Promise<RustLifecycleBridge> {
    await this.ensureModule();
    return RustLifecycleBridge.create(this.getLoadedWasmBridge(), manifest);
  }

  /**
   * Initialize engine state with a model path mounted in the wasm filesystem.
   */
  public async loadRuntimeModel(
    modelPathOrBundle: string | StagedModelBundle,
    config?: NativeRuntimeConfig
  ): Promise<void> {
    const module = await this.ensureModule();
    const bridge = this.getLoadedWasmBridge();
    const modelPath =
      typeof modelPathOrBundle === 'string' ? modelPathOrBundle : modelPathOrBundle.modelPath;
    if (!modelPath || modelPath.trim().length === 0) {
      throw new Error('modelPath must not be empty.');
    }
    if (this.engineInitialized) {
      this.rejectAllTrackedRequests(new Error('Engine runtime was reset during reinitialization.'));
      bridge.close();
      this.engineInitialized = false;
      this.resetRuntimeLifecycleState();
    }

    const effectiveConfig =
      typeof modelPathOrBundle === 'string' ||
        this.hasExplicitProjectorPath(config) ||
        modelPathOrBundle.projectorPath == null
        ? config
        : {
          ...config,
          multimodal: {
            ...config?.multimodal,
            projector_path: modelPathOrBundle.projectorPath,
          },
        };
    const expectsMediaSupport = effectiveConfig?.multimodal?.projector_path != null;
    this.runtimeObservabilityEnabled =
      effectiveConfig?.observability?.runtime_metrics === true ||
      effectiveConfig?.observability?.backend_profiling === true;
    this.backendProfilingEnabled = effectiveConfig?.observability?.backend_profiling === true;
    this.transportObservability.enabled = this.runtimeObservabilityEnabled;
    try {
      const result = await bridge.loadRuntimeModel(modelPath, effectiveConfig);
      if (result !== 0) {
        const detail = bridge.readLastEngineError();
        throw new Error(
          detail.length > 0
            ? `Failed to initialize engine. Code: ${result}. ${detail}`
            : `Failed to initialize engine. Code: ${result}`
        );
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

  private hasExplicitProjectorPath(config: NativeRuntimeConfig | undefined): boolean {
    return (
      typeof config?.multimodal?.projector_path === 'string' &&
      config.multimodal.projector_path.trim().length > 0
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
    this.rejectAllTrackedRequests(new Error('Engine runtime was closed.'));
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

    const cancelled = await this.withReadyWasmBridge((bridge) =>
      bridge.cancelQuery(requestId)
    );
    if (!cancelled) {
      return false;
    }

    if (this.tracker.has(requestId)) {
      this.tracker.requestCancel(requestId);
      const settled = await this.withReadyWasmBridge((bridge) =>
        this.scheduler.settleCompletedRequestIfPresent(bridge, requestId)
      );
      if (settled) {
        return true;
      }
    }

    if (!this.tracker.hasActive(requestId)) {
      await this.withReadyWasmBridge((bridge) => {
        this.finalizeRequest(bridge, requestId, {
          consumeCompletedResponse: true,
          deleteCompletion: true,
        });
      });
    }
    return true;
  }

  public async enqueueQuery(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<GenerateRequestId> {
    const request = this.buildGenerateRequest(contextKey, promptText, options);
    return this.enqueueNativeRequest(options, (bridge, emitTokens) => {
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
        return bridge.startMediaRequest(
          request.contextKey,
          request.promptText,
          request.maxOutputTokens,
          request.media,
          {
            grammar: request.grammar,
            stop: request.stop,
            sampling: request.sampling,
            emitTokens,
          }
        );
      }
      return bridge.startTextRequest(
        request.contextKey,
        request.promptText,
        request.maxOutputTokens,
        {
          grammar: request.grammar,
          stop: request.stop,
          sampling: request.sampling,
          emitTokens,
        }
      );
    });
  }

  public async enqueueChat(
    contextKey: string,
    messages: readonly ChatMessage[],
    options: number | PromptOptions = 128
  ): Promise<GenerateRequestId> {
    const media = this.resolvePromptMedia(options);
    if (media != null && media.length > 0 && this.cachedMediaMarker == null) {
      throw new Error(
        'Loaded runtime does not expose a media marker for the current model.'
      );
    }
    const maxOutputTokens = this.resolvePromptTokenCount(options);
    const grammar = this.resolvePromptGrammar(options);
    const stop = this.resolvePromptStop(options);
    const sampling = this.resolvePromptSampling(options);

    return this.enqueueNativeRequest(options, (bridge, emitTokens) =>
      bridge.startChatRequest(
        contextKey,
        messages,
        maxOutputTokens,
        media,
        {
          grammar,
          stop,
          sampling,
          emitTokens,
        }
      )
    );
  }

  public async enqueueEmbedding(
    contextKey: string,
    input: string,
    options: EmbedRuntimeOptions = {}
  ): Promise<GenerateRequestId> {
    const promptText = normalizePromptText(input);
    return this.enqueueNativeRequest(options, (bridge) =>
      bridge.startEmbeddingRequest(
        contextKey,
        promptText,
        options.normalize ?? true
      )
    );
  }

  private async enqueueNativeRequest(
    options: number | PromptOptions,
    startRequest: (bridge: WasmBridge, emitTokens: boolean) => GenerateRequestId
  ): Promise<GenerateRequestId> {
    const tokenBatchSink = typeof options === 'object' ? options.tokenBatchSink : undefined;
    const emitTokens =
      typeof options === 'object' &&
      (options.emitTokens === true || tokenBatchSink != null);
    const signal = typeof options === 'object' ? options.signal : undefined;

    if (signal?.aborted) {
      throw createAbortError('Prompt was aborted before it was enqueued.');
    }

    const { requestId, errorDetail } = await this.withReadyWasmBridge((bridge) => {
      const requestId = startRequest(bridge, emitTokens);
      return {
        requestId,
        errorDetail: requestId ? '' : bridge.readLastEngineError(),
      };
    });
    if (!requestId) {
      throw new Error(
        errorDetail.length > 0
          ? `Failed to enqueue request. ${errorDetail}`
          : 'Failed to enqueue request.'
      );
    }

    if (typeof options === 'object' && typeof options.onRequestStarted === 'function') {
      try {
        options.onRequestStarted(requestId);
      } catch {
        /* request-start observers must not abort enqueue */
      }
    }

    if (tokenBatchSink != null) {
      this.queuedPromptTokenBatchSinks.set(requestId, tokenBatchSink);
      this.refreshTokenTransportObservability();
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

    this.ensureTracked(requestId);
    const signal = options?.signal;
    const detachAbort =
      signal == null
        ? () => {}
        : this.tracker.attachSignal(requestId, signal, () => {
            void this.cancelQuery(requestId);
          });

    const responsePromise = this.tracker.beginWait(requestId);
    try {
      const response = await responsePromise;
      const tokenBatchSinkError = this.tracker.tokenBatchSinkError(requestId);
      if (tokenBatchSinkError != null) {
        throw tokenBatchSinkError;
      }
      if (response.cancelled || signal?.aborted) {
        throw createAbortError(response.errorMessage ?? 'Queued request cancelled.');
      }
      return response;
    } finally {
      detachAbort();
      this.tracker.endWait(requestId);
    }
  }

  public getRuntimeObservability(): RequestObservabilityMetrics | null {
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

  public async probeChatTemplateBoundaryInfo(): Promise<ChatBoundaryInfo> {
    return this.getReadyEngineBridge().probeChatTemplateBoundaryInfo();
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
