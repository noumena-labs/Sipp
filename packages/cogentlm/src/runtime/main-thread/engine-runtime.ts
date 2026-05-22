import { CogentConfig, EngineModuleOptions } from '../../engine/engine-options.js';
import {
  BackendObservability,
  ChatMessage,
  EngineExecutionMode,
  GenerateRequest,
  GenerateRequestId,
  GenerateResponse,
  InternalBundleDescriptor,
  ModelDetectionResult,
  NativeRuntimeConfig,
  PromptOptions,
  StagedModelBundle,
  StageModelBundleOptions,
  TokenBatch,
  RuntimeAggregateObservabilityMetrics,
  TransportObservability,
} from '../../types.js';
import type { ChatBoundaryInfo } from '../../core/chat-boundary-sanitizer.js';
import {
  RuntimePairingValidationError,
  type EngineRuntime,
  type RuntimePairingErrorCode,
} from '../engine-runtime.js';
import { MainThreadModelLoader } from './model-loader.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
} from './constants.js';
import { RequestTracker } from '../request-tracker.js';
import {
  TOKEN_EMISSION_NONE,
  TOKEN_EMISSION_STREAMING_BUFFER,
  parseBackendObservabilityJson,
  type TokenEmissionMode,
  WasmBridge,
} from '../../wasm/wasm-bridge.js';
import type { StreamingRingWriter } from '../streaming-ring.js';
import { EngineModule } from '../../wasm/engine-module.js';
import { createAbortError } from '../../utils/abort.js';
import { asErrorMessage } from '../../utils/error.js';
import { QueuedRequestScheduler } from '../scheduler.js';
import { resolveRuntimeUrls } from '../../engine/runtime-assets.js';
import type { BrowserRuntimeSmokeResult } from '../browser-smoke-types.js';
import type { ClassifiedAsset, PairingPlan } from '../../models/pairing-types.js';

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
  const filename = parsedModuleUrl.pathname.split('/').pop() ?? 'cogentlm-wasm.js';
  const stem = filename.endsWith('.js') ? filename.slice(0, -'.js'.length) : filename;
  return new URL(`${stem}${extension}`, parsedModuleUrl).toString();
}

const BROWSER_SMOKE_DIRECT_LOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024;
const BROWSER_SMOKE_SHARD_MAX_BYTES = 128;
const GGUF_MAGIC = 0x4655_4747;
const GGUF_VALUE_STRING = 8;
const GGUF_ALIGNMENT = 32;

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function appendU32(bytes: number[], value: number): void {
  bytes.push(value & 0xff, (value >>> 8) & 0xff, (value >>> 16) & 0xff, (value >>> 24) & 0xff);
}

function appendU64(bytes: number[], value: number): void {
  let remaining = BigInt(value);
  for (let index = 0; index < 8; index += 1) {
    bytes.push(Number(remaining & 0xffn));
    remaining >>= 8n;
  }
}

function appendString(bytes: number[], value: string): void {
  const encoded = new TextEncoder().encode(value);
  appendU64(bytes, encoded.byteLength);
  bytes.push(...encoded);
}

function buildBrowserSmokeGguf(): Uint8Array {
  const tensors = [
    { name: 'blk.0.weight', fill: 1 },
    { name: 'blk.1.weight', fill: 2 },
    { name: 'output.weight', fill: 3 },
  ];
  const metadata: number[] = [];
  const tensorData: number[] = [];
  const tensorOffsets: number[] = [];

  appendU32(metadata, GGUF_MAGIC);
  appendU32(metadata, 3);
  appendU64(metadata, tensors.length);
  appendU64(metadata, 1);
  appendString(metadata, 'general.architecture');
  appendU32(metadata, GGUF_VALUE_STRING);
  appendString(metadata, 'llama');

  for (const tensor of tensors) {
    const nextOffset = alignTo(tensorData.length, GGUF_ALIGNMENT);
    while (tensorData.length < nextOffset) {
      tensorData.push(0);
    }
    tensorOffsets.push(nextOffset);
    tensorData.push(...new Array<number>(64).fill(tensor.fill));
  }

  for (let index = 0; index < tensors.length; index += 1) {
    appendString(metadata, tensors[index].name);
    appendU32(metadata, 1);
    appendU64(metadata, 16);
    appendU32(metadata, 0);
    appendU64(metadata, tensorOffsets[index]);
  }

  const dataOffset = alignTo(metadata.length, GGUF_ALIGNMENT);
  while (metadata.length < dataOffset) {
    metadata.push(0);
  }
  metadata.push(...tensorData);
  return new Uint8Array(metadata);
}

function runBrowserGgufIngestSmoke(bridge: WasmBridge): BrowserRuntimeSmokeResult['ggufIngest'] {
  try {
    const layoutForLargeFile = bridge.browserCacheLayout(
      BROWSER_SMOKE_DIRECT_LOAD_MAX_BYTES + 1,
      true,
      BROWSER_SMOKE_DIRECT_LOAD_MAX_BYTES,
      512 * 1024 * 1024
    );
    const source = buildBrowserSmokeGguf();
    const readAt = (offset: number, target: Uint8Array): number => {
      const start = Math.trunc(offset);
      const end = start + target.byteLength;
      if (start < 0 || end > source.byteLength) {
        return -1;
      }
      target.set(source.subarray(start, end));
      return 0;
    };
    const plannedShardCount = bridge.planGgufSplitCount(
      source.byteLength,
      BROWSER_SMOKE_SHARD_MAX_BYTES,
      { readAt }
    );
    let activeShard = false;
    let streamedShardCount = 0;
    let streamedBytes = 0;

    bridge.splitGgufStream(
      source.byteLength,
      'browser-smoke-model',
      BROWSER_SMOKE_SHARD_MAX_BYTES,
      {
        readAt,
        openShard: (_path, _index, count) => {
          if (count !== plannedShardCount || activeShard) {
            return -1;
          }
          activeShard = true;
          return 0;
        },
        writeShard: (bytes) => {
          if (!activeShard) {
            return -1;
          }
          streamedBytes += bytes.byteLength;
          return 0;
        },
        closeShard: () => {
          if (!activeShard) {
            return -1;
          }
          activeShard = false;
          streamedShardCount += 1;
          return 0;
        },
      }
    );

    if (layoutForLargeFile !== 'split-gguf') {
      throw new Error(`unexpected browser cache layout: ${layoutForLargeFile}`);
    }
    if (plannedShardCount !== 2 || streamedShardCount !== plannedShardCount || streamedBytes <= 0) {
      throw new Error(
        `unexpected GGUF ingest result: planned=${plannedShardCount} streamed=${streamedShardCount} bytes=${streamedBytes}`
      );
    }

    return {
      available: true,
      layoutForLargeFile,
      plannedShardCount,
      streamedShardCount,
      streamedBytes,
      error: null,
    };
  } catch (error) {
    return {
      available: false,
      layoutForLargeFile: null,
      plannedShardCount: null,
      streamedShardCount: 0,
      streamedBytes: 0,
      error: asErrorMessage(error),
    };
  }
}

function runBrowserRustEngineSmoke(bridge: WasmBridge): BrowserRuntimeSmokeResult['rustEngine'] {
  let engine = 0;
  try {
    const abiVersion = bridge.rustBrowserEngineAbiVersion();
    if (abiVersion <= 0) {
      throw new Error(`Rust browser engine ABI is unavailable: version ${abiVersion}.`);
    }
    engine = bridge.rustBrowserEngineCreate();
    if (engine === 0) {
      throw new Error('Rust browser engine create returned a null handle.');
    }
    const engineId = bridge.rustBrowserEngineId(engine);
    if (engineId <= 0) {
      throw new Error(`Rust browser engine returned invalid id ${engineId}.`);
    }
    const closeStatus = bridge.rustBrowserEngineClose(engine);
    engine = 0;
    if (closeStatus !== 0) {
      throw new Error(`Rust browser engine close failed with status ${closeStatus}.`);
    }
    return {
      available: true,
      abiVersion,
      engineId,
      error: null,
    };
  } catch (error) {
    if (engine !== 0) {
      try {
        bridge.rustBrowserEngineClose(engine);
      } catch {
        /* ignore cleanup failure in smoke error path */
      }
    }
    return {
      available: false,
      abiVersion: 0,
      engineId: null,
      error: asErrorMessage(error),
    };
  }
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
    ((batch: TokenBatch) => void) | undefined
  >();
  private queuedPromptTokenFlushModes = new Map<GenerateRequestId, 'batch' | 'token'>();
  private queuedPromptCallbackErrors = new Map<GenerateRequestId, unknown>();
  private readonly tracker = new RequestTracker<GenerateResponse>();
  private readonly scheduler: QueuedRequestScheduler;
  private runtimeObservabilityEnabled = false;
  private backendProfilingEnabled = false;
  private transportObservability: TransportObservability;
  private streamingTickCallback: (() => void) | undefined;
  // Worker-side SAB ring writer.  When set, requests with onTokens use the
  // SAB fast path; otherwise the scheduler delivers TokenBatch values through
  // callbacks/postMessage.
  private streamingRingWriter: StreamingRingWriter | null = null;
  constructor(private config: CogentConfig = {}) {
    this.executionMode = config.executionMode === 'worker' ? 'worker' : 'main-thread';
    this.transportObservability = this.createTransportObservability();
    this.modelLoader = new MainThreadModelLoader(this.config);
    this.scheduler = new QueuedRequestScheduler({
      tracker: this.tracker,
      queuedPromptCallbacks: this.queuedPromptCallbacks,
      queuedPromptTokenFlushModes: this.queuedPromptTokenFlushModes,
      queuedPromptCallbackErrors: this.queuedPromptCallbackErrors,
      getTransportObservability: () => this.transportObservability,
      getBridge: () => this.getReadyEngineBridge(),
      finalizeRequest: (bridge, requestId, options) => {
        this.finalizeRequest(bridge, requestId, options);
      },
      cancelQuery: (requestId) => this.cancelQuery(requestId),
      getStreamingRingWriter: () => this.streamingRingWriter,
      onStreamingTick: () => this.streamingTickCallback?.(),
    });
  }

  // Wires the worker-side SAB streaming ring writer.  Called once by the
  // worker entry after the main thread allocates the ring SAB.
  public setStreamingRingWriter(writer: StreamingRingWriter | null): void {
    this.streamingRingWriter = writer;
  }

  public setStreamingTickCallback(callback: (() => void) | undefined): void {
    this.streamingTickCallback = callback;
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
    this.queuedPromptTokenFlushModes.delete(requestId);
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
    onTokens: ((batch: TokenBatch) => void) | undefined
  ): void {
    if (onTokens == null) {
      this.refreshTokenTransportObservability();
      return;
    }

    this.transportObservability.activeTokenTransport =
      this.streamingRingWriter != null ? 'streaming-buffer' : 'callback';
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
    const request = this.buildGenerateRequest(contextKey, promptText, options);
    return this.enqueueNativeRequest(options, (bridge, emissionMode) => {
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
          request.grammar,
          emissionMode
        );
      }
      return bridge.startTextRequest(
        request.contextKey,
        request.promptText,
        request.maxOutputTokens,
        request.grammar,
        emissionMode
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

    return this.enqueueNativeRequest(options, (bridge, emissionMode) =>
      bridge.startChatRequest(
        contextKey,
        messages,
        maxOutputTokens,
        media,
        grammar,
        emissionMode
      )
    );
  }

  private async enqueueNativeRequest(
    options: number | PromptOptions,
    startRequest: (bridge: WasmBridge, emissionMode: TokenEmissionMode) => GenerateRequestId
  ): Promise<GenerateRequestId> {
    const bridge = this.getReadyEngineBridge();
    const onTokens = typeof options === 'object' ? options.onTokens : undefined;
    const signal = typeof options === 'object' ? options.signal : undefined;

    if (signal?.aborted) {
      throw createAbortError('Prompt was aborted before it was enqueued.');
    }

    this.updateTokenTransportObservability(onTokens);

    const emissionMode =
      onTokens == null ? TOKEN_EMISSION_NONE : TOKEN_EMISSION_STREAMING_BUFFER;
    const requestId = startRequest(bridge, emissionMode);
    if (!requestId) {
      throw new Error('Failed to enqueue request.');
    }

    // Worker entry uses this hook to publish a streaming-claim message
    // before inference produces tokens. Errors are swallowed.
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

    if (onTokens != null) {
      this.queuedPromptCallbacks.set(requestId, onTokens);
      this.queuedPromptTokenFlushModes.set(
        requestId,
        typeof options === 'object' ? options.tokenFlush ?? 'token' : 'token'
      );
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

  public async runBrowserRuntimeSmoke(): Promise<BrowserRuntimeSmokeResult> {
    await this.initModule();
    const bridge = this.getLoadedWasmBridge();
    const rustEngine = runBrowserRustEngineSmoke(bridge);
    const ggufIngest = runBrowserGgufIngestSmoke(bridge);
    const rawBackend = await bridge.getBackendObservabilityJson();
    const backend =
      rawBackend == null ? null : parseBackendObservabilityJson(rawBackend);
    const webgpuReady = Boolean(
      backend?.webgpuCompiled &&
      backend.webgpuRegistered &&
      backend.webgpuDeviceCount > 0 &&
      backend.gpuOffloadSupported
    );

    return {
      rustEngine,
      ggufIngest,
      backend,
      webgpuReady,
    };
  }
}
