import type { SippClientOptions, EngineModuleOptions } from '../../engine/browser-client.js';
import type {
  BackendObservability,
  ChatMessage,
  EmbedRuntimeOptions,
  GenerateRequest,
  GenerateRequestHandle,
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
  ModelDetectionResult,
  PairingPlan,
  RegistryManifest,
  RuntimeBundleDescriptor,
  RuntimeSessionSnapshot,
  RuntimePairingErrorCode,
} from '../../models/types.js';
import type { ChatBoundaryInfo } from '../../engine/chat-boundary-sanitizer.js';
import type {
  EngineRuntime,
  RuntimeActivation,
  RuntimeActivationReport,
  RuntimeActivationResult,
} from '../engine-runtime.js';
import { WasmModelLoader } from './model-loader.js';
import { RequestTracker } from '../request-tracker.js';
import {
  COMPLETED_REQUEST_STATUS_PENDING,
  RustLifecycleBridge,
  parseBackendObservabilityJson,
  WasmBridge,
} from '../../wasm/wasm-bridge.js';
import { EngineModule } from '../../wasm/engine-module.js';
import { createAbortError } from '../../utils/abort.js';
import { AsyncSerialQueue } from '../../utils/async-queue.js';
import { QueuedRequestScheduler } from '../scheduler.js';
import { hasSamplingRuntimeOverrideFields } from '../../engine/inference-types.js';
import {
  resolveRuntimeThreadingMode,
  resolveRuntimeUrls,
  type RuntimeBackendConstraint,
  type WasmThreadingMode,
} from '../../engine/runtime-assets.js';
import { RuntimePairingValidationError } from '../../models/types.js';
import {
  attachCleanupFailures,
  releaseAllAsync,
} from '../../utils/cleanup.js';

interface ResolvedPromptOptions {
  readonly maxOutputTokens: number;
  readonly media: Uint8Array[] | undefined;
  readonly grammar: string | undefined;
  readonly stop: readonly string[] | undefined;
  readonly sampling: SamplingRuntimeOverride | undefined;
}

interface PreparedNativeActivation {
  readonly session: RuntimeSessionSnapshot;
  readonly report: RuntimeActivationReport;
  readonly sharedTokenRingDescriptor: SharedTokenRingDescriptor | null;
}

function resolvePromptMedia(
  media: PromptOptions['media']
): Uint8Array[] | undefined {
  if (media == null) {
    return undefined;
  }
  if (!Array.isArray(media)) {
    throw new Error('media must be an array of Uint8Array instances.');
  }
  if (media.length === 0) {
    return undefined;
  }
  if (media.some((image) => !(image instanceof Uint8Array))) {
    throw new Error('media entries must be Uint8Array instances.');
  }
  return media;
}

function resolvePromptGrammar(
  grammar: PromptOptions['grammar']
): string | undefined {
  if (grammar == null) {
    return undefined;
  }
  if (typeof grammar !== 'string') {
    throw new Error('grammar must be a string when provided.');
  }
  return grammar.length === 0 ? undefined : grammar;
}

function resolvePromptStop(
  stop: PromptOptions['stop']
): readonly string[] | undefined {
  if (stop == null) {
    return undefined;
  }
  if (!Array.isArray(stop)) {
    throw new Error('stop must be an array of strings.');
  }
  if (stop.some((value) => typeof value !== 'string')) {
    throw new Error('stop entries must be strings.');
  }
  return stop.length === 0 ? undefined : stop;
}

function resolvePromptSampling(
  sampling: PromptOptions['sampling']
): SamplingRuntimeOverride | undefined {
  if (sampling == null) {
    return undefined;
  }
  if (typeof sampling !== 'object' || Array.isArray(sampling)) {
    throw new Error('sampling must be an object when provided.');
  }
  return hasSamplingRuntimeOverrideFields(sampling) ? sampling : undefined;
}

function promptOptionsObject(
  input: PromptOptions | number | undefined
): PromptOptions | undefined {
  if (input === undefined || typeof input === 'number') {
    return undefined;
  }
  if (typeof input !== 'object' || input === null || Array.isArray(input)) {
    throw new Error('Prompt options must be an object or token count.');
  }
  return input;
}

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

const EXPECTED_RUST_BROWSER_ENGINE_ABI_VERSION = 15;
const DEFAULT_WASM_TRANSPORT_OBSERVABILITY: TransportObservability = {
  executionMode: 'worker',
  workerBacked: true,
  enabled: false,
  wasmRunLoopCalls: 0,
  wasmRunLoopMs: 0,
  activeTokenTransport: 'none',
  activeTokenEmission: false,
  tokenDrainCalls: 0,
  tokenDrainMs: 0,
};

interface WasmEngineRuntimeOptions {
  readonly backendConstraint?: RuntimeBackendConstraint | null;
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

/** Worker-owned Wasm runtime for one browser model session. */
export class WasmEngineRuntime implements EngineRuntime {
  public readonly backendConstraint: RuntimeBackendConstraint | null;
  private module: EngineModule | null = null;
  private wasmBridge: WasmBridge | null = null;
  private initPromise: Promise<void> | null = null;
  private moduleGeneration = 0;
  private engineInitialized = false;
  private runtimeSession: RuntimeSessionSnapshot | null = null;

  private readonly modelLoader: WasmModelLoader;
  private queuedPromptTokenBatchSinks = new Map<
    GenerateRequestId,
    (batch: TokenBatch) => void
  >();
  private readonly tracker = new RequestTracker<GenerateResponse>();
  private readonly scheduler: QueuedRequestScheduler;
  private runtimeObservabilityEnabled = false;
  private runtimeObservabilitySnapshot: RequestObservabilityMetrics | null = null;
  private sharedTokenRingDescriptor: SharedTokenRingDescriptor | null = null;
  private backendProfilingEnabled = false;
  private transportObservability: TransportObservability;
  private readonly wasmBridgeOperations = new AsyncSerialQueue();

  constructor(
    private readonly config: SippClientOptions = {},
    options: WasmEngineRuntimeOptions = {}
  ) {
    this.backendConstraint = options.backendConstraint ?? null;
    this.transportObservability = this.createTransportObservability();
    this.modelLoader = new WasmModelLoader(this.config);
    this.scheduler = new QueuedRequestScheduler({
      tracker: this.tracker,
      queuedPromptTokenBatchSinks: this.queuedPromptTokenBatchSinks,
      getTransportObservability: () => this.transportObservability,
      getRuntimeGeneration: () => this.requireRuntimeSession().generation,
      finalizeRequest: (bridge, requestId, options) => {
        this.finalizeRequest(bridge, requestId, options);
      },
      cancelQuery: (requestId) => this.cancelQuery(requestId),
      withWasmBridge: (operation) => this.withReadyWasmBridge(operation),
    });
  }

  public getWasmThreadingMode(): WasmThreadingMode {
    return resolveRuntimeThreadingMode(this.config);
  }

  public getTransportObservability(): TransportObservability {
    return { ...this.transportObservability };
  }

  /**
   * The activation-time token-ring descriptor. It is cached rather than read
   * on demand because callers are synchronous and must not enter Wasm outside
   * the bridge queue; the ring is fixed for the lifetime of the session.
   */
  public getSharedTokenRingDescriptor(): SharedTokenRingDescriptor | null {
    return this.sharedTokenRingDescriptor;
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

  /**
   * Resolves the generation payload fields at one runtime boundary.
   *
   * This package is called from plain JavaScript, so container shapes are
   * checked here rather than trusted from the type signature. Rust validates
   * the detailed sampler values when it deserializes the native request.
   */
  private resolvePromptOptions(
    input: PromptOptions | number | undefined,
    defaultTokens = 128
  ): ResolvedPromptOptions {
    const options = promptOptionsObject(input);
    return {
      maxOutputTokens: this.normalizeTokenCount(
        (typeof input === 'number' ? input : options?.nTokens) ?? defaultTokens
      ),
      media: resolvePromptMedia(options?.media),
      grammar: resolvePromptGrammar(options?.grammar),
      stop: resolvePromptStop(options?.stop),
      sampling: resolvePromptSampling(options?.sampling),
    };
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
    const resolved = this.resolvePromptOptions(options);
    return {
      contextKey,
      promptText: normalizePromptText(promptText),
      maxOutputTokens: resolved.maxOutputTokens,
      media: resolved.media,
      stop: resolved.stop,
      sampling: resolved.sampling,
      grammar: resolved.grammar,
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
      throw new Error('Engine runtime is not ready. Load a local endpoint first.');
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

  private requireRuntimeSession(): RuntimeSessionSnapshot {
    if (this.runtimeSession == null) {
      throw new Error('Engine runtime does not have an active session.');
    }
    return this.runtimeSession;
  }

  private withWasmBridgeOperation<T>(
    requireReady: boolean,
    operation: (bridge: WasmBridge) => T | Promise<T>
  ): Promise<T> {
    return this.wasmBridgeOperations.run(async () => {
      const bridge = requireReady
        ? this.getReadyEngineBridge()
        : this.getLoadedWasmBridge();
      try {
        return await operation(bridge);
      } finally {
        this.refreshRuntimeObservabilitySnapshot(bridge);
      }
    });
  }

  /**
   * Samples runtime metrics while still holding the bridge queue. Reading them
   * later from the synchronous accessor would re-enter Wasm outside the queue,
   * which is unsafe once a JSPI inference loop can be suspended mid-call.
   */
  private refreshRuntimeObservabilitySnapshot(bridge: WasmBridge): void {
    if (!this.runtimeObservabilityEnabled || !this.engineInitialized) {
      return;
    }
    try {
      this.runtimeObservabilitySnapshot = bridge.readRuntimeObservability();
    } catch {
      // Observability sampling must never fail the operation that produced it.
    }
  }

  private withReadyWasmBridge<T>(
    operation: (bridge: WasmBridge) => T | Promise<T>
  ): Promise<T> {
    return this.withWasmBridgeOperation(true, operation);
  }

  private withLoadedWasmBridge<T>(
    operation: (bridge: WasmBridge) => T | Promise<T>
  ): Promise<T> {
    return this.withWasmBridgeOperation(false, operation);
  }

  private releaseTokenState(request: GenerateRequestHandle): void {
    this.queuedPromptTokenBatchSinks.delete(request.requestId);
    this.refreshTokenTransportObservability();
  }

  private finalizeRequest(
    bridge: WasmBridge | null,
    requestId: GenerateRequestHandle,
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
    requestId: GenerateRequestHandle
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
    for (const tracked of this.tracker.records()) {
      this.releaseTokenState(tracked.request);
    }
    this.tracker.rejectAll(error);
  }

  private resetRuntimeLifecycleState(): void {
    this.scheduler.reset();
    this.tracker.clear();
    this.runtimeObservabilityEnabled = false;
    this.runtimeObservabilitySnapshot = null;
    this.sharedTokenRingDescriptor = null;
    this.backendProfilingEnabled = false;
    this.transportObservability = this.createTransportObservability();
  }

  private createTransportObservability(): TransportObservability {
    return { ...DEFAULT_WASM_TRANSPORT_OBSERVABILITY };
  }

  /**
   * Issues a best-effort cancellation from an abort listener. The awaiting
   * caller settles through the tracker either way, so a failed cancel must not
   * surface as an unhandled rejection.
   */
  private requestCancellation(request: GenerateRequestHandle): void {
    const tracked = this.tracker.get(request);
    if (tracked == null || tracked.cancelRequested) {
      return;
    }
    this.tracker.requestCancel(request);
    void this.cancelQuery(request).catch(() => {});
  }

  private assertActiveRequestHandle(request: GenerateRequestHandle): void {
    const generation = this.currentRuntimeSession()?.generation;
    if (generation !== request.generation) {
      throw new Error(
        `Browser request ${request.generation}:${request.requestId} does not belong to the active runtime generation.`
      );
    }
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
          const error = createAbortError('Module initialization was cancelled.');
          try {
            await new WasmBridge(module).close();
          } catch (cleanupError) {
            throw attachCleanupFailures(error, cleanupError);
          }
          throw error;
        }
        const bridge = new WasmBridge(module);
        try {
          verifyRustBrowserEngineAbi(bridge);
        } catch (error) {
          try {
            await bridge.close();
          } catch (cleanupError) {
            throw attachCleanupFailures(error, cleanupError);
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

  public async detectModelFromGgufFile(
    file: Blob & { name?: string },
    signal?: AbortSignal
  ): Promise<ModelDetectionResult> {
    await this.ensureModule();
    return await this.withLoadedWasmBridge((bridge) =>
      bridge.detectModelFromGgufFile(file, signal)
    );
  }

  public async browserCacheLayout(
    sourceBytes: number,
    sourceBytesKnown: boolean,
    directLoadMaxBytes: number,
    shardMaxBytes: number
  ) {
    await this.ensureModule();
    return await this.withLoadedWasmBridge((bridge) =>
      bridge.browserCacheLayout(
        sourceBytes,
        sourceBytesKnown,
        directLoadMaxBytes,
        shardMaxBytes
      )
    );
  }

  public async planGgufSplitCount(
    sourceBytes: number,
    shardMaxBytes: number,
    callbacks: Parameters<WasmBridge['planGgufSplitCount']>[2]
  ): Promise<number> {
    await this.ensureModule();
    return await this.withLoadedWasmBridge((bridge) =>
      bridge.planGgufSplitCount(sourceBytes, shardMaxBytes, callbacks)
    );
  }

  public async splitGgufStream(
    sourceBytes: number,
    outputPrefix: string,
    shardMaxBytes: number,
    callbacks: Parameters<WasmBridge['splitGgufStream']>[3]
  ): Promise<void> {
    await this.ensureModule();
    await this.withLoadedWasmBridge((bridge) => {
      bridge.splitGgufStream(sourceBytes, outputPrefix, shardMaxBytes, callbacks);
    });
  }

  public async resolvePairing(classified: readonly ClassifiedAsset[]): Promise<PairingPlan> {
    await this.ensureModule();
    const response = await this.withLoadedWasmBridge((bridge) =>
      bridge.validatePairing(classified)
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
    return await RustLifecycleBridge.create(
      (operation) => this.withLoadedWasmBridge(operation),
      manifest
    );
  }

  public async activateRuntime<TCommit>(
    bundle: RuntimeBundleDescriptor,
    activation: RuntimeActivation<TCommit>
  ): Promise<RuntimeActivationResult<TCommit>> {
    // Everything up to mountBundle fails before the loader owns the bundle, so
    // this method still owns closing its handles.
    let module: EngineModule;
    try {
      this.modelLoader.validateBundle(bundle);
      if (activation.signal?.aborted) {
        throw createAbortError('Model load aborted.');
      }
      module = await this.ensureModule();
      if (this.runtimeSession != null || this.engineInitialized) {
        throw new Error('A browser Worker can activate only one runtime session.');
      }
    } catch (error) {
      throw this.closeUntransferredBundle(bundle, error);
    }

    this.resetRuntimeLifecycleState();

    let bundleTransferred = false;
    let prepared: PreparedNativeActivation;
    try {
      prepared = await this.withLoadedWasmBridge(async (bridge) => {
        let nativeActivationStarted = false;
        try {
          if (activation.signal?.aborted) {
            throw createAbortError('Model load aborted.');
          }
          bundleTransferred = true;
          const mounted = this.modelLoader.mountBundle(module, bundle);
          const effectiveConfig = this.runtimeConfigForBundle(
            activation.config,
            mounted.projectorPath
          );
          this.configureRuntimeObservability(effectiveConfig);
          nativeActivationStarted = true;
          const result = await bridge.loadRuntimeModel(
            mounted.modelPath,
            activation.session,
            effectiveConfig
          );
          if (result !== 0) {
            const detail = bridge.readLastEngineError();
            throw new Error(
              detail.length > 0
                ? `Failed to initialize engine. Code: ${result}. ${detail}`
                : `Failed to initialize engine. Code: ${result}`
            );
          }
          const loadedSession = bridge.getRuntimeSession();
          if (mounted.projectorPath != null && loadedSession.mediaMarker == null) {
            throw new Error(
              'Failed to initialize multimodal runtime: loaded projector did not expose a media marker.'
            );
          }
          const report = await this.runtimeActivationReport(bridge, loadedSession);
          // Release the mounted bundle before committing. Native activation no
          // longer needs it, and a cleanup failure after commit would leave the
          // Rust catalog committed while this method reports activation failure
          // and publishes no session.
          this.modelLoader.cleanup(module);
          return {
            session: loadedSession,
            report,
            sharedTokenRingDescriptor: bridge.getSharedTokenRingDescriptor(),
          };
        } catch (error) {
          throw await this.cleanupFailedActivation(
            bridge,
            module,
            nativeActivationStarted,
            error
          );
        }
      });
    } catch (error) {
      throw bundleTransferred ? error : this.closeUntransferredBundle(bundle, error);
    }

    // The catalog commit enters the same Wasm instance through its lifecycle
    // bridge. Run it only after releasing the native activation queue; awaiting
    // it from inside withLoadedWasmBridge would queue behind itself forever.
    try {
      if (activation.signal?.aborted) {
        throw createAbortError('Model load aborted before activation was committed.');
      }
      const committed = await activation.commit(prepared.report);
      this.sharedTokenRingDescriptor = prepared.sharedTokenRingDescriptor;
      this.runtimeSession = prepared.session;
      this.engineInitialized = true;
      return { session: prepared.session, committed };
    } catch (error) {
      const failure = await this.withLoadedWasmBridge((bridge) =>
        this.cleanupFailedActivation(bridge, module, true, error)
      );
      throw failure;
    }
  }

  private async cleanupFailedActivation(
    bridge: WasmBridge,
    module: EngineModule,
    nativeActivationStarted: boolean,
    primary: unknown
  ): Promise<unknown> {
    let failure = primary;
    try {
      await releaseAllAsync('Failed to clean up an unsuccessful runtime activation.', [
        ...(nativeActivationStarted
          ? [{
            label: 'close native runtime',
            release: () => bridge.close(),
          }]
          : []),
        {
          label: 'release mounted model bundle',
          release: () => this.modelLoader.cleanup(module),
        },
      ]);
    } catch (cleanupError) {
      failure = attachCleanupFailures(primary, cleanupError);
    }
    this.engineInitialized = false;
    this.runtimeSession = null;
    this.resetRuntimeLifecycleState();
    return failure;
  }

  /**
   * Closes bundle handles the model loader never took ownership of, and
   * returns the failure to throw. A close failure is attached to `error`
   * rather than replacing it.
   */
  private closeUntransferredBundle(
    bundle: RuntimeBundleDescriptor,
    error: unknown
  ): unknown {
    try {
      this.modelLoader.closeBundle(bundle);
    } catch (cleanupError) {
      return attachCleanupFailures(error, cleanupError);
    }
    return error;
  }

  public currentRuntimeSession(): RuntimeSessionSnapshot | null {
    return this.runtimeSession;
  }

  private runtimeConfigForBundle(
    config: NativeRuntimeConfig,
    projectorPath: string | null
  ): NativeRuntimeConfig {
    if (projectorPath == null) {
      return config;
    }
    return {
      ...config,
      multimodal: {
        ...config.multimodal,
        projector_path: projectorPath,
      },
    };
  }

  private configureRuntimeObservability(config: NativeRuntimeConfig): void {
    this.runtimeObservabilityEnabled =
      config.observability?.runtime_metrics === true ||
      config.observability?.backend_profiling === true;
    this.backendProfilingEnabled = config.observability?.backend_profiling === true;
    this.transportObservability.enabled = this.runtimeObservabilityEnabled;
  }

  private async runtimeActivationReport(
    bridge: WasmBridge,
    session: RuntimeSessionSnapshot
  ): Promise<RuntimeActivationReport> {
    return {
      session,
      runtimeObservability: this.runtimeObservabilityEnabled
        ? bridge.readRuntimeObservability()
        : null,
      backendObservability: await this.readBackendObservability(bridge),
    };
  }

  /**
   * Shutdown engine instance.
   */
  public async close(): Promise<void> {
    this.moduleGeneration += 1;
    const module = this.module;
    this.runtimeSession = null;
    this.engineInitialized = false;
    this.rejectAllTrackedRequests(new Error('Engine runtime was closed.'));
    try {
      if (module != null) {
        await this.withLoadedWasmBridge(async (bridge) => {
          await releaseAllAsync('Failed to close the Wasm engine runtime.', [
            {
              label: 'close native runtime',
              release: () => bridge.close(),
            },
            {
              label: 'release mounted model bundle',
              release: () => this.modelLoader.cleanup(module),
            },
          ]);
        });
      }
    } finally {
      this.resetRuntimeLifecycleState();
      this.module = null;
      this.wasmBridge = null;
      this.initPromise = null;
    }
  }

  public async cancelQuery(requestId: GenerateRequestHandle): Promise<boolean> {
    this.assertActiveRequestHandle(requestId);
    const cancelled = await this.withReadyWasmBridge((bridge) =>
      bridge.cancelQuery(requestId)
    );
    if (!cancelled) {
      return false;
    }

    if (this.tracker.get(requestId) != null) {
      this.tracker.requestCancel(requestId);
      const settled = await this.withReadyWasmBridge((bridge) =>
        this.scheduler.settleCompletedRequestIfPresent(bridge, requestId)
      );
      if (settled) {
        return true;
      }
    }

    if (this.tracker.get(requestId)?.active !== true) {
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
  ): Promise<GenerateRequestHandle> {
    const request = this.buildGenerateRequest(contextKey, promptText, options);
    return this.enqueueNativeRequest(options, (bridge, generation, emitTokens) => {
      if (request.media != null && request.media.length > 0) {
        const mediaMarker = this.readMediaMarker();
        if (mediaMarker == null) {
          throw new Error(
            'Loaded runtime does not expose a media marker for the current model.'
          );
        }
        const markerCount = this.countMarkerOccurrences(
          request.promptText,
          mediaMarker
        );
        if (markerCount !== request.media.length) {
          throw new Error(
            `Prompt contains ${markerCount} media marker(s) but ${request.media.length} image(s) were provided. Use "${mediaMarker}" in your prompt to place each image.`
          );
        }
        return bridge.startMediaRequest(
          generation,
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
        generation,
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
  ): Promise<GenerateRequestHandle> {
    const resolved = this.resolvePromptOptions(options);
    if (resolved.media != null && this.readMediaMarker() == null) {
      throw new Error(
        'Loaded runtime does not expose a media marker for the current model.'
      );
    }

    return this.enqueueNativeRequest(options, (bridge, generation, emitTokens) =>
      bridge.startChatRequest(
        generation,
        contextKey,
        messages,
        resolved.maxOutputTokens,
        resolved.media,
        {
          grammar: resolved.grammar,
          stop: resolved.stop,
          sampling: resolved.sampling,
          emitTokens,
        }
      )
    );
  }

  public async enqueueEmbedding(
    contextKey: string,
    input: string,
    options: EmbedRuntimeOptions = {}
  ): Promise<GenerateRequestHandle> {
    const promptText = normalizePromptText(input);
    return this.enqueueNativeRequest(options, (bridge, generation) =>
      bridge.startEmbeddingRequest(
        generation,
        contextKey,
        promptText,
        options.normalize ?? true
      )
    );
  }

  public async enqueueListen(
    audio: Uint8Array,
    language: string,
    options: number | PromptOptions = 4096
  ): Promise<GenerateRequestHandle> {
    const { maxOutputTokens } = this.resolvePromptOptions(options, 4096);
    return this.enqueueNativeRequest(options, (bridge, generation) =>
      bridge.startListenRequest(generation, audio, language, maxOutputTokens)
    );
  }

  public async enqueueSpeak(
    text: string,
    language: string,
    speakerAudio: Uint8Array,
    maxDurationMs: number | undefined,
    options: PromptOptions = {}
  ): Promise<GenerateRequestHandle> {
    return this.enqueueNativeRequest(options, (bridge, generation) =>
      bridge.startSpeakRequest(generation, text, language, speakerAudio, maxDurationMs)
    );
  }

  private async enqueueNativeRequest(
    options: number | PromptOptions,
    startRequest: (
      bridge: WasmBridge,
      generation: number,
      emitTokens: boolean
    ) => GenerateRequestHandle | Promise<GenerateRequestHandle>
  ): Promise<GenerateRequestHandle> {
    const promptOptions = promptOptionsObject(options);
    const tokenBatchSink = promptOptions?.tokenBatchSink;
    const emitTokens =
      promptOptions != null &&
      (promptOptions.emitTokens === true || tokenBatchSink != null);
    const signal = promptOptions?.signal;

    if (signal?.aborted) {
      throw createAbortError('Prompt was aborted before it was enqueued.');
    }

    const { request, errorDetail } = await this.withReadyWasmBridge(async (bridge) => {
      const generation = this.requireRuntimeSession().generation;
      const request = await startRequest(bridge, generation, emitTokens);
      return {
        request,
        errorDetail: request.requestId === 0 ? bridge.readLastEngineError() : '',
      };
    });
    if (request.requestId === 0) {
      throw new Error(
        errorDetail.length > 0
          ? `Failed to enqueue request. ${errorDetail}`
          : 'Failed to enqueue request.'
      );
    }
    if (typeof promptOptions?.onRequestStarted === 'function') {
      try {
        promptOptions.onRequestStarted(request.requestId);
      } catch {
        /* request-start observers must not abort enqueue */
      }
    }

    if (tokenBatchSink != null) {
      this.queuedPromptTokenBatchSinks.set(request.requestId, tokenBatchSink);
      this.refreshTokenTransportObservability();
    }

    this.scheduler.track(request);
    if (signal != null) {
      this.tracker.attachSignal(request, signal, () => {
        this.requestCancellation(request);
      });
    }
    return request;
  }

  public async awaitQuery(
    requestId: GenerateRequestHandle,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    this.assertActiveRequestHandle(requestId);
    if (options?.signal?.aborted) {
      this.requestCancellation(requestId);
      throw createAbortError('Prompt was aborted before execution started.');
    }

    this.scheduler.track(requestId);
    const signal = options?.signal;
    const detachAbort =
      signal == null
        ? () => {}
        : this.tracker.attachSignal(requestId, signal, () => {
            this.requestCancellation(requestId);
          });

    const responsePromise = this.tracker.beginWait(requestId);
    try {
      const response = await responsePromise;
      const tracked = this.tracker.get(requestId);
      if (tracked?.tokenBatchSinkFailed === true) {
        throw tracked.tokenBatchSinkError;
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
    if (!this.engineInitialized || !this.runtimeObservabilityEnabled) {
      return null;
    }
    return this.runtimeObservabilitySnapshot;
  }

  public readMediaMarker(): string | null {
    return this.runtimeSession?.mediaMarker ?? null;
  }

  public getChatTemplate(): string | null {
    return this.runtimeSession?.chatTemplate ?? null;
  }

  public getBosText(): string {
    return this.runtimeSession?.bosText ?? '';
  }

  public getEosText(): string {
    return this.runtimeSession?.eosText ?? '';
  }

  public async probeChatTemplateBoundaryInfo(): Promise<ChatBoundaryInfo> {
    return await this.withReadyWasmBridge((bridge) =>
      bridge.probeChatTemplateBoundaryInfo()
    );
  }

  public async getBackendObservability(): Promise<BackendObservability | null> {
    return await this.withLoadedWasmBridge((bridge) =>
      this.readBackendObservability(bridge)
    );
  }

  private async readBackendObservability(
    bridge: WasmBridge
  ): Promise<BackendObservability | null> {
    const raw = await bridge.getBackendObservabilityJson();
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
