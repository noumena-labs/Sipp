import type {
  BackendObservability,
  ChatMessage,
  EmbedRuntimeOptions,
  GenerateRequestHandle,
  GenerateResponse,
  NativeRuntimeConfig,
  PromptOptions,
  RequestObservabilityMetrics,
  TransportObservability,
} from '../engine/inference-types.js';
import type {
  ClassifiedAsset,
  ModelDetectionResult,
  PairingPlan,
  RegistryManifest,
  RuntimeBundleDescriptor,
  RuntimeSessionDescriptor,
  RuntimeSessionSnapshot,
} from '../models/types.js';
import type { ChatBoundaryInfo } from '../engine/chat-boundary-sanitizer.js';
import type {
  BrowserCacheLayout,
  GgufReadAtCallbacks,
  GgufSplitStreamCallbacks,
  RustLifecycleBridge,
} from '../wasm/wasm-bridge.js';
import type {
  RuntimeBackendConstraint,
  WasmThreadingMode,
} from '../engine/runtime-assets.js';

export interface RuntimeActivationReport {
  readonly session: RuntimeSessionSnapshot;
  readonly runtimeObservability: RequestObservabilityMetrics | null;
  readonly backendObservability: BackendObservability | null;
}

export interface RuntimeActivation<TCommit> {
  readonly session: RuntimeSessionDescriptor;
  readonly config: NativeRuntimeConfig;
  readonly signal?: AbortSignal;
  readonly commit: (report: RuntimeActivationReport) => Promise<TCommit>;
}

export interface RuntimeActivationResult<TCommit> {
  readonly session: RuntimeSessionSnapshot;
  readonly committed: TCommit;
}

export interface EngineRuntime {
  readonly backendConstraint: RuntimeBackendConstraint | null;
  getWasmThreadingMode(): WasmThreadingMode;
  getTransportObservability(): TransportObservability;
  initModule(): Promise<void>;
  activateRuntime<TCommit>(
    bundle: RuntimeBundleDescriptor,
    activation: RuntimeActivation<TCommit>
  ): Promise<RuntimeActivationResult<TCommit>>;
  currentRuntimeSession(): RuntimeSessionSnapshot | null;
  close(): Promise<void>;
  getChatTemplate(): string | null;
  readMediaMarker(): string | null;
  /**
   * Returns the model's BOS token rendered as text, or '' if the model has
   * no BOS token. Used by the character-agent custom template builder to
   * emit the correct leading special token per model.
   */
  getBosText(): string;
  /** Returns the model's EOS token rendered as text, or '' if unavailable. */
  getEosText(): string;
  browserCacheLayout(
    sourceBytes: number,
    sourceBytesKnown: boolean,
    directLoadMaxBytes: number,
    shardMaxBytes: number
  ): Promise<BrowserCacheLayout>;
  planGgufSplitCount(
    sourceBytes: number,
    shardMaxBytes: number,
    callbacks: GgufReadAtCallbacks
  ): Promise<number>;
  splitGgufStream(
    sourceBytes: number,
    outputPrefix: string,
    shardMaxBytes: number,
    callbacks: GgufSplitStreamCallbacks
  ): Promise<void>;
  detectModelFromGgufFile(
    file: Blob & { name?: string },
    signal?: AbortSignal
  ): Promise<ModelDetectionResult>;
  resolvePairing(classified: readonly ClassifiedAsset[]): Promise<PairingPlan>;
  createRustLifecycleBridge(manifest: RegistryManifest): Promise<RustLifecycleBridge>;
  probeChatTemplateBoundaryInfo(): Promise<ChatBoundaryInfo>;
  enqueueChat(
    contextKey: string,
    messages: readonly ChatMessage[],
    options?: number | PromptOptions
  ): Promise<GenerateRequestHandle>;
  cancelQuery(request: GenerateRequestHandle): Promise<boolean>;
  enqueueQuery(
    contextKey: string,
    promptText: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestHandle>;
  enqueueEmbedding(
    contextKey: string,
    input: string,
    options?: EmbedRuntimeOptions
  ): Promise<GenerateRequestHandle>;
  enqueueListen(
    audio: Uint8Array,
    language: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestHandle>;
  enqueueSpeak(
    text: string,
    language: string,
    speakerAudio: Uint8Array,
    maxDurationMs: number | undefined,
    options?: PromptOptions
  ): Promise<GenerateRequestHandle>;
  awaitQuery(
    request: GenerateRequestHandle,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse>;
  getRuntimeObservability(): RequestObservabilityMetrics | null;
  getBackendObservability(): Promise<BackendObservability | null>;
}
