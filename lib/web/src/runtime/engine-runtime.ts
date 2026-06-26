import type {
  BackendObservability,
  ChatMessage,
  EmbedRuntimeOptions,
  EngineExecutionMode,
  GenerateRequestId,
  GenerateResponse,
  NativeRuntimeConfig,
  PromptOptions,
  RequestObservabilityMetrics,
  TransportObservability,
} from '../engine/inference-types.js';
import type {
  ClassifiedAsset,
  InternalBundleDescriptor,
  ModelDetectionResult,
  PairingPlan,
  RegistryManifest,
  StagedModelBundle,
  StageModelBundleOptions,
} from '../models/types.js';
import type { ChatBoundaryInfo } from '../engine/chat-boundary-sanitizer.js';
import type {
  BrowserCacheLayout,
  GgufReadAtCallbacks,
  GgufSplitStreamCallbacks,
  RustLifecycleBridge,
} from '../wasm/wasm-bridge.js';
import type { RuntimeBackendOverride, WasmThreadingMode } from '../engine/runtime-assets.js';

export interface EngineRuntime {
  getExecutionMode(): EngineExecutionMode;
  getWasmThreadingMode(): WasmThreadingMode;
  getDefaultBackendOverride(): RuntimeBackendOverride | null;
  getTransportObservability(): TransportObservability;
  initModule(): Promise<void>;
  stageModelBundle(
    descriptor: InternalBundleDescriptor,
    options?: StageModelBundleOptions
  ): Promise<StagedModelBundle>;
  loadRuntimeModel(
    modelPathOrBundle: string | StagedModelBundle,
    config?: NativeRuntimeConfig
  ): Promise<void>;
  close(): void;
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
  resolvePairing(
    classified: readonly ClassifiedAsset[],
    explicitProjectorId?: string | null
  ): Promise<PairingPlan>;
  createRustLifecycleBridge(manifest: RegistryManifest): Promise<RustLifecycleBridge>;
  probeChatTemplateBoundaryInfo(): Promise<ChatBoundaryInfo>;
  enqueueChat(
    contextKey: string,
    messages: readonly ChatMessage[],
    options?: number | PromptOptions
  ): Promise<GenerateRequestId>;
  cancelQuery(requestId: GenerateRequestId): Promise<boolean>;
  enqueueQuery(
    contextKey: string,
    promptText: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestId>;
  enqueueEmbedding(
    contextKey: string,
    input: string,
    options?: EmbedRuntimeOptions
  ): Promise<GenerateRequestId>;
  awaitQuery(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse>;
  getRuntimeObservability(): RequestObservabilityMetrics | null;
  getBackendObservability(): Promise<BackendObservability | null>;
}
