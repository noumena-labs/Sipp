import type { BackendObservability } from '../observability/backend-observability.js';
import type {
  ChatMessage,
  EngineExecutionMode,
  GenerateRequestId,
  GenerateResponse,
  NativeRuntimeConfig,
  PromptOptions,
} from '../core/inference-types.js';
import type {
  InternalBundleDescriptor,
  ModelDetectionResult,
  StagedModelBundle,
  StageModelBundleOptions,
} from '../bundle/model-bundle-types.js';
import type { RequestObservabilityMetrics } from '../observability/runtime-observability.js';
import type {
  TransportObservability,
} from '../observability/transport-observability.js';
import type { ChatBoundaryInfo } from '../core/chat-boundary-sanitizer.js';
import type { ClassifiedAsset, PairingPlan } from '../models/pairing-types.js';
import type {
  BrowserCacheLayout,
  GgufReadAtCallbacks,
  GgufSplitStreamCallbacks,
} from '../wasm/wasm-bridge.js';
import type { RustLifecycleBridge } from '../wasm/lifecycle-bridge.js';
import type { RegistryManifest } from '../models/types.js';
import type { WasmThreadingMode } from '../engine/runtime-assets.js';

export type RuntimePairingErrorCode =
  | 'INVALID_MODEL_SOURCE'
  | 'INVALID_MODEL_PAIRING'
  | 'MODEL_BROKEN';

export class RuntimePairingValidationError extends Error {
  public readonly code: RuntimePairingErrorCode;

  constructor(code: RuntimePairingErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'RuntimePairingValidationError';
    this.code = code;
  }
}

export interface EngineRuntime {
  getExecutionMode(): EngineExecutionMode;
  getWasmThreadingMode(): WasmThreadingMode;
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
  awaitQuery(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse>;
  getRuntimeObservability(): RequestObservabilityMetrics | null;
  getBackendObservability(): Promise<BackendObservability | null>;
}
