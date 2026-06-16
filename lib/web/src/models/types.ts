import type {
  BackendDeviceType,
  CacheSource,
  ChatMessage,
  KvReuseMode,
  NativeRuntimeConfig,
  PoolingType,
  SamplingRuntimeOverride,
  TokenEmissionStats,
  TokenBatch,
} from '../engine/inference-types.js';
import type { OpfsSyncAccessHandle } from '../engine/file-system-storage.js';

export type ModelModality = 'text' | 'vision';
export type ModelStatus = 'ready' | 'needs_projector' | 'broken';
export type ModelSourceKind = 'remote' | 'local';
export type BrowserBackendPreference = 'auto' | 'cpu' | 'webgpu';
export type ModelBundleSourceKind = 'installed';
export type ModelBundleProjectorStatus =
  | 'not-required'
  | 'explicit'
  | 'paired'
  | 'missing';
export type ModelDetectionMethod = 'gguf-metadata' | 'none';
export type AssetRole = 'model' | 'projector' | 'unknown';

export type ModelAssetKind = 'model' | 'projector' | 'shard';
export type ObservabilityMode = 'off' | 'runtime' | 'profile';
export type ObservabilityState = 'idle' | 'loading' | 'ready' | 'querying' | 'error' | 'closed';
export type ObservabilityEventType =
  | 'load-start'
  | 'load-complete'
  | 'query-start'
  | 'query-complete'
  | 'error'
  | 'close';

export interface ModelLoadProgress {
  phase: 'metadata' | 'download' | 'split' | 'store' | 'load';
  loadedBytes: number;
  totalBytes: number | null;
  percent: number | null;
  assetName?: string;
}

export interface ModelLoadOptions {
  signal?: AbortSignal;
  onProgress?: (progress: ModelLoadProgress) => void;
  backend?: BrowserBackendPreference;
  observability?: ObservabilityMode;
  runtime?: NativeRuntimeConfig;
}

export type ModelSource =
  | string
  | File
  | readonly string[]
  | readonly File[]
  | {
    model: string | File | readonly string[] | readonly File[];
    projector?: string | File;
  };

export interface ModelInfo {
  /** Installed model id persisted in OPFS. Pass this to a local `client.add(...)` descriptor. */
  id: string;
  name: string;
  modality: ModelModality;
  status: ModelStatus;
  source: ModelSourceKind;
  bytes: number;
  loaded: boolean;
  chatTemplate: string | null;
  bosText: string;
  eosText: string;
  mediaMarker: string | null;
  createdAt: string;
  updatedAt: string;
  capabilities?: ModelCapabilities;
}

export type ModelClass = 'decoder_only' | 'encoder_decoder' | 'encoder_only';

export interface ModelCapabilities {
  modelClass: ModelClass;
  supportsTextGeneration: boolean;
  supportsEmbeddings: boolean;
  hasChatTemplate: boolean;
  embedding?: {
    dimensions: number;
    pooling: PoolingType;
  };
}

export interface AssetInspection {
  version: 1;
  role: AssetRole;
  architecture: string | null;
  visionCapable: boolean;
  compatibleVisionProjectorTypes: string[];
  providedVisionProjectorType: string | null;
}

export interface ClassifiedAsset {
  assetId: string;
  inspection: AssetInspection;
  name: string;
}

export interface ClassifiedAssetFile extends ClassifiedAsset {
  file: File;
}

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

export interface PairingPlan {
  modelAssetIds: string[];
  projectorAssetId?: string | null;
  name: string;
  modality: ModelModality;
  status: ModelStatus;
  compatibleVisionProjectorTypes: string[];
}

export interface ModelBundleFileProjectorDescriptor {
  file: File;
  destFileName?: string;
}

export interface ModelBundleShard {
  name: string;
  handle: OpfsSyncAccessHandle;
  size: number;
}

export interface InternalBundleDescriptor {
  shards: ModelBundleShard[];
  projector?: ModelBundleFileProjectorDescriptor;
  detection: ModelDetectionResult;
}

export interface StageModelBundleOptions {
  signal?: AbortSignal;
}

export interface ModelDetectionResult {
  inspection: AssetInspection;
  detectionMethod: ModelDetectionMethod;
  modelName: string;
  modelType: string | null;
  modelArchitecture: string | null;
}

export interface StagedModelBundle {
  sourceKind: ModelBundleSourceKind;
  modelPath: string;
  projectorPath: string | null;
  isVisionModel: boolean;
  projectorStatus: ModelBundleProjectorStatus;
  modelName: string;
  detectionMethod: ModelDetectionMethod;
  modelType: string | null;
  modelArchitecture: string | null;
}

export type QueryInput =
  | string
  | {
    prompt: string;
    media?: Uint8Array[];
  };

export interface QueryOptions {
  /** Explicit endpoint for this request. Omitted requests use the current local endpoint. */
  endpoint?: EndpointRef;
  /** Local KV-cache context key for browser-local text requests. */
  contextKey?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  /** Local-only sparse sampler override using native sampling field names. */
  sampling?: SamplingRuntimeOverride;
  stop?: readonly string[];
  signal?: AbortSignal;
  emitTokens?: boolean;
  grammar?: string;
  /** Endpoint-specific request options passed only to gateway endpoints. */
  endpointOptions?: EndpointOptions;
  /** Provider-specific request options passed only to direct provider endpoints. */
  providerOptions?: ProviderOptions;
}

export type ChatInput =
  | readonly ChatMessage[]
  | {
      messages: readonly ChatMessage[];
      media?: Uint8Array[];
    };

/** Endpoint-specific options passed to gateway endpoint implementations. */
export type EndpointOptions = Record<string, unknown>;

/** Direct provider-specific options merged into provider request bodies after typed fields. */
export type ProviderOptions = Record<string, unknown>;

export type ChatOptions = QueryOptions;

export interface InternalTextRequestOptions extends QueryOptions {
  onRequestStarted?: (requestId: number) => void;
  tokenBatchSink?: (batch: TokenBatch) => void;
}

export interface QueryObservation {
  contextKey: string | null;
  status: 'running' | 'success' | 'cancelled' | 'failed';
  wallMs: number | null;
  ttftMs: number | null;
  outputTokens: number | null;
  errorCode?: string;
  errorMessage?: string;
}

export interface RuntimeObservation {
  // Unified Phase & Efficiency Metrics
  ttftMs: number;
  itlAvgMs: number;
  itlP99Ms: number;
  e2eMs: number;

  prefillMs: number;
  decodeMs: number;

  nativeGpuMs: number;
  nativeSyncMs: number;
  nativeLogicMs: number;

  inputTokens: number;
  outputTokens: number;
  cacheMode: KvReuseMode;
  cacheSource: CacheSource;
  cacheHits: number;
  prefillTokens: number;

  decodeTokensPerSecond: number | null;
  e2eTokensPerSecond: number | null;
  prefillTokensPerSecond: number | null;

  // JS Side & Transport Metadata
  execution: {
    mode: 'main-thread' | 'worker';
    workerBacked: boolean;
    tokenPath?: 'none' | 'token-stream';
  };

  /** Request-local ms spent draining native token records into JS token batches. */
  jsTokenDrainMs?: number;
  jsTokenDrainCalls?: number;
}

export interface BackendProfileObservation {
  profilingEnabled: boolean;
  webgpuCompiled: boolean;
  webgpuRegistered: boolean;
  webgpuDeviceCount: number;
  gpuOffloadSupported: boolean;
  availableBackends: Array<{
    name: string;
    deviceCount: number;
  }>;
  devices: Array<{
    name: string;
    description: string;
    type: BackendDeviceType;
    backendName: string;
  }>;
  webgpuAdapter?: WebGpuAdapterInfo | null;
}

/**
 * Identity of the WebGPU adapter observed in the execution context that runs
 * the engine. On hybrid-GPU machines this is the ground truth for which
 * physical GPU inference actually uses; backend name alone cannot tell an
 * integrated adapter from a discrete one.
 */
export interface WebGpuAdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

export type EngineStatus = 'idle' | 'loading' | 'ready' | 'running' | 'error' | 'closed';
export type EngineBackendName = 'cpu' | 'cuda' | 'metal' | 'vulkan' | 'webgpu' | 'unknown';
export type RequestStatus = 'queued' | 'prefill' | 'decode' | 'completed' | 'failed' | 'cancelled';
export type FinishReason = 'stop' | 'length' | 'cancelled' | 'error';

export interface BackendInfo {
  selected: EngineBackendName;
  available: string[];
  adapter: WebGpuAdapterInfo | null;
  devices: Array<{
    id: string | null;
    name: string;
    type: BackendDeviceType;
    memoryTotalBytes?: number;
    memoryFreeBytes?: number;
  }>;
}

export interface RequestState {
  id: string;
  status: RequestStatus;
  inputTokens: number;
  outputTokens: number;
}

export interface EngineStats {
  requestsRunning: number;
  requestsQueued: number;
  requestsCompleted: number;
  requestsFailed: number;
  inputTokens: number;
  outputTokens: number;
  cacheHits: number;
  prefillTokens: number;
  ttftMs: number | null;
  interTokenMs: number | null;
  e2eMs: number | null;
  decodeTokensPerSecond: number | null;
  e2eTokensPerSecond: number | null;
  prefillTokensPerSecond: number | null;
  prefillMs: number;
  decodeMs: number;
  backendMs: number;
  syncMs: number;
  engineOverheadMs: number;
}

export interface EngineState {
  status: EngineStatus;
  model: ModelInfo | null;
  backend: BackendInfo;
  requests: RequestState[];
  stats: EngineStats;
  updatedAt: string;
}

export interface RequestStats {
  inputTokens: number;
  outputTokens: number;
  cacheMode: KvReuseMode | null;
  cacheSource: CacheSource | null;
  cacheHits: number;
  prefillTokens: number | null;
  ttftMs: number | null;
  interTokenMs: number | null;
  e2eMs: number | null;
  decodeTokensPerSecond: number | null;
  e2eTokensPerSecond: number | null;
  prefillTokensPerSecond: number | null;
  prefillMs: number;
  decodeMs: number;
}

export interface GenerationResult {
  id: string;
  text: string;
  finishReason: FinishReason;
  stats: RequestStats;
}

export type { PoolingType };

export interface EmbedOptions {
  /** Explicit endpoint for this request. Omitted requests use the current local endpoint. */
  endpoint?: EndpointRef;
  /** L2-normalize the returned vector. Ignored for `pooling = 'rank'`. Default: true. */
  normalize?: boolean;
  contextKey?: string;
  signal?: AbortSignal;
  /** Endpoint-specific request options passed only to gateway endpoints. */
  endpointOptions?: EndpointOptions;
  /** Provider-specific request options passed only to direct provider endpoints. */
  providerOptions?: ProviderOptions;
}

export interface EmbeddingResult {
  id: string;
  values: number[];
  pooling: PoolingType;
  normalized: boolean;
  stats: RequestStats;
}

export type BrowserTokenBatches = AsyncIterable<TokenBatch>;

export interface BrowserTextRun {
  readonly response: Promise<GenerationResult>;
  readonly tokens: BrowserTokenBatches;
  cancel(reason?: unknown): void;
}

export interface BrowserEmbeddingRun {
  readonly response: Promise<EmbeddingResult>;
  cancel(reason?: unknown): void;
}

/** Stable reference returned by endpoint registration. */
export type EndpointRef =
  | {
      readonly kind: 'local';
      readonly id: string;
    }
  | {
      readonly kind: 'gateway';
      readonly id: string;
    }
  | {
      readonly kind: 'provider';
      readonly id: string;
    };

/** Supplies a short-lived direct provider key for BYOK browser calls. */
export type ProviderKeyProvider = () => string | Promise<string>;

/** Supplies a short-lived authentication value for browser gateway calls. */
export type GatewaySecretProvider = () => string | Promise<string>;

export type GatewayAuthentication =
  | { readonly kind: 'none' }
  | {
      readonly kind: 'bearer';
      readonly value?: string;
      readonly valueProvider?: GatewaySecretProvider;
    }
  | {
      readonly kind: 'header';
      readonly headerName: string;
      readonly value?: string;
      readonly valueProvider?: GatewaySecretProvider;
    };

/** Browser-safe gateway endpoint descriptor. */
export interface GatewayEndpointDescriptor {
  readonly kind: 'gateway';
  /** Target encoded in profile requests. */
  readonly target: string;
  /** Service base URL. */
  readonly baseUrl: string;
  readonly routes?: {
    readonly query?: string;
    readonly chat?: string;
    readonly embed?: string;
  };
  readonly authentication?: GatewayAuthentication;
  readonly staticHeaders?: Readonly<Record<string, string>>;
  /** Request timeout in milliseconds. */
  readonly timeoutMs?: number;
  readonly protocolOptions?: EndpointOptions;
}

export interface LocalEndpointDescriptor {
  readonly kind: 'local';
  readonly source: ModelSource;
  readonly options?: ModelLoadOptions;
}

export interface ProviderStaticHeader {
  readonly name: string;
  readonly value: string;
}

export interface ProviderEndpointDescriptor {
  readonly kind: 'provider';
  readonly provider: 'openai' | 'anthropic' | 'openai_compatible' | 'openai-compatible';
  readonly model: string;
  readonly apiKey?: string;
  readonly keyProvider?: ProviderKeyProvider;
  readonly baseUrl?: string;
  readonly timeoutMs?: number;
  readonly version?: string;
  readonly authHeaderName?: string;
  readonly authHeaderValue?: string;
  readonly authHeaderValueProvider?: ProviderKeyProvider;
  readonly staticHeaders?: readonly ProviderStaticHeader[];
}

export type EndpointDescriptor =
  | LocalEndpointDescriptor
  | GatewayEndpointDescriptor
  | ProviderEndpointDescriptor;

export type EngineEvent =
  | { type: 'state'; state: EngineState }
  | { type: 'load-progress'; loadedBytes: number; totalBytes: number | null; assetName?: string }
  | { type: 'request-started'; requestId: string; streamId: number }
  | { type: 'request-completed'; requestId: string }
  | { type: 'request-failed'; requestId: string; error: string }
  | { type: 'closed' };

export type { TokenEmissionStats, TokenBatch };

export interface ObservabilitySnapshot {
  mode: ObservabilityMode;
  state: ObservabilityState;
  updatedAt: string;
  model: ModelInfo | null;
  query: QueryObservation | null;
  runtime?: RuntimeObservation;
  profile?: BackendProfileObservation;
}

export interface ObservabilityEvent {
  type: ObservabilityEventType;
  snapshot: ObservabilitySnapshot;
}

export interface EngineObservability {
  current(): ObservabilitySnapshot;
  subscribe(listener: (event: ObservabilityEvent) => void): () => void;
}

export interface ModelLifecycleService {
  load(source: ModelSource, options?: ModelLoadOptions): Promise<ModelInfo>;
  unload(): void | Promise<void>;
  current(): ModelInfo | null;
  list(): Promise<ModelInfo[]>;
  remove(id: string): Promise<void>;
  runQuery(
    input: QueryInput,
    options: InternalTextRequestOptions
  ): Promise<GenerationResult>;
  runChat(
    input: ChatInput,
    options: InternalTextRequestOptions
  ): Promise<GenerationResult>;
  runEmbedding(input: string, options: EmbedOptions): Promise<EmbeddingResult>;
  state(): EngineState;
  subscribeEvents(listener: (event: EngineEvent) => void): () => void;
  currentObservability(): ObservabilitySnapshot;
  subscribeObservability(listener: (event: ObservabilityEvent) => void): () => void;
  close(): void | Promise<void>;
}

export interface SippClient {
  readonly observability: EngineObservability;
  /** Register or replace a local, gateway, or direct provider endpoint. */
  add(id: string, descriptor: EndpointDescriptor): Promise<EndpointRef>;
  /** Return the currently loaded local model, if one is active. */
  currentLocal(): ModelInfo | null;
  /** List installed local models. */
  listLocal(): Promise<ModelInfo[]>;
  /** Remove an installed local model by id. */
  removeLocal(id: string): Promise<void>;
  query(input: QueryInput, options?: QueryOptions): BrowserTextRun;
  chat(input: ChatInput, options?: ChatOptions): BrowserTextRun;
  embed(input: string, options?: EmbedOptions): BrowserEmbeddingRun;
  state(): EngineState;
  subscribeEvents(listener: (event: EngineEvent) => void): () => void;
  close(): Promise<void>;
}

export type QueryErrorCode =
  | 'ENGINE_CLOSED'
  | 'MODEL_NOT_READY'
  | 'MODEL_NOT_FOUND'
  | 'MODEL_BROKEN'
  | 'UNSUPPORTED_OPERATION'
  | 'INVALID_MODEL_SOURCE'
  | 'INVALID_MODEL_PAIRING'
  | 'STORAGE_UNAVAILABLE'
  | 'STORAGE_QUOTA_EXCEEDED'
  | 'STORAGE_CORRUPT'
  | 'REMOTE_METADATA_UNAVAILABLE'
  | 'REMOTE_LOAD_FAILED'
  | 'STREAMING_UNAVAILABLE'
  | 'QUERY_FAILED';

export class QueryError extends Error {
  public readonly code: QueryErrorCode;
  /** HTTP status returned by a gateway or provider endpoint, when available. */
  public readonly status?: number;
  /** Error code returned by the endpoint protocol. */
  public readonly protocolCode?: string;
  /** Direct provider label for provider endpoint failures. */
  public readonly provider?: string;
  /** Provider error code returned by the upstream provider. */
  public readonly providerCode?: string;
  /** Endpoint request id returned by the protocol. */
  public readonly requestId?: string;
  /** Retry delay in milliseconds returned by `retry-after-ms` or `retry-after`. */
  public readonly retryAfterMs?: number;

  constructor(code: QueryErrorCode, message: string, options?: QueryErrorOptions) {
    super(message, options);
    this.name = 'QueryError';
    this.code = code;
    this.status = options?.status;
    this.protocolCode = options?.protocolCode;
    this.provider = options?.provider;
    this.providerCode = options?.providerCode;
    this.requestId = options?.requestId;
    this.retryAfterMs = options?.retryAfterMs;
  }
}

/** Optional structured metadata attached to browser query failures. */
export interface QueryErrorOptions {
  readonly cause?: unknown;
  /** HTTP status returned by a gateway or provider endpoint, when available. */
  readonly status?: number;
  /** Error code returned by the endpoint protocol. */
  readonly protocolCode?: string;
  /** Direct provider label for provider endpoint failures. */
  readonly provider?: string;
  /** Provider error code returned by the upstream provider. */
  readonly providerCode?: string;
  /** Endpoint request id returned by the protocol. */
  readonly requestId?: string;
  /** Retry delay in milliseconds returned by `retry-after-ms` or `retry-after`. */
  readonly retryAfterMs?: number;
}

export interface AssetRecord {
  id: string;
  kind: ModelAssetKind;
  name: string;
  bytes: number;
  storagePath: string;
  sourceUrl?: string;
  sourceEtag?: string;
  sourceLastModified?: string;
  sourceBytes?: number;
  sourcePartIndex?: number;
  sourcePartCount?: number;
  sourceFileName?: string;
  sourceFileLastModified?: number;
  refCount: number;
  createdAt: string;
  inspection?: AssetInspection;
}

export interface ModelEntry {
  id: string;
  name: string;
  modality: ModelModality;
  status: ModelStatus;
  modelAssetIds: string[];
  projectorAssetId?: string;
  runtimeFingerprint?: string;
  createdAt: string;
  updatedAt: string;
  lastLoadedAt?: string;
}

export interface RegistryManifest {
  version: 3;
  projectorIndexRevision: number;
  assets: Record<string, AssetRecord>;
  models: Record<string, ModelEntry>;
}

export interface LoadedModelState {
  id: string;
  assetFingerprint: string;
  runtimeFingerprint: string;
}
