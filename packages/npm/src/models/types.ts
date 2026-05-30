import type {
  BackendDeviceType,
  ChatMessage,
  NativeRuntimeConfig,
  PoolingType,
  StreamStats,
  TokenBatch,
  TokenFlushMode,
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
  /** Installed model id persisted in OPFS. Pass this back to engine.models.load(id) to reload it. */
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
  session?: string;
  maxTokens?: number;
  signal?: AbortSignal;
  streamTokens?: boolean;
  tokenFlush?: TokenFlushMode;
  grammar?: string;
}

export type ChatInput =
  | readonly ChatMessage[]
  | {
      messages: readonly ChatMessage[];
      media?: Uint8Array[];
    };

export type ChatOptions = QueryOptions;

export interface QueryObservation {
  session: string | null;
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
  cacheHits: number;
  prefillTokens: number;

  tokensPerSecond: number | null;
  prefillTokensPerSecond: number | null;

  // JS Side & Transport Metadata
  execution: {
    mode: 'main-thread' | 'worker';
    workerBacked: boolean;
    tokenPath?: 'none' | 'streaming-buffer' | 'callback';
  };

  /** Cumulative ms spent in `_ce_yield_drain` (SAB ring writes from native scratch). */
  jsStreamingDrainMs?: number;
  jsStreamingDrainCount?: number;
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
}

export type EngineStatus = 'idle' | 'loading' | 'ready' | 'running' | 'error' | 'closed';
export type EngineBackendName = 'cpu' | 'cuda' | 'metal' | 'vulkan' | 'webgpu' | 'unknown';
export type RequestStatus = 'queued' | 'prefill' | 'decode' | 'completed' | 'failed' | 'cancelled';
export type FinishReason = 'stop' | 'length' | 'cancelled' | 'error';

export interface BackendInfo {
  selected: EngineBackendName;
  available: string[];
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
  tokensPerSecond: number | null;
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
  cacheHits: number;
  ttftMs: number | null;
  interTokenMs: number | null;
  e2eMs: number | null;
  tokensPerSecond: number | null;
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
  /** L2-normalize the returned vector. Ignored for `pooling = 'rank'`. Default: true. */
  normalize?: boolean;
  contextKey?: string;
  signal?: AbortSignal;
}

export interface EmbeddingResult {
  id: string;
  values: number[];
  pooling: PoolingType;
  normalized: boolean;
  stats: RequestStats;
}

export interface BrowserTextRun {
  readonly response: Promise<GenerationResult>;
  readonly tokens: AsyncIterable<TokenBatch>;
  cancel(reason?: unknown): void;
}

export interface BrowserEmbeddingRun {
  readonly response: Promise<EmbeddingResult>;
  cancel(reason?: unknown): void;
}

export type EngineEvent =
  | { type: 'state'; state: EngineState }
  | { type: 'load-progress'; loadedBytes: number; totalBytes: number | null; assetName?: string }
  | { type: 'request-started'; requestId: string; streamId: number }
  | { type: 'request-completed'; requestId: string }
  | { type: 'request-failed'; requestId: string; error: string }
  | { type: 'closed' };

export type { StreamStats, TokenBatch, TokenFlushMode };

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
  query(input: QueryInput, options?: QueryOptions): BrowserTextRun;
  chat(input: ChatInput, options?: ChatOptions): BrowserTextRun;
  embed(input: string, options?: EmbedOptions): BrowserEmbeddingRun;
  state(): EngineState;
  subscribeEvents(listener: (event: EngineEvent) => void): () => void;
  currentObservability(): ObservabilitySnapshot;
  subscribeObservability(listener: (event: ObservabilityEvent) => void): () => void;
  close(): void | Promise<void>;
}

export interface CogentClient {
  readonly models: {
    load(source: ModelSource, options?: ModelLoadOptions): Promise<ModelInfo>;
    current(): ModelInfo | null;
    list(): Promise<ModelInfo[]>;
    remove(id: string): Promise<void>;
  };
  readonly observability: EngineObservability;
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

  constructor(code: QueryErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'QueryError';
    this.code = code;
  }
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
