import type { AssetInspection } from '../model-bundle/model-bundle-types.js';
import type { ChatMessage } from '../core/inference-types.js';
import type { BackendDeviceType, InferenceInitConfig } from '../types.js';

export type ModelModality = 'text' | 'vision';
export type ModelStatus = 'ready' | 'needs_projector' | 'broken';
export type ModelSourceKind = 'remote' | 'local';

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
  phase: 'metadata' | 'download' | 'store' | 'load';
  loadedBytes: number;
  totalBytes: number | null;
  percent: number | null;
  assetName?: string;
}

export interface ModelRuntimeOptions {
  nCtx?: number;
  nBatch?: number;
  nUbatch?: number;
  nSeqMax?: number;
  nThreads?: number;
  nThreadsBatch?: number;
  /** `-1` keeps automatic/full accelerator offload; `0` forces CPU-only. */
  nGpuLayers?: number;
  flashAttention?: InferenceInitConfig['flashAttention'];
  kvUnified?: boolean;
  maxCachedSessions?: number;
  retainedPrefixTokens?: number;
  prefillChunkSize?: number;
  prefixCacheIntervalTokens?: number;
  maxPrefixCacheEntries?: number;
  schedulerPolicy?: InferenceInitConfig['schedulerPolicy'];
  decodeTokenReserve?: number;
  adaptivePrefillChunking?: boolean;
  multimodalUseGpu?: boolean;
  imageMinTokens?: number;
  imageMaxTokens?: number;
  sampling?: InferenceInitConfig['sampling'];
}

export interface ModelLoadOptions {
  signal?: AbortSignal;
  onProgress?: (progress: ModelLoadProgress) => void;
  observability?: ObservabilityMode;
  runtime?: ModelRuntimeOptions;
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
  onToken?: (token: string) => void;
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
  outputTokenCount: number | null;
  errorCode?: string;
  errorMessage?: string;
}

export interface RuntimeObservation {
  totalMs: number;
  ttftMs: number;
  tokensPerSecond: number | null;
  inputTokenCount: number;
  outputTokenCount: number;
  promptEvalMs?: number;
  decodeEvalMs?: number;
  sampleMs?: number;
  queueDelayMs?: number;
  meanItlMs?: number;
  tailItlMs?: number;
  promptEvalTokens?: number;
  decodeEvalCount?: number;
  batchParticipationCount?: number;
  decodeFirstTickCount?: number;
  chunkedPrefillTickCount?: number;
  mixedWorkloadTickCount?: number;
  lcpReuseTokens?: number;
  prefixCacheRestoreTokens?: number;
  prefixCacheHitCount?: number;
  prefixCacheStoreCount?: number;
  execution: {
    mode: 'main-thread' | 'worker';
    workerBacked: boolean;
    tokenPath?: 'none' | 'runtime-event';
  };
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

export type QueryErrorCode =
  | 'ENGINE_CLOSED'
  | 'MODEL_NOT_READY'
  | 'MODEL_NOT_FOUND'
  | 'MODEL_BROKEN'
  | 'INVALID_MODEL_SOURCE'
  | 'INVALID_MODEL_PAIRING'
  | 'STORAGE_UNAVAILABLE'
  | 'STORAGE_QUOTA_EXCEEDED'
  | 'STORAGE_CORRUPT'
  | 'REMOTE_METADATA_UNAVAILABLE'
  | 'REMOTE_LOAD_FAILED'
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
  hash: string;
  bytes: number;
  storagePath: string;
  sourceUrl?: string;
  sourceEtag?: string;
  sourceLastModified?: string;
  refCount: number;
  createdAt: string;
  inspection?: AssetInspection;
}

export type ModelPairingReasonCode =
  | 'BASE_NOT_VISION'
  | 'NO_MATCH'
  | 'MULTIPLE_MATCHES'
  | 'MISSING_METADATA';

export interface ModelPairingState {
  state: 'resolved' | 'unresolved';
  checkedProjectorIndexRevision: number;
  compatibleVisionProjectorTypes: string[];
  reasonCode?: ModelPairingReasonCode;
  updatedAt: string;
}

export interface ModelEntry {
  id: string;
  name: string;
  modality: ModelModality;
  status: ModelStatus;
  modelAssetIds: string[];
  projectorAssetId?: string;
  pairing?: ModelPairingState;
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
