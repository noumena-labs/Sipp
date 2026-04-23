import type { InferenceInitConfig, PromptFormatMode } from '../types.js';

export type ModelModality = 'text' | 'vision';
export type ModelStatus = 'ready' | 'needs_projector' | 'broken';
export type ModelSourceKind = 'remote' | 'local';

export type ModelAssetKind = 'model' | 'projector' | 'shard';

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
  enableRuntimeObservability?: boolean;
  enableBackendProfiling?: boolean;
  multimodalUseGpu?: boolean;
  debugCompareMultimodalEmbeddings?: boolean;
  imageMinTokens?: number;
  imageMaxTokens?: number;
  sampling?: InferenceInitConfig['sampling'];
}

export interface ModelLoadOptions {
  signal?: AbortSignal;
  onProgress?: (progress: ModelLoadProgress) => void;
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
  id: string;
  name: string;
  modality: ModelModality;
  status: ModelStatus;
  source: ModelSourceKind;
  bytes: number;
  loaded: boolean;
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
  format?: 'auto' | 'raw';
  signal?: AbortSignal;
  onToken?: (token: string) => void;
}

export type QueryErrorCode =
  | 'ENGINE_CLOSED'
  | 'MODEL_NOT_READY'
  | 'MODEL_NOT_FOUND'
  | 'MODEL_BROKEN'
  | 'INVALID_MODEL_SOURCE'
  | 'INVALID_MODEL_PAIRING'
  | 'STORAGE_UNAVAILABLE'
  | 'STORAGE_CORRUPT'
  | 'REMOTE_METADATA_UNAVAILABLE'
  | 'REMOTE_LOAD_FAILED'
  | 'QUERY_FAILED';

export class QueryError extends Error {
  public readonly code: QueryErrorCode;

  constructor(code: QueryErrorCode, message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = 'QueryError';
    this.code = code;
    if (options != null && 'cause' in Error.prototype) {
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
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
  assets: Record<string, AssetRecord>;
  models: Record<string, ModelEntry>;
}

export interface LoadedModelState {
  id: string;
  runtimeFingerprint: string;
}

export function toPromptFormatMode(format: QueryOptions['format']): PromptFormatMode {
  return format === 'raw' ? 'raw' : 'auto-chat';
}
