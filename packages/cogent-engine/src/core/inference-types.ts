import type { RequestObservabilityMetrics } from '../observability/runtime-observability.js';

export type FlashAttentionMode = 'auto' | 'enabled' | 'disabled';
export type PromptFormatMode = 'auto-chat' | 'raw';
export type SchedulerPolicyMode = 'latency-first' | 'balanced' | 'throughput-first';
export type EngineExecutionMode = 'main-thread' | 'worker';
export type ModelLoadSourceKind = 'url' | 'file' | 'buffer';
export type ModelLoadReuseMode =
  | 'network'
  | 'file-read'
  | 'persistent-cache'
  | 'page-local-reuse'
  | 'buffer';

export interface SamplingConfig {
  repeatLastN?: number;
  repeatPenalty?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  temperature?: number;
  seed?: number;
}

export interface InferenceInitConfig {
  nCtx?: number;
  nBatch?: number;
  nUbatch?: number;
  nSeqMax?: number;
  nThreads?: number;
  nThreadsBatch?: number;
  nGpuLayers?: number;
  flashAttention?: FlashAttentionMode;
  kvUnified?: boolean;
  maxCachedSessions?: number;
  retainedPrefixTokens?: number;
  prefillChunkSize?: number;
  prefixCacheIntervalTokens?: number;
  maxPrefixCacheEntries?: number;
  schedulerPolicy?: SchedulerPolicyMode;
  decodeTokenReserve?: number;
  adaptivePrefillChunking?: boolean;
  enableRuntimeObservability?: boolean;
  enableBackendProfiling?: boolean;
  multimodalProjectorPath?: string;
  multimodalUseGpu?: boolean;
  debugCompareMultimodalEmbeddings?: boolean;
  imageMinTokens?: number;
  imageMaxTokens?: number;
  sampling?: SamplingConfig;
}

export interface PromptOptions {
  nTokens?: number;
  promptFormat?: PromptFormatMode;
  signal?: AbortSignal;
  onToken?: (token: string) => void;
  media?: Uint8Array[];
}

export type GenerateRequestId = number;

export interface GenerateRequest {
  contextKey: string;
  promptText: string;
  maxOutputTokens: number;
  promptFormat: PromptFormatMode;
  media?: Uint8Array[];
}

export interface GenerateResponse {
  requestId: number;
  completed: boolean;
  failed: boolean;
  cancelled: boolean;
  outputText: string;
  errorMessage?: string | null;
  requestObservability?: RequestObservabilityMetrics | null;
  runtimeObservability?: RequestObservabilityMetrics | null;
}

export interface ModelLoadInfo {
  sourceKind: ModelLoadSourceKind;
  reuseMode: ModelLoadReuseMode;
  modelPath: string;
  fileName: string;
  byteLength: number | null;
  persistentCacheEnabled: boolean;
  persistentCacheKey: string | null;
  persistentCacheHit: boolean;
  persistentCacheStored: boolean;
}
