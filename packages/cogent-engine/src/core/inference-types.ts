import type { RuntimeObservabilityMetrics } from '../observability/runtime-observability.js';

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
}

export interface PromptOptions {
  nTokens?: number;
  promptFormat?: PromptFormatMode;
  signal?: AbortSignal;
  onToken?: (token: string) => void;
}

export type GenerateRequestId = number;

export interface GenerateRequest {
  contextKey: string;
  promptText: string;
  maxOutputTokens: number;
  promptFormat: PromptFormatMode;
}

export interface GenerateResponse {
  requestId: number;
  completed: boolean;
  failed: boolean;
  cancelled: boolean;
  outputText: string;
  errorMessage?: string | null;
  runtimeObservability?: RuntimeObservabilityMetrics | null;
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
