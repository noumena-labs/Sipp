export type FlashAttentionMode = "auto" | "enabled" | "disabled";
export type PromptFormatMode = "auto-chat" | "raw";
export type BackendDeviceType = "cpu" | "gpu" | "igpu" | "accel" | "unknown";
export type SchedulerPolicyMode = "latency-first" | "balanced" | "throughput-first";
export type EngineExecutionMode = "main-thread" | "worker";
export type ModelLoadSourceKind = "url" | "file" | "buffer";
export type ModelLoadReuseMode =
  | "network"
  | "file-read"
  | "persistent-cache"
  | "page-local-reuse"
  | "buffer";

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
  perf?: PromptPerformanceStats;
}

export interface PromptPerformanceStats {
  totalMs: number;
  promptEvalMs: number;
  decodeEvalMs: number;
  sampleMs: number;
  queueDelayMs: number;
  ttftMs: number;
  meanItlMs: number;
  tailItlMs: number;
  e2elMs: number;
  inputTokenCount: number;
  promptEvalTokens: number;
  decodeEvalCount: number;
  sampleCount: number;
  outputTokenCount: number;
  schedulerTickCount: number;
  batchParticipationCount: number;
  decodeFirstTickCount: number;
  chunkedPrefillTickCount: number;
  mixedWorkloadTickCount: number;
  lcpReuseTokens: number;
  prefixCacheRestoreTokens: number;
  prefixCacheHitCount: number;
  prefixCacheStoreCount: number;
}

export interface BackendDeviceCapabilities {
  async: boolean;
  hostBuffer: boolean;
  bufferFromHostPtr: boolean;
  events: boolean;
}

export interface BackendDeviceInfo {
  name: string;
  description: string;
  type: BackendDeviceType;
  backendName: string;
  deviceId: string | null;
  memoryFreeBytes: number;
  memoryTotalBytes: number;
  capabilities: BackendDeviceCapabilities;
}

export interface BackendRegistryInfo {
  name: string;
  deviceCount: number;
}

export interface BackendInfo {
  webgpuCompiled: boolean;
  webgpuRegistered: boolean;
  webgpuDeviceCount: number;
  gpuOffloadSupported: boolean;
  engineInitialized: boolean;
  availableBackends: BackendRegistryInfo[];
  devices: BackendDeviceInfo[];
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

export interface TransportInfo {
  executionMode: EngineExecutionMode;
  workerBacked: boolean;
  backpressureEnabled: boolean;
  maxBufferedTokenCount: number;
  flushIntervalMs: number;
  flushCount: number;
  coalescedTokenCount: number;
  maxObservedBufferedTokenCount: number;
}
