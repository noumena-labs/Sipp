export type FlashAttentionMode = "auto" | "enabled" | "disabled";
export type PromptFormatMode = "auto-chat" | "raw";
export type BackendDeviceType = "cpu" | "gpu" | "igpu" | "accel" | "unknown";
export type SchedulerPolicyMode = "latency-first" | "balanced" | "throughput-first";

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
  schedulerPolicy?: SchedulerPolicyMode;
  decodeTokenReserve?: number;
  adaptivePrefillChunking?: boolean;
}

export interface PromptGenerationOptions {
  nTokens?: number;
  promptFormat?: PromptFormatMode;
}

export interface PromptStreamOptions extends PromptGenerationOptions {
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
