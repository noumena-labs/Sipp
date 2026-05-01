import type { RequestObservabilityMetrics } from '../observability/runtime-observability.js';

export type FlashAttentionMode = 'auto' | 'enabled' | 'disabled';
export type PromptFormatMode = 'raw' | 'auto-chat';
export type SchedulerPolicyMode = 'latency-first' | 'balanced' | 'throughput-first';
export type EngineExecutionMode = 'main-thread' | 'worker';

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
  /**
   * Number of transformer layers to offload to the accelerator. `-1` keeps
   * llama.cpp's automatic/full offload behavior; `0` forces CPU-only loading.
   */
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
  /**
   * Optional GBNF grammar source applied to the sampler for this request.
   * When provided, the native runtime constrains token sampling to strings
   * matching the grammar. Must be <= 64 KiB when UTF-8 encoded.
   */
  grammar?: string;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export type GenerateRequestId = number;

export interface GenerateRequest {
  contextKey: string;
  promptText: string;
  maxOutputTokens: number;
  promptFormat: PromptFormatMode;
  media?: Uint8Array[];
  /** Optional GBNF grammar source (see {@link PromptOptions.grammar}). */
  grammar?: string;
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
