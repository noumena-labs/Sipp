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
  /**
   * Optional GBNF grammar source applied to the sampler for this request.
   * When provided, the native runtime constrains token sampling to strings
   * matching the grammar. Must be <= 64 KiB when UTF-8 encoded.
   */
  grammar?: string;
  /**
   * Structured chat messages. When provided, the runtime applies the
   * model's native chat template (via llama.cpp's `applyChatTemplate`) to
   * build the final prompt text, and `promptText` is ignored for
   * formatting purposes. This is the correct way to feed a multi-turn
   * conversation: the template emits the model's own role and turn-end
   * tokens, which allows generation to stop naturally at EOS.
   *
   * Requires the loaded model to expose a chat template (see
   * {@link EngineRuntime.getChatTemplate}). Incompatible with `media` in
   * the current runtime.
   */
  messages?: ChatMessage[];
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
