import type { RequestObservabilityMetrics } from '../observability/runtime-observability.js';

export type FlashAttentionMode = 'auto' | 'enabled' | 'disabled';
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
   * @internal — reserved for the worker streaming path.
   *
   * Invoked synchronously inside the runtime once a request has been
   * enqueued and assigned its native `GenerateRequestId`, BEFORE the
   * request begins decoding.  The worker entry uses this hook to publish
   * a `streaming-claim` postMessage to main, which lets the main-thread
   * SAB ring reader translate native request ids back to its own callIds
   * when dispatching streamed tokens to user `onToken` callbacks.
   *
   * Not part of the public API surface — added to PromptOptions purely so
   * worker-internal code can pass it through `runtime.enqueueQuery`
   * without a type-cast.  Public consumers should ignore it.
   */
  __internalRequestStarted?: (requestId: number) => void;
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
  /** Performance metrics for the request. */
  observability?: RequestObservabilityMetrics | null;
}
