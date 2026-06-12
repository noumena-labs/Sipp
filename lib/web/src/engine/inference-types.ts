export type FlashAttentionMode = 'auto' | 'enabled' | 'disabled';
export type SchedulerPolicyMode = 'latency_first' | 'balanced' | 'throughput_first';
export type EngineExecutionMode = 'main-thread' | 'worker';
export type BackendDeviceType = 'cpu' | 'gpu' | 'igpu' | 'accel' | 'unknown';

export type GpuLayerConfig = 'auto' | 'all' | { count: number };
export type SplitMode = 'none' | 'layer' | 'row' | 'tensor';
export type KvCacheType = 'f16' | 'f32' | 'q8_0' | 'q4_0' | 'q4_1' | 'iq4_nl' | 'q5_0' | 'q5_1';
export type RopeScaling = 'none' | 'linear' | 'yarn';
export type KvReuseMode = 'disabled' | 'live_slot_prefix' | 'state_snapshot' | 'live_slot_and_snapshot';
export type CacheSource = 'none' | 'live' | 'snapshot';
export type PoolingType = 'unspecified' | 'none' | 'mean' | 'cls' | 'last' | 'rank';
export type SamplerStage =
  | 'dry'
  | 'top_k'
  | 'typical_p'
  | 'top_p'
  | 'top_n_sigma'
  | 'min_p'
  | 'xtc'
  | 'temperature'
  | 'infill'
  | 'penalties'
  | 'adaptive_p';

export interface ModelPlacementConfig {
  devices?: string[];
  gpu_layers?: GpuLayerConfig;
  split_mode?: SplitMode;
  main_gpu?: number;
  tensor_split?: number[];
  use_mmap?: boolean;
  use_mlock?: boolean;
  fit_params?: boolean;
  fit_params_min_ctx?: number;
  fit_params_target_bytes?: number[];
  check_tensors?: boolean;
  no_extra_bufts?: boolean;
  no_host?: boolean;
}

export interface ContextRuntimeConfig {
  n_ctx?: number;
  n_batch?: number;
  n_ubatch?: number;
  n_parallel?: number;
  n_threads?: number;
  n_threads_batch?: number;
  flash_attention?: FlashAttentionMode;
  kv_unified?: boolean;
  cache_type_k?: KvCacheType;
  cache_type_v?: KvCacheType;
  offload_kqv?: boolean;
  op_offload?: boolean;
  swa_full?: boolean;
  warmup?: boolean;
  rope_scaling?: RopeScaling;
  rope_freq_base?: number;
  rope_freq_scale?: number;
  yarn_orig_ctx?: number;
  yarn_ext_factor?: number;
  yarn_attn_factor?: number;
  yarn_beta_fast?: number;
  yarn_beta_slow?: number;
  embeddings?: boolean;
  pooling?: PoolingType;
}

export interface LogitBiasConfig {
  token: number;
  bias: number;
}

export interface SamplingRuntimeConfig {
  samplers?: SamplerStage[];
  seed?: number;
  top_k?: number;
  top_p?: number;
  min_p?: number;
  typical_p?: number;
  xtc_probability?: number;
  xtc_threshold?: number;
  top_n_sigma?: number;
  temperature?: number;
  dynatemp_range?: number;
  dynatemp_exponent?: number;
  repeat_last_n?: number;
  repeat_penalty?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  dry_multiplier?: number;
  dry_base?: number;
  dry_allowed_length?: number;
  dry_penalty_last_n?: number;
  dry_sequence_breakers?: string[];
  mirostat?: number;
  mirostat_tau?: number;
  mirostat_eta?: number;
  min_keep?: number;
  n_probs?: number;
  logit_bias?: LogitBiasConfig[];
  ignore_eos?: boolean;
  grammar_lazy?: boolean;
  preserved_tokens?: number[];
  backend_sampling?: boolean;
}

export interface RequestSamplingPatch {
  temperature?: number;
  top_p?: number;
}

export interface SchedulerPolicyConfig {
  mode?: SchedulerPolicyMode;
  decode_token_reserve?: number;
  enable_adaptive_prefill_chunking?: boolean;
}

export interface SchedulerRuntimeConfig {
  continuous_batching?: boolean;
  policy?: SchedulerPolicyConfig;
  prefill_chunk_size?: number;
  max_running_requests?: number;
  max_queued_requests?: number;
}

export interface CacheRuntimeConfig {
  mode?: KvReuseMode;
  retained_prefix_tokens?: number;
  snapshot_interval_tokens?: number;
  max_snapshot_entries?: number;
  max_snapshot_bytes?: number;
}

export interface MultimodalRuntimeConfig {
  projector_path?: string;
  use_gpu?: boolean;
  image_min_tokens?: number;
  image_max_tokens?: number;
}

export interface ResidencyRuntimeConfig {
  max_gpu_models_per_device?: number;
  allow_cpu_models_while_gpu_loaded?: boolean;
  require_gpu_lease?: boolean;
  gpu_memory_safety_margin_bytes?: number;
}

export interface ObservabilityRuntimeConfig {
  runtime_metrics?: boolean;
  backend_profiling?: boolean;
}

export interface NativeRuntimeConfig {
  placement?: ModelPlacementConfig;
  context?: ContextRuntimeConfig;
  sampling?: SamplingRuntimeConfig;
  scheduler?: SchedulerRuntimeConfig;
  cache?: CacheRuntimeConfig;
  multimodal?: MultimodalRuntimeConfig;
  residency?: ResidencyRuntimeConfig;
  observability?: ObservabilityRuntimeConfig;
}

export interface PromptOptions {
  nTokens?: number;
  signal?: AbortSignal;
  emitTokens?: boolean;
  tokenBatchSink?: (batch: TokenBatch) => void;
  media?: Uint8Array[];
  stop?: readonly string[];
  sampling?: RequestSamplingPatch;
  /**
   * Optional GBNF grammar source applied to the sampler for this request.
   * When provided, the native runtime constrains token sampling to strings
   * matching the grammar. Must be <= 64 KiB when UTF-8 encoded.
   */
  grammar?: string;
  onRequestStarted?: (requestId: number) => void;
}

export interface EmbedRuntimeOptions {
  normalize?: boolean;
  signal?: AbortSignal;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface TokenEmissionStats {
  framesSent: number;
  bytesSent: number;
  batchesSent: number;
  drainMs: number;
  drainCalls: number;
}

export interface TokenBatch {
  requestId: string;
  streamId: number;
  sequenceStart: number;
  text: string;
  frameCount: number;
  byteCount: number;
  stats: TokenEmissionStats;
}

export type GenerateRequestId = number;

export interface GenerateRequest {
  contextKey: string;
  promptText: string;
  maxOutputTokens: number;
  media?: Uint8Array[];
  stop?: readonly string[];
  sampling?: RequestSamplingPatch;
  /** Optional GBNF grammar source (see {@link PromptOptions.grammar}). */
  grammar?: string;
}

export interface EmbeddingOutput {
  values: number[];
  pooling: PoolingType;
  normalized: boolean;
}

export interface RequestObservabilityMetrics {
  /**
   * Time to first token: enqueue -> first sampled token. Sampled when
   * llama_sampler_sample produces the first token, not when JS receives it.
   */
  ttftMs: number;
  /** Average inter-token latency (ms between consecutive emitted tokens). */
  itlAvgMs: number;
  /** Tail inter-token latency reported by the active runtime. */
  itlP99Ms: number;
  /** End-to-end latency: enqueue -> completion. */
  e2eMs: number;

  /** Wall-clock summed over ticks where this request had a prefill contribution. */
  prefillMs: number;
  /** Wall-clock summed over ticks where this request had a decode contribution. */
  decodeMs: number;

  /**
   * Raw wall-clock window around llama_decode. Backend synchronization is
   * reported separately as nativeSyncMs.
   */
  nativeGpuMs: number;
  /** Cumulative time spent in backend synchronization (llama_synchronize). */
  nativeSyncMs: number;
  /** Internal engine logic overhead (scheduling, batching, bookkeeping). */
  nativeLogicMs: number;

  /** Total number of tokens processed in the prompt. */
  inputTokens: number;
  /** Total number of tokens generated in the response. */
  outputTokens: number;
  /** Runtime cache mode used for this request or aggregate sample. */
  cacheMode: KvReuseMode;
  /** Request-level cache source; aggregate metrics report `none`. */
  cacheSource: CacheSource;
  /** Number of tokens reused from KV cache (LCP / prefix hits). */
  cacheHits: number;
  /** Number of tokens actually processed by the GPU during prefill. */
  prefillTokens: number;
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

export interface BackendCompiledInfo {
  readonly cuda?: boolean;
  readonly metal?: boolean;
  readonly vulkan?: boolean;
  readonly openmp?: boolean;
  readonly webgpu?: boolean;
}

export interface BackendObservability {
  profilingEnabled: boolean;
  webgpuCompiled: boolean;
  webgpuRegistered: boolean;
  webgpuDeviceCount: number;
  gpuOffloadSupported: boolean;
  engineInitialized: boolean;
  readonly dynamicBackendLoading?: boolean;
  readonly compiled?: BackendCompiledInfo;
  readonly backendCount?: number;
  readonly deviceCount?: number;
  availableBackends: BackendRegistryInfo[];
  devices: BackendDeviceInfo[];
}

export interface TransportObservability {
  executionMode: EngineExecutionMode;
  workerBacked: boolean;
  enabled: boolean;
  activeTokenTransport?: 'none' | 'token-stream';
  activeTokenEmission?: boolean;
  tokenDrainCalls?: number;
  tokenDrainMs?: number;
}

export function withDerivedObservabilityMetrics<T extends RequestObservabilityMetrics>(
  metrics: T
): T & {
  decodeTokensPerSecond: number | null;
  e2eTokensPerSecond: number | null;
  prefillTokensPerSecond: number | null;
} {
  return {
    ...metrics,
    decodeTokensPerSecond:
      metrics.decodeMs > 0 && metrics.outputTokens > 0
        ? (metrics.outputTokens / metrics.decodeMs) * 1000
        : null,
    e2eTokensPerSecond:
      metrics.e2eMs > 0 && metrics.outputTokens > 0
        ? (metrics.outputTokens / metrics.e2eMs) * 1000
        : null,
    prefillTokensPerSecond:
      metrics.prefillMs >= 0.1 && metrics.prefillTokens >= 1
        ? (metrics.prefillTokens / metrics.prefillMs) * 1000
        : null,
  };
}

interface BaseGenerateResponse {
  requestId: number;
  completed: boolean;
  failed: boolean;
  cancelled: boolean;
  errorMessage?: string | null;
  /** Performance metrics for the request. */
  observability?: RequestObservabilityMetrics | null;
}

export interface TextGenerateResponse extends BaseGenerateResponse {
  outputText: string;
  embedding?: undefined;
}

export interface EmbeddingGenerateResponse extends BaseGenerateResponse {
  embedding: EmbeddingOutput;
  outputText?: undefined;
}

export type GenerateResponse = TextGenerateResponse | EmbeddingGenerateResponse;
