import type { RequestObservabilityMetrics } from '../observability/runtime-observability.js';

export type FlashAttentionMode = 'auto' | 'enabled' | 'disabled';
export type SchedulerPolicyMode = 'latency-first' | 'balanced' | 'throughput-first';
export type EngineExecutionMode = 'main-thread' | 'worker';

export type GpuLayerConfig = 'auto' | 'all' | number;
export type SplitMode = 'none' | 'layer' | 'row' | 'tensor';
export type KvCacheType = 'f16' | 'f32' | 'q8_0' | 'q4_0' | 'q4_1' | 'iq4_nl' | 'q5_0' | 'q5_1';
export type RopeScaling = 'none' | 'linear' | 'yarn';
export type KvReuseMode = 'disabled' | 'live-slot-prefix' | 'state-snapshot' | 'live-slot-and-snapshot';
export type CacheKeyPolicy = 'context-key' | 'prompt-hash';
export type SamplerStage =
  | 'dry'
  | 'top-k'
  | 'typical-p'
  | 'top-p'
  | 'top-n-sigma'
  | 'min-p'
  | 'xtc'
  | 'temperature'
  | 'infill'
  | 'penalties'
  | 'adaptive-p';

export interface ModelPlacementConfig {
  devices?: string[];
  gpuLayers?: GpuLayerConfig;
  splitMode?: SplitMode;
  mainGpu?: number;
  tensorSplit?: number[];
  useMmap?: boolean;
  useMlock?: boolean;
  fitParams?: boolean;
  fitParamsMinCtx?: number;
  fitParamsTargetBytes?: number[];
  checkTensors?: boolean;
  noExtraBufts?: boolean;
  noHost?: boolean;
}

export interface ContextRuntimeConfig {
  nCtx?: number;
  nBatch?: number;
  nUbatch?: number;
  nParallel?: number;
  nThreads?: number;
  nThreadsBatch?: number;
  flashAttention?: FlashAttentionMode;
  kvUnified?: boolean;
  cacheTypeK?: KvCacheType;
  cacheTypeV?: KvCacheType;
  offloadKqv?: boolean;
  opOffload?: boolean;
  swaFull?: boolean;
  warmup?: boolean;
  ropeScaling?: RopeScaling;
  ropeFreqBase?: number;
  ropeFreqScale?: number;
  yarnOrigCtx?: number;
  yarnExtFactor?: number;
  yarnAttnFactor?: number;
  yarnBetaFast?: number;
  yarnBetaSlow?: number;
}

export interface LogitBiasConfig {
  token: number;
  bias: number;
}

export interface SamplingRuntimeConfig {
  samplers?: SamplerStage[];
  seed?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  typicalP?: number;
  xtcProbability?: number;
  xtcThreshold?: number;
  topNSigma?: number;
  temperature?: number;
  dynatempRange?: number;
  dynatempExponent?: number;
  repeatLastN?: number;
  repeatPenalty?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  dryMultiplier?: number;
  dryBase?: number;
  dryAllowedLength?: number;
  dryPenaltyLastN?: number;
  drySequenceBreakers?: string[];
  mirostat?: number;
  mirostatTau?: number;
  mirostatEta?: number;
  minKeep?: number;
  nProbs?: number;
  logitBias?: LogitBiasConfig[];
  ignoreEos?: boolean;
  grammarLazy?: boolean;
  preservedTokens?: number[];
  backendSampling?: boolean;
}

export interface SchedulerRuntimeConfig {
  continuousBatching?: boolean;
  policy?: SchedulerPolicyMode;
  decodeTokenReserve?: number;
  adaptivePrefillChunking?: boolean;
  prefillChunkSize?: number;
  maxRunningRequests?: number;
  maxQueuedRequests?: number;
}

export interface CacheRuntimeConfig {
  mode?: KvReuseMode;
  retainedPrefixTokens?: number;
  snapshotIntervalTokens?: number;
  maxSnapshotEntries?: number;
  maxSnapshotBytes?: number;
  maxSessionEntries?: number;
  cacheKeyPolicy?: CacheKeyPolicy;
  enableContextCheckpoints?: boolean;
  checkpointEveryTokens?: number;
}

export interface MultimodalRuntimeConfig {
  projectorPath?: string;
  useGpu?: boolean;
  imageMinTokens?: number;
  imageMaxTokens?: number;
}

export interface ResidencyRuntimeConfig {
  maxGpuModelsPerDevice?: number;
  allowCpuModelsWhileGpuLoaded?: boolean;
  requireGpuLease?: boolean;
  gpuMemorySafetyMarginBytes?: number;
}

export interface ObservabilityRuntimeConfig {
  runtimeMetrics?: boolean;
  backendProfiling?: boolean;
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
  onTokens?: (batch: TokenBatch) => void;
  /**
   * Controls how aggressively the browser scheduler flushes token batches.
   * `token` favors interactive rendering; `batch` favors throughput.
   */
  tokenFlush?: TokenFlushMode;
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
   * when dispatching streamed tokens to user `onTokens` callbacks.
   *
   * Not part of the public API surface — added to PromptOptions purely so
   * worker-internal code can pass it through `runtime.enqueueQuery`
   * without a type-cast.  Public consumers should ignore it.
   */
  __internalRequestStarted?: (requestId: number) => void;
}

export type TokenFlushMode = 'batch' | 'token';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface StreamStats {
  framesSent: number;
  bytesSent: number;
  framesDropped: number;
  batchesSent: number;
}

export interface TokenBatch {
  requestId: string;
  streamId: number;
  sequenceStart: number;
  text: string;
  frameCount: number;
  byteCount: number;
  stats: StreamStats;
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
