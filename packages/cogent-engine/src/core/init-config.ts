import {
  FlashAttentionMode,
  InferenceInitConfig,
  SamplingConfig,
  SchedulerPolicyMode,
} from '../types.js';

export interface NormalizedInitConfig {
  nCtx: number;
  nBatch: number;
  nUbatch: number;
  nSeqMax: number;
  nThreads: number;
  nThreadsBatch: number;
  nGpuLayers: number;
  flashAttention: number;
  kvUnified: number;
  maxCachedSessions: number;
  retainedPrefixTokens: number;
  prefillChunkSize: number;
  prefixCacheIntervalTokens: number;
  maxPrefixCacheEntries: number;
  schedulerPolicy: number;
  decodeTokenReserve: number;
  adaptivePrefillChunking: number;
  enableRuntimeObservability: number;
  enableBackendProfiling: number;
  multimodalProjectorPath: string | null;
  multimodalUseGpu: number;
  debugCompareMultimodalEmbeddings: number;
  imageMinTokens: number;
  imageMaxTokens: number;
  samplingRepeatLastN: number;
  samplingRepeatPenalty: number;
  samplingFrequencyPenalty: number;
  samplingPresencePenalty: number;
  samplingTopK: number;
  samplingTopP: number;
  samplingMinP: number;
  samplingTemperature: number;
  samplingSeed: number;
}

const DEFAULT_VALUE = 0;
const DEFAULT_GPU_LAYERS = -1;
const DEFAULT_FLASH_ATTENTION = -1;
const DEFAULT_KV_UNIFIED = -1;
const DEFAULT_MULTIMODAL_USE_GPU = -1;
const DEFAULT_SCHEDULER_POLICY = 1;
const DEFAULT_REPEAT_LAST_N = 64;
const DEFAULT_REPEAT_PENALTY = 1.05;
const DEFAULT_FREQUENCY_PENALTY = 0;
const DEFAULT_PRESENCE_PENALTY = 0;
const DEFAULT_TOP_K = 40;
const DEFAULT_TOP_P = 0.8;
const DEFAULT_MIN_P = 0;
const DEFAULT_TEMPERATURE = 0.7;
const DEFAULT_SAMPLING_SEED = -1;

function normalizeInteger(
  fieldName: string,
  value: number | undefined,
  minimum: number,
  allowDefault = true
): number {
  if (value == null) {
    return allowDefault ? DEFAULT_VALUE : minimum;
  }
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`"${fieldName}" must be an integer >= ${minimum}.`);
  }
  return value;
}

function normalizeOptionalInteger(
  fieldName: string,
  value: number | undefined,
  fallback: number,
  minimum: number
): number {
  if (value == null) {
    return fallback;
  }
  return normalizeInteger(fieldName, value, minimum, false);
}

function normalizeNumber(
  fieldName: string,
  value: number | undefined,
  fallback: number,
  minimum: number,
  maximum?: number
): number {
  if (value == null) {
    return fallback;
  }
  if (!Number.isFinite(value) || value < minimum || (maximum != null && value > maximum)) {
    const maximumClause = maximum != null ? ` and <= ${maximum}` : '';
    throw new Error(`"${fieldName}" must be a finite number >= ${minimum}${maximumClause}.`);
  }
  return value;
}

function normalizeSignedNumber(
  fieldName: string,
  value: number | undefined,
  fallback: number
): number {
  if (value == null) {
    return fallback;
  }
  if (!Number.isFinite(value)) {
    throw new Error(`"${fieldName}" must be a finite number.`);
  }
  return value;
}

function normalizeSamplingSeed(value: number | undefined): number {
  if (value == null) {
    return DEFAULT_SAMPLING_SEED;
  }
  if (!Number.isInteger(value)) {
    throw new Error('"sampling.seed" must be an integer.');
  }
  return value;
}

function normalizeOptionalString(
  fieldName: string,
  value: string | undefined
): string | null {
  if (value == null) {
    return null;
  }
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    throw new Error(`"${fieldName}" must not be empty.`);
  }
  return trimmed;
}

function normalizeOptionalBoolean(value: boolean | undefined): number {
  if (value == null) {
    return DEFAULT_KV_UNIFIED;
  }
  return value ? 1 : 0;
}

function normalizeOptionalBooleanWithDefault(
  value: boolean | undefined,
  fallback: number
): number {
  if (value == null) {
    return fallback;
  }
  return value ? 1 : 0;
}

function normalizeRequiredBoolean(value: boolean | undefined): number {
  return value ? 1 : 0;
}

function normalizeFlashAttention(value: FlashAttentionMode | undefined): number {
  if (value == null || value === 'auto') {
    return DEFAULT_FLASH_ATTENTION;
  }
  return value === 'enabled' ? 1 : 0;
}

function normalizeSchedulerPolicy(value: SchedulerPolicyMode | undefined): number {
  if (value == null || value === 'balanced') {
    return DEFAULT_SCHEDULER_POLICY;
  }
  if (value === 'latency-first') {
    return 0;
  }
  return 2;
}

function normalizeSamplingConfig(
  sampling: SamplingConfig | undefined
): Pick<
  NormalizedInitConfig,
  | 'samplingRepeatLastN'
  | 'samplingRepeatPenalty'
  | 'samplingFrequencyPenalty'
  | 'samplingPresencePenalty'
  | 'samplingTopK'
  | 'samplingTopP'
  | 'samplingMinP'
  | 'samplingTemperature'
  | 'samplingSeed'
> {
  return {
    samplingRepeatLastN: normalizeOptionalInteger(
      'sampling.repeatLastN',
      sampling?.repeatLastN,
      DEFAULT_REPEAT_LAST_N,
      0
    ),
    samplingRepeatPenalty: normalizeNumber(
      'sampling.repeatPenalty',
      sampling?.repeatPenalty,
      DEFAULT_REPEAT_PENALTY,
      Number.EPSILON
    ),
    samplingFrequencyPenalty: normalizeSignedNumber(
      'sampling.frequencyPenalty',
      sampling?.frequencyPenalty,
      DEFAULT_FREQUENCY_PENALTY
    ),
    samplingPresencePenalty: normalizeSignedNumber(
      'sampling.presencePenalty',
      sampling?.presencePenalty,
      DEFAULT_PRESENCE_PENALTY
    ),
    samplingTopK: normalizeOptionalInteger(
      'sampling.topK',
      sampling?.topK,
      DEFAULT_TOP_K,
      0
    ),
    samplingTopP: normalizeNumber(
      'sampling.topP',
      sampling?.topP,
      DEFAULT_TOP_P,
      0,
      1
    ),
    samplingMinP: normalizeNumber(
      'sampling.minP',
      sampling?.minP,
      DEFAULT_MIN_P,
      0,
      1
    ),
    samplingTemperature: normalizeNumber(
      'sampling.temperature',
      sampling?.temperature,
      DEFAULT_TEMPERATURE,
      0
    ),
    samplingSeed: normalizeSamplingSeed(sampling?.seed),
  };
}

export function normalizeInitConfig(config: InferenceInitConfig | undefined): NormalizedInitConfig {
  const enableBackendProfiling = normalizeRequiredBoolean(
    config?.enableBackendProfiling
  );
  const enableRuntimeObservability =
    enableBackendProfiling || normalizeRequiredBoolean(config?.enableRuntimeObservability);
  const multimodalProjectorPath = normalizeOptionalString(
    'multimodalProjectorPath',
    config?.multimodalProjectorPath
  );
  const imageMinTokens = normalizeInteger('imageMinTokens', config?.imageMinTokens, 0);
  const imageMaxTokens = normalizeInteger('imageMaxTokens', config?.imageMaxTokens, 0);
  if (multimodalProjectorPath == null && (imageMinTokens > 0 || imageMaxTokens > 0)) {
    throw new Error(
      '"imageMinTokens" and "imageMaxTokens" require "multimodalProjectorPath".'
    );
  }
  const multimodalUseGpu = normalizeOptionalBooleanWithDefault(
    config?.multimodalUseGpu,
    DEFAULT_MULTIMODAL_USE_GPU
  );
  const debugCompareMultimodalEmbeddings = normalizeRequiredBoolean(
    config?.debugCompareMultimodalEmbeddings
  );
  if (imageMinTokens > 0 && imageMaxTokens > 0 && imageMaxTokens < imageMinTokens) {
    throw new Error('"imageMaxTokens" must be >= "imageMinTokens".');
  }
  const samplingConfig = normalizeSamplingConfig(config?.sampling);

  return {
    nCtx: normalizeInteger('nCtx', config?.nCtx, 1),
    nBatch: normalizeInteger('nBatch', config?.nBatch, 1),
    nUbatch: normalizeInteger('nUbatch', config?.nUbatch, 1),
    nSeqMax: normalizeInteger('nSeqMax', config?.nSeqMax, 1),
    nThreads: normalizeInteger('nThreads', config?.nThreads, 1),
    nThreadsBatch: normalizeInteger('nThreadsBatch', config?.nThreadsBatch, 1),
    nGpuLayers:
      config?.nGpuLayers == null
        ? DEFAULT_GPU_LAYERS
        : normalizeInteger('nGpuLayers', config.nGpuLayers, 0, false),
    flashAttention: normalizeFlashAttention(config?.flashAttention),
    kvUnified: normalizeOptionalBoolean(config?.kvUnified),
    maxCachedSessions: normalizeInteger('maxCachedSessions', config?.maxCachedSessions, 1),
    retainedPrefixTokens: normalizeInteger('retainedPrefixTokens', config?.retainedPrefixTokens, 0),
    prefillChunkSize: normalizeInteger('prefillChunkSize', config?.prefillChunkSize, 0),
    prefixCacheIntervalTokens: normalizeInteger(
      'prefixCacheIntervalTokens',
      config?.prefixCacheIntervalTokens,
      1
    ),
    maxPrefixCacheEntries: normalizeInteger(
      'maxPrefixCacheEntries',
      config?.maxPrefixCacheEntries,
      1
    ),
    schedulerPolicy: normalizeSchedulerPolicy(config?.schedulerPolicy),
    decodeTokenReserve: normalizeInteger('decodeTokenReserve', config?.decodeTokenReserve, 0),
    adaptivePrefillChunking: normalizeOptionalBoolean(config?.adaptivePrefillChunking),
    enableRuntimeObservability,
    enableBackendProfiling,
    multimodalProjectorPath,
    multimodalUseGpu,
    debugCompareMultimodalEmbeddings,
    imageMinTokens,
    imageMaxTokens,
    ...samplingConfig,
  };
}
