import {
  FlashAttentionMode,
  InferenceInitConfig,
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
}

const DEFAULT_VALUE = 0;
const DEFAULT_GPU_LAYERS = -1;
const DEFAULT_FLASH_ATTENTION = -1;
const DEFAULT_KV_UNIFIED = -1;
const DEFAULT_SCHEDULER_POLICY = 1;

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

function normalizeOptionalBoolean(value: boolean | undefined): number {
  if (value == null) {
    return DEFAULT_KV_UNIFIED;
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

export function normalizeInitConfig(config: InferenceInitConfig | undefined): NormalizedInitConfig {
  const enableBackendProfiling = normalizeRequiredBoolean(
    config?.enableBackendProfiling
  );
  const enableRuntimeObservability =
    enableBackendProfiling || normalizeRequiredBoolean(config?.enableRuntimeObservability);

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
  };
}
