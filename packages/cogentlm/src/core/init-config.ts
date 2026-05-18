import type {
  CacheKeyPolicy,
  FlashAttentionMode,
  GpuLayerConfig,
  KvCacheType,
  KvReuseMode,
  NativeRuntimeConfig,
  RopeScaling,
  SamplerStage,
  SamplingRuntimeConfig,
  SchedulerPolicyMode,
  SplitMode,
} from '../types.js';

export interface NormalizedInitConfig {
  runtimeConfigJson: string;
}

type JsonRecord = Record<string, unknown>;

function put(record: JsonRecord, key: string, value: unknown): void {
  if (value !== undefined) {
    record[key] = value;
  }
}

function hasKeys(record: JsonRecord): boolean {
  return Object.keys(record).length > 0;
}

function normalizeInteger(
  fieldName: string,
  value: number | undefined,
  minimum: number
): number | undefined {
  if (value == null) {
    return undefined;
  }
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`"${fieldName}" must be an integer >= ${minimum}.`);
  }
  return value;
}

function normalizeSafeInteger(
  fieldName: string,
  value: number | undefined,
  minimum: number
): number | undefined {
  if (value == null) {
    return undefined;
  }
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(`"${fieldName}" must be a safe integer >= ${minimum}.`);
  }
  return value;
}

function normalizeFiniteNumber(
  fieldName: string,
  value: number | undefined,
  minimum?: number,
  maximum?: number
): number | undefined {
  if (value == null) {
    return undefined;
  }
  if (
    !Number.isFinite(value) ||
    (minimum != null && value < minimum) ||
    (maximum != null && value > maximum)
  ) {
    const range =
      minimum != null
        ? ` >= ${minimum}${maximum != null ? ` and <= ${maximum}` : ''}`
        : '';
    throw new Error(`"${fieldName}" must be a finite number${range}.`);
  }
  return value;
}

function normalizeOptionalString(fieldName: string, value: string | undefined): string | undefined {
  if (value == null) {
    return undefined;
  }
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    throw new Error(`"${fieldName}" must not be empty.`);
  }
  return trimmed;
}

function normalizeStringArray(
  fieldName: string,
  values: string[] | undefined,
  trimValues = true
): string[] | undefined {
  if (values == null) {
    return undefined;
  }
  return values.map((value, index) => {
    const normalized = trimValues ? value.trim() : value;
    if (normalized.length === 0) {
      throw new Error(`"${fieldName}[${index}]" must not be empty.`);
    }
    return normalized;
  });
}

function snakeCase(value: string): string {
  return value.replaceAll('-', '_');
}

function normalizeGpuLayers(value: GpuLayerConfig | undefined): string | { count: number } | undefined {
  if (value == null) {
    return undefined;
  }
  if (value === 'auto' || value === 'all') {
    return value;
  }
  return { count: normalizeInteger('placement.gpuLayers', value, 0)! };
}

function normalizeEnum<T extends string>(
  _fieldName: string,
  value: T | undefined
): string | undefined {
  return value == null ? undefined : snakeCase(value);
}

function normalizeSamplerStages(values: SamplerStage[] | undefined): string[] | undefined {
  return values?.map(snakeCase);
}

function normalizePlacement(config: NativeRuntimeConfig | undefined): JsonRecord | undefined {
  const placement = config?.placement;
  if (placement == null) {
    return undefined;
  }
  const out: JsonRecord = {};
  put(out, 'devices', normalizeStringArray('placement.devices', placement.devices));
  put(out, 'gpu_layers', normalizeGpuLayers(placement.gpuLayers));
  put(out, 'split_mode', normalizeEnum<SplitMode>('placement.splitMode', placement.splitMode));
  put(out, 'main_gpu', normalizeInteger('placement.mainGpu', placement.mainGpu, 0));
  put(
    out,
    'tensor_split',
    placement.tensorSplit?.map((value, index) =>
      normalizeFiniteNumber(`placement.tensorSplit[${index}]`, value, 0)
    )
  );
  put(out, 'use_mmap', placement.useMmap);
  put(out, 'use_mlock', placement.useMlock);
  put(out, 'fit_params', placement.fitParams);
  put(out, 'fit_params_min_ctx', normalizeInteger('placement.fitParamsMinCtx', placement.fitParamsMinCtx, 1));
  put(
    out,
    'fit_params_target_bytes',
    placement.fitParamsTargetBytes?.map((value, index) =>
      normalizeSafeInteger(`placement.fitParamsTargetBytes[${index}]`, value, 0)
    )
  );
  put(out, 'check_tensors', placement.checkTensors);
  put(out, 'no_extra_bufts', placement.noExtraBufts);
  put(out, 'no_host', placement.noHost);
  return hasKeys(out) ? out : undefined;
}

function normalizeContext(config: NativeRuntimeConfig | undefined): JsonRecord | undefined {
  const context = config?.context;
  if (context == null) {
    return undefined;
  }
  const out: JsonRecord = {};
  put(out, 'n_ctx', normalizeInteger('context.nCtx', context.nCtx, 1));
  put(out, 'n_batch', normalizeInteger('context.nBatch', context.nBatch, 1));
  put(out, 'n_ubatch', normalizeInteger('context.nUbatch', context.nUbatch, 1));
  put(out, 'n_parallel', normalizeInteger('context.nParallel', context.nParallel, 1));
  put(out, 'n_threads', normalizeInteger('context.nThreads', context.nThreads, 0));
  put(out, 'n_threads_batch', normalizeInteger('context.nThreadsBatch', context.nThreadsBatch, 0));
  put(
    out,
    'flash_attention',
    normalizeEnum<FlashAttentionMode>('context.flashAttention', context.flashAttention)
  );
  put(out, 'kv_unified', context.kvUnified);
  put(out, 'cache_type_k', normalizeEnum<KvCacheType>('context.cacheTypeK', context.cacheTypeK));
  put(out, 'cache_type_v', normalizeEnum<KvCacheType>('context.cacheTypeV', context.cacheTypeV));
  put(out, 'offload_kqv', context.offloadKqv);
  put(out, 'op_offload', context.opOffload);
  put(out, 'swa_full', context.swaFull);
  put(out, 'warmup', context.warmup);
  put(out, 'rope_scaling', normalizeEnum<RopeScaling>('context.ropeScaling', context.ropeScaling));
  put(out, 'rope_freq_base', normalizeFiniteNumber('context.ropeFreqBase', context.ropeFreqBase));
  put(out, 'rope_freq_scale', normalizeFiniteNumber('context.ropeFreqScale', context.ropeFreqScale));
  put(out, 'yarn_orig_ctx', normalizeInteger('context.yarnOrigCtx', context.yarnOrigCtx, 1));
  put(out, 'yarn_ext_factor', normalizeFiniteNumber('context.yarnExtFactor', context.yarnExtFactor));
  put(out, 'yarn_attn_factor', normalizeFiniteNumber('context.yarnAttnFactor', context.yarnAttnFactor));
  put(out, 'yarn_beta_fast', normalizeFiniteNumber('context.yarnBetaFast', context.yarnBetaFast));
  put(out, 'yarn_beta_slow', normalizeFiniteNumber('context.yarnBetaSlow', context.yarnBetaSlow));
  return hasKeys(out) ? out : undefined;
}

function normalizeSampling(sampling: SamplingRuntimeConfig | undefined): JsonRecord | undefined {
  if (sampling == null) {
    return undefined;
  }
  const out: JsonRecord = {};
  put(out, 'samplers', normalizeSamplerStages(sampling.samplers));
  put(out, 'seed', normalizeSafeInteger('sampling.seed', sampling.seed, 0));
  put(out, 'top_k', normalizeInteger('sampling.topK', sampling.topK, 0));
  put(out, 'top_p', normalizeFiniteNumber('sampling.topP', sampling.topP, 0, 1));
  put(out, 'min_p', normalizeFiniteNumber('sampling.minP', sampling.minP, 0, 1));
  put(out, 'typical_p', normalizeFiniteNumber('sampling.typicalP', sampling.typicalP, 0, 1));
  put(out, 'xtc_probability', normalizeFiniteNumber('sampling.xtcProbability', sampling.xtcProbability, 0, 1));
  put(out, 'xtc_threshold', normalizeFiniteNumber('sampling.xtcThreshold', sampling.xtcThreshold, 0, 1));
  put(out, 'top_n_sigma', normalizeFiniteNumber('sampling.topNSigma', sampling.topNSigma, 0));
  put(out, 'temperature', normalizeFiniteNumber('sampling.temperature', sampling.temperature, 0));
  put(out, 'dynatemp_range', normalizeFiniteNumber('sampling.dynatempRange', sampling.dynatempRange, 0));
  put(out, 'dynatemp_exponent', normalizeFiniteNumber('sampling.dynatempExponent', sampling.dynatempExponent));
  put(out, 'repeat_last_n', normalizeInteger('sampling.repeatLastN', sampling.repeatLastN, 0));
  put(out, 'repeat_penalty', normalizeFiniteNumber('sampling.repeatPenalty', sampling.repeatPenalty, Number.EPSILON));
  put(out, 'frequency_penalty', normalizeFiniteNumber('sampling.frequencyPenalty', sampling.frequencyPenalty));
  put(out, 'presence_penalty', normalizeFiniteNumber('sampling.presencePenalty', sampling.presencePenalty));
  put(out, 'dry_multiplier', normalizeFiniteNumber('sampling.dryMultiplier', sampling.dryMultiplier, 0));
  put(out, 'dry_base', normalizeFiniteNumber('sampling.dryBase', sampling.dryBase));
  put(out, 'dry_allowed_length', normalizeInteger('sampling.dryAllowedLength', sampling.dryAllowedLength, 0));
  put(out, 'dry_penalty_last_n', normalizeInteger('sampling.dryPenaltyLastN', sampling.dryPenaltyLastN, 0));
  put(
    out,
    'dry_sequence_breakers',
    normalizeStringArray('sampling.drySequenceBreakers', sampling.drySequenceBreakers, false)
  );
  put(out, 'mirostat', normalizeInteger('sampling.mirostat', sampling.mirostat, 0));
  put(out, 'mirostat_tau', normalizeFiniteNumber('sampling.mirostatTau', sampling.mirostatTau, 0));
  put(out, 'mirostat_eta', normalizeFiniteNumber('sampling.mirostatEta', sampling.mirostatEta, 0));
  put(out, 'min_keep', normalizeInteger('sampling.minKeep', sampling.minKeep, 0));
  put(out, 'n_probs', normalizeInteger('sampling.nProbs', sampling.nProbs, 0));
  put(
    out,
    'logit_bias',
    sampling.logitBias?.map((entry, index) => ({
      token: normalizeInteger(`sampling.logitBias[${index}].token`, entry.token, 0),
      bias: normalizeFiniteNumber(`sampling.logitBias[${index}].bias`, entry.bias),
    }))
  );
  put(out, 'ignore_eos', sampling.ignoreEos);
  put(out, 'grammar_lazy', sampling.grammarLazy);
  put(
    out,
    'preserved_tokens',
    sampling.preservedTokens?.map((token, index) =>
      normalizeInteger(`sampling.preservedTokens[${index}]`, token, 0)
    )
  );
  put(out, 'backend_sampling', sampling.backendSampling);
  return hasKeys(out) ? out : undefined;
}

function normalizeScheduler(config: NativeRuntimeConfig | undefined): JsonRecord | undefined {
  const scheduler = config?.scheduler;
  if (scheduler == null) {
    return undefined;
  }
  const policy: JsonRecord = {};
  put(policy, 'mode', normalizeEnum<SchedulerPolicyMode>('scheduler.policy', scheduler.policy));
  put(
    policy,
    'decode_token_reserve',
    normalizeInteger('scheduler.decodeTokenReserve', scheduler.decodeTokenReserve, 0)
  );
  put(policy, 'enable_adaptive_prefill_chunking', scheduler.adaptivePrefillChunking);

  const out: JsonRecord = {};
  put(out, 'continuous_batching', scheduler.continuousBatching);
  put(out, 'policy', hasKeys(policy) ? policy : undefined);
  put(out, 'prefill_chunk_size', normalizeInteger('scheduler.prefillChunkSize', scheduler.prefillChunkSize, 0));
  put(out, 'max_running_requests', normalizeInteger('scheduler.maxRunningRequests', scheduler.maxRunningRequests, 1));
  put(out, 'max_queued_requests', normalizeInteger('scheduler.maxQueuedRequests', scheduler.maxQueuedRequests, 1));
  return hasKeys(out) ? out : undefined;
}

function normalizeCache(config: NativeRuntimeConfig | undefined): JsonRecord | undefined {
  const cache = config?.cache;
  if (cache == null) {
    return undefined;
  }
  const out: JsonRecord = {};
  put(out, 'mode', normalizeEnum<KvReuseMode>('cache.mode', cache.mode));
  put(out, 'retained_prefix_tokens', normalizeInteger('cache.retainedPrefixTokens', cache.retainedPrefixTokens, 0));
  put(out, 'snapshot_interval_tokens', normalizeInteger('cache.snapshotIntervalTokens', cache.snapshotIntervalTokens, 0));
  put(out, 'max_snapshot_entries', normalizeInteger('cache.maxSnapshotEntries', cache.maxSnapshotEntries, 1));
  put(out, 'max_snapshot_bytes', normalizeSafeInteger('cache.maxSnapshotBytes', cache.maxSnapshotBytes, 1));
  put(out, 'max_session_entries', normalizeInteger('cache.maxSessionEntries', cache.maxSessionEntries, 1));
  put(out, 'cache_key_policy', normalizeEnum<CacheKeyPolicy>('cache.cacheKeyPolicy', cache.cacheKeyPolicy));
  put(out, 'enable_context_checkpoints', cache.enableContextCheckpoints);
  put(out, 'checkpoint_every_tokens', normalizeInteger('cache.checkpointEveryTokens', cache.checkpointEveryTokens, 0));
  return hasKeys(out) ? out : undefined;
}

function normalizeMultimodal(config: NativeRuntimeConfig | undefined): JsonRecord | undefined {
  const multimodal = config?.multimodal;
  if (multimodal == null) {
    return undefined;
  }
  const projectorPath = normalizeOptionalString('multimodal.projectorPath', multimodal.projectorPath);
  const imageMinTokens = normalizeInteger('multimodal.imageMinTokens', multimodal.imageMinTokens, 0);
  const imageMaxTokens = normalizeInteger('multimodal.imageMaxTokens', multimodal.imageMaxTokens, 0);
  if (projectorPath == null && ((imageMinTokens ?? 0) > 0 || (imageMaxTokens ?? 0) > 0)) {
    throw new Error(
      '"multimodal.imageMinTokens" and "multimodal.imageMaxTokens" require "multimodal.projectorPath".'
    );
  }
  if (imageMinTokens != null && imageMaxTokens != null && imageMaxTokens > 0 && imageMaxTokens < imageMinTokens) {
    throw new Error('"multimodal.imageMaxTokens" must be >= "multimodal.imageMinTokens".');
  }

  const out: JsonRecord = {};
  put(out, 'projector_path', projectorPath);
  put(out, 'use_gpu', multimodal.useGpu);
  put(out, 'image_min_tokens', imageMinTokens);
  put(out, 'image_max_tokens', imageMaxTokens);
  return hasKeys(out) ? out : undefined;
}

function normalizeResidency(config: NativeRuntimeConfig | undefined): JsonRecord | undefined {
  const residency = config?.residency;
  if (residency == null) {
    return undefined;
  }
  const out: JsonRecord = {};
  put(out, 'max_gpu_models_per_device', normalizeInteger('residency.maxGpuModelsPerDevice', residency.maxGpuModelsPerDevice, 1));
  put(out, 'allow_cpu_models_while_gpu_loaded', residency.allowCpuModelsWhileGpuLoaded);
  put(out, 'require_gpu_lease', residency.requireGpuLease);
  put(
    out,
    'gpu_memory_safety_margin_bytes',
    normalizeSafeInteger('residency.gpuMemorySafetyMarginBytes', residency.gpuMemorySafetyMarginBytes, 0)
  );
  return hasKeys(out) ? out : undefined;
}

function normalizeObservability(config: NativeRuntimeConfig | undefined): JsonRecord | undefined {
  const observability = config?.observability;
  if (observability == null) {
    return undefined;
  }
  const backendProfiling = observability.backendProfiling === true;
  const out: JsonRecord = {};
  put(out, 'backend_profiling', observability.backendProfiling);
  put(out, 'runtime_metrics', backendProfiling || observability.runtimeMetrics);
  return hasKeys(out) ? out : undefined;
}

export function normalizeInitConfig(config: NativeRuntimeConfig | undefined): NormalizedInitConfig {
  const runtimeConfig: JsonRecord = {};
  put(runtimeConfig, 'placement', normalizePlacement(config));
  put(runtimeConfig, 'context', normalizeContext(config));
  put(runtimeConfig, 'sampling', normalizeSampling(config?.sampling));
  put(runtimeConfig, 'scheduler', normalizeScheduler(config));
  put(runtimeConfig, 'cache', normalizeCache(config));
  put(runtimeConfig, 'multimodal', normalizeMultimodal(config));
  put(runtimeConfig, 'residency', normalizeResidency(config));
  put(runtimeConfig, 'observability', normalizeObservability(config));
  return {
    runtimeConfigJson: JSON.stringify(runtimeConfig),
  };
}
