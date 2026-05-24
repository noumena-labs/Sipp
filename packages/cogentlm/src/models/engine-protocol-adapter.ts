import type { GenerateResponse } from '../core/inference-types.js';
import type {
  BackendInfo,
  BackendProfileObservation,
  EngineBackendName,
  EngineEvent,
  EngineState,
  EngineStats,
  EmbeddingResult,
  FinishReason,
  ObservabilityEvent,
  ObservabilitySnapshot,
  QueryObservation,
  GenerationResult,
  RequestState,
  RequestStats,
  RuntimeObservation,
} from './types.js';

const emptyStats: EngineStats = {
  requestsRunning: 0,
  requestsQueued: 0,
  requestsCompleted: 0,
  requestsFailed: 0,
  inputTokens: 0,
  outputTokens: 0,
  cacheHits: 0,
  prefillTokens: 0,
  ttftMs: null,
  interTokenMs: null,
  e2eMs: null,
  tokensPerSecond: null,
  prefillTokensPerSecond: null,
  prefillMs: 0,
  decodeMs: 0,
  backendMs: 0,
  syncMs: 0,
  engineOverheadMs: 0,
};

const emptyBackend: BackendInfo = {
  selected: 'unknown',
  available: [],
  devices: [],
};

export function observabilityEventToStateEvent(event: ObservabilityEvent): EngineEvent {
  const state = observabilitySnapshotToEngineState(event.snapshot);
  return state.status === 'closed' ? { type: 'closed' } : { type: 'state', state };
}

export function observabilitySnapshotToEngineState(
  snapshot: ObservabilitySnapshot
): EngineState {
  return {
    status: toEngineStatus(snapshot.state),
    model: snapshot.model,
    backend: toBackendInfo(snapshot.profile),
    requests: snapshot.query == null ? [] : [toRequestState(snapshot.query, snapshot.runtime)],
    stats: toEngineStats(snapshot),
    updatedAt: snapshot.updatedAt,
  };
}

export function generationResultFromGenerateResponse(
  response: GenerateResponse,
  options: {
    text?: string;
    maxTokens?: number;
    finishReason?: FinishReason;
  } = {}
): GenerationResult {
  const text = options.text ?? textOutputFromGenerateResponse(response);
  const finishReason = options.finishReason ?? finishReasonFromGenerateResponse(response, options.maxTokens);
  return generationResultFromText({
    id: response.requestId,
    text,
    finishReason,
    metrics: response.observability ?? null,
  });
}

export function generationResultFromText(input: {
  id: string | number;
  text: string;
  finishReason: FinishReason;
  metrics?: GenerateResponse['observability'] | null;
}): GenerationResult {
  return {
    id: String(input.id),
    text: input.text,
    finishReason: input.finishReason,
    stats: requestStatsFromMetrics(input.metrics ?? null),
  };
}

export function embeddingResultFromGenerateResponse(response: GenerateResponse): EmbeddingResult {
  if (response.embedding == null) {
    throw new Error('Runtime completed embed() without embedding output.');
  }
  return {
    id: String(response.requestId),
    values: response.embedding.values,
    pooling: response.embedding.pooling,
    normalized: response.embedding.normalized,
    stats: requestStatsFromMetrics(response.observability ?? null),
  };
}

function textOutputFromGenerateResponse(response: GenerateResponse): string {
  if (response.outputText == null) {
    throw new Error('Runtime completed text generation without text output.');
  }
  return response.outputText;
}

function toEngineStatus(state: ObservabilitySnapshot['state']): EngineState['status'] {
  switch (state) {
    case 'idle':
    case 'loading':
    case 'ready':
    case 'error':
    case 'closed':
      return state;
    case 'querying':
      return 'running';
  }
}

function toRequestStatus(status: QueryObservation['status']): RequestState['status'] {
  switch (status) {
    case 'running':
      return 'decode';
    case 'success':
      return 'completed';
    case 'cancelled':
      return 'cancelled';
    case 'failed':
      return 'failed';
  }
}

function toRequestState(query: QueryObservation, runtime?: RuntimeObservation): RequestState {
  return {
    id: query.session ?? 'default',
    status: toRequestStatus(query.status),
    inputTokens: runtime?.inputTokens ?? 0,
    outputTokens: query.outputTokens ?? runtime?.outputTokens ?? 0,
  };
}

function toEngineStats(snapshot: ObservabilitySnapshot): EngineStats {
  const runtime = snapshot.runtime;
  const query = snapshot.query;
  if (runtime == null && query == null) {
    return { ...emptyStats };
  }

  return {
    requestsRunning: query?.status === 'running' ? 1 : 0,
    requestsQueued: 0,
    requestsCompleted: query?.status === 'success' ? 1 : 0,
    requestsFailed: query?.status === 'failed' ? 1 : 0,
    inputTokens: runtime?.inputTokens ?? 0,
    outputTokens: query?.outputTokens ?? runtime?.outputTokens ?? 0,
    cacheHits: runtime?.cacheHits ?? 0,
    prefillTokens: runtime?.prefillTokens ?? 0,
    ttftMs: runtime?.ttftMs ?? query?.ttftMs ?? null,
    interTokenMs: runtime?.itlAvgMs ?? null,
    e2eMs: runtime?.e2eMs ?? query?.wallMs ?? null,
    tokensPerSecond: runtime?.tokensPerSecond ?? null,
    prefillTokensPerSecond: runtime?.prefillTokensPerSecond ?? null,
    prefillMs: runtime?.prefillMs ?? 0,
    decodeMs: runtime?.decodeMs ?? 0,
    backendMs: runtime?.nativeGpuMs ?? 0,
    syncMs: runtime?.nativeSyncMs ?? 0,
    engineOverheadMs: runtime?.nativeLogicMs ?? 0,
  };
}

function toBackendInfo(profile: BackendProfileObservation | undefined): BackendInfo {
  if (profile == null) {
    return { ...emptyBackend, devices: [] };
  }

  return {
    selected: selectBackend(profile),
    available: profile.availableBackends.map((backend) => backend.name),
    devices: profile.devices.map((device) => ({
      id: null,
      name: device.name,
      type: device.type,
      memoryTotalBytes: undefined,
      memoryFreeBytes: undefined,
    })),
  };
}

function selectBackend(profile: BackendProfileObservation): EngineBackendName {
  if (profile.webgpuRegistered && profile.webgpuDeviceCount > 0 && profile.gpuOffloadSupported) {
    return 'webgpu';
  }

  const acceleratedDevice = profile.devices.find((device) => device.type !== 'cpu');
  const backendName = acceleratedDevice?.backendName ?? profile.availableBackends.at(0)?.name;
  return normalizeBackendName(backendName);
}

function normalizeBackendName(name: string | undefined): EngineBackendName {
  const normalized = name?.toLowerCase() ?? '';
  if (normalized.includes('cuda')) return 'cuda';
  if (normalized.includes('metal')) return 'metal';
  if (normalized.includes('vulkan')) return 'vulkan';
  if (normalized.includes('webgpu') || normalized.includes('wgpu')) return 'webgpu';
  if (normalized.includes('cpu')) return 'cpu';
  return 'unknown';
}

function requestStatsFromMetrics(metrics: GenerateResponse['observability']): RequestStats {
  return {
    inputTokens: metrics?.inputTokens ?? 0,
    outputTokens: metrics?.outputTokens ?? 0,
    cacheHits: metrics?.cacheHits ?? 0,
    ttftMs: metrics?.ttftMs ?? null,
    interTokenMs: metrics?.itlAvgMs ?? null,
    e2eMs: metrics?.e2eMs ?? null,
    tokensPerSecond:
      metrics != null && metrics.decodeMs > 0
        ? (metrics.outputTokens / metrics.decodeMs) * 1000
        : null,
    prefillMs: metrics?.prefillMs ?? 0,
    decodeMs: metrics?.decodeMs ?? 0,
  };
}

function finishReasonFromGenerateResponse(
  response: GenerateResponse,
  maxTokens: number | undefined
): FinishReason {
  if (response.cancelled) return 'cancelled';
  if (response.failed) return 'error';
  if (
    maxTokens != null &&
    response.observability != null &&
    response.observability.outputTokens >= maxTokens
  ) {
    return 'length';
  }
  return 'stop';
}
