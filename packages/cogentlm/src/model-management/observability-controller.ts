import type {
  BackendObservability,
  RequestObservabilityMetrics,
  RuntimeAggregateObservabilityMetrics,
  TransportObservability,
} from '../types.js';
import type {
  BackendProfileObservation,
  EngineObservability,
  ObservabilityEvent,
  ObservabilityEventType,
  ObservabilityMode,
  ObservabilitySnapshot,
  RuntimeObservation,
} from './model-types.js';

type SnapshotPatch = Partial<
  Omit<ObservabilitySnapshot, 'updatedAt' | 'runtime' | 'profile'> & {
    runtime: RuntimeObservation | null | undefined;
    profile: BackendProfileObservation | null | undefined;
  }
>;

function cloneSnapshot(snapshot: ObservabilitySnapshot): ObservabilitySnapshot {
  return {
    ...snapshot,
    model: snapshot.model == null ? null : { ...snapshot.model },
    query: snapshot.query == null ? null : { ...snapshot.query },
    runtime:
      snapshot.runtime == null
        ? undefined
        : {
            ...snapshot.runtime,
            execution: { ...snapshot.runtime.execution },
          },
    profile:
      snapshot.profile == null
        ? undefined
        : {
            ...snapshot.profile,
            availableBackends: snapshot.profile.availableBackends.map((backend) => ({ ...backend })),
            devices: snapshot.profile.devices.map((device) => ({ ...device })),
          },
  };
}

function includeFinite(
  target: RuntimeObservation,
  key: keyof RuntimeObservation,
  value: unknown
): void {
  if (typeof value === 'number' && Number.isFinite(value)) {
    (target as unknown as Record<string, unknown>)[key] = value;
  }
}

export function resolveObservabilityMode(mode: ObservabilityMode | undefined): ObservabilityMode {
  return mode ?? 'off';
}

export function applyObservabilityMode<T extends object>(
  runtime: T | undefined,
  mode: ObservabilityMode
): T & {
  enableRuntimeObservability?: boolean;
  enableBackendProfiling?: boolean;
} {
  return {
    ...(runtime ?? ({} as T)),
    enableRuntimeObservability: mode === 'runtime' || mode === 'profile',
    enableBackendProfiling: mode === 'profile',
  };
}

export function toRuntimeObservation(
  metrics: RuntimeAggregateObservabilityMetrics | RequestObservabilityMetrics | null,
  transport: TransportObservability
): RuntimeObservation | undefined {
  if (metrics == null) {
    return undefined;
  }

  const tokenPath =
    transport.activeTokenTransport === 'runtime-events'
      ? 'runtime-event'
      : transport.activeTokenTransport === 'none'
        ? 'none'
        : undefined;

  const observation: RuntimeObservation = {
    totalMs: metrics.totalMs,
    ttftMs: metrics.ttftMs,
    tokensPerSecond: metrics.tokensPerSecond,
    inputTokenCount: metrics.inputTokenCount,
    outputTokenCount: metrics.outputTokenCount,
    execution: {
      mode: transport.executionMode,
      workerBacked: transport.workerBacked,
      tokenPath,
    },
  };

  includeFinite(observation, 'promptEvalMs', (metrics as { promptEvalMs?: unknown }).promptEvalMs);
  includeFinite(observation, 'decodeEvalMs', (metrics as { decodeEvalMs?: unknown }).decodeEvalMs);
  includeFinite(observation, 'sampleMs', (metrics as { sampleMs?: unknown }).sampleMs);
  includeFinite(observation, 'queueDelayMs', (metrics as { queueDelayMs?: unknown }).queueDelayMs);
  includeFinite(observation, 'meanItlMs', (metrics as { meanItlMs?: unknown }).meanItlMs);
  includeFinite(observation, 'tailItlMs', (metrics as { tailItlMs?: unknown }).tailItlMs);
  includeFinite(observation, 'promptEvalTokens', (metrics as { promptEvalTokens?: unknown }).promptEvalTokens);
  includeFinite(observation, 'decodeEvalCount', (metrics as { decodeEvalCount?: unknown }).decodeEvalCount);
  includeFinite(
    observation,
    'batchParticipationCount',
    (metrics as { batchParticipationCount?: unknown }).batchParticipationCount
  );
  includeFinite(
    observation,
    'decodeFirstTickCount',
    (metrics as { decodeFirstTickCount?: unknown }).decodeFirstTickCount
  );
  includeFinite(
    observation,
    'chunkedPrefillTickCount',
    (metrics as { chunkedPrefillTickCount?: unknown }).chunkedPrefillTickCount
  );
  includeFinite(
    observation,
    'mixedWorkloadTickCount',
    (metrics as { mixedWorkloadTickCount?: unknown }).mixedWorkloadTickCount
  );
  includeFinite(observation, 'lcpReuseTokens', (metrics as { lcpReuseTokens?: unknown }).lcpReuseTokens);
  includeFinite(
    observation,
    'prefixCacheRestoreTokens',
    (metrics as { prefixCacheRestoreTokens?: unknown }).prefixCacheRestoreTokens
  );
  includeFinite(
    observation,
    'prefixCacheHitCount',
    (metrics as { prefixCacheHitCount?: unknown }).prefixCacheHitCount
  );
  includeFinite(
    observation,
    'prefixCacheStoreCount',
    (metrics as { prefixCacheStoreCount?: unknown }).prefixCacheStoreCount
  );

  return observation;
}

export function toBackendProfileObservation(
  backend: BackendObservability | null
): BackendProfileObservation | undefined {
  if (backend == null) {
    return undefined;
  }
  return {
    profilingEnabled: backend.profilingEnabled,
    webgpuCompiled: backend.webgpuCompiled,
    webgpuRegistered: backend.webgpuRegistered,
    webgpuDeviceCount: backend.webgpuDeviceCount,
    gpuOffloadSupported: backend.gpuOffloadSupported,
    availableBackends: backend.availableBackends.map((item) => ({ ...item })),
    devices: backend.devices.map((device) => ({
      name: device.name,
      description: device.description,
      type: device.type,
      backendName: device.backendName,
    })),
  };
}

export class ObservabilityController implements EngineObservability {
  private snapshot: ObservabilitySnapshot = {
    mode: 'off',
    state: 'idle',
    updatedAt: new Date().toISOString(),
    model: null,
    query: null,
  };
  private readonly listeners = new Set<(event: ObservabilityEvent) => void>();

  public current(): ObservabilitySnapshot {
    return cloneSnapshot(this.snapshot);
  }

  public subscribe(listener: (event: ObservabilityEvent) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  public emit(type: ObservabilityEventType, patch: SnapshotPatch = {}): void {
    this.snapshot = this.buildSnapshot(patch);
    const event: ObservabilityEvent = {
      type,
      snapshot: this.current(),
    };
    for (const listener of this.listeners) {
      listener(event);
    }
  }

  public ingest(event: ObservabilityEvent): void {
    this.snapshot = cloneSnapshot(event.snapshot);
    const localEvent: ObservabilityEvent = {
      type: event.type,
      snapshot: this.current(),
    };
    for (const listener of this.listeners) {
      listener(localEvent);
    }
  }

  public update(patch: SnapshotPatch = {}): void {
    this.snapshot = this.buildSnapshot(patch);
  }

  public markClosed(): void {
    this.emit('close', {
      state: 'closed',
      model: null,
      query: null,
      runtime: null,
      profile: null,
    });
  }

  private buildSnapshot(patch: SnapshotPatch): ObservabilitySnapshot {
    const next = {
      ...this.snapshot,
      ...patch,
      updatedAt: new Date().toISOString(),
    } as ObservabilitySnapshot;
    if ('runtime' in patch && patch.runtime == null) {
      delete next.runtime;
    }
    if ('profile' in patch && patch.profile == null) {
      delete next.profile;
    }
    return next;
  }
}
