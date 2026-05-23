import type { BackendObservability } from '../observability/backend-observability.js';
import type {
  RequestObservabilityMetrics,
  RuntimeAggregateObservabilityMetrics,
} from '../observability/runtime-observability.js';
import type { TransportObservability } from '../observability/transport-observability.js';
import type {
  BackendProfileObservation,
  EngineObservability,
  ObservabilityEvent,
  ObservabilityEventType,
  ObservabilityMode,
  ObservabilitySnapshot,
  RuntimeObservation,
} from './types.js';

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

export function toRuntimeObservation(
  metrics: RuntimeAggregateObservabilityMetrics | RequestObservabilityMetrics | null,
  transport: TransportObservability
): RuntimeObservation | undefined {
  if (metrics == null) {
    return undefined;
  }

  const tokenPath =
    transport.activeTokenTransport === 'streaming-buffer'
      ? 'streaming-buffer'
      : transport.activeTokenTransport === 'callback'
        ? 'callback'
      : transport.activeTokenTransport === 'none'
        ? 'none'
        : undefined;

  const observation: RuntimeObservation = {
    ttftMs: metrics.ttftMs,
    itlAvgMs: metrics.itlAvgMs,
    itlP99Ms: metrics.itlP99Ms,
    e2eMs: metrics.e2eMs,
    prefillMs: metrics.prefillMs,
    decodeMs: metrics.decodeMs,
    nativeGpuMs: metrics.nativeGpuMs,
    nativeSyncMs: metrics.nativeSyncMs,
    nativeLogicMs: metrics.nativeLogicMs,
    inputTokens: metrics.inputTokens,
    outputTokens: metrics.outputTokens,
    cacheHits: metrics.cacheHits,
    prefillTokens: metrics.prefillTokens,
    tokensPerSecond: metrics.decodeMs > 0 ? (metrics.outputTokens / metrics.decodeMs) * 1000 : 0,
    prefillTokensPerSecond:
      metrics.prefillMs >= 0.1 && metrics.prefillTokens >= 1
        ? (metrics.prefillTokens / metrics.prefillMs) * 1000
        : 0,


    execution: {
      mode: transport.executionMode,
      workerBacked: transport.workerBacked,
      tokenPath,
    },
  };

  includeFinite(observation, 'jsStreamingDrainMs', transport.streamingDrainMs);
  includeFinite(observation, 'jsStreamingDrainCount', transport.streamingDrainCount);
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
