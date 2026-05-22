import type {
  ModelInfo,
  ObservabilityEvent,
  ObservabilityEventType,
  ObservabilitySnapshot,
  QueryErrorCode,
  RegistryManifest,
} from '../models/types.js';
import { QueryError } from '../models/types.js';
import type {
  RustLifecycleCommitLoad,
  RustLifecycleCommitLoadValue,
  RustLifecycleCreateValue,
  RustLifecycleHandle,
  RustLifecycleLoadOptions,
  RustLifecycleLoadSource,
  RustLifecyclePrepareLoadValue,
  RustLifecycleRemoveValue,
  RustLifecycleResponse,
} from './wasm-bridge.js';
import { WasmBridge } from './wasm-bridge.js';

export class RustLifecycleBridge {
  private closed = false;

  private constructor(
    private readonly bridge: WasmBridge,
    private readonly handle: RustLifecycleHandle
  ) {}

  public static create(bridge: WasmBridge, manifest: RegistryManifest): RustLifecycleBridge {
    const created = unwrapLifecycleResponse<RustLifecycleCreateValue>(
      bridge.modelServiceCreate({ manifest }),
      'create model lifecycle service'
    );
    return new RustLifecycleBridge(bridge, created.handle);
  }

  public list(): ModelInfo[] {
    return unwrapLifecycleResponse(this.bridge.modelServiceList(this.handle), 'list models');
  }

  public current(): ModelInfo | null {
    return unwrapLifecycleResponse(this.bridge.modelServiceCurrent(this.handle), 'read current model');
  }

  public manifest(): RegistryManifest {
    return unwrapLifecycleResponse(this.bridge.modelServiceManifest(this.handle), 'read manifest');
  }

  public prepareLoad(
    source: RustLifecycleLoadSource,
    options: RustLifecycleLoadOptions
  ): RustLifecyclePrepareLoadValue {
    return unwrapLifecycleResponse(
      this.bridge.modelServicePrepareLoad(this.handle, source, options),
      'prepare model load'
    );
  }

  public commitLoad(commit: RustLifecycleCommitLoad): RustLifecycleCommitLoadValue {
    return unwrapLifecycleResponse(
      this.bridge.modelServiceCommitLoad(this.handle, commit),
      'commit model load'
    );
  }

  public abortLoad(error: { message?: string }): ObservabilitySnapshot {
    return unwrapLifecycleResponse(
      this.bridge.modelServiceAbortLoad(this.handle, error),
      'abort model load'
    );
  }

  public remove(modelId: string): RustLifecycleRemoveValue {
    return unwrapLifecycleResponse(
      this.bridge.modelServiceRemove(this.handle, modelId),
      'remove model'
    );
  }

  public unload(): ObservabilitySnapshot {
    return unwrapLifecycleResponse(this.bridge.modelServiceUnload(this.handle), 'unload model');
  }

  public snapshot(): ObservabilitySnapshot {
    return unwrapLifecycleResponse(
      this.bridge.modelServiceSnapshot(this.handle),
      'read lifecycle snapshot'
    );
  }

  public drainEvents(): ObservabilityEvent[] {
    return unwrapLifecycleResponse(
      this.bridge.modelServiceDrainEvents(this.handle),
      'drain lifecycle events'
    );
  }

  public recordEvent(
    type: ObservabilityEventType,
    patch: Record<string, unknown>
  ): ObservabilitySnapshot {
    return unwrapLifecycleResponse(
      this.bridge.modelServiceRecordEvent(this.handle, type, patch),
      'record lifecycle event'
    );
  }

  public close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.bridge.modelServiceClose(this.handle);
  }
}

export function unwrapLifecycleResponse<T>(
  response: RustLifecycleResponse<T>,
  label: string
): T {
  if (response.ok && 'value' in response) {
    return response.value as T;
  }
  const code = normalizeLifecycleErrorCode(response.error?.code);
  const message = response.error?.message ?? `Rust lifecycle failed to ${label}.`;
  throw new QueryError(code, message);
}

function normalizeLifecycleErrorCode(code: string | undefined): QueryErrorCode {
  switch (code) {
    case 'ENGINE_CLOSED':
    case 'MODEL_NOT_READY':
    case 'MODEL_NOT_FOUND':
    case 'MODEL_BROKEN':
    case 'INVALID_MODEL_SOURCE':
    case 'INVALID_MODEL_PAIRING':
    case 'STORAGE_UNAVAILABLE':
    case 'STORAGE_QUOTA_EXCEEDED':
    case 'STORAGE_CORRUPT':
    case 'REMOTE_METADATA_UNAVAILABLE':
    case 'REMOTE_LOAD_FAILED':
    case 'STREAMING_UNAVAILABLE':
    case 'QUERY_FAILED':
      return code;
    default:
      return 'QUERY_FAILED';
  }
}
