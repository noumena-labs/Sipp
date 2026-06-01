import type {
  BackendObservability,
  CacheSource,
  EmbeddingOutput,
  GenerateRequestId,
  GenerateResponse,
  KvReuseMode,
  NativeRuntimeConfig,
  PoolingType,
  RequestObservabilityMetrics,
} from '../engine/inference-types.js';
import type {
  ClassifiedAsset,
  ModelDetectionMethod,
  ModelDetectionResult,
  PairingPlan,
  RuntimePairingErrorCode,
} from '../models/types.js';
import {
  QueryError,
  type AssetRecord,
  type ModelInfo,
  type ObservabilityEvent,
  type ObservabilityEventType,
  type ObservabilitySnapshot,
  type QueryErrorCode,
  type RegistryManifest,
} from '../models/types.js';
import type { ChatBoundaryInfo } from '../engine/chat-boundary-sanitizer.js';
import type { ChatMessage } from '../engine/inference-types.js';
import { EngineModule } from './engine-module.js';
import { withDerivedObservabilityMetrics } from '../engine/inference-types.js';
import type { SharedTokenRingDescriptor } from '../runtime/shared-token-ring.js';
import { createAbortError } from '../utils/abort.js';
import { assertGrammarByteSize } from '../utils/grammar.js';

export const COMPLETED_REQUEST_STATUS_PENDING = 0;
export const COMPLETED_REQUEST_STATUS_COMPLETED = 1;
const COMPLETED_REQUEST_STATUS_CANCELLED = 2;
const COMPLETED_REQUEST_STATUS_FAILED = 3;
const COMPLETED_REQUEST_STATUS_UNKNOWN = 4;
const COMPLETED_REQUEST_OUTPUT_TEXT = 1;
const COMPLETED_REQUEST_OUTPUT_EMBEDDING = 2;

const RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES = 96;
const RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT = 9;
const SCHEDULER_LOOP_RESULT_SIZE_BYTES = 16;
const utf8Decoder = new TextDecoder('utf-8', { fatal: false });

function decodeWasmUtf8(bytes: Uint8Array): string {
  const input = bytes.buffer instanceof SharedArrayBuffer ? new Uint8Array(bytes) : bytes;
  return utf8Decoder.decode(input);
}

function validateGrammarSize(grammar: string | undefined): void {
  assertGrammarByteSize(grammar);
}

export type WasmSchedulerProgressResult = {
  stepResult: number;
  completedResponseCount: number;
};

export type BrowserCacheLayout = 'single-file' | 'split-gguf';

const DEFAULT_GGUF_METADATA_PREFIX_BYTES = 8 * 1024 * 1024;

interface GgufJsonResponse<T> {
  ok: boolean;
  value?: T;
  error?: {
    code: string;
    message: string;
  };
}

type RustModelDetectionResult = Omit<ModelDetectionResult, 'detectionMethod'> & {
  detectionMethod: ModelDetectionMethod | 'gguf_metadata';
};

interface PairingValidationResponse {
  ok: boolean;
  plan?: PairingPlan;
  error?: {
    code: RuntimePairingErrorCode | string;
    message: string;
  };
}

interface RustLifecycleResponse<T> {
  ok: boolean;
  value?: T;
  error?: {
    code: QueryErrorCode | string;
    message: string;
  };
}

type RustLifecycleHandle = number;
type RustLifecycleBackendPreference = 'auto' | 'cpu' | 'webgpu';

interface RustLifecycleCreateValue {
  handle: RustLifecycleHandle;
  manifest: RegistryManifest;
  snapshot: ObservabilitySnapshot;
}

interface RustLifecycleLoadSourceInstalled {
  kind: 'installed';
  id: string;
  classifiedProjectors?: ClassifiedAsset[];
}

interface RustLifecycleLoadSourceAssets {
  kind: 'assets';
  assets: AssetRecord[];
  classified: ClassifiedAsset[];
  explicitProjectorAssetId?: string | null;
  classifiedProjectors?: ClassifiedAsset[];
}

export type RustLifecycleLoadSource =
  | RustLifecycleLoadSourceInstalled
  | RustLifecycleLoadSourceAssets;

interface RustLifecycleLoadOptions {
  backend?: RustLifecycleBackendPreference;
  runtime?: NativeRuntimeConfig;
  observability?: 'off' | 'runtime' | 'profile';
}

interface RustLifecyclePlannedAsset {
  assetId: string;
  kind: AssetRecord['kind'];
  storagePath: string;
  mountName: string;
  bytes: number;
}

export interface RustLifecyclePrepareLoadValue {
  loadId: string;
  model: ModelInfo;
  runtimeFingerprint: string;
  runtimeConfig: NativeRuntimeConfig;
  loadRequired: boolean;
  assets: RustLifecyclePlannedAsset[];
  projector?: RustLifecyclePlannedAsset | null;
  manifest: RegistryManifest;
  snapshot: ObservabilitySnapshot;
  events: ObservabilityEvent[];
}

interface RustLifecycleCommitLoad {
  loadId: string;
  modelId: string;
  runtimeFingerprint: string;
  chatTemplate?: string | null;
  bosText?: string;
  eosText?: string;
  mediaMarker?: string | null;
  runtime?: unknown;
  profile?: unknown;
}

interface RustLifecycleCommitLoadValue {
  model: ModelInfo;
  manifest: RegistryManifest;
  snapshot: ObservabilitySnapshot;
  events: ObservabilityEvent[];
}

interface RustLifecycleRemoveValue {
  removed: unknown;
  orphanedAssets: AssetRecord[];
  manifest: RegistryManifest;
  snapshot: ObservabilitySnapshot;
  events: ObservabilityEvent[];
}

export interface GgufSplitStreamCallbacks {
  readAt(offset: number, target: Uint8Array): number | void;
  openShard(path: string, index: number, count: number): number | void;
  writeShard(bytes: Uint8Array): number | void;
  closeShard(): number | void;
}

export interface GgufReadAtCallbacks {
  readAt(offset: number, target: Uint8Array): number | void;
}

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
    case 'UNSUPPORTED_OPERATION':
    case 'INVALID_MODEL_SOURCE':
    case 'INVALID_MODEL_PAIRING':
    case 'STORAGE_UNAVAILABLE':
    case 'STORAGE_QUOTA_EXCEEDED':
    case 'STORAGE_CORRUPT':
    case 'REMOTE_METADATA_UNAVAILABLE':
    case 'REMOTE_LOAD_FAILED':
    case 'QUERY_FAILED':
      return code;
    default:
      return 'QUERY_FAILED';
  }
}

export class WasmBridge {
  private _cachedDataView: DataView | null = null;

  public constructor(public readonly module: EngineModule) { }

  private ensureHeapView(): DataView {
    if (
      this._cachedDataView == null ||
      this._cachedDataView.buffer !== this.module.HEAPU8.buffer
    ) {
      this._cachedDataView = new DataView(this.module.HEAPU8.buffer);
    }
    return this._cachedDataView;
  }

  private byteOffset(ptr: number | bigint): number {
    const n = typeof ptr === 'bigint' ? Number(ptr) : ptr;
    if (!Number.isSafeInteger(n) || n < 0) {
      throw new RangeError(`Invalid wasm pointer: ${String(ptr)}`);
    }
    return n;
  }

  private heapIndex(ptr: number | bigint, bytesPerElement: number): number {
    const n = this.byteOffset(ptr);
    if (n % bytesPerElement !== 0) {
      throw new RangeError(
        `Unaligned wasm pointer ${n} for element size ${bytesPerElement}`
      );
    }
    return Math.floor(n / bytesPerElement);
  }

  public callNumber(
    ident: string,
    argTypes: string[] = [],
    args: unknown[] = []
  ): number {
    const result = this.module.ccall(ident, 'number', argTypes, args);
    if (result instanceof Promise) {
      throw new Error(`Unexpected async result while calling ${ident}.`);
    }
    return Number(result);
  }

  public async callNumberAsync(
    ident: string,
    argTypes: string[] = [],
    args: unknown[] = []
  ): Promise<number> {
    const result = this.module.ccall(ident, 'number', argTypes, args, {
      async: true,
    });
    return Number(await result);
  }

  public async loadRuntimeModel(
    modelPath: string,
    config?: NativeRuntimeConfig
  ): Promise<number> {
    const result = await this.module.ccall('CE_Init', 'number', ['string', 'string'], [
      modelPath,
      JSON.stringify(config ?? {}),
    ], {
      async: true,
    });
    return Number(result);
  }

  public readLastEngineError(): string {
    return this.copyText(
      'CE_GetLastEngineErrorSize',
      'CE_CopyLastEngineError',
      'last engine error'
    );
  }

  public close(): void {
    try {
      this.module.ccall('CE_Close', null, [], []);
    } finally {
      this.releaseReusableBuffers();
    }
  }

  public startTextRequest(
    contextKey: string,
    promptText: string,
    maxOutputTokens: number,
    grammar?: string,
    emitTokens = false
  ): GenerateRequestId {
    validateGrammarSize(grammar);
    const grammarArg = grammar ?? '';
    const requestId = this.module.ccall(
      'CE_StartTextRequest',
      'number',
      ['string', 'string', 'number', 'number', 'string'],
      [contextKey, promptText, maxOutputTokens, emitTokens ? 1 : 0, grammarArg]
    );
    if (requestId instanceof Promise) {
      throw new Error('Unexpected async result while enqueuing a request.');
    }
    return requestId as GenerateRequestId;
  }

  public startMediaRequest(
    contextKey: string,
    promptText: string,
    maxOutputTokens: number,
    media: Uint8Array[],
    grammar?: string,
    emitTokens = false
  ): GenerateRequestId {
    validateGrammarSize(grammar);
    const grammarArg = grammar ?? '';
    return this.withWasmMediaBuffers(media, (flatPtr, sizesPtr) =>
      this.callNumber(
        'CE_StartMediaRequest',
        [
          'string',
          'string',
          'number',
          'number',
          'pointer',
          'pointer',
          'number',
          'string',
        ],
        [
          contextKey,
          promptText,
          maxOutputTokens,
          media.length,
          flatPtr,
          sizesPtr,
          emitTokens ? 1 : 0,
          grammarArg,
        ]
      ) as GenerateRequestId
    );
  }

  public startChatRequest(
    contextKey: string,
    messages: readonly ChatMessage[],
    maxOutputTokens: number,
    media: Uint8Array[] = [],
    grammar?: string,
    emitTokens = false
  ): GenerateRequestId {
    validateGrammarSize(grammar);
    const grammarArg = grammar ?? '';
    return this.withWasmMediaBuffers(media, (flatPtr, sizesPtr) =>
      this.callNumber(
        'CE_StartChatRequest',
        [
          'string',
          'string',
          'number',
          'number',
          'pointer',
          'pointer',
          'number',
          'string',
        ],
        [
          contextKey,
          JSON.stringify(messages),
          maxOutputTokens,
          media.length,
          flatPtr,
          sizesPtr,
          emitTokens ? 1 : 0,
          grammarArg,
        ]
      ) as GenerateRequestId
    );
  }

  public startEmbeddingRequest(
    contextKey: string,
    input: string,
    normalize: boolean
  ): GenerateRequestId {
    const requestId = this.module.ccall(
      'CE_StartEmbeddingRequest',
      'number',
      ['string', 'string', 'number'],
      [contextKey, input, normalize ? 1 : 0]
    );
    if (requestId instanceof Promise) {
      throw new Error('Unexpected async result while enqueuing an embedding request.');
    }
    return requestId as GenerateRequestId;
  }

  public readMediaMarker(): string | null {
    const ptr = this.callNumber('CE_GetMediaMarker');
    if (!ptr) {
      return null;
    }
    const marker = this.readUtf8String(ptr);
    return marker.length > 0 ? marker : null;
  }

  public readNativeChatTemplate(): string | null {
    const ptr = this.callNumber('CE_GetChatTemplate');
    if (!ptr) {
      return null;
    }
    const template = this.readUtf8String(ptr);
    return template.length > 0 ? template : null;
  }

  public getBosText(): string {
    return this.callOwnedString('CE_GetBosText');
  }

  public getEosText(): string {
    return this.callOwnedString('CE_GetEosText');
  }

  /**
   * Applies llama.cpp's native chat template (via common_chat_format_single)
   * to a set of OpenAI-style chat messages and returns the formatted prompt
   * text. Returns '' when the model has no embedded chat template.
   */
  public probeChatTemplateBoundaryInfo(): ChatBoundaryInfo {
    const raw = this.callOwnedString('CE_ProbeChatBoundaryInfo');
    if (raw.trim().length === 0) {
      throw new Error('Rust chat template boundary probe returned an empty response.');
    }
    try {
      return JSON.parse(raw) as ChatBoundaryInfo;
    } catch (error) {
      throw new Error('Rust chat template boundary probe returned invalid JSON.', {
        cause: error,
      });
    }
  }

  public validatePairing(
    classified: readonly ClassifiedAsset[],
    explicitProjectorId?: string | null
  ): PairingValidationResponse {
    const raw = this.callOwnedString(
      'CE_PairingValidate',
      ['string', 'string'],
      [JSON.stringify(classified), explicitProjectorId ?? '']
    );
    try {
      return JSON.parse(raw) as PairingValidationResponse;
    } catch (error) {
      throw new Error('Rust pairing validation returned invalid JSON.', { cause: error });
    }
  }

  public async cancelQuery(requestId: GenerateRequestId): Promise<boolean> {
    const result = this.module.ccall(
      'CE_CancelRequest',
      'number',
      ['number'],
      [requestId]
    );
    return result instanceof Promise ? Boolean(await result) : Boolean(result);
  }

  public getCompletedRequestStatus(requestId: GenerateRequestId): number {
    return this.callNumber('CE_GetCompletedRequestStatus', ['number'], [requestId]);
  }

  public consumeCompletedRequest(requestId: GenerateRequestId): boolean {
    return Boolean(this.callNumber('CE_ConsumeCompletedRequest', ['number'], [requestId]));
  }

  public consumeCompletedResponseIfPresent(requestId: GenerateRequestId): boolean {
    const status = this.getCompletedRequestStatus(requestId);
    if (status === COMPLETED_REQUEST_STATUS_UNKNOWN) {
      return false;
    }
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      return false;
    }
    if (!this.consumeCompletedRequest(requestId)) {
      throw new Error('Failed to consume completed queued request response.');
    }
    return true;
  }

  public async getBackendObservabilityJson(): Promise<string | null> {
    const rawPtr = await this.module.ccall('CE_GetBackendObservabilityJson', 'pointer', [], [], {
      async: true,
    });
    const ptr = rawPtr as number;
    if (!ptr) {
      return null;
    }

    try {
      return this.readUtf8String(ptr);
    } finally {
      this.module.ccall('CE_FreeString', null, ['pointer'], [ptr]);
    }
  }

  public rustBrowserEngineAbiVersion(): number {
    return this.callNumber('CE_RustBrowserEngineAbiVersion');
  }

  public rustBrowserEngineCreate(): number {
    return this.callNumber('CE_RustBrowserEngineCreate');
  }

  public rustBrowserEngineId(engine: number): number {
    return this.callNumber('CE_RustBrowserEngineId', ['number'], [engine]);
  }

  public rustBrowserEngineClose(engine: number): number {
    return this.callNumber('CE_RustBrowserEngineClose', ['number'], [engine]);
  }

  public modelServiceCreate(config: {
    manifest?: RegistryManifest | null;
  } = {}): RustLifecycleResponse<RustLifecycleCreateValue> {
    return this.callLifecycleJson<RustLifecycleCreateValue>(
      'CE_ModelServiceCreate',
      ['string'],
      [JSON.stringify(config)]
    );
  }

  public modelServiceClose(handle: RustLifecycleHandle): boolean {
    return Boolean(this.callNumber('CE_ModelServiceClose', ['number'], [handle]));
  }

  public modelServiceList(
    handle: RustLifecycleHandle
  ): RustLifecycleResponse<ModelInfo[]> {
    return this.callLifecycleJson<ModelInfo[]>('CE_ModelServiceList', ['number'], [handle]);
  }

  public modelServiceCurrent(
    handle: RustLifecycleHandle
  ): RustLifecycleResponse<ModelInfo | null> {
    return this.callLifecycleJson<ModelInfo | null>('CE_ModelServiceCurrent', ['number'], [handle]);
  }

  public modelServiceManifest(
    handle: RustLifecycleHandle
  ): RustLifecycleResponse<RegistryManifest> {
    return this.callLifecycleJson<RegistryManifest>('CE_ModelServiceManifest', ['number'], [handle]);
  }

  public modelServicePrepareLoad(
    handle: RustLifecycleHandle,
    source: RustLifecycleLoadSource,
    options: RustLifecycleLoadOptions = {}
  ): RustLifecycleResponse<RustLifecyclePrepareLoadValue> {
    return this.callLifecycleJson<RustLifecyclePrepareLoadValue>(
      'CE_ModelServicePrepareLoad',
      ['number', 'string', 'string'],
      [handle, JSON.stringify(source), JSON.stringify(options)]
    );
  }

  public modelServiceCommitLoad(
    handle: RustLifecycleHandle,
    commit: RustLifecycleCommitLoad
  ): RustLifecycleResponse<RustLifecycleCommitLoadValue> {
    return this.callLifecycleJson<RustLifecycleCommitLoadValue>(
      'CE_ModelServiceCommitLoad',
      ['number', 'string'],
      [handle, JSON.stringify(commit)]
    );
  }

  public modelServiceAbortLoad(
    handle: RustLifecycleHandle,
    error: { message?: string } = {}
  ): RustLifecycleResponse<ObservabilitySnapshot> {
    return this.callLifecycleJson<ObservabilitySnapshot>(
      'CE_ModelServiceAbortLoad',
      ['number', 'string'],
      [handle, JSON.stringify(error)]
    );
  }

  public modelServiceRemove(
    handle: RustLifecycleHandle,
    modelId: string
  ): RustLifecycleResponse<RustLifecycleRemoveValue> {
    return this.callLifecycleJson<RustLifecycleRemoveValue>(
      'CE_ModelServiceRemove',
      ['number', 'string'],
      [handle, modelId]
    );
  }

  public modelServiceUnload(
    handle: RustLifecycleHandle
  ): RustLifecycleResponse<ObservabilitySnapshot> {
    return this.callLifecycleJson<ObservabilitySnapshot>('CE_ModelServiceUnload', ['number'], [handle]);
  }

  public modelServiceSnapshot(
    handle: RustLifecycleHandle
  ): RustLifecycleResponse<ObservabilitySnapshot> {
    return this.callLifecycleJson<ObservabilitySnapshot>('CE_ModelServiceSnapshot', ['number'], [handle]);
  }

  public modelServiceDrainEvents(
    handle: RustLifecycleHandle
  ): RustLifecycleResponse<ObservabilityEvent[]> {
    return this.callLifecycleJson<ObservabilityEvent[]>('CE_ModelServiceDrainEvents', ['number'], [handle]);
  }

  public modelServiceRecordEvent(
    handle: RustLifecycleHandle,
    type: ObservabilityEventType,
    patch: Record<string, unknown>
  ): RustLifecycleResponse<ObservabilitySnapshot> {
    return this.callLifecycleJson<ObservabilitySnapshot>(
      'CE_ModelServiceRecordEvent',
      ['number', 'string', 'string'],
      [handle, type, JSON.stringify(patch)]
    );
  }

  public sha256Text(value: string): string {
    const bytes = new TextEncoder().encode(value);
    return this.withSha256((handle) => {
      this.updateSha256(handle, bytes);
    });
  }

  public async sha256Blob(blob: Blob, signal?: AbortSignal): Promise<string> {
    if (signal?.aborted) {
      throw createAbortError('Hashing aborted.');
    }
    const reader = blob.stream().getReader();
    try {
      return await this.withSha256((handle) => {
        return (async () => {
          while (true) {
            if (signal?.aborted) {
              throw createAbortError('Hashing aborted.');
            }
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            if (value != null && value.byteLength > 0) {
              this.updateSha256(handle, value);
            }
          }
        })();
      });
    } catch (error) {
      try {
        await reader.cancel(error);
      } catch {}
      throw error;
    }
  }

  public browserCacheLayout(
    sourceBytes: number,
    sourceBytesKnown: boolean,
    directLoadMaxBytes: number,
    shardMaxBytes: number
  ): BrowserCacheLayout {
    const layout = this.callNumber(
      'CE_BrowserCacheLayout',
      ['number', 'number', 'number', 'number'],
      [sourceBytes, sourceBytesKnown ? 1 : 0, directLoadMaxBytes, shardMaxBytes]
    );
    if (layout === 0) {
      return 'single-file';
    }
    if (layout === 1) {
      return 'split-gguf';
    }
    throw new Error(`Rust browser cache layout failed with status ${layout}.`);
  }

  public async detectModelFromGgufFile(
    file: Blob & { name?: string },
    signal?: AbortSignal
  ): Promise<ModelDetectionResult> {
    const bytes = await this.readGgufMetadataPrefix(file, signal);
    const fileName =
      typeof file.name === 'string' && file.name.trim().length > 0
        ? file.name
        : 'model.gguf';
    const detection = this.withWasmBytes(bytes, (ptr, len) => {
      const raw = this.callOwnedString(
        'CE_DetectModelFromGgufBytes',
        ['string', 'pointer', 'number'],
        [fileName, ptr, len]
      );
      return this.unwrapGgufResponse<RustModelDetectionResult>(
        raw,
        'GGUF model detection'
      );
    });
    return {
      ...detection,
      detectionMethod: normalizeModelDetectionMethod(detection.detectionMethod),
    };
  }

  public planGgufSplitCount(
    sourceBytes: number,
    shardMaxBytes: number,
    callbacks: GgufReadAtCallbacks
  ): number {
    const readAtPtr = this.module.addFunction(
      (_userData: number, offset: bigint | number, dstPtr: number, len: number) => {
        const start = this.byteOffset(dstPtr);
        const target = this.module.HEAPU8.subarray(start, start + len);
        return callbacks.readAt(this.byteOffset(offset), target) ?? 0;
      },
      'iijii'
    );

    try {
      const count = this.callNumber(
        'CE_GgufPlanSplitCount',
        ['number', 'number', 'number', 'number'],
        [sourceBytes, shardMaxBytes, 0, readAtPtr]
      );
      if (count <= 0) {
        throw new Error(`Rust GGUF split planning failed with status ${count}.`);
      }
      return count;
    } finally {
      this.module.removeFunction(readAtPtr);
    }
  }

  public splitGgufStream(
    sourceBytes: number,
    outputPrefix: string,
    shardMaxBytes: number,
    callbacks: GgufSplitStreamCallbacks
  ): void {
    const readAtPtr = this.module.addFunction(
      (_userData: number, offset: bigint | number, dstPtr: number, len: number) => {
        const start = this.byteOffset(dstPtr);
        const target = this.module.HEAPU8.subarray(start, start + len);
        return callbacks.readAt(this.byteOffset(offset), target) ?? 0;
      },
      'iijii'
    );
    const openShardPtr = this.module.addFunction(
      (_userData: number, pathPtr: number, index: number, count: number) =>
        callbacks.openShard(this.readUtf8String(pathPtr), index, count) ?? 0,
      'iiiii'
    );
    const writeShardPtr = this.module.addFunction(
      (_userData: number, bytesPtr: number, len: number) => {
        const start = this.byteOffset(bytesPtr);
        const bytes = this.module.HEAPU8.subarray(start, start + len);
        return callbacks.writeShard(bytes) ?? 0;
      },
      'iiii'
    );
    const closeShardPtr = this.module.addFunction(
      () => callbacks.closeShard() ?? 0,
      'ii'
    );

    try {
      const status = this.callNumber(
        'CE_GgufSplitStream',
        ['number', 'string', 'number', 'number', 'number', 'number', 'number', 'number'],
        [
          sourceBytes,
          outputPrefix,
          shardMaxBytes,
          0,
          readAtPtr,
          openShardPtr,
          writeShardPtr,
          closeShardPtr,
        ]
      );
      if (status !== 0) {
        throw new Error(`Rust GGUF stream split failed with status ${status}.`);
      }
    } finally {
      this.module.removeFunction(readAtPtr);
      this.module.removeFunction(openShardPtr);
      this.module.removeFunction(writeShardPtr);
      this.module.removeFunction(closeShardPtr);
    }
  }

  public readRuntimeObservability(): RequestObservabilityMetrics | null {
    return this.readRuntimeObservabilityViaCall('CE_GetRuntimeObservability', [], []);
  }

  public readCompletedRequestRuntimeObservability(
    requestId: GenerateRequestId
  ): RequestObservabilityMetrics | null {
    return this.readRuntimeObservabilityViaCall(
      'CE_GetCompletedRequestRuntimeObservability',
      ['number'],
      [requestId]
    );
  }

  public takeCompletedResponse(requestId: GenerateRequestId): GenerateResponse {
    const status = this.getCompletedRequestStatus(requestId);
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      throw new Error('Queued request reached a terminal step without a completed response.');
    }
    if (status === COMPLETED_REQUEST_STATUS_UNKNOWN) {
      throw new Error('Queued request response is no longer available.');
    }

    const outputKind = this.callNumber('CE_GetCompletedRequestOutputKind', ['number'], [requestId]);
    if (outputKind === COMPLETED_REQUEST_OUTPUT_TEXT) {
      const outputText = this.copyText(
        'CE_GetCompletedRequestOutputSize',
        'CE_CopyCompletedRequestOutput',
        'output',
        ['number'],
        [requestId]
      );
      return {
        ...this.completedResponseBase(requestId, status),
        outputText,
      };
    }
    if (outputKind === COMPLETED_REQUEST_OUTPUT_EMBEDDING) {
      const embedding = this.readCompletedEmbedding(requestId);
      return {
        ...this.completedResponseBase(requestId, status),
        embedding,
      };
    }
    throw new Error(`Completed request ${requestId} has unknown output kind ${outputKind}.`);
  }

  private completedResponseBase(
    requestId: GenerateRequestId,
    status: number
  ): Omit<GenerateResponse, 'outputText' | 'embedding'> {
    const errorText = this.copyText(
      'CE_GetCompletedRequestErrorSize',
      'CE_CopyCompletedRequestError',
      'error',
      ['number'],
      [requestId]
    );
    const runtimeObservability = this.readCompletedRequestRuntimeObservability(requestId);
    if (!this.consumeCompletedRequest(requestId)) {
      throw new Error('Failed to consume completed queued request response.');
    }

    return {
      requestId,
      completed: status === COMPLETED_REQUEST_STATUS_COMPLETED,
      failed: status === COMPLETED_REQUEST_STATUS_FAILED,
      cancelled: status === COMPLETED_REQUEST_STATUS_CANCELLED,
      errorMessage: errorText.length > 0 ? errorText : null,
      observability: runtimeObservability,
    };
  }

  public async runInferenceLoop(
    maxTicks: number,
    maxCompletedResponses: number,
    maxGeneratedTokens: number,
    options: {
      maxDurationUs?: number;
    } = {}
  ): Promise<WasmSchedulerProgressResult> {
    const maxDurationUs = Math.max(0, options.maxDurationUs ?? 0);
    const resultPtr = this.ensureLoopResultBuffer();

    const stepResult = await this.callNumberAsync(
      'CE_RunSchedulerLoop',
      ['number', 'number', 'number', 'number', 'pointer'],
      [
        maxTicks,
        maxCompletedResponses,
        maxGeneratedTokens,
        maxDurationUs,
        resultPtr,
      ]
    );

    const loopResult = this.readSchedulerLoopResult(resultPtr);
    return {
      stepResult,
      completedResponseCount: loopResult.completedResponseCount,
    };
  }

  public getSharedTokenRingDescriptor(): SharedTokenRingDescriptor {
    const headerOffset = this.callNumber('CE_GetTokenRingHeaderAddress');
    const bodyOffset = this.callNumber('CE_GetTokenRingBodyAddress');
    const bodyCapacity = this.callNumber('CE_GetTokenRingCapacity');
    return {
      buffer: this.module.HEAPU8.buffer,
      headerOffset,
      bodyOffset,
      bodyCapacity,
    };
  }

  public releaseReusableBuffers(): void {
    if (this.reusableLoopResultPtr !== 0) {
      this.free(this.reusableLoopResultPtr);
      this.reusableLoopResultPtr = 0;
    }
  }

  private allocate(size: number): number {
    if (!Number.isSafeInteger(size) || size <= 0) {
      throw new RangeError(`Invalid wasm allocation size: ${size}`);
    }
    const ptr = Number(this.module._malloc(size));
    if (ptr === 0) {
      throw new Error(`WASM allocation failed for ${size} bytes.`);
    }
    return ptr;
  }

  private free(ptr: number): void {
    this.module._free(ptr);
  }

  private async readGgufMetadataPrefix(
    blob: Blob,
    signal?: AbortSignal
  ): Promise<Uint8Array> {
    if (signal?.aborted) {
      throw createAbortError('GGUF metadata read aborted.');
    }
    const byteLength = Math.min(blob.size, DEFAULT_GGUF_METADATA_PREFIX_BYTES);
    const bytes = new Uint8Array(await blob.slice(0, byteLength).arrayBuffer());
    if (signal?.aborted) {
      throw createAbortError('GGUF metadata read aborted.');
    }
    return bytes;
  }

  private withWasmBytes<T>(
    bytes: Uint8Array,
    operation: (ptr: number, len: number) => T
  ): T {
    const ptr = this.allocate(Math.max(1, bytes.byteLength));
    try {
      if (bytes.byteLength > 0) {
        this.module.HEAPU8.set(bytes, ptr);
      }
      return operation(ptr, bytes.byteLength);
    } finally {
      this.free(ptr);
    }
  }

  private withWasmMediaBuffers<T>(
    media: readonly Uint8Array[],
    operation: (flatPtr: number, sizesPtr: number) => T
  ): T {
    const totalBytes = media.reduce((sum, image) => sum + image.byteLength, 0);
    const flatPtr = this.allocate(Math.max(1, totalBytes));
    const sizesPtr = this.allocate(Math.max(1, media.length * 4));

    try {
      let offset = 0;
      for (let index = 0; index < media.length; index += 1) {
        const image = media[index];
        this.module.HEAPU8.set(image, flatPtr + offset);
        this.module.HEAP32[this.heapIndex(sizesPtr, 4) + index] = image.byteLength;
        offset += image.byteLength;
      }
      return operation(flatPtr, sizesPtr);
    } finally {
      this.free(flatPtr);
      this.free(sizesPtr);
    }
  }

  private callOwnedString(
    ident: string,
    argTypes: string[] = [],
    args: unknown[] = []
  ): string {
    const ptr = this.callNumber(ident, argTypes, args);
    if (!ptr) {
      return '';
    }
    try {
      return this.readUtf8String(ptr);
    } finally {
      this.module.ccall('CE_FreeString', null, ['pointer'], [ptr]);
    }
  }

  private readUtf8String(ptr: number | bigint, byteLength?: number): string {
    const start = this.byteOffset(ptr);
    const heap = this.module.HEAPU8;
    let end = start;
    if (byteLength == null) {
      while (end < heap.length && heap[end] !== 0) {
        end += 1;
      }
    } else {
      end = start + byteLength;
    }
    return decodeWasmUtf8(heap.subarray(start, end));
  }

  private unwrapGgufResponse<T>(raw: string, label: string): T {
    let parsed: GgufJsonResponse<T>;
    try {
      parsed = JSON.parse(raw) as GgufJsonResponse<T>;
    } catch (error) {
      throw new Error(`Rust ${label} returned invalid JSON.`, { cause: error });
    }
    if (parsed.ok) {
      if (!Object.prototype.hasOwnProperty.call(parsed, 'value')) {
        throw new Error(`Rust ${label} response omitted value.`);
      }
      return parsed.value as T;
    }
    throw new Error(parsed.error?.message ?? `Rust ${label} failed.`);
  }

  private callLifecycleJson<T>(
    ident: string,
    argTypes: string[] = [],
    args: unknown[] = []
  ): RustLifecycleResponse<T> {
    const raw = this.callOwnedString(ident, argTypes, args);
    try {
      return JSON.parse(raw) as RustLifecycleResponse<T>;
    } catch (error) {
      return {
        ok: false,
        error: {
          code: 'STORAGE_CORRUPT',
          message: `Rust lifecycle response from ${ident} was invalid JSON.`,
        },
      };
    }
  }

  private withSha256<T>(operation: (handle: number) => T): T extends Promise<unknown> ? Promise<string> : string {
    const handle = this.callNumber('CE_Sha256Create');
    if (!handle) {
      throw new Error('Failed to create Rust SHA-256 hasher.');
    }
    let finalized = false;
    const finalize = (): string => {
      finalized = true;
      const digest = this.callOwnedString('CE_Sha256Finalize', ['number'], [handle]);
      if (digest.length !== 64) {
        throw new Error('Rust SHA-256 hasher returned an invalid digest.');
      }
      return digest;
    };
    try {
      const result = operation(handle);
      if (result instanceof Promise) {
        return result.then(finalize).finally(() => {
          if (!finalized) {
            this.callNumber('CE_Sha256Close', ['number'], [handle]);
          }
        }) as T extends Promise<unknown> ? Promise<string> : string;
      }
      return finalize() as T extends Promise<unknown> ? Promise<string> : string;
    } catch (error) {
      if (!finalized) {
        this.callNumber('CE_Sha256Close', ['number'], [handle]);
      }
      throw error;
    }
  }

  private updateSha256(handle: number, bytes: Uint8Array): void {
    this.withWasmBytes(bytes, (ptr, len) => {
      const status = this.callNumber('CE_Sha256Update', ['number', 'pointer', 'number'], [
        handle,
        ptr,
        len,
      ]);
      if (status !== 0) {
        throw new Error(`Rust SHA-256 update failed with status ${status}.`);
      }
    });
  }

  private reusableLoopResultPtr = 0;
  private ensureLoopResultBuffer(): number {
    if (this.reusableLoopResultPtr === 0) {
      this.reusableLoopResultPtr = this.allocate(SCHEDULER_LOOP_RESULT_SIZE_BYTES);
    }
    return this.reusableLoopResultPtr;
  }



  private readSchedulerLoopResult(ptr: number): {
    ticksExecuted: number;
    progressedTicks: number;
    completedResponseCount: number;
    emittedTokenCount: number;
  } {
    const view = this.ensureHeapView();
    const offset = this.byteOffset(ptr);
    return {
      ticksExecuted: view.getInt32(offset, true),
      progressedTicks: view.getInt32(offset + 4, true),
      completedResponseCount: view.getInt32(offset + 8, true),
      emittedTokenCount: view.getInt32(offset + 12, true),
    };
  }

  private readRuntimeObservabilityViaCall(
    ident: string,
    argTypes: string[],
    args: unknown[]
  ): RequestObservabilityMetrics | null {
    const metricsPtr = this.allocate(RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES);
    try {
      const status = this.callNumber(ident, [...argTypes, 'pointer'], [...args, metricsPtr]);
      if (status !== 0) {
        return null;
      }

      const view = this.ensureHeapView();
      const offset = this.byteOffset(metricsPtr);
      const doublesOffset = offset;
      const intsOffset = offset + RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT * 8;

      return withDerivedObservabilityMetrics({
        ttftMs: view.getFloat64(doublesOffset, true),
        itlAvgMs: view.getFloat64(doublesOffset + 8, true),
        itlP99Ms: view.getFloat64(doublesOffset + 16, true),
        e2eMs: view.getFloat64(doublesOffset + 24, true),
        prefillMs: view.getFloat64(doublesOffset + 32, true),
        decodeMs: view.getFloat64(doublesOffset + 40, true),
        nativeGpuMs: view.getFloat64(doublesOffset + 48, true),
        nativeSyncMs: view.getFloat64(doublesOffset + 56, true),
        nativeLogicMs: view.getFloat64(doublesOffset + 64, true),

        inputTokens: view.getInt32(intsOffset, true),
        outputTokens: view.getInt32(intsOffset + 4, true),
        cacheMode: cacheModeFromCode(view.getInt32(intsOffset + 8, true)),
        cacheSource: cacheSourceFromCode(view.getInt32(intsOffset + 12, true)),
        cacheHits: view.getInt32(intsOffset + 16, true),
        prefillTokens: view.getInt32(intsOffset + 20, true),
      });
    } finally {
      this.free(metricsPtr);
    }
  }

  private readCompletedEmbedding(requestId: GenerateRequestId): EmbeddingOutput {
    const length = this.callNumber('CE_GetCompletedRequestEmbeddingLength', ['number'], [requestId]);
    if (length < 0) {
      throw new Error('Completed request did not expose an embedding vector.');
    }
    const pooling = poolingTypeFromCode(
      this.callNumber('CE_GetCompletedRequestEmbeddingPooling', ['number'], [requestId])
    );
    const normalizedValue = this.callNumber(
      'CE_GetCompletedRequestEmbeddingNormalized',
      ['number'],
      [requestId]
    );
    if (normalizedValue < 0) {
      throw new Error('Failed to read embedding normalization flag.');
    }
    const normalized = normalizedValue !== 0;
    const bufferPtr = this.allocate(Math.max(1, length * 4));
    try {
      const copied = this.callNumber(
        'CE_CopyCompletedRequestEmbedding',
        ['number', 'pointer', 'number'],
        [requestId, bufferPtr, length]
      );
      if (copied !== length) {
        throw new Error('Failed to copy embedding output.');
      }
      const values = Array.from(
        this.module.HEAPF32.subarray(
          this.heapIndex(bufferPtr, 4),
          this.heapIndex(bufferPtr, 4) + length
        )
      );
      return {
        values,
        pooling,
        normalized,
      };
    } finally {
      this.free(bufferPtr);
    }
  }

  private copyText(
    sizeFunction: string,
    copyFunction: string,
    fieldName: string,
    argTypes: string[] = [],
    args: unknown[] = []
  ): string {
    const byteLength = this.callNumber(sizeFunction, argTypes, args);
    if (byteLength < 0) {
      throw new Error(`Failed to read ${fieldName} size.`);
    }
    if (byteLength === 0) {
      return '';
    }

    const bufferPtr = this.allocate(byteLength + 1);
    try {
      const copied = this.callNumber(copyFunction, [...argTypes, 'pointer', 'number'], [
        ...args,
        bufferPtr,
        byteLength + 1,
      ]);
      if (copied !== byteLength) {
        throw new Error(`Failed to copy ${fieldName}.`);
      }
      return this.readUtf8String(bufferPtr, byteLength);
    } finally {
      this.free(bufferPtr);
    }
  }
}

export function parseBackendObservabilityJson(raw: string): BackendObservability {
  return JSON.parse(raw) as BackendObservability;
}

function poolingTypeFromCode(value: number): PoolingType {
  switch (value) {
    case -1:
      return 'unspecified';
    case 0:
      return 'none';
    case 1:
      return 'mean';
    case 2:
      return 'cls';
    case 3:
      return 'last';
    case 4:
      return 'rank';
    default:
      throw new Error(`Unknown embedding pooling type ${value}.`);
  }
}

function cacheModeFromCode(value: number): KvReuseMode {
  switch (value) {
    case 0:
      return 'disabled';
    case 1:
      return 'live_slot_prefix';
    case 2:
      return 'state_snapshot';
    case 3:
      return 'live_slot_and_snapshot';
    default:
      throw new Error(`Unknown cache mode ${value}.`);
  }
}

function cacheSourceFromCode(value: number): CacheSource {
  switch (value) {
    case 0:
      return 'none';
    case 1:
      return 'live';
    case 2:
      return 'snapshot';
    default:
      throw new Error(`Unknown cache source ${value}.`);
  }
}

function normalizeModelDetectionMethod(
  value: RustModelDetectionResult['detectionMethod']
): ModelDetectionMethod {
  return value === 'gguf_metadata' ? 'gguf-metadata' : value;
}
