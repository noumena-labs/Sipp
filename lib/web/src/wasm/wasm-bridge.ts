import type {
  BackendObservability,
  CacheSource,
  EmbeddingOutput,
  GenerateRequestHandle,
  GenerateRequestId,
  GenerateResponse,
  KvReuseMode,
  NativeRuntimeConfig,
  PoolingType,
  SamplingRuntimeOverride,
  RequestObservabilityMetrics,
  ChatMessage,
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
  type CatalogModelInfo,
  type CatalogObservabilityEvent,
  type CatalogObservabilitySnapshot,
  type ObservabilityEventType,
  type QueryErrorCode,
  type RegistryManifest,
  type RuntimeSessionDescriptor,
  type RuntimeSessionSnapshot,
} from '../models/types.js';
import type { ChatBoundaryInfo } from '../engine/chat-boundary-sanitizer.js';
import { EngineModule } from './engine-module.js';
import {
  hasSamplingRuntimeOverrideFields,
  withDerivedObservabilityMetrics,
} from '../engine/inference-types.js';
import type { SharedTokenRingDescriptor } from '../runtime/shared-token-ring.js';
import { createAbortError } from '../utils/abort.js';
import { assertGrammarByteSize } from '../utils/grammar.js';

export const COMPLETED_REQUEST_STATUS_PENDING = 0;
export const COMPLETED_REQUEST_STATUS_COMPLETED = 1;
const COMPLETED_REQUEST_STATUS_CANCELLED = 2;
const COMPLETED_REQUEST_STATUS_FAILED = 3;
const COMPLETED_REQUEST_STATUS_UNKNOWN = 4;
const STATUS_STALE_RUNTIME_SESSION = -3;
const COMPLETED_REQUEST_OUTPUT_TEXT = 1;
const COMPLETED_REQUEST_OUTPUT_EMBEDDING = 2;
const COMPLETED_REQUEST_OUTPUT_AUDIO = 3;

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

interface WasmTextRequestOptions {
  grammar?: string;
  stop?: readonly string[];
  sampling?: SamplingRuntimeOverride;
  emitTokens?: boolean;
}

function serializeStop(stop: readonly string[] | undefined): string {
  return stop == null || stop.length === 0 ? '' : JSON.stringify(stop);
}

function serializeSampling(sampling: SamplingRuntimeOverride | undefined): string {
  return hasSamplingRuntimeOverrideFields(sampling) ? JSON.stringify(sampling) : '';
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

export interface RustLifecycleError {
  readonly code: QueryErrorCode | string;
  readonly message: string;
  readonly status?: number;
  readonly retryAfterMs?: number;
}

interface RustLifecycleResponse<T> {
  ok: boolean;
  value?: T;
  error?: RustLifecycleError;
}

type RustLifecycleHandle = number;
type RustLifecycleBackendPreference = 'auto' | 'cpu' | 'webgpu';

interface RustLifecycleCreateValue {
  handle: RustLifecycleHandle;
  manifest: RegistryManifest;
  snapshot: CatalogObservabilitySnapshot;
}

export interface RustLifecycleLoadSource {
  modelId: string;
}

export interface RustLifecycleInstallSource {
  assets: AssetRecord[];
  classified: ClassifiedAsset[];
}

export interface RustRemoteMetadata {
  readonly url: string;
  readonly name: string;
  readonly bytes: number;
  readonly etag?: string;
  readonly lastModified?: string;
}

export interface RustRemoteCacheCandidate {
  readonly candidateId: string;
  readonly assetIds: readonly string[];
  readonly metadata: RustRemoteMetadata;
}

export type RustRemoteAction =
  | {
    readonly kind: 'fetch_metadata';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
    readonly url: string;
  }
  | {
    readonly kind: 'wait';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
    readonly delayMs: number;
  }
  | {
    readonly kind: 'validate_cache';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
    readonly candidate: RustRemoteCacheCandidate;
  }
  | {
    readonly kind: 'download';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
    readonly metadata: RustRemoteMetadata;
  }
  | {
    readonly kind: 'cleanup';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
    readonly assetIds: readonly string[];
  };

export type RustRemoteFailure = {
  readonly phase: 'metadata' | 'download' | 'cache_validation' | 'cleanup';
  readonly kind: 'transport' | 'http' | 'invalid_response' | 'integrity' | 'storage';
  readonly status?: number;
  readonly retryAfter?: string;
  readonly reason: string;
};

export type RustRemoteEvent =
  | {
    readonly kind: 'metadata_succeeded';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
    readonly headers: {
      readonly contentLength?: number;
      readonly linkedSize?: number;
      readonly etag?: string;
      readonly linkedEtag?: string;
      readonly lastModified?: string;
    };
  }
  | {
    readonly kind: 'operation_failed';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
    readonly failure: RustRemoteFailure;
    readonly createdAssetIds: readonly string[];
  }
  | {
    readonly kind: 'wait_completed';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
  }
  | {
    readonly kind: 'cache_validated';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
    readonly assetIds: readonly string[];
  }
  | {
    readonly kind: 'download_succeeded';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
    readonly assetIds: readonly string[];
    readonly createdAssetIds: readonly string[];
  }
  | {
    readonly kind: 'cleanup_succeeded';
    readonly acquisitionId: string;
    readonly memberId: number;
    readonly attempt: number;
  };

export type RustRemoteCommand =
  | {
    readonly command: 'begin';
    readonly urls: readonly string[];
  }
  | {
    readonly command: 'advance';
    readonly event: RustRemoteEvent;
    readonly assets?: readonly AssetRecord[];
    readonly classified?: readonly ClassifiedAsset[];
  }
  | {
    readonly command: 'cancel';
    readonly acquisitionId: string;
  };

export type RustRemoteCommandValue =
  | { readonly kind: 'action'; readonly action: RustRemoteAction }
  | { readonly kind: 'installed'; readonly installed: RustLifecycleInstallValue }
  | { readonly kind: 'cancelled'; readonly snapshot: CatalogObservabilitySnapshot }
  | { readonly kind: 'failed'; readonly error: RustLifecycleError };

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
  model: CatalogModelInfo;
  runtimeFingerprint: string;
  runtimeConfig: NativeRuntimeConfig;
  assets: RustLifecyclePlannedAsset[];
  projector?: RustLifecyclePlannedAsset | null;
  manifest: RegistryManifest;
  snapshot: CatalogObservabilitySnapshot;
  events: CatalogObservabilityEvent[];
}

export interface RustLifecycleInstallValue {
  model: CatalogModelInfo;
  manifest: RegistryManifest;
  snapshot: CatalogObservabilitySnapshot;
  events: CatalogObservabilityEvent[];
}

interface RustLifecycleCommitLoad {
  loadId: string;
  modelId: string;
  runtimeFingerprint: string;
  runtime?: unknown;
  profile?: unknown;
}

interface RustLifecycleCommitLoadValue {
  model: CatalogModelInfo;
  manifest: RegistryManifest;
  snapshot: CatalogObservabilitySnapshot;
  events: CatalogObservabilityEvent[];
}

interface RustLifecycleRemoveValue {
  removed: unknown;
  orphanedAssets: AssetRecord[];
  manifest: RegistryManifest;
  snapshot: CatalogObservabilitySnapshot;
  events: CatalogObservabilityEvent[];
}

interface RustLifecycleRemoveRequest {
  readonly modelId: string;
  readonly activeModelId?: string;
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

/**
 * Runs an operation inside the owning runtime's serialized Wasm-bridge queue.
 *
 * Every entry into a given Wasm instance must go through one queue: a JSPI
 * inference loop can be suspended mid-call, and a concurrent catalog call would
 * re-enter the same instance while it is suspended.
 */
export type WasmBridgeRunner = <T>(operation: (bridge: WasmBridge) => T) => Promise<T>;

interface GgufCallbackFailure {
  readonly error: unknown;
}

/**
 * Catalog lifecycle handle. Every method is asynchronous because each one
 * queues behind whatever else is currently inside the Wasm instance.
 */
export class RustLifecycleBridge {
  private closed = false;

  private constructor(
    private readonly run: WasmBridgeRunner,
    private readonly handle: RustLifecycleHandle
  ) {}

  public static async create(
    run: WasmBridgeRunner,
    manifest: RegistryManifest
  ): Promise<RustLifecycleBridge> {
    const created = await run((bridge) =>
      unwrapLifecycleResponse<RustLifecycleCreateValue>(
        bridge.modelServiceCreate({ manifest }),
        'create model lifecycle service'
      )
    );
    return new RustLifecycleBridge(run, created.handle);
  }

  public async list(): Promise<CatalogModelInfo[]> {
    return await this.run((bridge) =>
      unwrapLifecycleResponse(bridge.modelServiceList(this.handle), 'list models')
    );
  }

  public async manifest(): Promise<RegistryManifest> {
    return await this.run((bridge) =>
      unwrapLifecycleResponse(bridge.modelServiceManifest(this.handle), 'read manifest')
    );
  }

  public async prepareLoad(
    source: RustLifecycleLoadSource,
    options: RustLifecycleLoadOptions
  ): Promise<RustLifecyclePrepareLoadValue> {
    return await this.run((bridge) =>
      unwrapLifecycleResponse(
        bridge.modelServicePrepareLoad(this.handle, source, options),
        'prepare model load'
      )
    );
  }

  public async install(
    source: RustLifecycleInstallSource
  ): Promise<RustLifecycleInstallValue> {
    return await this.run((bridge) =>
      unwrapLifecycleResponse(
        bridge.modelServiceInstall(this.handle, source),
        'install model'
      )
    );
  }

  public async remoteAcquisition(
    command: RustRemoteCommand
  ): Promise<RustRemoteCommandValue> {
    return await this.run((bridge) =>
      unwrapLifecycleResponse(
        bridge.modelServiceRemoteAcquisitionCommand(this.handle, command),
        'advance remote model acquisition'
      )
    );
  }

  public async commitLoad(
    commit: RustLifecycleCommitLoad
  ): Promise<RustLifecycleCommitLoadValue> {
    return await this.run((bridge) =>
      unwrapLifecycleResponse(
        bridge.modelServiceCommitLoad(this.handle, commit),
        'commit model load'
      )
    );
  }

  public async remove(
    modelId: string,
    activeModelId: string | null
  ): Promise<RustLifecycleRemoveValue> {
    return await this.run((bridge) =>
      unwrapLifecycleResponse(
        bridge.modelServiceRemove(this.handle, {
          modelId,
          ...(activeModelId == null ? {} : { activeModelId }),
        }),
        'remove model'
      )
    );
  }

  public async snapshot(): Promise<CatalogObservabilitySnapshot> {
    return await this.run((bridge) =>
      unwrapLifecycleResponse(
        bridge.modelServiceSnapshot(this.handle),
        'read lifecycle snapshot'
      )
    );
  }

  public async drainEvents(): Promise<CatalogObservabilityEvent[]> {
    return await this.run((bridge) =>
      unwrapLifecycleResponse(
        bridge.modelServiceDrainEvents(this.handle),
        'drain lifecycle events'
      )
    );
  }

  public async recordEvent(
    type: ObservabilityEventType,
    patch: Record<string, unknown>
  ): Promise<CatalogObservabilitySnapshot> {
    return await this.run((bridge) =>
      unwrapLifecycleResponse(
        bridge.modelServiceRecordEvent(this.handle, type, patch),
        'record lifecycle event'
      )
    );
  }

  public async close(): Promise<void> {
    if (this.closed) {
      return;
    }
    this.closed = true;
    await this.run((bridge) => bridge.modelServiceClose(this.handle));
  }
}

export function unwrapLifecycleResponse<T>(
  response: RustLifecycleResponse<T>,
  label: string
): T {
  if (response.ok && 'value' in response) {
    return response.value as T;
  }
  throw queryErrorFromLifecycleError(response.error, `Rust lifecycle failed to ${label}.`);
}

export function queryErrorFromLifecycleError(
  error: RustLifecycleError | undefined,
  fallbackMessage: string
): QueryError {
  const code = normalizeLifecycleErrorCode(error?.code);
  return new QueryError(code, error?.message ?? fallbackMessage, {
    status: error?.status,
    retryAfterMs: error?.retryAfterMs,
  });
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
    case 'ACQUISITION_CANCELLED':
    case 'STALE_ACQUISITION_RESULT':
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
    session: RuntimeSessionDescriptor,
    config?: NativeRuntimeConfig
  ): Promise<number> {
    const result = await this.module.ccall('CE_Init', 'number', ['string', 'string', 'string'], [
      modelPath,
      JSON.stringify(config ?? {}),
      JSON.stringify(session),
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

  public async close(): Promise<void> {
    try {
      await this.module.ccall('CE_Close', null, [], [], { async: true });
    } finally {
      this.releaseReusableBuffers();
    }
  }

  public getRuntimeSession(): RuntimeSessionSnapshot {
    const rawPtr = this.module.ccall('CE_GetRuntimeSessionJson', 'pointer', [], []);
    if (rawPtr instanceof Promise) {
      throw new Error('Unexpected async result while reading the runtime session.');
    }
    const ptr = Number(rawPtr);
    if (ptr === 0) {
      throw new Error(this.readLastEngineError());
    }
    try {
      return JSON.parse(this.readUtf8String(ptr)) as RuntimeSessionSnapshot;
    } finally {
      this.module.ccall('CE_FreeString', null, ['pointer'], [ptr]);
    }
  }

  public startTextRequest(
    generation: number,
    contextKey: string,
    promptText: string,
    maxOutputTokens: number,
    options: WasmTextRequestOptions = {}
  ): GenerateRequestHandle {
    validateGrammarSize(options.grammar);
    const grammarArg = options.grammar ?? '';
    const stopArg = serializeStop(options.stop);
    const samplingArg = serializeSampling(options.sampling);
    const requestId = this.module.ccall(
      'CE_StartTextRequest',
      'number',
      ['number', 'string', 'string', 'number', 'number', 'string', 'string', 'string'],
      [
        generation,
        contextKey,
        promptText,
        maxOutputTokens,
        options.emitTokens === true ? 1 : 0,
        grammarArg,
        stopArg,
        samplingArg,
      ]
    );
    if (requestId instanceof Promise) {
      throw new Error('Unexpected async result while enqueuing a request.');
    }
    return { generation, requestId: requestId as GenerateRequestId };
  }

  public startMediaRequest(
    generation: number,
    contextKey: string,
    promptText: string,
    maxOutputTokens: number,
    media: Uint8Array[],
    options: WasmTextRequestOptions = {}
  ): GenerateRequestHandle {
    validateGrammarSize(options.grammar);
    const grammarArg = options.grammar ?? '';
    const stopArg = serializeStop(options.stop);
    const samplingArg = serializeSampling(options.sampling);
    const requestId = this.withWasmMediaBuffers(media, (flatPtr, sizesPtr) =>
      this.callNumber(
        'CE_StartMediaRequest',
        [
          'number',
          'string',
          'string',
          'number',
          'number',
          'pointer',
          'pointer',
          'number',
          'string',
          'string',
          'string',
        ],
        [
          generation,
          contextKey,
          promptText,
          maxOutputTokens,
          media.length,
          flatPtr,
          sizesPtr,
          options.emitTokens === true ? 1 : 0,
          grammarArg,
          stopArg,
          samplingArg,
        ]
      ) as GenerateRequestId
    );
    return { generation, requestId };
  }

  public startChatRequest(
    generation: number,
    contextKey: string,
    messages: readonly ChatMessage[],
    maxOutputTokens: number,
    media: Uint8Array[] = [],
    options: WasmTextRequestOptions = {}
  ): GenerateRequestHandle {
    validateGrammarSize(options.grammar);
    const grammarArg = options.grammar ?? '';
    const stopArg = serializeStop(options.stop);
    const samplingArg = serializeSampling(options.sampling);
    const requestId = this.withWasmMediaBuffers(media, (flatPtr, sizesPtr) =>
      this.callNumber(
        'CE_StartChatRequest',
        [
          'number',
          'string',
          'string',
          'number',
          'number',
          'pointer',
          'pointer',
          'number',
          'string',
          'string',
          'string',
        ],
        [
          generation,
          contextKey,
          JSON.stringify(messages),
          maxOutputTokens,
          media.length,
          flatPtr,
          sizesPtr,
          options.emitTokens === true ? 1 : 0,
          grammarArg,
          stopArg,
          samplingArg,
        ]
      ) as GenerateRequestId
    );
    return { generation, requestId };
  }

  public startEmbeddingRequest(
    generation: number,
    contextKey: string,
    input: string,
    normalize: boolean
  ): GenerateRequestHandle {
    const requestId = this.module.ccall(
      'CE_StartEmbeddingRequest',
      'number',
      ['number', 'string', 'string', 'number'],
      [generation, contextKey, input, normalize ? 1 : 0]
    );
    if (requestId instanceof Promise) {
      throw new Error('Unexpected async result while enqueuing an embedding request.');
    }
    return { generation, requestId: requestId as GenerateRequestId };
  }

  public async startListenRequest(
    generation: number,
    audio: Uint8Array,
    language: string,
    maxOutputTokens: number
  ): Promise<GenerateRequestHandle> {
    const requestId = await this.withWasmBytesAsync(audio, (audioPtr, audioLength) =>
      this.callNumberAsync(
        'CE_StartListenRequest',
        ['number', 'pointer', 'number', 'string', 'number'],
        [generation, audioPtr, audioLength, language, maxOutputTokens]
      ) as Promise<GenerateRequestId>
    );
    return { generation, requestId };
  }

  public async startSpeakRequest(
    generation: number,
    text: string,
    language: string,
    speakerAudio: Uint8Array,
    maxDurationMs: number | undefined
  ): Promise<GenerateRequestHandle> {
    const requestId = await this.withWasmBytesAsync(speakerAudio, (speakerPtr, speakerLength) =>
      this.callNumberAsync(
        'CE_StartSpeakRequest',
        ['number', 'string', 'string', 'pointer', 'number', 'number', 'number'],
        [
          generation,
          text,
          language,
          speakerPtr,
          speakerLength,
          maxDurationMs == null ? 0 : 1,
          maxDurationMs ?? 0,
        ]
      ) as Promise<GenerateRequestId>
    );
    return { generation, requestId };
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

  public validatePairing(classified: readonly ClassifiedAsset[]): PairingValidationResponse {
    const raw = this.callOwnedString(
      'CE_PairingValidate',
      ['string'],
      [JSON.stringify(classified)]
    );
    try {
      return JSON.parse(raw) as PairingValidationResponse;
    } catch (error) {
      throw new Error('Rust pairing validation returned invalid JSON.', { cause: error });
    }
  }

  public async cancelQuery(request: GenerateRequestHandle): Promise<boolean> {
    const result = this.module.ccall(
      'CE_CancelRequest',
      'number',
      ['number', 'number'],
      [request.generation, request.requestId]
    );
    const status = Number(result instanceof Promise ? await result : result);
    if (status < 0) {
      throw new Error(this.readLastEngineError());
    }
    return Boolean(status);
  }

  public getCompletedRequestStatus(request: GenerateRequestHandle): number {
    const status = this.callNumber(
      'CE_GetCompletedRequestStatus',
      ['number', 'number'],
      [request.generation, request.requestId]
    );
    if (status < 0) {
      throw new Error(this.readLastEngineError());
    }
    return status;
  }

  public consumeCompletedRequest(request: GenerateRequestHandle): boolean {
    const result = this.callNumber(
      'CE_ConsumeCompletedRequest',
      ['number', 'number'],
      [request.generation, request.requestId]
    );
    if (result === STATUS_STALE_RUNTIME_SESSION) {
      throw new Error(this.readLastEngineError());
    }
    return Boolean(result);
  }

  public consumeCompletedResponseIfPresent(request: GenerateRequestHandle): boolean {
    const status = this.getCompletedRequestStatus(request);
    if (status === COMPLETED_REQUEST_STATUS_UNKNOWN) {
      return false;
    }
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      return false;
    }
    if (!this.consumeCompletedRequest(request)) {
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
  ): RustLifecycleResponse<CatalogModelInfo[]> {
    return this.callLifecycleJson<CatalogModelInfo[]>('CE_ModelServiceList', ['number'], [handle]);
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

  public modelServiceInstall(
    handle: RustLifecycleHandle,
    source: RustLifecycleInstallSource
  ): RustLifecycleResponse<RustLifecycleInstallValue> {
    return this.callLifecycleJson<RustLifecycleInstallValue>(
      'CE_ModelServiceInstall',
      ['number', 'string'],
      [handle, JSON.stringify(source)]
    );
  }

  public modelServiceRemoteAcquisitionCommand(
    handle: RustLifecycleHandle,
    command: RustRemoteCommand
  ): RustLifecycleResponse<RustRemoteCommandValue> {
    return this.callLifecycleJson<RustRemoteCommandValue>(
      'CE_ModelServiceRemoteAcquisitionCommand',
      ['number', 'string'],
      [handle, JSON.stringify(command)]
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

  public modelServiceRemove(
    handle: RustLifecycleHandle,
    request: RustLifecycleRemoveRequest
  ): RustLifecycleResponse<RustLifecycleRemoveValue> {
    return this.callLifecycleJson<RustLifecycleRemoveValue>(
      'CE_ModelServiceRemove',
      ['number', 'string'],
      [handle, JSON.stringify(request)]
    );
  }

  public modelServiceSnapshot(
    handle: RustLifecycleHandle
  ): RustLifecycleResponse<CatalogObservabilitySnapshot> {
    return this.callLifecycleJson<CatalogObservabilitySnapshot>(
      'CE_ModelServiceSnapshot',
      ['number'],
      [handle]
    );
  }

  public modelServiceDrainEvents(
    handle: RustLifecycleHandle
  ): RustLifecycleResponse<CatalogObservabilityEvent[]> {
    return this.callLifecycleJson<CatalogObservabilityEvent[]>(
      'CE_ModelServiceDrainEvents',
      ['number'],
      [handle]
    );
  }

  public modelServiceRecordEvent(
    handle: RustLifecycleHandle,
    type: ObservabilityEventType,
    patch: Record<string, unknown>
  ): RustLifecycleResponse<CatalogObservabilitySnapshot> {
    return this.callLifecycleJson<CatalogObservabilitySnapshot>(
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
    let callbackFailure: GgufCallbackFailure | null = null;
    const readAtPtr = this.module.addFunction(
      (_userData: number, offset: bigint | number, dstPtr: number, len: number) => {
        try {
          const start = this.byteOffset(dstPtr);
          const target = this.module.HEAPU8.subarray(start, start + len);
          return callbacks.readAt(this.byteOffset(offset), target) ?? 0;
        } catch (error) {
          callbackFailure ??= { error };
          return -1;
        }
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
        throw this.ggufCallbackError(
          `Rust GGUF split planning failed with status ${count}.`,
          callbackFailure
        );
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
    let callbackFailure: GgufCallbackFailure | null = null;
    const readAtPtr = this.module.addFunction(
      (_userData: number, offset: bigint | number, dstPtr: number, len: number) => {
        try {
          const start = this.byteOffset(dstPtr);
          const target = this.module.HEAPU8.subarray(start, start + len);
          return callbacks.readAt(this.byteOffset(offset), target) ?? 0;
        } catch (error) {
          callbackFailure ??= { error };
          return -1;
        }
      },
      'iijii'
    );
    const openShardPtr = this.module.addFunction(
      (_userData: number, pathPtr: number, index: number, count: number) => {
        try {
          return callbacks.openShard(this.readUtf8String(pathPtr), index, count) ?? 0;
        } catch (error) {
          callbackFailure ??= { error };
          return -1;
        }
      },
      'iiiii'
    );
    const writeShardPtr = this.module.addFunction(
      (_userData: number, bytesPtr: number, len: number) => {
        try {
          const start = this.byteOffset(bytesPtr);
          const bytes = this.module.HEAPU8.subarray(start, start + len);
          return callbacks.writeShard(bytes) ?? 0;
        } catch (error) {
          callbackFailure ??= { error };
          return -1;
        }
      },
      'iiii'
    );
    const closeShardPtr = this.module.addFunction(
      () => {
        try {
          return callbacks.closeShard() ?? 0;
        } catch (error) {
          callbackFailure ??= { error };
          return -1;
        }
      },
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
        throw this.ggufCallbackError(
          `Rust GGUF stream split failed with status ${status}.`,
          callbackFailure
        );
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
    request: GenerateRequestHandle
  ): RequestObservabilityMetrics | null {
    return this.readRuntimeObservabilityViaCall(
      'CE_GetCompletedRequestRuntimeObservability',
      ['number', 'number'],
      [request.generation, request.requestId]
    );
  }

  public takeCompletedResponse(request: GenerateRequestHandle): GenerateResponse {
    const status = this.getCompletedRequestStatus(request);
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      throw new Error('Queued request reached a terminal step without a completed response.');
    }
    if (status === COMPLETED_REQUEST_STATUS_UNKNOWN) {
      throw new Error('Queued request response is no longer available.');
    }

    const outputKind = this.callNumber(
      'CE_GetCompletedRequestOutputKind',
      ['number', 'number'],
      [request.generation, request.requestId]
    );
    if (outputKind === COMPLETED_REQUEST_OUTPUT_TEXT) {
      const outputText = this.copyText(
        'CE_GetCompletedRequestOutputSize',
        'CE_CopyCompletedRequestOutput',
        'output',
        ['number', 'number'],
        [request.generation, request.requestId]
      );
      return {
        ...this.completedResponseBase(request, status),
        outputText,
      };
    }
    if (outputKind === COMPLETED_REQUEST_OUTPUT_EMBEDDING) {
      const embedding = this.readCompletedEmbedding(request);
      return {
        ...this.completedResponseBase(request, status),
        embedding,
      };
    }
    if (outputKind === COMPLETED_REQUEST_OUTPUT_AUDIO) {
      const audio = this.readCompletedAudio(request);
      return {
        ...this.completedResponseBase(request, status),
        audio,
      };
    }
    throw new Error(
      `Completed request ${request.generation}:${request.requestId} has unknown output kind ${outputKind}.`
    );
  }

  private completedResponseBase(
    request: GenerateRequestHandle,
    status: number
  ): Omit<GenerateResponse, 'outputText' | 'embedding' | 'audio'> {
    const errorText = this.copyText(
      'CE_GetCompletedRequestErrorSize',
      'CE_CopyCompletedRequestError',
      'error',
      ['number', 'number'],
      [request.generation, request.requestId]
    );
    const runtimeObservability = this.readCompletedRequestRuntimeObservability(request);
    if (!this.consumeCompletedRequest(request)) {
      throw new Error('Failed to consume completed queued request response.');
    }

    return {
      requestId: request.requestId,
      completed: status === COMPLETED_REQUEST_STATUS_COMPLETED,
      failed: status === COMPLETED_REQUEST_STATUS_FAILED,
      cancelled: status === COMPLETED_REQUEST_STATUS_CANCELLED,
      errorMessage: errorText.length > 0 ? errorText : null,
      observability: runtimeObservability,
    };
  }

  private readCompletedAudio(request: GenerateRequestHandle): import('../engine/inference-types.js').AudioOutput {
    const length = this.callNumber(
      'CE_GetCompletedRequestAudioLength',
      ['number', 'number'],
      [request.generation, request.requestId]
    );
    if (length < 0) {
      throw new Error(
        `Failed to read completed audio length for request ${request.generation}:${request.requestId}.`
      );
    }
    const ptr = this.allocate(Math.max(1, length));
    try {
      const copied = this.callNumber(
        'CE_CopyCompletedRequestAudio',
        ['number', 'number', 'pointer', 'number'],
        [request.generation, request.requestId, ptr, length]
      );
      if (copied !== length) {
        throw new Error(
          `Failed to copy completed audio for request ${request.generation}:${request.requestId}.`
        );
      }
      return {
        data: this.module.HEAPU8.slice(ptr, ptr + length),
        sampleRateHz: this.callNumber(
          'CE_GetCompletedRequestAudioSampleRate',
          ['number', 'number'],
          [request.generation, request.requestId]
        ),
        channels: this.callNumber(
          'CE_GetCompletedRequestAudioChannels',
          ['number', 'number'],
          [request.generation, request.requestId]
        ),
        durationMs: this.callNumber(
          'CE_GetCompletedRequestAudioDurationMs',
          ['number', 'number'],
          [request.generation, request.requestId]
        ),
      };
    } finally {
      this.free(ptr);
    }
  }

  public async runInferenceLoop(
    generation: number,
    maxTicks: number,
    maxCompletedResponses: number,
    maxGeneratedTokens: number
  ): Promise<WasmSchedulerProgressResult> {
    const resultPtr = this.ensureLoopResultBuffer();

    const stepResult = await this.callNumberAsync(
      'CE_RunSchedulerLoop',
      ['number', 'number', 'number', 'number', 'pointer'],
      [generation, maxTicks, maxCompletedResponses, maxGeneratedTokens, resultPtr]
    );
    if (stepResult === STATUS_STALE_RUNTIME_SESSION) {
      throw new Error(this.readLastEngineError());
    }

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

  private async withWasmBytesAsync<T>(
    bytes: Uint8Array,
    operation: (ptr: number, len: number) => Promise<T>
  ): Promise<T> {
    const ptr = this.allocate(Math.max(1, bytes.byteLength));
    try {
      if (bytes.byteLength > 0) {
        this.module.HEAPU8.set(bytes, ptr);
      }
      return await operation(ptr, bytes.byteLength);
    } finally {
      this.free(ptr);
    }
  }

  private ggufCallbackError(
    message: string,
    callbackFailure: GgufCallbackFailure | null
  ): Error {
    if (callbackFailure == null) {
      return new Error(message);
    }
    const callbackError = callbackFailure.error;
    const detail = callbackError instanceof Error ? callbackError.message : String(callbackError);
    return new Error(`${message} Callback failed: ${detail}`, { cause: callbackError });
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

  private readCompletedEmbedding(request: GenerateRequestHandle): EmbeddingOutput {
    const length = this.callNumber(
      'CE_GetCompletedRequestEmbeddingLength',
      ['number', 'number'],
      [request.generation, request.requestId]
    );
    if (length < 0) {
      throw new Error('Completed request did not expose an embedding vector.');
    }
    const pooling = poolingTypeFromCode(
      this.callNumber(
        'CE_GetCompletedRequestEmbeddingPooling',
        ['number', 'number'],
        [request.generation, request.requestId]
      )
    );
    const normalizedValue = this.callNumber(
      'CE_GetCompletedRequestEmbeddingNormalized',
      ['number', 'number'],
      [request.generation, request.requestId]
    );
    if (normalizedValue < 0) {
      throw new Error('Failed to read embedding normalization flag.');
    }
    const normalized = normalizedValue !== 0;
    const bufferPtr = this.allocate(Math.max(1, length * 4));
    try {
      const copied = this.callNumber(
        'CE_CopyCompletedRequestEmbedding',
        ['number', 'number', 'pointer', 'number'],
        [request.generation, request.requestId, bufferPtr, length]
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
