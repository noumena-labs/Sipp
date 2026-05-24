import type { BackendObservability } from '../observability/backend-observability.js';
import type {
  GenerateRequestId,
  GenerateResponse,
  NativeRuntimeConfig,
} from '../core/inference-types.js';
import type { RuntimePairingErrorCode } from '../runtime/engine-runtime.js';
import type { ClassifiedAsset, PairingPlan } from '../models/pairing-types.js';
import type {
  AssetRecord,
  ModelInfo,
  ObservabilityEvent,
  ObservabilityEventType,
  ObservabilitySnapshot,
  QueryErrorCode,
  RegistryManifest,
} from '../models/types.js';
import type {
  ModelDetectionMethod,
  ModelDetectionResult,
} from '../bundle/model-bundle-types.js';
import type { ChatBoundaryInfo } from '../core/chat-boundary-sanitizer.js';
import type { ChatMessage } from '../core/inference-types.js';
import { EngineModule } from './engine-module.js';
import {
  withDerivedObservabilityMetrics,
  type RequestObservabilityMetrics,
} from '../observability/runtime-observability.js';
import {
  COMPLETED_REQUEST_STATUS_CANCELLED,
  COMPLETED_REQUEST_STATUS_COMPLETED,
  COMPLETED_REQUEST_STATUS_FAILED,
  COMPLETED_REQUEST_STATUS_PENDING,
  COMPLETED_REQUEST_STATUS_UNKNOWN,
  RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT,
  RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES,
  SCHEDULER_LOOP_RESULT_SIZE_BYTES,
} from '../runtime/main-thread/constants.js';
import { createAbortError } from '../utils/abort.js';
import { assertGrammarByteSize } from '../utils/grammar.js';

// Mirror of CE_TokenEmissionMode in native/api/ffi_types.h.  Native exposes
// only NONE (no emission) and STREAMING_BUFFER (SAB ring).
export const TOKEN_EMISSION_NONE = 0;
export const TOKEN_EMISSION_STREAMING_BUFFER = 1;

export type TokenEmissionMode =
  | typeof TOKEN_EMISSION_NONE
  | typeof TOKEN_EMISSION_STREAMING_BUFFER;

function validateGrammarSize(grammar: string | undefined): void {
  assertGrammarByteSize(grammar);
}

function validateTokenEmissionMode(mode: TokenEmissionMode): void {
  if (mode !== TOKEN_EMISSION_NONE && mode !== TOKEN_EMISSION_STREAMING_BUFFER) {
    throw new Error(`invalid token emission mode ${mode}.`);
  }
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

export interface PairingValidationResponse {
  ok: boolean;
  plan?: PairingPlan;
  error?: {
    code: RuntimePairingErrorCode | string;
    message: string;
  };
}

export interface RustLifecycleResponse<T> {
  ok: boolean;
  value?: T;
  error?: {
    code: QueryErrorCode | string;
    message: string;
  };
}

export type RustLifecycleHandle = number;
export type RustLifecycleBackendPreference = 'auto' | 'cpu' | 'webgpu';

export interface RustLifecycleCreateValue {
  handle: RustLifecycleHandle;
  manifest: RegistryManifest;
  snapshot: ObservabilitySnapshot;
}

export interface RustLifecycleLoadSourceInstalled {
  kind: 'installed';
  id: string;
  classifiedProjectors?: ClassifiedAsset[];
}

export interface RustLifecycleLoadSourceAssets {
  kind: 'assets';
  assets: AssetRecord[];
  classified: ClassifiedAsset[];
  explicitProjectorAssetId?: string | null;
  classifiedProjectors?: ClassifiedAsset[];
}

export type RustLifecycleLoadSource =
  | RustLifecycleLoadSourceInstalled
  | RustLifecycleLoadSourceAssets;

export interface RustLifecycleLoadOptions {
  backend?: RustLifecycleBackendPreference;
  runtime?: NativeRuntimeConfig;
  observability?: 'off' | 'runtime' | 'profile';
}

export interface RustLifecyclePlannedAsset {
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

export interface RustLifecycleCommitLoad {
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

export interface RustLifecycleCommitLoadValue {
  model: ModelInfo;
  manifest: RegistryManifest;
  snapshot: ObservabilitySnapshot;
  events: ObservabilityEvent[];
}

export interface RustLifecycleRemoveValue {
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

export class WasmBridge {
  private _cachedDataView: DataView | null = null;
  private _cachedHeapU8: Uint8Array | null = null;

  public constructor(public readonly module: EngineModule) { }

  private ensureHeapView(): DataView {
    if (
      this._cachedDataView == null ||
      this._cachedDataView.buffer !== this.module.HEAPU8.buffer
    ) {
      this._cachedDataView = new DataView(this.module.HEAPU8.buffer);
      this._cachedHeapU8 = this.module.HEAPU8;
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
    tokenEmissionMode: TokenEmissionMode = TOKEN_EMISSION_NONE
  ): GenerateRequestId {
    validateGrammarSize(grammar);
    validateTokenEmissionMode(tokenEmissionMode);
    const grammarArg = grammar ?? '';
    const requestId = this.module.ccall(
      'CE_StartTextRequestWithTokenEmissionMode',
      'number',
      ['string', 'string', 'number', 'number', 'string'],
      [contextKey, promptText, maxOutputTokens, tokenEmissionMode, grammarArg]
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
    tokenEmissionMode: TokenEmissionMode = TOKEN_EMISSION_NONE
  ): GenerateRequestId {
    validateGrammarSize(grammar);
    validateTokenEmissionMode(tokenEmissionMode);
    const grammarArg = grammar ?? '';
    return this.withWasmMediaBuffers(media, (flatPtr, sizesPtr) =>
      this.callNumber(
        'CE_StartMediaRequestWithTokenEmissionMode',
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
          tokenEmissionMode,
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
    tokenEmissionMode: TokenEmissionMode = TOKEN_EMISSION_NONE
  ): GenerateRequestId {
    validateGrammarSize(grammar);
    validateTokenEmissionMode(tokenEmissionMode);
    const grammarArg = grammar ?? '';
    return this.withWasmMediaBuffers(media, (flatPtr, sizesPtr) =>
      this.callNumber(
        'CE_StartChatRequestWithTokenEmissionMode',
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
          tokenEmissionMode,
          grammarArg,
        ]
      ) as GenerateRequestId
    );
  }

  public readMediaMarker(): string | null {
    const ptr = this.callNumber('CE_GetMediaMarker');
    if (!ptr) {
      return null;
    }
    const marker = this.module.UTF8ToString(ptr);
    return marker.length > 0 ? marker : null;
  }

  public readNativeChatTemplate(): string | null {
    const ptr = this.callNumber('CE_GetChatTemplate');
    if (!ptr) {
      return null;
    }
    const template = this.module.UTF8ToString(ptr);
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
      return this.module.UTF8ToString(ptr);
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
        callbacks.openShard(this.module.UTF8ToString(pathPtr), index, count) ?? 0,
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

    const outputText = this.copyCompletedRequestText(
      requestId,
      'CE_GetCompletedRequestOutputSize',
      'CE_CopyCompletedRequestOutput',
      'output'
    );
    const errorText = this.copyCompletedRequestText(
      requestId,
      'CE_GetCompletedRequestErrorSize',
      'CE_CopyCompletedRequestError',
      'error'
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
      outputText,
      errorMessage: errorText.length > 0 ? errorText : null,
      observability: runtimeObservability,
    };
  }

  public async runInferenceLoop(
    maxTicks: number,
    maxCompletedResponses: number,
    maxEmittedTokens: number,
    options: {
      maxDurationUs?: number;
      // Tells the native scheduler whether to use the per-emitted-token
      // yield path (true, for streaming requests) or the monolithic loop
      // (false, for bulk requests). Setting this incorrectly is not a
      // correctness bug — only a performance one: streaming requests with
      // false won't deliver tokens until the loop returns; bulk requests
      // with true pay per-burst yielding overhead for no reason.
      streamingActive?: boolean;
    } = {}
  ): Promise<WasmSchedulerProgressResult> {
    const maxDurationUs = Math.max(0, options.maxDurationUs ?? 0);
    const streamingActive = options.streamingActive === true ? 1 : 0;
    const resultPtr = this.ensureLoopResultBuffer();

    const stepResult = await this.callNumberAsync(
      'CE_RunSchedulerLoop',
      ['number', 'number', 'number', 'number', 'number', 'pointer'],
      [
        maxTicks,
        maxCompletedResponses,
        maxEmittedTokens,
        maxDurationUs,
        streamingActive,
        resultPtr,
      ]
    );

    const loopResult = this.readSchedulerLoopResult(resultPtr);
    return {
      stepResult,
      completedResponseCount: loopResult.completedResponseCount,
    };
  }

  // Streaming buffer init-time accessors.  Stable wasm-heap addresses; the
  // caller caches them once and afterwards touches the buffer and counter
  // cells via HEAPU8 / HEAP32 directly (zero ccalls).  Returns 0 when no
  // engine is initialized.
  public getStreamingBufferPointer(): number {
    return this.callNumber('CE_GetStreamingBufferPointer');
  }

  public getStreamingBufferUsedAddress(): number {
    return this.callNumber('CE_GetStreamingBufferUsedAddress');
  }

  public getStreamingBufferDropCountAddress(): number {
    return this.callNumber('CE_GetStreamingBufferDropCountAddress');
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
      return this.module.UTF8ToString(ptr);
    } finally {
      this.module.ccall('CE_FreeString', null, ['pointer'], [ptr]);
    }
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
        cacheHits: view.getInt32(intsOffset + 8, true),
        prefillTokens: view.getInt32(intsOffset + 12, true),
      } as any);
    } finally {
      this.free(metricsPtr);
    }
  }

  private copyCompletedRequestText(
    requestId: GenerateRequestId,
    sizeFunction: string,
    copyFunction: string,
    fieldName: string
  ): string {
    return this.copyText(sizeFunction, copyFunction, fieldName, ['number'], [requestId]);
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
      return this.module.UTF8ToString(bufferPtr, byteLength);
    } finally {
      this.free(bufferPtr);
    }
  }
}

export function parseBackendObservabilityJson(raw: string): BackendObservability {
  return JSON.parse(raw) as BackendObservability;
}

function normalizeModelDetectionMethod(
  value: RustModelDetectionResult['detectionMethod']
): ModelDetectionMethod {
  return value === 'gguf_metadata' ? 'gguf-metadata' : value;
}
