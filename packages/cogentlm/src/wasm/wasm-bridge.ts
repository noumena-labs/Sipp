import { NormalizedInitConfig } from '../core/init-config.js';
import {
  BackendObservability,
  GenerateRequestId,
  GenerateResponse,
} from '../types.js';
import { EngineModule } from './engine-module.js';
import {
  DetailedRequestObservabilityMetrics,
  DetailedRuntimeAggregateObservabilityMetrics,
  DetailedRuntimeObservabilityMetrics,
  withDerivedObservabilityMetrics,
} from '../observability/runtime-observability-detail.js';
import {
  COMPLETED_REQUEST_STATUS_CANCELLED,
  COMPLETED_REQUEST_STATUS_COMPLETED,
  COMPLETED_REQUEST_STATUS_FAILED,
  COMPLETED_REQUEST_STATUS_PENDING,
  COMPLETED_REQUEST_STATUS_UNKNOWN,
  RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT,
  RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES,
  SCHEDULER_LOOP_RESULT_SIZE_BYTES,
} from '../runtime/main-thread-runtime-constants.js';
import { assertGrammarByteSize } from '../utils/grammar.js';
export { MAX_GRAMMAR_BYTES } from '../utils/grammar.js';

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
 * Shape of an OpenAI-compatible chat message accepted by
 * `WasmBridge.applyChatTemplate`. Corresponds to the JSON array parsed by
 * `common_chat_msgs_parse_oaicompat` on the native side.
 */
export type ChatTemplateMessage = {
  role: string;
  content: string;
};

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

  private safeNumber(value: number | bigint): number {
    const n = typeof value === 'bigint' ? Number(value) : value;
    if (!Number.isSafeInteger(n) || n < 0) {
      throw new RangeError(`Invalid wasm integer: ${String(value)}`);
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
    normalizedConfig: NormalizedInitConfig
  ): Promise<number> {
    const result = await this.module.ccall('CE_Init', 'number', ['string', 'string'], [
      modelPath,
      normalizedConfig.runtimeConfigJson,
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

      return this.callNumber(
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
      ) as GenerateRequestId;
    } finally {
      this.free(flatPtr);
      this.free(sizesPtr);
    }
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
  public applyChatTemplate(
    messages: ChatTemplateMessage[],
    addAssistant: boolean
  ): string {
    return this.callOwnedString(
      'CE_ApplyChatTemplate',
      ['string', 'number'],
      [JSON.stringify(messages), addAssistant ? 1 : 0]
    );
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

  public splitGgufFile(
    inputPath: string,
    outputPrefix: string,
    shardMaxBytes: number
  ): void {
    const status = this.callNumber(
      'CE_GgufSplitFile',
      ['string', 'string', 'number'],
      [inputPath, outputPrefix, shardMaxBytes]
    );
    if (status !== 0) {
      throw new Error(`Rust GGUF file split failed with status ${status}.`);
    }
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
        return callbacks.readAt(this.safeNumber(offset), target) ?? 0;
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
        return callbacks.readAt(this.safeNumber(offset), target) ?? 0;
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

  public readRuntimeObservability(): DetailedRuntimeAggregateObservabilityMetrics | null {
    return this.readRuntimeObservabilityViaCall('CE_GetRuntimeObservability', [], []);
  }

  public readCompletedRequestRuntimeObservability(
    requestId: GenerateRequestId
  ): DetailedRequestObservabilityMetrics | null {
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

  public getStreamingBufferCapacity(): number {
    return this.callNumber('CE_GetStreamingBufferCapacity');
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
  ): DetailedRuntimeObservabilityMetrics | null {
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
