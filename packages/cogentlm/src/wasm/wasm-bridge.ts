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
// only NONE (no emission) and STREAMING_BUFFER (SAB ring); the legacy FFI
// callback / runtime-event paths were removed.
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
  private reusableBurstResultPtr = 0;
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
    normalizedConfig: NormalizedInitConfig
  ): Promise<number> {
    const hasMultimodalConfig =
      normalizedConfig.multimodalProjectorPath != null ||
      normalizedConfig.imageMinTokens > 0 ||
      normalizedConfig.imageMaxTokens > 0;
    const ident = hasMultimodalConfig ? 'CE_InitWithMultimodal' : 'CE_Init';
    const argTypes = hasMultimodalConfig
      ? [
        'string',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'string',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
      ]
      : [
        'string',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
      ];
    const args = hasMultimodalConfig
      ? [
        modelPath,
        normalizedConfig.nCtx,
        normalizedConfig.nBatch,
        normalizedConfig.nUbatch,
        normalizedConfig.nSeqMax,
        normalizedConfig.nThreads,
        normalizedConfig.nThreadsBatch,
        normalizedConfig.nGpuLayers,
        normalizedConfig.flashAttention,
        normalizedConfig.kvUnified,
        normalizedConfig.maxCachedSessions,
        normalizedConfig.retainedPrefixTokens,
        normalizedConfig.prefillChunkSize,
        normalizedConfig.prefixCacheIntervalTokens,
        normalizedConfig.maxPrefixCacheEntries,
        normalizedConfig.schedulerPolicy,
        normalizedConfig.decodeTokenReserve,
        normalizedConfig.adaptivePrefillChunking,
        normalizedConfig.enableRuntimeObservability,
        normalizedConfig.enableBackendProfiling,
        normalizedConfig.multimodalProjectorPath ?? '',
        normalizedConfig.multimodalUseGpu,
        normalizedConfig.imageMinTokens,
        normalizedConfig.imageMaxTokens,
        normalizedConfig.samplingRepeatLastN,
        normalizedConfig.samplingRepeatPenalty,
        normalizedConfig.samplingFrequencyPenalty,
        normalizedConfig.samplingPresencePenalty,
        normalizedConfig.samplingTopK,
        normalizedConfig.samplingTopP,
        normalizedConfig.samplingMinP,
        normalizedConfig.samplingTemperature,
        normalizedConfig.samplingSeed,
      ]
      : [
        modelPath,
        normalizedConfig.nCtx,
        normalizedConfig.nBatch,
        normalizedConfig.nUbatch,
        normalizedConfig.nSeqMax,
        normalizedConfig.nThreads,
        normalizedConfig.nThreadsBatch,
        normalizedConfig.nGpuLayers,
        normalizedConfig.flashAttention,
        normalizedConfig.kvUnified,
        normalizedConfig.maxCachedSessions,
        normalizedConfig.retainedPrefixTokens,
        normalizedConfig.prefillChunkSize,
        normalizedConfig.prefixCacheIntervalTokens,
        normalizedConfig.maxPrefixCacheEntries,
        normalizedConfig.schedulerPolicy,
        normalizedConfig.decodeTokenReserve,
        normalizedConfig.adaptivePrefillChunking,
        normalizedConfig.enableRuntimeObservability,
        normalizedConfig.enableBackendProfiling,
        normalizedConfig.multimodalUseGpu,
        normalizedConfig.samplingRepeatLastN,
        normalizedConfig.samplingRepeatPenalty,
        normalizedConfig.samplingFrequencyPenalty,
        normalizedConfig.samplingPresencePenalty,
        normalizedConfig.samplingTopK,
        normalizedConfig.samplingTopP,
        normalizedConfig.samplingMinP,
        normalizedConfig.samplingTemperature,
        normalizedConfig.samplingSeed,
      ];
    const result = await this.module.ccall(ident, 'number', argTypes, args, {
      async: true,
    });
    return Number(result);
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
    } = {}
  ): Promise<WasmSchedulerProgressResult> {
    const maxDurationUs = Math.max(0, options.maxDurationUs ?? 0);
    const resultPtr = this.ensureLoopResultBuffer();
    
    const stepResult = await this.callNumberAsync(
      'CE_RunSchedulerLoop',
      ['number', 'number', 'number', 'number', 'pointer'],
      [maxTicks, maxCompletedResponses, maxEmittedTokens, maxDurationUs, resultPtr]
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
    const byteLength = this.callNumber(sizeFunction, ['number'], [requestId]);
    if (byteLength < 0) {
      throw new Error(`Failed to read queued request ${fieldName} size.`);
    }
    if (byteLength === 0) {
      return '';
    }

    const bufferPtr = this.allocate(byteLength + 1);
    try {
      const copied = this.callNumber(copyFunction, ['number', 'pointer', 'number'], [
        requestId,
        bufferPtr,
        byteLength + 1,
      ]);
      if (copied !== byteLength) {
        throw new Error(`Failed to copy queued request ${fieldName}.`);
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
