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
  RUNTIME_EVENT_DRAIN_RESULT_SIZE_BYTES,
  RUNTIME_EVENT_KIND_TERMINAL,
  RUNTIME_EVENT_KIND_TOKEN,
  RUNTIME_EVENT_SIZE_BYTES,
  RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT,
  RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES,
  SCHEDULER_LOOP_RESULT_SIZE_BYTES,
} from '../runtime/main-thread-runtime-constants.js';
import { assertGrammarByteSize } from '../utils/grammar.js';
export { MAX_GRAMMAR_BYTES } from '../utils/grammar.js';

const RUNTIME_EVENT_DRAIN_TEXT_BUFFER_SIZE_BYTES = 64 * 1024;

export const TOKEN_EMISSION_NONE = 0;
export const TOKEN_EMISSION_RUNTIME_EVENTS = 1;
export const TOKEN_EMISSION_DIRECT_CALLBACK = 2;

export type TokenEmissionMode =
  | typeof TOKEN_EMISSION_NONE
  | typeof TOKEN_EMISSION_RUNTIME_EVENTS
  | typeof TOKEN_EMISSION_DIRECT_CALLBACK;

function validateGrammarSize(grammar: string | undefined): void {
  assertGrammarByteSize(grammar);
}

function validateTokenEmissionMode(mode: TokenEmissionMode): void {
  if (
    mode !== TOKEN_EMISSION_NONE &&
    mode !== TOKEN_EMISSION_RUNTIME_EVENTS &&
    mode !== TOKEN_EMISSION_DIRECT_CALLBACK
  ) {
    throw new Error(`invalid token emission mode ${mode}.`);
  }
}

export type WasmRuntimeTokenEvent = {
  requestId: GenerateRequestId;
  token: string;
  textLength: number;
};

export type WasmRuntimeEventDrainResult = {
  eventCount: number;
  terminalRequestIds: GenerateRequestId[];
  tokenEvents: WasmRuntimeTokenEvent[];
  textBytes: number;
};

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
  private reusableRuntimeEventBufferPtr = 0;
  private reusableRuntimeEventBufferCapacity = 0;
  private reusableRuntimeEventTextBufferPtr = 0;
  private reusableRuntimeEventTextBufferCapacity = 0;
  private reusableRuntimeEventDrainResultPtr = 0;
  private readonly tokenDecoders = new Map<GenerateRequestId, TextDecoder>();
  private _cachedDataView: DataView | null = null;
  private _cachedHeapU8: Uint8Array | null = null;

  public constructor(private readonly module: EngineModule) { }

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

  private getDecoder(requestId: GenerateRequestId): TextDecoder {
    let decoder = this.tokenDecoders.get(requestId);
    if (!decoder) {
      decoder = new TextDecoder();
      this.tokenDecoders.set(requestId, decoder);
    }
    return decoder;
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
    callbackPtr: number,
    grammar?: string,
    tokenEmissionMode: TokenEmissionMode = TOKEN_EMISSION_NONE
  ): GenerateRequestId {
    validateGrammarSize(grammar);
    validateTokenEmissionMode(tokenEmissionMode);
    const grammarArg = grammar ?? '';
    const requestId = this.module.ccall(
      'CE_StartTextRequestWithTokenEmissionMode',
      'number',
      ['string', 'string', 'number', 'pointer', 'number', 'string'],
      [
        contextKey,
        promptText,
        maxOutputTokens,
        callbackPtr,
        tokenEmissionMode,
        grammarArg,
      ]
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
    callbackPtr: number,
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
          callbackPtr,
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

  public registerTokenCallback(
    onToken: (text: string) => boolean
  ): number {
    // Native signature: int32_t (*CE_TokenCallback)(const char *text, int32_t length)
    //
    // Emscripten addFunction signature is one char per slot: <return><arg1><arg2>...
    // Crucially, 'i' is always i32 but 'p' tracks the pointer width of the
    // build: in this WASM_BIGINT/MEMORY64 build, pointers are i64, so the
    // native call site produces wasm sig (i64, i32) -> i32. Registering as
    // 'iii' (i.e. (i32, i32) -> i32) traps with "function signature mismatch"
    // the moment native invokes the callback. 'ipi' lets emscripten emit i64
    // for the pointer slot when the build is 64-bit and i32 when it is 32-bit.
    //
    // textPtr arrives as a BigInt under WASM_BIGINT (the 'p' slot is i64 in
    // this build).  Although UTF8ToString accepts `number | bigint` in its
    // type signature, it forwards the value into UTF8ArrayToString, which does
    // `endPtr - idx` and `heapOrArray.subarray(idx, endPtr)` — mixing the
    // BigInt idx with the Number endPtr from findStringEnd throws
    // "Cannot mix BigInt and other types, use explicit conversions".  Coerce
    // to Number once at the boundary using byteOffset (which validates the
    // pointer is a safe integer).
    //
    // Return-value contract: native treats 0 as "continue" and any non-zero
    // value as "cancel" — the C++ lambda is `on_token(...) == 0`, which then
    // becomes the `bool` the runtime interprets as "keep going".  Our `onToken`
    // wrapper returns true to mean "continue", so the JS-side mapping is
    // continue → 0, cancel → 1.  Returning the obvious `? 1 : 0` inverts the
    // contract and silently cancels every successful token, surfacing as
    // "Request cancelled." after the first emission.
    const callbackPtr = this.module.addFunction((textPtr: number | bigint, length: number) => {
      const text = this.module.UTF8ToString(this.byteOffset(textPtr), length);
      return onToken(text) ? 0 : 1;
    }, 'ipi');
    return callbackPtr;
  }

  public unregisterTokenCallback(ptr: number): void {
    if (ptr !== 0) {
      this.module.removeFunction(ptr);
    }
  }

  public drainRuntimeEvents(
    maxEventCount: number,
    textBufferSizeBytes: number = RUNTIME_EVENT_DRAIN_TEXT_BUFFER_SIZE_BYTES
  ): WasmRuntimeEventDrainResult {
    const eventBufferPtr = this.ensureRuntimeEventBuffer(maxEventCount);
    const textBufferPtr = this.ensureRuntimeEventTextBuffer(textBufferSizeBytes);
    const resultPtr = this.ensureRuntimeEventDrainResultBuffer();

    const status = this.callNumber(
      'CE_DrainRuntimeEventsDirectly',
      ['pointer', 'number', 'pointer', 'number', 'pointer'],
      [eventBufferPtr, maxEventCount, textBufferPtr, textBufferSizeBytes, resultPtr]
    );
    if (status !== 0) {
      throw new Error(`Failed to drain runtime events. Code: ${status}`);
    }

    const view = this.ensureHeapView();
    const resultOffset = this.byteOffset(resultPtr);
    const eventCount = view.getInt32(resultOffset, true);
    const terminalRequestIds: GenerateRequestId[] = [];
    const tokenEvents: WasmRuntimeTokenEvent[] = [];
    let totalTextBytes = 0;

    const eventBaseOffset = this.byteOffset(eventBufferPtr);
    const textBaseOffset = this.byteOffset(textBufferPtr);
    const heapU8 = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);

    let i = 0;
    while (i < eventCount) {
      const eventOffset = eventBaseOffset + i * RUNTIME_EVENT_SIZE_BYTES;
      const requestId = view.getUint32(eventOffset, true);
      const kind = view.getInt32(eventOffset + 4, true);
      const status = view.getInt32(eventOffset + 8, true);
      const textOffset = view.getInt32(eventOffset + 12, true);
      const textLength = view.getInt32(eventOffset + 16, true);

      if (kind === RUNTIME_EVENT_KIND_TERMINAL) {
        terminalRequestIds.push(requestId);
        this.tokenDecoders.delete(requestId);
        i++;
        continue;
      }

      if (kind === RUNTIME_EVENT_KIND_TOKEN) {
        // Coalesce contiguous tokens for the same request
        let batchTextLength = textLength;
        let batchEnd = i + 1;
        while (batchEnd < eventCount) {
          const nextOffset = eventBaseOffset + batchEnd * RUNTIME_EVENT_SIZE_BYTES;
          const nextRequestId = view.getUint32(nextOffset, true);
          const nextKind = view.getInt32(nextOffset + 4, true);
          const nextTextLength = view.getInt32(nextOffset + 16, true);

          if (nextKind === RUNTIME_EVENT_KIND_TOKEN && nextRequestId === requestId) {
            batchTextLength += nextTextLength;
            batchEnd++;
          } else {
            break;
          }
        }

        const textBuffer = new Uint8Array(batchTextLength);
        let writeOffset = 0;
        for (let j = i; j < batchEnd; j++) {
          const chunkOffset = eventBaseOffset + j * RUNTIME_EVENT_SIZE_BYTES;
          const chunkTextOffset = view.getInt32(chunkOffset + 12, true);
          const chunkTextLen = view.getInt32(chunkOffset + 16, true);
          const absoluteTextOffset = textBaseOffset + chunkTextOffset;
          textBuffer.set(heapU8.subarray(absoluteTextOffset, absoluteTextOffset + chunkTextLen), writeOffset);
          writeOffset += chunkTextLen;
        }

        const decoder = this.getDecoder(requestId);
        const token = decoder.decode(textBuffer, { stream: true });
        tokenEvents.push({ requestId, token, textLength: batchTextLength });
        totalTextBytes += batchTextLength;
        i = batchEnd;
      } else {
        i++;
      }
    }

    return {
      eventCount,
      terminalRequestIds,
      tokenEvents,
      textBytes: totalTextBytes,
    };
  }

  public releaseReusableBuffers(): void {
    this.tokenDecoders.clear();
    if (this.reusableLoopResultPtr !== 0) {
      this.free(this.reusableLoopResultPtr);
      this.reusableLoopResultPtr = 0;
    }
    if (this.reusableRuntimeEventBufferPtr !== 0) {
      this.free(this.reusableRuntimeEventBufferPtr);
      this.reusableRuntimeEventBufferPtr = 0;
      this.reusableRuntimeEventBufferCapacity = 0;
    }
    if (this.reusableRuntimeEventTextBufferPtr !== 0) {
      this.free(this.reusableRuntimeEventTextBufferPtr);
      this.reusableRuntimeEventTextBufferPtr = 0;
      this.reusableRuntimeEventTextBufferCapacity = 0;
    }
    if (this.reusableRuntimeEventDrainResultPtr !== 0) {
      this.free(this.reusableRuntimeEventDrainResultPtr);
      this.reusableRuntimeEventDrainResultPtr = 0;
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



  private ensureRuntimeEventBuffer(maxEventCount: number): number {
    const requiredCapacity = Math.max(1, maxEventCount) * RUNTIME_EVENT_SIZE_BYTES;
    if (
      this.reusableRuntimeEventBufferPtr !== 0 &&
      this.reusableRuntimeEventBufferCapacity >= requiredCapacity
    ) {
      return this.reusableRuntimeEventBufferPtr;
    }

    if (this.reusableRuntimeEventBufferPtr !== 0) {
      this.free(this.reusableRuntimeEventBufferPtr);
    }
    this.reusableRuntimeEventBufferPtr = this.allocate(requiredCapacity);
    this.reusableRuntimeEventBufferCapacity = requiredCapacity;
    return this.reusableRuntimeEventBufferPtr;
  }

  private ensureRuntimeEventTextBuffer(textBufferSizeBytes: number): number {
    const requiredCapacity = Math.max(1, textBufferSizeBytes);
    if (
      this.reusableRuntimeEventTextBufferPtr !== 0 &&
      this.reusableRuntimeEventTextBufferCapacity >= requiredCapacity
    ) {
      return this.reusableRuntimeEventTextBufferPtr;
    }

    if (this.reusableRuntimeEventTextBufferPtr !== 0) {
      this.free(this.reusableRuntimeEventTextBufferPtr);
    }
    this.reusableRuntimeEventTextBufferPtr = this.allocate(requiredCapacity);
    this.reusableRuntimeEventTextBufferCapacity = requiredCapacity;
    return this.reusableRuntimeEventTextBufferPtr;
  }

  private ensureRuntimeEventDrainResultBuffer(): number {
    if (this.reusableRuntimeEventDrainResultPtr === 0) {
      this.reusableRuntimeEventDrainResultPtr = this.allocate(
        RUNTIME_EVENT_DRAIN_RESULT_SIZE_BYTES
      );
    }
    return this.reusableRuntimeEventDrainResultPtr;
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
