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
  RUNTIME_EVENT_DRAIN_RESULT_SIZE_BYTES,
  RUNTIME_EVENT_KIND_TERMINAL,
  RUNTIME_EVENT_KIND_TOKEN,
  RUNTIME_EVENT_SIZE_BYTES,
  RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT,
  RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES,
  SCHEDULER_BURST_RESULT_SIZE_BYTES,
} from '../runtime/main-thread-runtime-constants.js';

const RUNTIME_EVENT_DRAIN_TEXT_BUFFER_SIZE_BYTES = 64 * 1024;

export const TOKEN_EMISSION_NONE = 0;
export const TOKEN_EMISSION_RUNTIME_EVENTS = 1;
export const TOKEN_EMISSION_DIRECT_CALLBACK = 2;

export type TokenEmissionMode =
  | typeof TOKEN_EMISSION_NONE
  | typeof TOKEN_EMISSION_RUNTIME_EVENTS
  | typeof TOKEN_EMISSION_DIRECT_CALLBACK;

/**
 * Maximum accepted size of a GBNF grammar source (UTF-8 byte length).
 * Enforced at the bridge boundary before any ccall to the native runtime.
 */
export const MAX_GRAMMAR_BYTES = 64 * 1024;

function validateGrammarSize(grammar: string | undefined): void {
  if (grammar == null) {
    return;
  }
  // Fast path: if the string length in UTF-16 code units is under the limit,
  // UTF-8 size is guaranteed to be under 4x that. We only need the precise
  // byte length when close to the limit.
  if (grammar.length <= MAX_GRAMMAR_BYTES) {
    return;
  }
  const byteLength =
    typeof TextEncoder !== 'undefined'
      ? new TextEncoder().encode(grammar).byteLength
      : grammar.length;
  if (byteLength > MAX_GRAMMAR_BYTES) {
    throw new Error(
      `grammar exceeds maximum size of ${MAX_GRAMMAR_BYTES} bytes (got ${byteLength}).`
    );
  }
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

  public constructor(private readonly module: EngineModule) { }

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


  public getModule(): EngineModule {
    return this.module;
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
    try {
      const ptr = this.callNumber('CE_GetBosText');
      if (!ptr) {
        return '';
      }
      try {
        return this.module.UTF8ToString(ptr);
      } finally {
        this.module.ccall('CE_FreeString', null, ['pointer'], [ptr]);
      }
    } catch (error) {
      if (this.isMissingOptionalRuntimeApiError('CE_GetBosText', error)) {
        return '';
      }
      throw error;
    }
  }

  public getEosText(): string {
    try {
      const ptr = this.callNumber('CE_GetEosText');
      if (!ptr) {
        return '';
      }
      try {
        return this.module.UTF8ToString(ptr);
      } finally {
        this.module.ccall('CE_FreeString', null, ['pointer'], [ptr]);
      }
    } catch (error) {
      if (this.isMissingOptionalRuntimeApiError('CE_GetEosText', error)) {
        return '';
      }
      throw error;
    }
  }

  public tokenToString(tokenId: number): string {
    try {
      const ptr = this.callNumber('CE_TokenToString', ['number'], [tokenId]);
      if (!ptr) {
        return '';
      }
      try {
        return this.module.UTF8ToString(ptr);
      } finally {
        this.module.ccall('CE_FreeString', null, ['pointer'], [ptr]);
      }
    } catch (error) {
      if (this.isMissingOptionalRuntimeApiError('CE_TokenToString', error)) {
        return '';
      }
      throw error;
    }
  }

  /**
   * Applies llama.cpp's native chat template (via common_chat_format_single)
   * to a set of OpenAI-style chat messages and returns the formatted prompt
   * text. Returns '' when the runtime lacks the export (older WASM builds)
   * or when the model has no embedded chat template.
   *
   * Retained as a general-purpose bridge API for callers that want the
   * model-native chat formatting path. CharacterRuntime now uses this same
   * template-application path via the runtime surface.
   */
  public applyChatTemplate(
    messages: ChatTemplateMessage[],
    addAssistant: boolean
  ): string {
    try {
      const ptr = this.callNumber(
        'CE_ApplyChatTemplate',
        ['string', 'number'],
        [JSON.stringify(messages), addAssistant ? 1 : 0]
      );
      if (!ptr) {
        return '';
      }
      try {
        return this.module.UTF8ToString(ptr);
      } finally {
        this.module.ccall('CE_FreeString', null, ['pointer'], [ptr]);
      }
    } catch (error) {
      if (this.isMissingOptionalRuntimeApiError('CE_ApplyChatTemplate', error)) {
        return '';
      }
      throw error;
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
      requestObservability: runtimeObservability,
      runtimeObservability,
    };
  }

  public async runSchedulerProgress(
    maxTicks: number,
    maxCompletedResponses: number,
    maxEmittedTokens: number,
    options: {
      maxDurationUs?: number;
    } = {}
  ): Promise<WasmSchedulerProgressResult> {
    const maxDurationUs = Math.max(0, options.maxDurationUs ?? 0);
    const resultPtr = this.ensureBurstResultBuffer();
    const stepResult =
      maxDurationUs > 0
        ? await this.callNumberAsync(
          'CE_RunSchedulerBurstWithDeadline',
          ['number', 'number', 'number', 'number', 'pointer'],
          [maxTicks, maxCompletedResponses, maxEmittedTokens, maxDurationUs, resultPtr]
        )
        : await this.callNumberAsync(
          'CE_RunSchedulerBurst',
          ['number', 'number', 'number', 'pointer'],
          [maxTicks, maxCompletedResponses, maxEmittedTokens, resultPtr]
        );
    const burstResult = this.readSchedulerBurstResult(resultPtr);
    return {
      stepResult,
      completedResponseCount: burstResult.completedResponseCount,
    };
  }

  public drainRuntimeEvents(
    maxEventCount: number,
    textBufferSizeBytes: number = RUNTIME_EVENT_DRAIN_TEXT_BUFFER_SIZE_BYTES
  ): WasmRuntimeEventDrainResult {
    const eventBufferPtr = this.ensureRuntimeEventBuffer(maxEventCount);
    const textBufferPtr = this.ensureRuntimeEventTextBuffer(textBufferSizeBytes);
    const resultPtr = this.ensureRuntimeEventDrainResultBuffer();

    const status = this.callNumber(
      'CE_DrainRuntimeEvents',
      ['pointer', 'number', 'pointer', 'number', 'pointer'],
      [eventBufferPtr, maxEventCount, textBufferPtr, textBufferSizeBytes, resultPtr]
    );
    if (status !== 0) {
      throw new Error(`Failed to drain runtime events. Code: ${status}`);
    }

    const resultOffset = this.heapIndex(resultPtr, 4);
    const eventCount = this.module.HEAP32[resultOffset];
    const terminalRequestIds: GenerateRequestId[] = [];
    const tokenEvents: WasmRuntimeTokenEvent[] = [];
    let textBytes = 0;

    for (let index = 0; index < eventCount; index += 1) {
      const eventBytePtr = this.byteOffset(eventBufferPtr) + index * RUNTIME_EVENT_SIZE_BYTES;
      const eventOffset = this.heapIndex(eventBytePtr, 4);
      const requestId = this.module.HEAP32[eventOffset] as GenerateRequestId;
      const kind = this.module.HEAP32[eventOffset + 1];
      const textOffset = this.module.HEAP32[eventOffset + 3];
      const textLength = this.module.HEAP32[eventOffset + 4];

      if (kind === RUNTIME_EVENT_KIND_TOKEN) {
        if (requestId !== 0 && textLength > 0) {
          let decoder = this.tokenDecoders.get(requestId);
          if (!decoder) {
            decoder = new TextDecoder('utf-8', { fatal: false });
            this.tokenDecoders.set(requestId, decoder);
          }
          const start = this.byteOffset(textBufferPtr + textOffset);
          const end = start + textLength;
          const token = decoder.decode(this.module.HEAPU8.subarray(start, end), { stream: true });

          tokenEvents.push({
            requestId,
            token,
            textLength,
          });
          textBytes += textLength;
        }
        continue;
      }

      if (kind === RUNTIME_EVENT_KIND_TERMINAL && requestId !== 0) {
        terminalRequestIds.push(requestId);
        this.tokenDecoders.delete(requestId);
      }
    }

    return {
      terminalRequestIds,
      tokenEvents,
      textBytes,
    };
  }

  public releaseReusableBuffers(): void {
    if (this.reusableBurstResultPtr !== 0) {
      this.free(this.reusableBurstResultPtr);
      this.reusableBurstResultPtr = 0;
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
    return Number(this.module._malloc(size));
  }

  private free(ptr: number): void {
    this.module._free(ptr);
  }

  private isMissingOptionalRuntimeApiError(ident: string, error: unknown): boolean {
    const message = this.asErrorMessage(error).toLowerCase();
    const normalizedIdent = ident.toLowerCase();
    if (!message.includes(normalizedIdent)) {
      return false;
    }
    return (
      message.includes('unexpected ccall') ||
      message.includes('unknown function') ||
      message.includes('not a function') ||
      message.includes('is not exported') ||
      message.includes('missing')
    );
  }

  private asErrorMessage(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  }

  private ensureBurstResultBuffer(): number {
    if (this.reusableBurstResultPtr === 0) {
      this.reusableBurstResultPtr = this.allocate(SCHEDULER_BURST_RESULT_SIZE_BYTES);
    }
    return this.reusableBurstResultPtr;
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

  private readSchedulerBurstResult(ptr: number): {
    ticksExecuted: number;
    progressedTicks: number;
    completedResponseCount: number;
    emittedTokenCount: number;
  } {
    const i32Offset = this.heapIndex(ptr, 4);
    return {
      ticksExecuted: this.module.HEAP32[i32Offset],
      progressedTicks: this.module.HEAP32[i32Offset + 1],
      completedResponseCount: this.module.HEAP32[i32Offset + 2],
      emittedTokenCount: this.module.HEAP32[i32Offset + 3],
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

      const f64Offset = Math.floor(this.byteOffset(metricsPtr) / 8);
      const i32Offset = Math.floor(this.byteOffset(metricsPtr + RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT * 8) / 4);

      return withDerivedObservabilityMetrics({
        totalMs: this.module.HEAPF64[f64Offset],
        promptEvalMs: this.module.HEAPF64[f64Offset + 1],
        decodeEvalMs: this.module.HEAPF64[f64Offset + 2],
        sampleMs: this.module.HEAPF64[f64Offset + 3],
        queueDelayMs: this.module.HEAPF64[f64Offset + 4],
        ttftMs: this.module.HEAPF64[f64Offset + 5],
        meanItlMs: this.module.HEAPF64[f64Offset + 6],
        tailItlMs: this.module.HEAPF64[f64Offset + 7],
        e2elMs: this.module.HEAPF64[f64Offset + 8],
        inputTokenCount: this.module.HEAP32[i32Offset],
        promptEvalTokens: this.module.HEAP32[i32Offset + 1],
        decodeEvalCount: this.module.HEAP32[i32Offset + 2],
        sampleCount: this.module.HEAP32[i32Offset + 3],
        outputTokenCount: this.module.HEAP32[i32Offset + 4],
        firstSampledTokenId: this.module.HEAP32[i32Offset + 5],
        batchParticipationCount: this.module.HEAP32[i32Offset + 6],
        decodeFirstTickCount: this.module.HEAP32[i32Offset + 7],
        chunkedPrefillTickCount: this.module.HEAP32[i32Offset + 8],
        mixedWorkloadTickCount: this.module.HEAP32[i32Offset + 9],
        lcpReuseTokens: this.module.HEAP32[i32Offset + 10],
        prefixCacheRestoreTokens: this.module.HEAP32[i32Offset + 11],
        prefixCacheHitCount: this.module.HEAP32[i32Offset + 12],
        prefixCacheStoreCount: this.module.HEAP32[i32Offset + 13],
      });
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
