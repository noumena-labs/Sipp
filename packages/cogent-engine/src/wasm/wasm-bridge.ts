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

export class WasmBridge {
  private schedulerBurstApiAvailable: boolean | null = null;
  private completedRequestDrainApiAvailable: boolean | null = null;
  private runtimeEventDrainApiAvailable: boolean | null = null;

  public constructor(private readonly module: EngineModule) {}

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

  public registerTokenCallback(
    onToken: (token: string, length: number) => number
  ): number | bigint {
    return this.module.addFunction((rawPtr: number | bigint, length: number) => {
      return onToken(this.module.UTF8ToString(Number(rawPtr), length), length);
    }, 'ipi');
  }

  public unregisterCallback(callbackPtr: number | bigint): void {
    this.module.removeFunction(callbackPtr);
  }

  public async initEngine(
    modelPath: string,
    normalizedConfig: NormalizedInitConfig
  ): Promise<number> {
    const result = await this.module.ccall(
      'CE_Init',
      'number',
      [
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
      ],
      [
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
      ],
      { async: true }
    );
    return Number(result);
  }

  public close(): void {
    this.module.ccall('CE_Close', null, [], []);
  }

  public enqueuePrompt(
    contextKey: string,
    promptText: string,
    maxOutputTokens: number,
    callbackPtr: number
  ): GenerateRequestId {
    const requestId = this.module.ccall(
      'CE_EnqueuePrompt',
      'number',
      ['string', 'string', 'number', 'pointer'],
      [contextKey, promptText, maxOutputTokens, callbackPtr]
    );
    if (requestId instanceof Promise) {
      throw new Error('Unexpected async result while enqueuing a request.');
    }
    return requestId as GenerateRequestId;
  }

  public async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
    const result = this.module.ccall(
      'CE_CancelQueuedRequest',
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

  public supportsRuntimeEventDrain(): boolean {
    if (this.runtimeEventDrainApiAvailable != null) {
      return this.runtimeEventDrainApiAvailable;
    }

    const resultPtr = this.allocate(RUNTIME_EVENT_DRAIN_RESULT_SIZE_BYTES);
    try {
      this.callNumber(
        'CE_DrainRuntimeEvents',
        ['pointer', 'number', 'pointer', 'number', 'pointer'],
        [0, 0, 0, 0, resultPtr]
      );
      this.runtimeEventDrainApiAvailable = true;
    } catch (error) {
      if (!this.isMissingOptionalRuntimeApiError('CE_DrainRuntimeEvents', error)) {
        throw error;
      }
      this.runtimeEventDrainApiAvailable = false;
    } finally {
      this.free(resultPtr);
    }

    return this.runtimeEventDrainApiAvailable;
  }

  public async runSchedulerProgress(
    maxTicks: number,
    maxCompletedResponses: number,
    maxEmittedTokens: number
  ): Promise<WasmSchedulerProgressResult> {
    if (this.schedulerBurstApiAvailable === false) {
      return {
        stepResult: await this.callNumberAsync('CE_RunSchedulerTick'),
        completedResponseCount: 0,
      };
    }

    const resultPtr = this.allocate(SCHEDULER_BURST_RESULT_SIZE_BYTES);
    try {
      const stepResult = await this.callNumberAsync(
        'CE_RunSchedulerBurst',
        ['number', 'number', 'number', 'pointer'],
        [maxTicks, maxCompletedResponses, maxEmittedTokens, resultPtr]
      );
      this.schedulerBurstApiAvailable = true;
      const burstResult = this.readSchedulerBurstResult(resultPtr);
      return {
        stepResult,
        completedResponseCount: burstResult.completedResponseCount,
      };
    } catch (error) {
      if (!this.isMissingOptionalRuntimeApiError('CE_RunSchedulerBurst', error)) {
        throw error;
      }
      this.schedulerBurstApiAvailable = false;
      return {
        stepResult: await this.callNumberAsync('CE_RunSchedulerTick'),
        completedResponseCount: 0,
      };
    } finally {
      this.free(resultPtr);
    }
  }

  public drainCompletedRequestIds(maxCount: number): GenerateRequestId[] | null {
    if (this.completedRequestDrainApiAvailable === false) {
      return null;
    }

    const bufferPtr = this.allocate(maxCount * 4);
    try {
      const drainedCount = this.callNumber(
        'CE_DrainCompletedRequestIds',
        ['pointer', 'number'],
        [bufferPtr, maxCount]
      );
      this.completedRequestDrainApiAvailable = true;
      if (drainedCount <= 0) {
        return [];
      }

      const i32Offset = bufferPtr >> 2;
      return Array.from(
        this.module.HEAP32.subarray(i32Offset, i32Offset + drainedCount),
        (requestId) => requestId as GenerateRequestId
      ).filter((requestId) => requestId !== 0);
    } catch (error) {
      if (!this.isMissingOptionalRuntimeApiError('CE_DrainCompletedRequestIds', error)) {
        throw error;
      }
      this.completedRequestDrainApiAvailable = false;
      return null;
    } finally {
      this.free(bufferPtr);
    }
  }

  public drainRuntimeEvents(
    maxEventCount: number,
    textBufferSizeBytes: number = RUNTIME_EVENT_DRAIN_TEXT_BUFFER_SIZE_BYTES
  ): WasmRuntimeEventDrainResult | null {
    if (!this.supportsRuntimeEventDrain()) {
      return null;
    }

    const eventBufferPtr = this.allocate(maxEventCount * RUNTIME_EVENT_SIZE_BYTES);
    const textBufferPtr = this.allocate(textBufferSizeBytes);
    const resultPtr = this.allocate(RUNTIME_EVENT_DRAIN_RESULT_SIZE_BYTES);

    try {
      const status = this.callNumber(
        'CE_DrainRuntimeEvents',
        ['pointer', 'number', 'pointer', 'number', 'pointer'],
        [eventBufferPtr, maxEventCount, textBufferPtr, textBufferSizeBytes, resultPtr]
      );
      if (status !== 0) {
        throw new Error(`Failed to drain runtime events. Code: ${status}`);
      }

      const resultOffset = resultPtr >> 2;
      const eventCount = this.module.HEAP32[resultOffset];
      const terminalRequestIds: GenerateRequestId[] = [];
      const tokenEvents: WasmRuntimeTokenEvent[] = [];
      let textBytes = 0;

      for (let index = 0; index < eventCount; index += 1) {
        const eventOffset = (eventBufferPtr + index * RUNTIME_EVENT_SIZE_BYTES) >> 2;
        const requestId = this.module.HEAP32[eventOffset] as GenerateRequestId;
        const kind = this.module.HEAP32[eventOffset + 1];
        const textOffset = this.module.HEAP32[eventOffset + 3];
        const textLength = this.module.HEAP32[eventOffset + 4];

        if (kind === RUNTIME_EVENT_KIND_TOKEN) {
          if (requestId !== 0 && textLength > 0) {
            tokenEvents.push({
              requestId,
              token: this.module.UTF8ToString(textBufferPtr + textOffset, textLength),
              textLength,
            });
            textBytes += textLength;
          }
          continue;
        }

        if (kind === RUNTIME_EVENT_KIND_TERMINAL && requestId !== 0) {
          terminalRequestIds.push(requestId);
        }
      }

      return {
        terminalRequestIds,
        tokenEvents,
        textBytes,
      };
    } catch (error) {
      if (!this.isMissingOptionalRuntimeApiError('CE_DrainRuntimeEvents', error)) {
        throw error;
      }
      this.runtimeEventDrainApiAvailable = false;
      return null;
    } finally {
      this.free(eventBufferPtr);
      this.free(textBufferPtr);
      this.free(resultPtr);
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

  private readSchedulerBurstResult(ptr: number): {
    ticksExecuted: number;
    progressedTicks: number;
    completedResponseCount: number;
    emittedTokenCount: number;
  } {
    const i32Offset = ptr >> 2;
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

      const f64Offset = (metricsPtr / 8) | 0;
      const i32Offset = ((metricsPtr + RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT * 8) / 4) | 0;

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
        batchParticipationCount: this.module.HEAP32[i32Offset + 5],
        decodeFirstTickCount: this.module.HEAP32[i32Offset + 6],
        chunkedPrefillTickCount: this.module.HEAP32[i32Offset + 7],
        mixedWorkloadTickCount: this.module.HEAP32[i32Offset + 8],
        lcpReuseTokens: this.module.HEAP32[i32Offset + 9],
        prefixCacheRestoreTokens: this.module.HEAP32[i32Offset + 10],
        prefixCacheHitCount: this.module.HEAP32[i32Offset + 11],
        prefixCacheStoreCount: this.module.HEAP32[i32Offset + 12],
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
