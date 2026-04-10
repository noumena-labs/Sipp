import {
  GenerateRequestId,
  GenerateResponse,
  RequestObservabilityMetrics,
  RuntimeAggregateObservabilityMetrics,
  RuntimeObservabilityMetrics,
} from '../types.js';
import {
  COMPLETED_REQUEST_STATUS_CANCELLED,
  COMPLETED_REQUEST_STATUS_COMPLETED,
  COMPLETED_REQUEST_STATUS_FAILED,
  COMPLETED_REQUEST_STATUS_PENDING,
  EngineModule,
  RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT,
  RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES,
} from './main-thread-runtime-shared.js';

type CallNumberFn = (
  module: EngineModule,
  ident: string,
  argTypes?: string[],
  args?: unknown[]
) => number;

export function readRuntimeObservabilityFromModule(
  module: EngineModule,
  callNumber: CallNumberFn
): RuntimeAggregateObservabilityMetrics | null {
  return readRuntimeObservabilityViaCall(
    module,
    'CE_GetRuntimeObservability',
    ['pointer'],
    [],
    callNumber
  );
}

export function readCompletedRequestRuntimeObservability(
  module: EngineModule,
  requestId: GenerateRequestId,
  callNumber: CallNumberFn
): RequestObservabilityMetrics | null {
  return readRuntimeObservabilityViaCall(
    module,
    'CE_GetCompletedRequestRuntimeObservability',
    ['number'],
    [requestId],
    callNumber
  );
}

export function takeCompletedResponse(
  module: EngineModule,
  requestId: GenerateRequestId,
  callNumber: CallNumberFn
): GenerateResponse {
  const status = callNumber(module, 'CE_GetCompletedRequestStatus', ['number'], [requestId]);
  if (status === COMPLETED_REQUEST_STATUS_PENDING) {
    throw new Error('Queued request reached a terminal step without a completed response.');
  }

  const outputText = copyCompletedRequestText(
    module,
    requestId,
    'CE_GetCompletedRequestOutputSize',
    'CE_CopyCompletedRequestOutput',
    'output',
    callNumber
  );
  const errorText = copyCompletedRequestText(
    module,
    requestId,
    'CE_GetCompletedRequestErrorSize',
    'CE_CopyCompletedRequestError',
    'error',
    callNumber
  );
  const runtimeObservability = readCompletedRequestRuntimeObservability(
    module,
    requestId,
    callNumber
  );
  const consumed = callNumber(module, 'CE_ConsumeCompletedRequest', ['number'], [requestId]);
  if (!consumed) {
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

function readRuntimeObservabilityViaCall(
  module: EngineModule,
  ident: string,
  argTypes: string[],
  args: unknown[],
  callNumber: CallNumberFn
): RuntimeObservabilityMetrics | null {
  const metricsPtr = Number(module._malloc(RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES));
  if (!metricsPtr) {
    throw new Error('Failed to allocate runtime observability buffer.');
  }

  try {
    const status = callNumber(module, ident, [...argTypes, 'pointer'], [...args, metricsPtr]);
    if (status !== 0) {
      return null;
    }

    const f64Offset = (metricsPtr / 8) | 0;
    const i32Offset = ((metricsPtr + RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT * 8) / 4) | 0;

    return {
      totalMs: module.HEAPF64[f64Offset],
      promptEvalMs: module.HEAPF64[f64Offset + 1],
      decodeEvalMs: module.HEAPF64[f64Offset + 2],
      sampleMs: module.HEAPF64[f64Offset + 3],
      queueDelayMs: module.HEAPF64[f64Offset + 4],
      ttftMs: module.HEAPF64[f64Offset + 5],
      meanItlMs: module.HEAPF64[f64Offset + 6],
      tailItlMs: module.HEAPF64[f64Offset + 7],
      e2elMs: module.HEAPF64[f64Offset + 8],
      inputTokenCount: module.HEAP32[i32Offset],
      promptEvalTokens: module.HEAP32[i32Offset + 1],
      decodeEvalCount: module.HEAP32[i32Offset + 2],
      sampleCount: module.HEAP32[i32Offset + 3],
      outputTokenCount: module.HEAP32[i32Offset + 4],
      batchParticipationCount: module.HEAP32[i32Offset + 5],
      decodeFirstTickCount: module.HEAP32[i32Offset + 6],
      chunkedPrefillTickCount: module.HEAP32[i32Offset + 7],
      mixedWorkloadTickCount: module.HEAP32[i32Offset + 8],
      lcpReuseTokens: module.HEAP32[i32Offset + 9],
      prefixCacheRestoreTokens: module.HEAP32[i32Offset + 10],
      prefixCacheHitCount: module.HEAP32[i32Offset + 11],
      prefixCacheStoreCount: module.HEAP32[i32Offset + 12],
    };
  } finally {
    module._free(metricsPtr);
  }
}

function copyCompletedRequestText(
  module: EngineModule,
  requestId: GenerateRequestId,
  sizeFunction: string,
  copyFunction: string,
  fieldName: string,
  callNumber: CallNumberFn
): string {
  const byteLength = callNumber(module, sizeFunction, ['number'], [requestId]);
  if (byteLength < 0) {
    throw new Error(`Failed to read queued request ${fieldName} size.`);
  }
  if (byteLength === 0) {
    return '';
  }

  const rawBufferPtr = module._malloc(byteLength + 1);
  if (!rawBufferPtr) {
    throw new Error(`Failed to allocate queued request ${fieldName} buffer.`);
  }
  const bufferPtr = Number(rawBufferPtr);

  try {
    const copied = callNumber(module, copyFunction, ['number', 'pointer', 'number'], [
      requestId,
      bufferPtr,
      byteLength + 1,
    ]);
    if (copied !== byteLength) {
      throw new Error(`Failed to copy queued request ${fieldName}.`);
    }
    return module.UTF8ToString(bufferPtr, byteLength);
  } finally {
    module._free(bufferPtr);
  }
}
