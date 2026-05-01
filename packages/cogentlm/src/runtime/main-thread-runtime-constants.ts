import { TransportObservability } from '../types.js';

export type MountableModelFile = Blob & { name?: string };

export const MAX_PROMPT_TOKENS = 2048;
export const DEFAULT_MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024;
export const REQUEST_STEP_RESULT_INVALID = -1;
export const REQUEST_STEP_RESULT_FATAL_NO_PROGRESS = -2;
export const REQUEST_STEP_RESULT_WAITING = 0;
export const REQUEST_STEP_RESULT_PROGRESSED = 1;
export const REQUEST_STEP_RESULT_TERMINAL = 2;
export const COMPLETED_REQUEST_STATUS_PENDING = 0;
export const COMPLETED_REQUEST_STATUS_COMPLETED = 1;
export const COMPLETED_REQUEST_STATUS_CANCELLED = 2;
export const COMPLETED_REQUEST_STATUS_FAILED = 3;
export const RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES = 128;
export const RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT = 9;
export const SCHEDULER_BURST_RESULT_SIZE_BYTES = 16;
export const REQUEST_ID_SIZE_BYTES = 4;
export const RUNTIME_EVENT_SIZE_BYTES = 20;
export const RUNTIME_EVENT_DRAIN_RESULT_SIZE_BYTES = 8;
export const RUNTIME_EVENT_KIND_TOKEN = 1;
export const RUNTIME_EVENT_KIND_TERMINAL = 2;

export const DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY: TransportObservability = {
  executionMode: 'main-thread',
  workerBacked: false,
  enabled: false,
  bufferedTokenLimit: 0,
  flushIntervalMs: 0,
  flushCount: 0,
  coalescedTokenCount: 0,
  maxObservedBufferedTokenCount: 0,
  activeTokenTransport: 'none',
  runtimeEventDrainCount: 0,
  runtimeEventTokenCount: 0,
  runtimeEventTerminalCount: 0,
  runtimeEventTextBytes: 0,
};

export function normalizeModelFileName(fileName: string): string {
  const trimmed = fileName.trim();
  if (!trimmed) {
    throw new Error('Model file name must not be empty.');
  }
  if (trimmed.includes('/') || trimmed.includes('\\') || trimmed.includes('..')) {
    throw new Error(
      `Invalid model file name "${fileName}". Provide a simple file name, not a path.`
    );
  }
  return trimmed;
}

export function createMountableModelFile(
  blob: Blob,
  fileName: string
): MountableModelFile {
  const normalizedFileName = normalizeModelFileName(fileName);
  const existingName = (blob as MountableModelFile).name;
  if (existingName === normalizedFileName) {
    return blob as MountableModelFile;
  }

  if (typeof File === 'function') {
    return new File([blob], normalizedFileName, {
      type: blob.type,
    }) as MountableModelFile;
  }

  const copiedBlob = blob.slice(0, blob.size, blob.type) as MountableModelFile;
  Object.defineProperty(copiedBlob, 'name', {
    configurable: true,
    value: normalizedFileName,
    writable: false,
  });
  return copiedBlob;
}
