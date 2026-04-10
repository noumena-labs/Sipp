import { CogentConfig } from '../cogent-config.js';
import { TransportObservability } from '../types.js';
import { WorkerSerializableCogentConfig } from './engine-runtime-worker-protocol.js';

export interface PendingWorkerCall {
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  onProgress?: (pct: number) => void;
}

export interface QueuedRequestCompletionState {
  promise: Promise<import('./engine-runtime-worker-protocol.js').WorkerRunQueuedRequestResult>;
  settled: boolean;
  consumed: boolean;
  waiterCount: number;
  callbackError: unknown;
}

export type WithoutCallId<T> = T extends { callId: number } ? Omit<T, 'callId'> : never;

export function createDefaultTransportObservability(): TransportObservability {
  return {
    executionMode: 'worker',
    workerBacked: true,
    enabled: false,
    bufferedTokenLimit: 0,
    flushIntervalMs: 0,
    flushCount: 0,
    coalescedTokenCount: 0,
    maxObservedBufferedTokenCount: 0,
  };
}

export function toTransferableChunkBuffer(chunk: Uint8Array): ArrayBuffer {
  const { buffer, byteOffset, byteLength } = chunk;
  if (buffer instanceof ArrayBuffer && byteOffset === 0 && byteLength === buffer.byteLength) {
    return buffer;
  }
  return chunk.slice().buffer;
}

export function toWorkerSerializableConfig(
  config: CogentConfig
): WorkerSerializableCogentConfig {
  if (typeof config.moduleOptions?.locateFile === 'function') {
    throw new Error(
      'Worker mode does not support moduleOptions.locateFile. Provide explicit moduleUrl/wasmUrl instead.'
    );
  }

  const persistentModelCache =
    config.persistentModelCache == null
      ? undefined
      : {
          enabled: config.persistentModelCache.enabled,
        };

  return {
    moduleUrl: config.moduleUrl,
    wasmUrl: config.wasmUrl,
    moduleOptions: config.moduleOptions,
    maxModelBytes: config.maxModelBytes,
    trustedOrigins: config.trustedOrigins,
    workerMaxBufferedTokens: config.workerMaxBufferedTokens,
    workerTokenFlushIntervalMs: config.workerTokenFlushIntervalMs,
    persistentModelCache,
  };
}
