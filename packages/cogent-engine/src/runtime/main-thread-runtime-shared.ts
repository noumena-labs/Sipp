import { BrowserModelCacheIdentity } from '../storage/browser-model-cache.js';
import { TransportObservability } from '../types.js';

export interface EmscriptenFs {
  analyzePath(path: string): { exists: boolean };
  mkdir(path: string): void;
  writeFile(path: string, data: Uint8Array): void;
  unlink(path: string): void;
  mount(type: any, opts: any, mountpoint: string): void;
  unmount(mountpoint: string): void;
}

export interface EngineModule {
  FS: EmscriptenFs;
  WORKERFS: any;
  HEAP32: Int32Array;
  HEAPF64: Float64Array;
  _free(ptr: number | bigint): void;
  _malloc(size: number | bigint): number | bigint;
  addFunction(func: (...args: any[]) => any, signature: string): number | bigint;
  ccall(
    ident: string,
    returnType: string | null,
    argTypes: string[],
    args: any[],
    opts?: { async?: boolean }
  ): Promise<any> | any;
  removeFunction(ptr: number | bigint): void;
  UTF8ToString(ptr: number | bigint, maxBytesToRead?: number): string;
}

export type MountableModelFile = Blob & { name?: string };

export type UrlShardMetadata = {
  url: string;
  fileName: string;
  contentLength: number;
  cacheIdentity: BrowserModelCacheIdentity;
};

export interface QueuedRequestCompletionState {
  promise: Promise<import('../types.js').GenerateResponse>;
  resolve: (value: import('../types.js').GenerateResponse) => void;
  reject: (error: unknown) => void;
  settled: boolean;
  consumed: boolean;
  waiterCount: number;
  callbackError: unknown;
  cancelRequested: boolean;
}

export const MAX_PROMPT_TOKENS = 2048;
export const DEFAULT_MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024;
export const DEFAULT_PROMPT_FORMAT = 'auto-chat';
export const URL_METADATA_FETCH_CONCURRENCY = 4;
export const URL_DOWNLOAD_CONCURRENCY_OPFS = 4;
export const URL_DOWNLOAD_CONCURRENCY_MEMORY = 2;
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

export const DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY: TransportObservability = {
  executionMode: 'main-thread',
  workerBacked: false,
  enabled: false,
  bufferedTokenLimit: 0,
  flushIntervalMs: 0,
  flushCount: 0,
  coalescedTokenCount: 0,
  maxObservedBufferedTokenCount: 0,
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

export function callModuleNumber(
  module: EngineModule,
  ident: string,
  argTypes: string[] = [],
  args: unknown[] = []
): number {
  const result = module.ccall(ident, 'number', argTypes, args);
  if (result instanceof Promise) {
    throw new Error(`Unexpected async result while calling ${ident}.`);
  }
  return Number(result);
}

export async function callModuleNumberAsync(
  module: EngineModule,
  ident: string,
  argTypes: string[] = [],
  args: unknown[] = []
): Promise<number> {
  const result = module.ccall(ident, 'number', argTypes, args, {
    async: true,
  });
  return Number(await result);
}

export async function mapWithConcurrency<T, TResult>(
  items: readonly T[],
  concurrency: number,
  mapper: (item: T, index: number) => Promise<TResult>,
  onError?: (error: unknown) => void
): Promise<TResult[]> {
  if (items.length === 0) {
    return [];
  }

  const results = new Array<TResult>(items.length);
  const workerCount = Math.min(Math.max(1, concurrency), items.length);
  let nextIndex = 0;
  let firstError: unknown = null;

  const workers = Array.from({ length: workerCount }, async () => {
    while (true) {
      if (firstError != null) {
        return;
      }

      const currentIndex = nextIndex;
      nextIndex += 1;
      if (currentIndex >= items.length) {
        return;
      }

      try {
        results[currentIndex] = await mapper(items[currentIndex], currentIndex);
      } catch (error) {
        if (firstError == null) {
          firstError = error;
          onError?.(error);
        }
        throw error;
      }
    }
  });

  await Promise.allSettled(workers);
  if (firstError != null) {
    throw firstError;
  }
  return results;
}
