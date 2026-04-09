import { FileSystemStorage } from '../storage/file-system-storage.js';
import { CogentConfig, EngineModuleOptions } from '../cogent-config.js';
import { normalizeInitConfig } from '../core/init-config.js';
import { formatPromptText } from '../core/prompt-format.js';
import {
  BackendObservability,
  EngineExecutionMode,
  GenerateRequest,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  ModelLoadInfo,
  ModelLoadSourceKind,
  ModelLoadReuseMode,
  PromptOptions,
  RuntimeObservabilityMetrics,
  TransportObservability,
} from '../types.js';
import { EngineRuntime } from './engine-runtime.js';

interface FsStream {
  fd: number;
  position: number;
}

interface EmscriptenFs {
  analyzePath(path: string): { exists: boolean };
  mkdir(path: string): void;
  writeFile(path: string, data: Uint8Array): void;
  unlink(path: string): void;
  open(path: string, flags: string): FsStream;
  write(stream: FsStream, buffer: Uint8Array, offset: number, length: number, position: number): number;
  close(stream: FsStream): void;
  mount(type: any, opts: any, mountpoint: string): void;
  unmount(mountpoint: string): void;
}

interface EngineModule {
  FS: EmscriptenFs;
  WORKERFS: any;
  HEAP32: Int32Array;
  HEAPF64: Float64Array;
  HEAP64: BigInt64Array;
  HEAPU64: BigUint64Array;
  _CE_FreeString(ptr: number | bigint): void;
  _free(ptr: number | bigint): void;
  _malloc(size: number | bigint): number | bigint;
  addFunction(func: (...args: any[]) => any, signature: string): number | bigint;
  ccall(ident: string, returnType: string | null, argTypes: string[], args: any[], opts?: { async?: boolean }): Promise<any> | any;
  removeFunction(ptr: number | bigint): void;
  UTF8ToString(ptr: number | bigint, maxBytesToRead?: number): string;
}

type MountableModelFile = Blob & { name?: string };
type HeaderLookup = { get(name: string): string | null };

const MAX_PROMPT_TOKENS = 2048;
const DEFAULT_MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024;
const DEFAULT_PROMPT_FORMAT = 'auto-chat';
const MEMFS_FILE_SIZE_LIMIT = 2 * 1024 * 1024 * 1024 - 1024 * 1024; // ~2GB
const REQUEST_STEP_RESULT_INVALID = -1;
const REQUEST_STEP_RESULT_FATAL_NO_PROGRESS = -2;
const REQUEST_STEP_RESULT_WAITING = 0;
const REQUEST_STEP_RESULT_PROGRESSED = 1;
const REQUEST_STEP_RESULT_TERMINAL = 2;
const COMPLETED_REQUEST_STATUS_PENDING = 0;
const COMPLETED_REQUEST_STATUS_COMPLETED = 1;
const COMPLETED_REQUEST_STATUS_CANCELLED = 2;
const COMPLETED_REQUEST_STATUS_FAILED = 3;
const RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES = 128;
const RUNTIME_OBSERVABILITY_DOUBLE_FIELD_COUNT = 9;

const DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY: TransportObservability = {
  executionMode: 'main-thread',
  workerBacked: false,
  enabled: false,
  bufferedTokenLimit: 0,
  flushIntervalMs: 0,
  flushCount: 0,
  coalescedTokenCount: 0,
  maxObservedBufferedTokenCount: 0,
};

function createAbortError(message = 'The operation was aborted.'): Error {
  if (typeof DOMException === 'function') {
    return new DOMException(message, 'AbortError');
  }
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function normalizeModelFileName(fileName: string): string {
  const trimmed = fileName.trim();
  if (!trimmed) {
    throw new Error('Model file name must not be empty.');
  }
  if (trimmed.includes('/') || trimmed.includes('\\') || trimmed.includes('..')) {
    throw new Error(`Invalid model file name "${fileName}". Provide a simple file name, not a path.`);
  }
  return trimmed;
}

function asErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function hashCacheIdentity(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let hash = 0x811c9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

export class MainThreadEngineRuntime implements EngineRuntime {
  private module: EngineModule | null = null;
  private initPromise: Promise<void> | null = null;
  private engineInitialized = false;
  private loadedModelPaths: string[] = [];
  private workerFsMountPath: string | null = null;
  private readonly opfs = new FileSystemStorage();
  private queuedPromptCallbacks = new Map<
    GenerateRequestId,
    ((token: string) => void) | undefined
  >();
  private queuedPromptCallbackPtrs = new Map<GenerateRequestId, number | bigint>();
  private queuedPromptTokenBuffers = new Map<GenerateRequestId, string[]>();
  private queuedPromptCallbackErrors = new Map<GenerateRequestId, unknown>();
  private activeQueuedRequestRuns = new Set<GenerateRequestId>();
  private lastModelLoadInfo: ModelLoadInfo | null = null;
  private runtimeObservabilityEnabled = false;
  private backendProfilingEnabled = false;
  private transportObservability: TransportObservability = {
    ...DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
  };
  constructor(private config: CogentConfig = {}) { }

  private resolveWasmUrls(): { moduleUrl: string; wasmUrl: string } {
    const moduleUrl = this.config.moduleUrl?.trim();
    const wasmUrl = this.config.wasmUrl?.trim();

    if (!moduleUrl || !wasmUrl) {
      throw new Error(
        'Both "moduleUrl" and "wasmUrl" must be provided in CogentEngine config. Use getBundledRuntimeUrls() for the package defaults.'
      );
    }

    const module = this.parseConfiguredUrl(moduleUrl, 'moduleUrl');
    const wasm = this.parseConfiguredUrl(wasmUrl, 'wasmUrl');
    const trustedOrigins = this.resolveTrustedOrigins();

    if (trustedOrigins.size > 0) {
      if (!trustedOrigins.has(module.origin)) {
        throw new Error(`Blocked moduleUrl origin "${module.origin}". Add it to trustedOrigins to allow it.`);
      }
      if (!trustedOrigins.has(wasm.origin)) {
        throw new Error(`Blocked wasmUrl origin "${wasm.origin}". Add it to trustedOrigins to allow it.`);
      }
    }

    return { moduleUrl: module.toString(), wasmUrl: wasm.toString() };
  }

  private parseConfiguredUrl(rawUrl: string, fieldName: string): URL {
    try {
      if (typeof window !== 'undefined' && typeof window.location?.href === 'string') {
        return new URL(rawUrl, window.location.href);
      }
      return new URL(rawUrl);
    } catch {
      throw new Error(`Invalid ${fieldName} value "${rawUrl}".`);
    }
  }

  private resolveTrustedOrigins(): Set<string> {
    const configuredOrigins = this.config.trustedOrigins ?? [];
    if (configuredOrigins.length > 0) {
      const allowed = new Set<string>();
      for (const originValue of configuredOrigins) {
        const normalizedOrigin = this.parseConfiguredUrl(originValue, 'trustedOrigins').origin;
        allowed.add(normalizedOrigin);
      }
      return allowed;
    }

    if (typeof window !== 'undefined' && typeof window.location?.origin === 'string') {
      return new Set([window.location.origin]);
    }

    return new Set();
  }

  private resolveMaxModelBytes(): number {
    const maxModelBytes = this.config.maxModelBytes ?? DEFAULT_MAX_MODEL_BYTES;
    if (!Number.isInteger(maxModelBytes) || maxModelBytes <= 0) {
      throw new Error('"maxModelBytes" must be a positive integer.');
    }
    return maxModelBytes;
  }

  private setLastModelLoadInfo(info: ModelLoadInfo): void {
    this.lastModelLoadInfo = info;
  }

  public getExecutionMode(): EngineExecutionMode {
    return 'main-thread';
  }

  public getLastModelLoadInfo(): ModelLoadInfo | null {
    return this.lastModelLoadInfo;
  }

  public getTransportObservability(): TransportObservability {
    return { ...this.transportObservability };
  }

  private normalizeTokenCount(nTokens: number): number {
    if (!Number.isInteger(nTokens)) {
      throw new Error('nTokens must be an integer.');
    }
    if (nTokens <= 0 || nTokens > MAX_PROMPT_TOKENS) {
      throw new Error(`nTokens must be between 1 and ${MAX_PROMPT_TOKENS}.`);
    }
    return nTokens;
  }

  private resolvePromptTokenCount(
    input: number | PromptOptions | undefined
  ): number {
    if (typeof input === 'number' || input === undefined) {
      return this.normalizeTokenCount(input ?? 128);
    }
    return this.normalizeTokenCount(input.nTokens ?? 128);
  }

  private resolvePromptFormat(
    input: PromptOptions | number | undefined
  ): 'auto-chat' | 'raw' {
    if (typeof input === 'number' || input === undefined) {
      return DEFAULT_PROMPT_FORMAT;
    }
    return input.promptFormat ?? DEFAULT_PROMPT_FORMAT;
  }

  private buildGenerateRequest(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions
  ): GenerateRequest {
    const promptFormat = this.resolvePromptFormat(options);
    return {
      contextKey,
      promptText: formatPromptText(promptText, promptFormat),
      maxOutputTokens: this.resolvePromptTokenCount(options),
      promptFormat,
    };
  }

  private getLoadedModule(): EngineModule {
    if (!this.module) {
      throw new Error('Module is not initialized. Call initModule() first.');
    }
    return this.module;
  }

  private getReadyEngineModule(): EngineModule {
    const module = this.getLoadedModule();
    if (!this.engineInitialized) {
      throw new Error('Engine is not initialized. Call initEngine(modelPath, config?) first.');
    }
    return module;
  }

  private releaseQueuedPromptCallback(module: EngineModule, requestId: GenerateRequestId): void {
    const callbackPtr = this.queuedPromptCallbackPtrs.get(requestId);
    if (callbackPtr != null) {
      module.removeFunction(callbackPtr);
    }
    this.queuedPromptCallbacks.delete(requestId);
    this.queuedPromptTokenBuffers.delete(requestId);
    this.queuedPromptCallbackPtrs.delete(requestId);
    this.queuedPromptCallbackErrors.delete(requestId);
  }

  private consumeCompletedResponseIfPresent(
    module: EngineModule,
    requestId: GenerateRequestId
  ): boolean {
    const status = this.callNumber(module, 'CE_GetCompletedRequestStatus', ['number'], [requestId]);
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      return false;
    }

    const consumed = this.callNumber(module, 'CE_ConsumeCompletedRequest', ['number'], [requestId]);
    if (!consumed) {
      throw new Error('Failed to consume completed queued request response.');
    }
    return true;
  }

  private releaseCancelledQueuedRequestState(
    module: EngineModule,
    requestId: GenerateRequestId
  ): void {
    this.consumeCompletedResponseIfPresent(module, requestId);
    this.releaseQueuedPromptCallback(module, requestId);
  }

  private releaseAllQueuedPromptCallbacks(module: EngineModule): void {
    for (const callbackPtr of this.queuedPromptCallbackPtrs.values()) {
      module.removeFunction(callbackPtr);
    }
    this.queuedPromptCallbacks.clear();
    this.queuedPromptTokenBuffers.clear();
    this.queuedPromptCallbackPtrs.clear();
    this.queuedPromptCallbackErrors.clear();
    this.activeQueuedRequestRuns.clear();
  }

  private resetRuntimeLifecycleState(): void {
    this.activeQueuedRequestRuns.clear();
    this.runtimeObservabilityEnabled = false;
    this.backendProfilingEnabled = false;
    this.transportObservability = {
      ...DEFAULT_MAIN_THREAD_TRANSPORT_OBSERVABILITY,
    };
  }

  private bufferQueuedTokenPiece(requestId: GenerateRequestId, token: string): void {
    const buffered = this.queuedPromptTokenBuffers.get(requestId);
    if (buffered != null) {
      buffered.push(token);
      return;
    }
    this.queuedPromptTokenBuffers.set(requestId, [token]);
  }

  private flushQueuedTokenPieces(requestId: GenerateRequestId): void {
    const onToken = this.queuedPromptCallbacks.get(requestId);
    const bufferedPieces = this.queuedPromptTokenBuffers.get(requestId);
    if (onToken == null || bufferedPieces == null || bufferedPieces.length === 0) {
      if (bufferedPieces != null) {
        bufferedPieces.length = 0;
      }
      return;
    }

    while (bufferedPieces.length > 0) {
      const piece = bufferedPieces.shift();
      if (piece == null) {
        continue;
      }
      try {
        onToken(piece);
      } catch (error) {
        this.queuedPromptCallbackErrors.set(requestId, error);
        break;
      }
    }
  }

  private waitForNextSchedulerStep(): Promise<void> {
    return new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  }

  private callNumber(
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

  private async callNumberAsync(
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


  private readRuntimeObservabilityFromModule(
    module: EngineModule
  ): RuntimeObservabilityMetrics | null {
    return this.readRuntimeObservabilityViaCall(
      module,
      'CE_GetRuntimeObservability',
      ['pointer'],
      []
    );
  }

  private readRuntimeObservabilityViaCall(
    module: EngineModule,
    ident: string,
    argTypes: string[],
    args: unknown[]
  ): RuntimeObservabilityMetrics | null {
    const metricsPtr = Number(module._malloc(RUNTIME_OBSERVABILITY_METRICS_SIZE_BYTES));
    if (!metricsPtr) {
      throw new Error('Failed to allocate runtime observability buffer.');
    }

    try {
      const status = this.callNumber(module, ident, [...argTypes, 'pointer'], [...args, metricsPtr]);
      if (status !== 0) {
        return null;
      }

      // Integer division for typed array index: ptr must be aligned.
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

  private readCompletedRequestRuntimeObservability(
    module: EngineModule,
    requestId: GenerateRequestId
  ): RuntimeObservabilityMetrics | null {
    return this.readRuntimeObservabilityViaCall(
      module,
      'CE_GetCompletedRequestRuntimeObservability',
      ['number'],
      [requestId]
    );
  }

  private copyCompletedRequestText(
    module: EngineModule,
    requestId: GenerateRequestId,
    sizeFunction: string,
    copyFunction: string,
    fieldName: string
  ): string {
    const byteLength = this.callNumber(module, sizeFunction, ['number'], [requestId]);
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
      const copied = this.callNumber(module, copyFunction, ['number', 'pointer', 'number'], [
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

  private takeCompletedResponse(
    module: EngineModule,
    requestId: GenerateRequestId
  ): GenerateResponse {
    const status = this.callNumber(module, 'CE_GetCompletedRequestStatus', ['number'], [requestId]);
    if (status === COMPLETED_REQUEST_STATUS_PENDING) {
      throw new Error('Queued request reached a terminal step without a completed response.');
    }

    const outputText = this.copyCompletedRequestText(
      module,
      requestId,
      'CE_GetCompletedRequestOutputSize',
      'CE_CopyCompletedRequestOutput',
      'output'
    );
    const errorText = this.copyCompletedRequestText(
      module,
      requestId,
      'CE_GetCompletedRequestErrorSize',
      'CE_CopyCompletedRequestError',
      'error'
    );
    const runtimeObservability = this.readCompletedRequestRuntimeObservability(
      module,
      requestId
    );
    const consumed = this.callNumber(module, 'CE_ConsumeCompletedRequest', ['number'], [requestId]);
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
      runtimeObservability,
    };
  }

  private removeFileIfExists(module: EngineModule, path: string): void {
    if (module.FS.analyzePath(path).exists) {
      module.FS.unlink(path);
    }
  }

  private createMountableModelFile(blob: Blob, fileName: string): MountableModelFile {
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

  private buildPersistentCacheKey(
    rawUrl: string,
    fileName: string,
    headers?: HeaderLookup | null
  ): string {
    const canonicalUrl = this.parseConfiguredUrl(rawUrl, 'modelUrl').toString();
    const etag = headers?.get('ETag')?.trim() ?? '';
    const lastModified = headers?.get('Last-Modified')?.trim() ?? '';
    const contentLength = headers?.get('Content-Length')?.trim() ?? '';
    const identity = [
      canonicalUrl,
      etag,
      lastModified,
      contentLength,
    ].join('\n');
    return `${hashCacheIdentity(identity)}-${normalizeModelFileName(fileName)}`;
  }

  private removeAllLoadedModelFiles(module: EngineModule): void {
    for (const p of this.loadedModelPaths) {
      // Don't try to unlink files inside a WORKERFS mount point.
      // They will be "cleaned up" when the mount is unmounted.
      if (this.workerFsMountPath && p.startsWith(this.workerFsMountPath)) {
        continue;
      }
      this.removeFileIfExists(module, p);
    }
    this.loadedModelPaths = [];
  }

  private commitLoadedModelPaths(module: EngineModule, paths: string[]): void {
    // Clean up any previously loaded model files that aren't in the new set
    const newSet = new Set(paths);
    for (const p of this.loadedModelPaths) {
      if (!newSet.has(p)) {
        this.removeFileIfExists(module, p);
      }
    }
    this.loadedModelPaths = [...paths];
  }

  private prepareModelPath(module: EngineModule, destFileName: string): string {
    const safeName = normalizeModelFileName(destFileName);
    const modelPath = `/models/${safeName}`;
    this.ensureModelsDir(module);
    this.removeFileIfExists(module, modelPath);
    return modelPath;
  }

  private async importModuleFactory(moduleUrl: string): Promise<(options: EngineModuleOptions) => Promise<EngineModule>> {
    const importedModule = await import(/* @vite-ignore */ moduleUrl);
    const createModule = importedModule.default;
    if (typeof createModule !== 'function') {
      throw new Error(`Invalid Emscripten module at "${moduleUrl}"`);
    }
    return createModule as (options: EngineModuleOptions) => Promise<EngineModule>;
  }

  private async ensureModule(): Promise<EngineModule> {
    if (this.module) {
      return this.module;
    }
    await this.initModule();
    return this.getLoadedModule();
  }

  /**
   * Initializes the underlying WebAssembly module.
   */
  public async initModule() {
    if (this.module) {
      return;
    }
    if (!this.initPromise) {
      this.initPromise = (async () => {
        const { moduleUrl, wasmUrl } = this.resolveWasmUrls();
        const createModule = await this.importModuleFactory(moduleUrl);
        const moduleConfig: EngineModuleOptions = { ...(this.config.moduleOptions ?? {}) };
        const userLocateFile = moduleConfig.locateFile;

        moduleConfig.locateFile = (path: string, prefix?: string) => {
          if (path.endsWith('.wasm')) {
            return wasmUrl;
          }
          if (userLocateFile) {
            return userLocateFile(path, prefix);
          }
          return prefix ? `${prefix}${path}` : path;
        };

        this.module = await createModule(moduleConfig);
      })().catch((error) => {
        this.initPromise = null;
        this.module = null;
        throw error;
      });
    }
    await this.initPromise;
  }

  private ensureModelsDir(module: EngineModule) {
    const modelsPath = '/models';
    if (!module.FS.analyzePath(modelsPath).exists) {
      module.FS.mkdir(modelsPath);
    }
  }


  /**
   * Load a GGUF model from a URL.
   * 
   * This streams the model directly to OPFS (Persistent Storage) if supported,
   * then mounts it into the WASM filesystem via WORKERFS. This is zero-copy
   * and handles files >2GB.
   */
  public async loadModelFromUrl(
    url: string,
    destFileName: string = 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    return this.loadModelFromUrls([url], onProgress, signal);
  }

  /**
   * Load a model from a ReadableStream.
   * 
   * The stream is piped to OPFS then zero-copy mounted.
   */
  public async loadModelFromReadableStream(
    stream: ReadableStream<Uint8Array>,
    destFileName: string = 'model.gguf',
    options: {
      expectedBytes?: number;
      onProgress?: (pct: number) => void;
      signal?: AbortSignal;
    } = {}
  ): Promise<string> {
    const module = await this.ensureModule();
    const opfsEnabled = FileSystemStorage.isSupported() && this.config.persistentModelCache?.enabled !== false;

    let modelFile: Blob;
    if (opfsEnabled) {
      modelFile = await this.opfs.streamToDisk(
        destFileName,
        stream,
        options.onProgress,
        options.signal
      );
    } else {
      // Fallback for environments without OPFS
      const reader = stream.getReader();
      const chunks: Uint8Array[] = [];
      const abortListener =
        options.signal == null
          ? null
          : () => {
              void reader.cancel(createAbortError('Model load aborted.'));
            };
      if (abortListener != null) {
        options.signal?.addEventListener('abort', abortListener, { once: true });
      }
      try {
        while (true) {
          if (options.signal?.aborted) {
            throw createAbortError('Model load aborted.');
          }
          const { done, value } = await reader.read();
          if (done) {
            if (options.signal?.aborted) {
              throw createAbortError('Model load aborted.');
            }
            break;
          }
          if (value != null) {
            chunks.push(value);
          }
        }
      } catch (error) {
        if (isAbortError(error) || options.signal?.aborted) {
          throw createAbortError('Model load aborted.');
        }
        throw error;
      } finally {
        if (abortListener != null) {
          options.signal?.removeEventListener('abort', abortListener);
        }
        reader.releaseLock();
      }
      modelFile = this.createMountableModelFile(new Blob(chunks as any), destFileName);
    }

    const modelPath = await this.mountModelFiles(module, [modelFile]);

    this.setLastModelLoadInfo({
      sourceKind: 'buffer',
      reuseMode: 'buffer',
      modelPath,
      fileName: destFileName,
      byteLength: modelFile.size,
      persistentCacheEnabled: FileSystemStorage.isSupported(),
      persistentCacheKey: null,
      persistentCacheHit: false,
      persistentCacheStored: FileSystemStorage.isSupported(),
    });

    return modelPath;
  }

  /**
   * Internal helper to mount one or more Blob/File objects into the WASM filesystem
   * using the zero-copy WORKERFS driver.
   *
   * @param module The Emscripten module instance.
   * @param files Array of Blob or File objects to mount.
   * @param mountDir The path in the virtual filesystem where files will be mounted.
   * @returns The virtual path to the first file in the set.
   */
  private async mountModelFiles(
    module: EngineModule,
    files: MountableModelFile[],
    mountDir = '/workerfs_model'
  ): Promise<string> {
    const fs = module.FS;

    // Ensure mount directory exists
    if (!fs.analyzePath(mountDir).exists) {
      fs.mkdir(mountDir);
    } else if (this.workerFsMountPath) {
      // If we already have something mounted, unmount first
      try {
        fs.unmount(this.workerFsMountPath);
      } catch (e) { }
    }

    if (!module.WORKERFS) {
      throw new Error(
        'WORKERFS is not available in the Emscripten module. ' +
        'Ensure the module was linked with -lworkerfs.js and WORKERFS is exported.'
      );
    }

    fs.mount(module.WORKERFS, { files }, mountDir);
    this.workerFsMountPath = mountDir;

    // Path in WORKERFS is /mountDir/fileName
    const firstFileName = files[0].name || 'model.gguf';
    const firstModelPath = `${mountDir}/${firstFileName}`;

    this.commitLoadedModelPaths(module, files.map((file) => `${mountDir}/${file.name || 'model.gguf'}`));

    return firstModelPath;
  }

  /**
   * Load a model from a local File object.
   *
   * This uses WORKERFS for zero-copy, low-RAM loading, bypassing any 2GB MEMFS size limits.
   */
  public async loadModelFromFile(
    file: File,
    destFileName?: string,
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    const module = await this.ensureModule();

    if (onProgress) {
      onProgress(100); // WORKERFS mount is instant
    }

    const modelPath = await this.mountModelFiles(module, [file]);

    this.setLastModelLoadInfo({
      sourceKind: 'file',
      reuseMode: 'file-read',
      modelPath,
      fileName: normalizeModelFileName(destFileName || file.name),
      byteLength: file.size,
      persistentCacheEnabled: false,
      persistentCacheKey: null,
      persistentCacheHit: false,
      persistentCacheStored: false,
    });

    return modelPath;
  }

  /**
   * Load a GGUF model from a local buffer into MEMFS.
   */
  public loadModelFromBuffer(buffer: Uint8Array, destFileName: string = 'model.gguf'): string {
    const module = this.getLoadedModule();
    const maxModelBytes = this.resolveMaxModelBytes();
    if (buffer.byteLength === 0) {
      throw new Error('Model buffer is empty.');
    }
    if (buffer.byteLength > maxModelBytes) {
      throw new Error(`Model exceeds configured maxModelBytes (${maxModelBytes} bytes).`);
    }

    const modelPath = this.prepareModelPath(module, destFileName);
    module.FS.writeFile(modelPath, buffer);
    this.commitLoadedModelPaths(module, [modelPath]);
    this.setLastModelLoadInfo({
      sourceKind: 'buffer',
      reuseMode: 'buffer',
      modelPath,
      fileName: normalizeModelFileName(destFileName),
      byteLength: buffer.byteLength,
      persistentCacheEnabled: false,
      persistentCacheKey: null,
      persistentCacheHit: false,
      persistentCacheStored: false,
    });
    return modelPath;
  }

  /**
   * Load a split GGUF model from an array of File objects.
   *
   * Use `gguf-split` to split a large model into shards (<2GB each) so that
   * each shard fits within the MEMFS file-size limit. llama.cpp natively
   * detects the split naming convention (e.g. `model-00001-of-00010.gguf`)
   * and loads all shards automatically when given the first shard's path.
   *
   * @param files Array of File objects, one per shard, in order.
   * @param onProgress Optional progress callback (0–100 across all shards).
   * @param signal Optional AbortSignal.
   * @returns The MEMFS path to the first shard (pass this to initEngine).
   */
  public async loadModelFromFileShards(
    files: File[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    if (!files || files.length === 0) {
      throw new Error('No shard files provided.');
    }
    if (files.length === 1) {
      return this.loadModelFromFile(files[0], files[0].name, onProgress, signal);
    }

    const module = await this.ensureModule();
    const maxModelBytes = this.resolveMaxModelBytes();
    this.ensureModelsDir(module);

    const totalBytes = files.reduce((sum, f) => sum + f.size, 0);
    if (totalBytes <= 0) {
      throw new Error('Model shards are empty.');
    }
    if (totalBytes > maxModelBytes) {
      throw new Error(`Total model size (${totalBytes} bytes) exceeds configured maxModelBytes (${maxModelBytes} bytes).`);
    }

    try {
      // Mount all shards at once as a single WORKERFS directory
      const modelPath = await this.mountModelFiles(module, files);

      this.commitLoadedModelPaths(module, files.map(f => `/workerfs_model/${f.name}`));
      this.setLastModelLoadInfo({
        sourceKind: 'file',
        reuseMode: 'file-read',
        modelPath,
        fileName: normalizeModelFileName(files[0].name),
        byteLength: totalBytes,
        persistentCacheEnabled: false,
        persistentCacheKey: null,
        persistentCacheHit: false,
        persistentCacheStored: false,
      });

      if (onProgress) onProgress(100);
      return modelPath;
    } catch (error) {
      if (isAbortError(error) || signal?.aborted) {
        throw createAbortError('Model load aborted.');
      }
      throw new Error(`Failed while loading model shards: ${asErrorMessage(error)}`);
    }
  }

  /**
   * Load a split GGUF model from an array of URLs.
   */
  public async loadModelFromUrls(
    urls: string[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    if (!urls || urls.length === 0) {
      throw new Error('No shard URLs provided.');
    }

    const module = await this.ensureModule();
    const opfsSupported = FileSystemStorage.isSupported() && this.config.persistentModelCache?.enabled !== false;

    const shardBlobs: MountableModelFile[] = [];
    let bytesLoadedSoFar = 0;

    // Step 1: Resolve metadata for all shards
    const shardMeta: { url: string; fileName: string; contentLength: number; cacheKey: string }[] = [];
    for (const url of urls) {
      const parsed = this.parseConfiguredUrl(url, 'modelUrl');
      const fileName = normalizeModelFileName(parsed.pathname.split('/').pop() || 'model.gguf');
      try {
        const headResp = await fetch(url, { method: 'HEAD', signal });
        const cl = Number.parseInt(headResp.headers.get('Content-Length') ?? '0', 10) || 0;
        shardMeta.push({
          url,
          fileName,
          contentLength: cl,
          cacheKey: this.buildPersistentCacheKey(url, fileName, headResp.headers),
        });
      } catch {
        shardMeta.push({
          url,
          fileName,
          contentLength: 0,
          cacheKey: this.buildPersistentCacheKey(url, fileName),
        });
      }
    }

    const totalBytes = shardMeta.reduce((sum, s) => sum + s.contentLength, 0);

    // Step 2: Fetch or Retrieve each shard
    try {
      for (const shard of shardMeta) {
        if (signal?.aborted) throw createAbortError();

        // Check OPFS cache
        const cachedFile = opfsSupported ? await this.opfs.getFile(shard.cacheKey) : null;
        if (cachedFile && (shard.contentLength === 0 || cachedFile.size === shard.contentLength)) {
          shardBlobs.push(this.createMountableModelFile(cachedFile, shard.fileName));
          bytesLoadedSoFar += cachedFile.size;
          if (onProgress && totalBytes > 0) {
            onProgress(Math.round((bytesLoadedSoFar / totalBytes) * 100));
          }
          continue;
        }

        // Cache miss: Download
        const response = await fetch(shard.url, { signal });
        if (!response.ok) throw new Error(`HTTP ${response.status} for ${shard.fileName}`);
        if (!response.body) throw new Error(`Empty body for ${shard.fileName}`);

        const shardStart = bytesLoadedSoFar;
        let finalShardBlob: MountableModelFile;

        if (opfsSupported) {
          finalShardBlob = this.createMountableModelFile(await this.opfs.streamToDisk(
            shard.cacheKey,
            response.body,
            (written) => {
              if (onProgress && totalBytes > 0) {
                onProgress(Math.round(((shardStart + written) / totalBytes) * 100));
              }
            },
            signal
          ), shard.fileName);
        } else {
          const buffer = await response.arrayBuffer();
          finalShardBlob = this.createMountableModelFile(new Blob([buffer]), shard.fileName);
          bytesLoadedSoFar += finalShardBlob.size;
          if (onProgress && totalBytes > 0) {
            onProgress(Math.round((bytesLoadedSoFar / totalBytes) * 100));
          }
        }

        shardBlobs.push(finalShardBlob);
      }

      const modelPath = await this.mountModelFiles(module, shardBlobs);

      this.setLastModelLoadInfo({
        sourceKind: 'url',
        reuseMode: 'network',
        modelPath,
        fileName: shardMeta[0].fileName,
        byteLength: shardBlobs.reduce((sum, b) => sum + b.size, 0),
        persistentCacheEnabled: opfsSupported,
        persistentCacheKey: shardMeta.map((shard) => shard.cacheKey).join(','),
        persistentCacheHit: false,
        persistentCacheStored: opfsSupported,
      });

      return modelPath;
    } catch (e) {
      if (isAbortError(e)) throw createAbortError();
      throw new Error(`Model load from URLs failed: ${asErrorMessage(e)}`);
    }
  }

  /**
   * Initialize engine state with a model path in MEMFS.
   */
  public async initEngine(
    modelPath: string,
    config?: InferenceInitConfig
  ): Promise<void> {
    const module = await this.ensureModule();
    if (!modelPath || modelPath.trim().length === 0) {
      throw new Error('modelPath must not be empty.');
    }
    if (this.engineInitialized) {
      this.releaseAllQueuedPromptCallbacks(module);
      module.ccall('CE_Close', null, [], []);
      this.engineInitialized = false;
      this.resetRuntimeLifecycleState();
    }

    const normalizedConfig = normalizeInitConfig(config);
    this.runtimeObservabilityEnabled =
      normalizedConfig.enableRuntimeObservability > 0;
    this.backendProfilingEnabled = normalizedConfig.enableBackendProfiling > 0;
    this.transportObservability.enabled = this.runtimeObservabilityEnabled;
    const result = await module.ccall(
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
    if (result !== 0) {
      this.engineInitialized = false;
      this.resetRuntimeLifecycleState();
      throw new Error(`Failed to initialize engine. Code: ${result}`);
    }
    this.engineInitialized = true;

    this.removeAllLoadedModelFiles(module);
  }

  /**
   * Shutdown engine instance.
   */
  public close(): void {
    const module = this.module;
    if (!module) {
      return;
    }
    module.ccall('CE_Close', null, [], []);

    if (this.workerFsMountPath) {
      try {
        module.FS.unmount(this.workerFsMountPath);
      } catch (e) {
        // Ignore
      }
      this.workerFsMountPath = null;
    }

    this.releaseAllQueuedPromptCallbacks(module);
    this.engineInitialized = false;
    this.loadedModelPaths = [];
    this.workerFsMountPath = null;
    this.resetRuntimeLifecycleState();
    this.lastModelLoadInfo = null;
    this.module = null;
    this.initPromise = null;
  }

  public async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
    const module = this.getReadyEngineModule();
    if (!Number.isInteger(requestId) || requestId <= 0) {
      return false;
    }

    const result = module.ccall(
      'CE_CancelQueuedRequest',
      'number',
      ['number'],
      [requestId]
    );
    const cancelled = result instanceof Promise ? Boolean(await result) : Boolean(result);
    if (cancelled && !this.activeQueuedRequestRuns.has(requestId)) {
      this.releaseCancelledQueuedRequestState(module, requestId);
    }
    return cancelled;
  }

  public async queuePrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<GenerateRequestId> {
    const module = this.getReadyEngineModule();
    const request = this.buildGenerateRequest(contextKey, promptText, options);
    const onToken = typeof options === 'object' ? options.onToken : undefined;
    const signal = typeof options === 'object' ? options.signal : undefined;

    if (signal?.aborted) {
      throw createAbortError('Prompt was aborted before it was enqueued.');
    }

    let requestId: GenerateRequestId = 0;
    const callbackPtr =
      onToken == null && signal == null
        ? 0
        : module.addFunction((rawPtr: bigint, length: number) => {
          if (signal?.aborted) {
            return 1;
          }
          if (onToken != null && requestId !== 0) {
            this.bufferQueuedTokenPiece(requestId, module.UTF8ToString(Number(rawPtr), length));
          }
          return signal?.aborted ? 1 : 0;
        }, 'ipi');

    const requestIdResult = module.ccall(
      'CE_EnqueuePrompt',
      'number',
      ['string', 'string', 'number', 'pointer'],
      [request.contextKey, request.promptText, request.maxOutputTokens, Number(callbackPtr)]
    );
    if (requestIdResult instanceof Promise) {
      if (callbackPtr !== 0) {
        module.removeFunction(callbackPtr);
      }
      throw new Error('Unexpected async result while enqueuing a request.');
    }

    requestId = requestIdResult as GenerateRequestId;
    if (!requestId) {
      if (callbackPtr !== 0) {
        module.removeFunction(callbackPtr);
      }
      throw new Error('Failed to enqueue request.');
    }

    if (callbackPtr !== 0) {
      this.queuedPromptCallbacks.set(requestId, onToken);
      this.queuedPromptTokenBuffers.set(requestId, []);
      this.queuedPromptCallbackPtrs.set(requestId, callbackPtr);
    }

    return requestId;
  }

  public async runQueuedRequest(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    const module = this.getReadyEngineModule();
    if (!Number.isInteger(requestId) || requestId <= 0) {
      throw new Error('requestId must be a positive integer.');
    }
    if (options?.signal?.aborted) {
      await this.cancelQueuedRequest(requestId);
      throw createAbortError('Prompt was aborted before execution started.');
    }

    this.activeQueuedRequestRuns.add(requestId);
    let abortRequested = false;
    const abortListener =
      options?.signal == null
        ? null
        : () => {
          abortRequested = true;
          void this.cancelQueuedRequest(requestId);
        };
    if (abortListener != null) {
      options?.signal?.addEventListener('abort', abortListener, { once: true });
    }

    try {
      const resetStatus = this.callNumber(module, 'CE_ResetRuntimeObservability');
      if (resetStatus !== 0) {
        throw new Error('Failed to reset runtime observability before queued execution.');
      }

      let callbackFailure: unknown = null;
      while (true) {
        const stepResult = await this.callNumberAsync(
          module,
          'CE_RunRequestStep',
          ['number'],
          [requestId]
        );
        this.flushQueuedTokenPieces(requestId);

        const callbackError = this.queuedPromptCallbackErrors.get(requestId);
        if (callbackError != null && callbackFailure == null) {
          callbackFailure = callbackError;
          await this.cancelQueuedRequest(requestId);
        }

        if (stepResult === REQUEST_STEP_RESULT_TERMINAL) {
          break;
        }
        if (stepResult === REQUEST_STEP_RESULT_INVALID) {
          throw new Error('Queued request became invalid during execution.');
        }
        if (stepResult === REQUEST_STEP_RESULT_FATAL_NO_PROGRESS) {
          throw new Error('Queued request execution failed to make progress.');
        }
        if (stepResult === REQUEST_STEP_RESULT_WAITING) {
          await this.waitForNextSchedulerStep();
          continue;
        }
        if (stepResult !== REQUEST_STEP_RESULT_PROGRESSED) {
          throw new Error(`Queued request returned unknown step result ${stepResult}.`);
        }
      }

      this.flushQueuedTokenPieces(requestId);
      const response = this.takeCompletedResponse(module, requestId);
      const finalCallbackError = this.queuedPromptCallbackErrors.get(requestId);
      if (finalCallbackError != null) {
        throw finalCallbackError;
      }
      if (callbackFailure != null) {
        throw callbackFailure;
      }
      if (response.cancelled || abortRequested || options?.signal?.aborted) {
        throw createAbortError(response.errorMessage ?? 'Queued request cancelled.');
      }

      return response;
    } finally {
      if (abortListener != null) {
        options?.signal?.removeEventListener('abort', abortListener);
      }
      this.releaseQueuedPromptCallback(module, requestId);
      this.activeQueuedRequestRuns.delete(requestId);
    }
  }

  /**
   * Submit a generation prompt.
   */
  public async submitPrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<string> {
    const request = this.buildGenerateRequest(contextKey, promptText, options);
    const onToken = typeof options === 'object' ? options.onToken : undefined;
    const signal = typeof options === 'object' ? options.signal : undefined;
    const requestId = await this.queuePrompt(request.contextKey, request.promptText, {
      nTokens: request.maxOutputTokens,
      promptFormat: request.promptFormat,
      onToken,
      signal,
    });
    const response = await this.runQueuedRequest(requestId, { signal });
    if (response.cancelled) {
      throw createAbortError(response.errorMessage ?? 'Queued prompt cancelled.');
    }
    if (response.failed) {
      throw new Error(response.errorMessage ?? 'Queued prompt failed.');
    }
    return response.outputText;
  }

  public getRuntimeObservability(): RuntimeObservabilityMetrics | null {
    if (!this.runtimeObservabilityEnabled) {
      return null;
    }

    const module = this.getReadyEngineModule();
    return this.readRuntimeObservabilityFromModule(module);
  }

  public async getBackendObservability(): Promise<BackendObservability | null> {
    const module = this.getLoadedModule();
    const rawPtr = await module.ccall('CE_GetBackendObservabilityJson', 'pointer', [], [], {
      async: true,
    });

    // ccall with 'pointer' return type already converts BigInt → Number.
    const ptr = rawPtr as number;

    if (!ptr) {
      return null;
    }

    try {
      const raw = module.UTF8ToString(ptr);
      const parsed = JSON.parse(raw) as BackendObservability;
      parsed.profilingEnabled = this.backendProfilingEnabled;
      return parsed;
    } catch (error) {
      throw new Error(`Failed to parse backend observability: ${asErrorMessage(error)}`);
    } finally {
      module.ccall('CE_FreeString', null, ['pointer'], [ptr]);
    }
  }
}
