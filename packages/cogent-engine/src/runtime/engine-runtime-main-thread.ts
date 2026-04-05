import { CogentConfig, EngineModuleOptions } from '../cogent-config.js';
import { normalizeInitConfig } from '../core/init-config.js';
import {
  BrowserModelCache,
  BrowserModelCacheWriter,
  ModelCacheSourceDescriptor,
} from '../storage/browser-model-cache.js';
import { formatPromptText } from '../core/prompt-format.js';
import {
  BackendInfo,
  EngineExecutionMode,
  GenerateRequest,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  ModelLoadInfo,
  ModelLoadSourceKind,
  ModelLoadReuseMode,
  PromptPerformanceStats,
  PromptOptions,
  TransportInfo,
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
}

interface EngineModule {
  FS: EmscriptenFs;
  _CE_FreeString(ptr: number): void;
  addFunction(func: (...args: number[]) => number, signature: string): number;
  ccall(ident: string, returnType: string | null, argTypes: string[], args: unknown[], opts?: { async?: boolean }): Promise<number> | number;
  removeFunction(ptr: number): void;
  UTF8ToString(ptr: number, maxBytesToRead?: number): string;
}

const MAX_PROMPT_TOKENS = 2048;
const DEFAULT_MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024;
const DEFAULT_PROMPT_FORMAT = 'auto-chat';
const DEFAULT_PERSISTENT_CACHE_NAMESPACE = 'cogent-engine-model-cache';

interface ModelStreamSink {
  write(chunk: Uint8Array): Promise<void>;
  close(finalSizeBytes: number): Promise<void>;
  abort(): Promise<void>;
}

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

export class MainThreadEngineRuntime implements EngineRuntime {
  private module: EngineModule | null = null;
  private initPromise: Promise<void> | null = null;
  private engineInitialized = false;
  private loadedModelPath: string | null = null;
  private queuedPromptCallbackPtrs = new Map<GenerateRequestId, number>();
  private queuedPromptCallbackErrors = new Map<GenerateRequestId, unknown>();
  private lastModelLoadInfo: ModelLoadInfo | null = null;
  private readonly transportInfo: TransportInfo = {
    executionMode: 'main-thread',
    workerBacked: false,
    backpressureEnabled: false,
    maxBufferedTokenCount: 0,
    flushIntervalMs: 0,
    flushCount: 0,
    coalescedTokenCount: 0,
    maxObservedBufferedTokenCount: 0,
  };
  private readonly persistentModelCache: BrowserModelCache;

  constructor(private config: CogentConfig = {}) {
    this.persistentModelCache = new BrowserModelCache({
      enabled: config.persistentModelCache?.enabled ?? true,
      namespace:
        config.persistentModelCache?.namespace ??
        DEFAULT_PERSISTENT_CACHE_NAMESPACE,
      cacheLocalFiles: config.persistentModelCache?.cacheLocalFiles ?? false,
      maxEntryBytes:
        config.persistentModelCache?.maxEntryBytes ??
        this.resolveMaxModelBytes(),
    });
  }

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

  private buildModelCacheSource(
    kind: ModelLoadSourceKind,
    identity: string,
    fileName: string
  ): ModelCacheSourceDescriptor {
    return {
      kind,
      identity,
      fileName: normalizeModelFileName(fileName),
    };
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

  public getTransportInfo(): TransportInfo {
    return { ...this.transportInfo };
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
    if (callbackPtr == null) {
      return;
    }
    module.removeFunction(callbackPtr);
    this.queuedPromptCallbackPtrs.delete(requestId);
    this.queuedPromptCallbackErrors.delete(requestId);
  }

  private releaseAllQueuedPromptCallbacks(module: EngineModule): void {
    for (const callbackPtr of this.queuedPromptCallbackPtrs.values()) {
      module.removeFunction(callbackPtr);
    }
    this.queuedPromptCallbackPtrs.clear();
    this.queuedPromptCallbackErrors.clear();
  }

  private removeFileIfExists(module: EngineModule, path: string): void {
    if (module.FS.analyzePath(path).exists) {
      module.FS.unlink(path);
    }
  }

  private commitLoadedModelPath(module: EngineModule, path: string): void {
    if (this.loadedModelPath && this.loadedModelPath !== path) {
      this.removeFileIfExists(module, this.loadedModelPath);
    }
    this.loadedModelPath = path;
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

  private async writeModelStream(
    module: EngineModule,
    path: string,
    stream: ReadableStream<Uint8Array>,
    maxModelBytes: number,
    expectedBytes: number,
    onProgress?: (pct: number) => void,
    signal?: AbortSignal,
    sink?: ModelStreamSink
  ): Promise<number> {
    if (expectedBytes > 0 && expectedBytes > maxModelBytes) {
      throw new Error(`Model exceeds configured maxModelBytes (${maxModelBytes} bytes).`);
    }

    const fileStream = module.FS.open(path, 'w+');
    if (!Number.isFinite(fileStream.position)) {
      fileStream.position = 0;
    }

    let receivedLength = 0;
    const reader = stream.getReader();

    try {
      while (true) {
        if (signal?.aborted) {
          throw new Error('Model load aborted.');
        }

        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (!value || value.length === 0) {
          continue;
        }

        receivedLength += value.length;
        if (receivedLength > maxModelBytes) {
          throw new Error(`Model exceeds configured maxModelBytes (${maxModelBytes} bytes).`);
        }

        module.FS.write(fileStream, value, 0, value.length, fileStream.position);
        fileStream.position += value.length;
        if (sink) {
          await sink.write(value);
        }

        if (expectedBytes > 0 && onProgress) {
          onProgress(Math.round((receivedLength / expectedBytes) * 100));
        }
      }
    } finally {
      module.FS.close(fileStream);
      reader.releaseLock();
    }

    if (receivedLength === 0) {
      throw new Error('Model file is empty.');
    }

    if (sink) {
      await sink.close(receivedLength);
    }

    return receivedLength;
  }

  private async loadModelFromReadableStreamInternal(
    stream: ReadableStream<Uint8Array>,
    destFileName: string,
    source: ModelCacheSourceDescriptor,
    reuseMode: ModelLoadReuseMode,
    options: {
      expectedBytes?: number;
      onProgress?: (pct: number) => void;
      signal?: AbortSignal;
      persistentCacheHit?: boolean;
      persistentCacheKey?: string | null;
      persistentCacheWriter?: BrowserModelCacheWriter | null;
    } = {}
  ): Promise<string> {
    const module = await this.ensureModule();
    const modelPath = this.prepareModelPath(module, destFileName);
    const maxModelBytes = this.resolveMaxModelBytes();
    const expectedBytes = options.expectedBytes ?? 0;
    const persistentCacheEnabled =
      options.persistentCacheHit === true ||
      options.persistentCacheWriter != null ||
      this.persistentModelCache.isEnabledForSource(source);

    try {
      const finalBytes = await this.writeModelStream(
        module,
        modelPath,
        stream,
        maxModelBytes,
        expectedBytes,
        options.onProgress,
        options.signal,
        options.persistentCacheWriter ?? undefined
      );

      this.commitLoadedModelPath(module, modelPath);
      this.setLastModelLoadInfo({
        sourceKind: source.kind,
        reuseMode,
        modelPath,
        fileName: source.fileName,
        byteLength: finalBytes,
        persistentCacheEnabled,
        persistentCacheKey: options.persistentCacheKey ?? null,
        persistentCacheHit: options.persistentCacheHit === true,
        persistentCacheStored: options.persistentCacheWriter != null,
      });
      return modelPath;
    } catch (error) {
      this.removeFileIfExists(module, modelPath);
      if (options.persistentCacheWriter != null) {
        await options.persistentCacheWriter.abort();
      }
      if (isAbortError(error) || options.signal?.aborted) {
        throw createAbortError('Model load aborted.');
      }
      throw new Error(`Failed while streaming model: ${asErrorMessage(error)}`);
    }
  }

  /**
   * Load a GGUF model into MEMFS from a URL.
   */
  public async loadModelFromUrl(
    url: string,
    destFileName: string = 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    const source = this.buildModelCacheSource(
      'url',
      this.parseConfiguredUrl(url, 'modelUrl').toString(),
      destFileName
    );
    const restored = await this.persistentModelCache.restore(source);
    if (restored != null) {
      return this.loadModelFromReadableStreamInternal(
        restored.stream,
        destFileName,
        source,
        'persistent-cache',
        {
          expectedBytes: restored.byteLength,
          onProgress,
          signal,
          persistentCacheHit: true,
          persistentCacheKey: restored.persistentCacheKey,
        }
      );
    }

    const maxModelBytes = this.resolveMaxModelBytes();
    const response = await fetch(url, { signal });
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }
    if (!response.body) {
      throw new Error('Model response body is empty.');
    }

    const contentLength = Number.parseInt(response.headers.get('Content-Length') ?? '0', 10) || 0;
    if (contentLength <= 0 && !this.config.allowUnknownContentLength) {
      throw new Error('Model response must include a valid Content-Length header.');
    }
    if (contentLength > maxModelBytes) {
      throw new Error(`Model exceeds configured maxModelBytes (${maxModelBytes} bytes).`);
    }

    const persistentCacheWriter =
      contentLength <= maxModelBytes
        ? await this.persistentModelCache.createWriter(source)
        : null;

    return this.loadModelFromReadableStreamInternal(
      response.body,
      destFileName,
      source,
      'network',
      {
        expectedBytes: contentLength,
        onProgress,
        signal,
        persistentCacheKey: persistentCacheWriter?.persistentCacheKey ?? null,
        persistentCacheWriter,
      }
    );
  }

  public async loadModelFromReadableStream(
    stream: ReadableStream<Uint8Array>,
    destFileName: string = 'model.gguf',
    options: {
      expectedBytes?: number;
      onProgress?: (pct: number) => void;
      signal?: AbortSignal;
    } = {}
  ): Promise<string> {
    const source = this.buildModelCacheSource(
      'buffer',
      `stream:${destFileName}:${options.expectedBytes ?? 0}`,
      destFileName
    );

    return this.loadModelFromReadableStreamInternal(
      stream,
      destFileName,
      source,
      'buffer',
      options
    );
  }

  public async loadModelFromFile(
    file: File,
    destFileName: string = file.name || 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    if (file.size <= 0) {
      throw new Error('Model file is empty.');
    }

    const source = this.buildModelCacheSource(
      'file',
      `${file.name}:${file.size}:${file.lastModified}`,
      destFileName
    );
    const restored = await this.persistentModelCache.restore(source);
    if (restored != null) {
      return this.loadModelFromReadableStreamInternal(
        restored.stream,
        destFileName,
        source,
        'persistent-cache',
        {
          expectedBytes: restored.byteLength,
          onProgress,
          signal,
          persistentCacheHit: true,
          persistentCacheKey: restored.persistentCacheKey,
        }
      );
    }

    const persistentCacheWriter = await this.persistentModelCache.createWriter(source);
    return this.loadModelFromReadableStreamInternal(
      file.stream(),
      destFileName,
      source,
      'file-read',
      {
        expectedBytes: file.size,
        onProgress,
        signal,
        persistentCacheKey: persistentCacheWriter?.persistentCacheKey ?? null,
        persistentCacheWriter,
      }
    );
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
    this.commitLoadedModelPath(module, modelPath);
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
      module.ccall('CE_Close', null, [], []);
      this.engineInitialized = false;
    }

    const normalizedConfig = normalizeInitConfig(config);
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
      ],
      { async: true }
    );
    if (result !== 0) {
      this.engineInitialized = false;
      throw new Error(`Failed to initialize engine. Code: ${result}`);
    }
    this.engineInitialized = true;
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
    this.releaseAllQueuedPromptCallbacks(module);
    this.engineInitialized = false;
    this.loadedModelPath = null;
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
    if (result instanceof Promise) {
      return Boolean(await result);
    }
    return Boolean(result);
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
        : module.addFunction((ptr: number, length: number) => {
            if (signal?.aborted) {
              return 1;
            }
            try {
              onToken?.(module.UTF8ToString(ptr, length));
            } catch (error) {
              if (requestId !== 0) {
                this.queuedPromptCallbackErrors.set(requestId, error);
              }
              return 1;
            }
            return signal?.aborted ? 1 : 0;
          }, 'iii');

    const requestIdResult = module.ccall(
      'CE_EnqueuePrompt',
      'number',
      ['string', 'string', 'number', 'number'],
      [request.contextKey, request.promptText, request.maxOutputTokens, callbackPtr]
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

    const abortListener =
      options?.signal == null
        ? null
        : () => {
            void this.cancelQueuedRequest(requestId);
          };
    if (abortListener != null) {
      options?.signal?.addEventListener('abort', abortListener, { once: true });
    }
    let ptr = 0;

    try {
      ptr = await module.ccall(
        'CE_RunQueuedRequestJson',
        'number',
        ['number'],
        [requestId],
        { async: true }
      );

      if (!ptr) {
        this.releaseQueuedPromptCallback(module, requestId);
        throw new Error('Queued request returned no response payload.');
      }

      const raw = module.UTF8ToString(ptr);
      const response = JSON.parse(raw) as GenerateResponse;
      const callbackError = this.queuedPromptCallbackErrors.get(requestId);
      if (callbackError != null) {
        throw callbackError;
      }
      if (response.cancelled || options?.signal?.aborted) {
        throw createAbortError(response.errorMessage ?? 'Queued request cancelled.');
      }
      return response;
    } catch (error) {
      if (error instanceof SyntaxError) {
        throw new Error(`Failed to parse queued generate response: ${asErrorMessage(error)}`);
      }
      throw error;
    } finally {
      if (abortListener != null) {
        options?.signal?.removeEventListener('abort', abortListener);
      }
      if (ptr) {
        module._CE_FreeString(ptr);
      }
      this.releaseQueuedPromptCallback(module, requestId);
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

  public getLastPromptPerformance(): PromptPerformanceStats | null {
    const module = this.getReadyEngineModule();
    const ptrResult = module.ccall('CE_GetLastPromptPerfJson', 'number', [], []);
    if (ptrResult instanceof Promise) {
      throw new Error('Unexpected async result while reading prompt performance stats.');
    }
    const ptr = ptrResult;

    if (!ptr) {
      return null;
    }

    try {
      const raw = module.UTF8ToString(ptr);
      return JSON.parse(raw) as PromptPerformanceStats;
    } catch (error) {
      throw new Error(`Failed to parse prompt performance stats: ${asErrorMessage(error)}`);
    } finally {
      module._CE_FreeString(ptr);
    }
  }

  public async getBackendInfo(): Promise<BackendInfo | null> {
    const module = this.getLoadedModule();
    // WebGPU backend enumeration is not a pure metadata read on the web path.
    // ggml-webgpu may need to touch emdawnwebgpu adapter/device creation while
    // building its backend/device list, which can suspend under JSPI.
    const ptr = await module.ccall('CE_GetBackendInfoJson', 'number', [], [], {
      async: true,
    });

    if (!ptr) {
      return null;
    }

    try {
      const raw = module.UTF8ToString(ptr);
      return JSON.parse(raw) as BackendInfo;
    } catch (error) {
      throw new Error(`Failed to parse backend info: ${asErrorMessage(error)}`);
    } finally {
      module._CE_FreeString(ptr);
    }
  }
}
