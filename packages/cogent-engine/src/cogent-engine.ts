import { normalizeInitConfig } from './config.js';
import { formatPromptText } from './prompt-format.js';
import {
  BackendInfo,
  GenerateRequest,
  GenerateResponse,
  InferenceInitConfig,
  PromptGenerationOptions,
  PromptPerformanceStats,
  PromptStreamOptions,
} from './types.js';

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
  addFunction(func: (...args: number[]) => void, signature: string): number;
  ccall(ident: string, returnType: string | null, argTypes: string[], args: unknown[], opts?: { async?: boolean }): Promise<number> | number;
  removeFunction(ptr: number): void;
  UTF8ToString(ptr: number, maxBytesToRead?: number): string;
}

interface EngineModuleOptions {
  locateFile?: (path: string, prefix?: string) => string;
  [key: string]: unknown;
}

interface ActivePromptStream {
  chunks: string[];
  onToken?: (token: string) => void;
  callbackError: unknown;
}

export interface CogentConfig {
  moduleUrl?: string;
  wasmUrl?: string;
  moduleOptions?: EngineModuleOptions;
  maxModelBytes?: number;
  trustedOrigins?: string[];
  allowUnknownContentLength?: boolean;
}

const MAX_PROMPT_TOKENS = 2048;
const DEFAULT_MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024;
const DEFAULT_PROMPT_FORMAT = 'auto-chat';

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

export class CogentEngine {
  private module: EngineModule | null = null;
  private initPromise: Promise<void> | null = null;
  private engineInitialized = false;
  private loadedModelPath: string | null = null;
  private streamCallbackPtr: number | null = null;
  private activePromptStream: ActivePromptStream | null = null;

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
    input: number | PromptGenerationOptions | undefined
  ): number {
    if (typeof input === 'number' || input === undefined) {
      return this.normalizeTokenCount(input ?? 128);
    }
    return this.normalizeTokenCount(input.nTokens ?? 128);
  }

  private resolvePromptFormat(
    input: PromptGenerationOptions | PromptStreamOptions | number | undefined
  ): 'auto-chat' | 'raw' {
    if (typeof input === 'number' || input === undefined) {
      return DEFAULT_PROMPT_FORMAT;
    }
    return input.promptFormat ?? DEFAULT_PROMPT_FORMAT;
  }

  private buildGenerateRequest(
    contextKey: string,
    promptText: string,
    options: number | PromptGenerationOptions | PromptStreamOptions
  ): GenerateRequest {
    const promptFormat = this.resolvePromptFormat(options);
    return {
      contextKey,
      promptText: formatPromptText(promptText, promptFormat),
      maxOutputTokens: this.resolvePromptTokenCount(options),
      promptFormat,
    };
  }

  private async runQueuedGenerateRequest(
    request: GenerateRequest
  ): Promise<GenerateResponse> {
    const module = this.getReadyEngineModule();
    const callbackPtr = this.ensureStreamCallback(module);
    const ptr = await module.ccall(
      'CE_RunQueuedPromptJson',
      'number',
      ['string', 'string', 'number', 'number'],
      [request.contextKey, request.promptText, request.maxOutputTokens, callbackPtr],
      { async: true }
    );

    if (!ptr) {
      throw new Error('Queued prompt returned no response payload.');
    }

    try {
      const raw = module.UTF8ToString(ptr);
      return JSON.parse(raw) as GenerateResponse;
    } catch (error) {
      throw new Error(`Failed to parse queued generate response: ${asErrorMessage(error)}`);
    } finally {
      module._CE_FreeString(ptr);
    }
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

  private releaseStreamCallback(module: EngineModule): void {
    if (this.streamCallbackPtr == null) {
      return;
    }
    module.removeFunction(this.streamCallbackPtr);
    this.streamCallbackPtr = null;
  }

  private ensureStreamCallback(module: EngineModule): number {

    if (this.streamCallbackPtr != null) {
      return this.streamCallbackPtr;
    }

    this.streamCallbackPtr = module.addFunction((ptr: number, length: number) => {
      const activeStream = this.activePromptStream;
      if (!activeStream || activeStream.callbackError != null) {
        return;
      }

      try {
        const token = module.UTF8ToString(ptr, length);
        activeStream.chunks.push(token);
        activeStream.onToken?.(token);
      } catch (error) {
        activeStream.callbackError = error;
      }
    }, 'vii');

    return this.streamCallbackPtr;
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
    const importedModule = await import(moduleUrl);
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
    signal?: AbortSignal
  ): Promise<void> {
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

    return this.loadModelFromReadableStream(response.body, destFileName, {
      expectedBytes: contentLength,
      onProgress,
      signal
    });
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
    const module = await this.ensureModule();
    const modelPath = this.prepareModelPath(module, destFileName);
    const maxModelBytes = this.resolveMaxModelBytes();
    const expectedBytes = options.expectedBytes ?? 0;

    try {
      await this.writeModelStream(
        module,
        modelPath,
        stream,
        maxModelBytes,
        expectedBytes,
        options.onProgress,
        options.signal
      );
    } catch (error) {
      this.removeFileIfExists(module, modelPath);
      throw new Error(`Failed while streaming model: ${asErrorMessage(error)}`);
    }

    this.commitLoadedModelPath(module, modelPath);
    return modelPath;
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

    return this.loadModelFromReadableStream(file.stream(), destFileName, {
      expectedBytes: file.size,
      onProgress,
      signal
    });
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
    this.releaseStreamCallback(module);
    this.activePromptStream = null;
    this.engineInitialized = false;
    this.loadedModelPath = null;
    this.module = null;
    this.initPromise = null;
  }

  /**
   * Submit a generation prompt.
   */
  public async streamPrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptStreamOptions = 128
  ): Promise<string> {
    this.getReadyEngineModule();
    if (this.activePromptStream != null) {
      throw new Error('Concurrent streamPrompt() calls are not supported yet.');
    }

    const request = this.buildGenerateRequest(contextKey, promptText, options);
    const onToken = typeof options === 'object' ? options.onToken : undefined;
    const activeStream: ActivePromptStream = {
      chunks: [],
      onToken,
      callbackError: null,
    };
    this.activePromptStream = activeStream;

    try {
      const response = await this.runQueuedGenerateRequest(request);
      if (activeStream.callbackError != null) {
        throw activeStream.callbackError;
      }
      if (response.failed) {
        throw new Error(response.errorMessage ?? 'Queued prompt failed.');
      }

      return response.outputText;
    } finally {
      if (this.activePromptStream === activeStream) {
        this.activePromptStream = null;
      }
    }
  }

  public async prompt(
    contextKey: string,
    promptText: string,
    options: number | PromptGenerationOptions = 128
  ): Promise<string> {
    return this.streamPrompt(contextKey, promptText, options);
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
