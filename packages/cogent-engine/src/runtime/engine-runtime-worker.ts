import { CogentConfig } from '../cogent-config.js';
import {
  BackendObservability,
  EngineExecutionMode,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  ModelLoadInfo,
  PromptOptions,
  RuntimeObservabilityMetrics,
  TransportObservability,
} from '../types.js';
import { EngineRuntime } from './engine-runtime.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  WorkerSerializableCogentConfig,
  WorkerLoadModelResult,
  WorkerRunQueuedRequestResult,
  WorkerBackendObservabilityResult,
} from './engine-runtime-worker-protocol.js';

interface PendingWorkerCall {
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  onProgress?: (pct: number) => void;
}

type WithoutCallId<T> = T extends { callId: number } ? Omit<T, 'callId'> : never;

function createAbortError(message = 'The operation was aborted.'): Error {
  if (typeof DOMException === 'function') {
    return new DOMException(message, 'AbortError');
  }
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

function toWorkerSerializableConfig(config: CogentConfig): WorkerSerializableCogentConfig {
  if (typeof config.moduleOptions?.locateFile === 'function') {
    throw new Error(
      'Worker mode does not support moduleOptions.locateFile. Provide explicit moduleUrl/wasmUrl instead.'
    );
  }

  return {
    moduleUrl: config.moduleUrl,
    wasmUrl: config.wasmUrl,
    moduleOptions: config.moduleOptions,
    maxModelBytes: config.maxModelBytes,
    trustedOrigins: config.trustedOrigins,
    allowUnknownContentLength: config.allowUnknownContentLength,
    workerMaxBufferedTokens: config.workerMaxBufferedTokens,
    workerTokenFlushIntervalMs: config.workerTokenFlushIntervalMs,
    persistentModelCache: config.persistentModelCache,
  };
}

export class WorkerEngineRuntime implements EngineRuntime {
  private worker: Worker | null = null;
  private workerInitialized = false;
  private nextCallId = 1;
  private readonly pendingWorkerCalls = new Map<number, PendingWorkerCall>();
  private readonly queuedTokenCallbacks = new Map<
    GenerateRequestId,
    (token: string) => void
  >();
  private readonly pendingQueuedTokenCallbacks = new Map<
    number,
    ((token: string) => void) | undefined
  >();
  private readonly queuedTokenErrors = new Map<GenerateRequestId, unknown>();
  private readonly queuedSignals = new Map<GenerateRequestId, AbortSignal>();
  private runtimeObservability: RuntimeObservabilityMetrics | null = null;
  private lastModelLoadInfo: ModelLoadInfo | null = null;
  private transportObservability: TransportObservability = {
    executionMode: 'worker',
    workerBacked: true,
    enabled: false,
    bufferedTokenLimit: 0,
    flushIntervalMs: 0,
    flushCount: 0,
    coalescedTokenCount: 0,
    maxObservedBufferedTokenCount: 0,
  };

  constructor(private readonly config: CogentConfig = {}) {}

  public getExecutionMode(): EngineExecutionMode {
    return 'worker';
  }

  public getLastModelLoadInfo(): ModelLoadInfo | null {
    return this.lastModelLoadInfo;
  }

  public getTransportObservability(): TransportObservability {
    return { ...this.transportObservability };
  }

  public async initModule(): Promise<void> {
    await this.ensureWorkerInitialized();
  }

  public async loadModelFromUrl(
    url: string,
    destFileName = 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    await this.ensureWorkerInitialized();
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    const result = (await this.callWorker<Extract<WorkerRequestMessage, { kind: 'load-model-url' }>>(
      {
        kind: 'load-model-url',
        url,
        destFileName,
      },
      onProgress
    )) as WorkerLoadModelResult;
    this.lastModelLoadInfo = result.modelLoadInfo;
    this.transportObservability = result.transportObservability;
    return result.modelPath;
  }

  public async loadModelFromFile(
    file: File,
    destFileName = file.name || 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    await this.ensureWorkerInitialized();
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    const result = (await this.callWorker<Extract<WorkerRequestMessage, { kind: 'load-model-file' }>>(
      {
        kind: 'load-model-file',
        file,
        destFileName,
      },
      onProgress
    )) as WorkerLoadModelResult;
    this.lastModelLoadInfo = result.modelLoadInfo;
    this.transportObservability = result.transportObservability;
    return result.modelPath;
  }

  public async loadModelFromReadableStream(
    stream: ReadableStream<Uint8Array>,
    destFileName = 'model.gguf',
    options: {
      expectedBytes?: number;
      onProgress?: (pct: number) => void;
      signal?: AbortSignal;
    } = {}
  ): Promise<string> {
    const chunks: Uint8Array[] = [];
    const reader = stream.getReader();
    let totalBytes = 0;
    try {
      while (true) {
        if (options.signal?.aborted) {
          throw createAbortError('Model load aborted.');
        }
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (value == null || value.byteLength === 0) {
          continue;
        }
        chunks.push(value);
        totalBytes += value.byteLength;
        if (options.expectedBytes && options.onProgress) {
          options.onProgress(Math.round((totalBytes / options.expectedBytes) * 100));
        }
      }
    } finally {
      reader.releaseLock();
    }

    const buffer = new Uint8Array(totalBytes);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.byteLength;
    }

    const modelPath = await this.callWorkerLoadBuffer(buffer, destFileName);
    return modelPath;
  }

  public loadModelFromBuffer(buffer: Uint8Array, destFileName = 'model.gguf'): string {
    throw new Error(
      'loadModelFromBuffer() is not available synchronously in worker runtime. Use loadModelFromFile() or loadModelFromUrl().'
    );
  }

  public async initEngine(
    modelPath: string,
    config?: InferenceInitConfig
  ): Promise<void> {
    await this.ensureWorkerInitialized();
    await this.callWorker<Extract<WorkerRequestMessage, { kind: 'init-engine' }>>({
      kind: 'init-engine',
      modelPath,
      config,
    });
  }

  public close(): void {
    if (this.worker == null) {
      return;
    }
    this.worker.terminate();
    this.worker = null;
    this.workerInitialized = false;
    for (const call of this.pendingWorkerCalls.values()) {
      call.reject(new Error('Worker runtime was closed.'));
    }
    this.pendingWorkerCalls.clear();
    this.queuedTokenCallbacks.clear();
    this.pendingQueuedTokenCallbacks.clear();
    this.queuedTokenErrors.clear();
    this.queuedSignals.clear();
  }

  public async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
    await this.ensureWorkerInitialized();
    return (await this.callWorker<Extract<WorkerRequestMessage, { kind: 'cancel-request' }>>({
      kind: 'cancel-request',
      requestId,
    })) as boolean;
  }

  public async queuePrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<GenerateRequestId> {
    await this.ensureWorkerInitialized();
    const onToken = typeof options === 'object' ? options.onToken : undefined;
    const signal = typeof options === 'object' ? options.signal : undefined;
    const callId = this.nextCallId++;
    this.pendingQueuedTokenCallbacks.set(callId, onToken);

    const requestId = (await this.callWorkerWithId<
      Extract<WorkerRequestMessage, { kind: 'queue-prompt' }>
    >(callId, {
      kind: 'queue-prompt',
      contextKey,
      promptText,
      options: {
        nTokens: typeof options === 'number' ? options : options.nTokens,
        promptFormat: typeof options === 'number' ? undefined : options.promptFormat,
      },
    })) as GenerateRequestId;

    const pendingCallback = this.pendingQueuedTokenCallbacks.get(callId);
    this.pendingQueuedTokenCallbacks.delete(callId);
    if (pendingCallback) {
      this.queuedTokenCallbacks.set(requestId, pendingCallback);
    }
    if (signal) {
      this.queuedSignals.set(requestId, signal);
    }
    return requestId;
  }

  public async runQueuedRequest(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    await this.ensureWorkerInitialized();
    const signal = options?.signal ?? this.queuedSignals.get(requestId);
    if (signal?.aborted) {
      await this.cancelQueuedRequest(requestId);
      throw createAbortError('Queued request cancelled.');
    }

    const abortListener =
      signal == null
        ? null
        : () => {
            void this.cancelQueuedRequest(requestId);
          };
    if (abortListener != null) {
      signal?.addEventListener('abort', abortListener, { once: true });
    }

    try {
      const result = (await this.callWorker<
        Extract<WorkerRequestMessage, { kind: 'run-queued-request' }>
      >({
        kind: 'run-queued-request',
        requestId,
      })) as WorkerRunQueuedRequestResult;
      this.runtimeObservability = result.runtimeObservability;
      this.transportObservability = result.transportObservability;

      const tokenError = this.queuedTokenErrors.get(requestId);
      if (tokenError != null) {
        throw tokenError;
      }
      if (result.response.cancelled || signal?.aborted) {
        throw createAbortError(result.response.errorMessage ?? 'Queued request cancelled.');
      }
      return result.response;
    } finally {
      if (abortListener != null) {
        signal?.removeEventListener('abort', abortListener);
      }
      this.queuedTokenCallbacks.delete(requestId);
      this.queuedTokenErrors.delete(requestId);
      this.queuedSignals.delete(requestId);
    }
  }

  public async submitPrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<string> {
    const requestId = await this.queuePrompt(contextKey, promptText, options);
    const signal = typeof options === 'object' ? options.signal : undefined;
    const response = await this.runQueuedRequest(requestId, { signal });
    if (response.failed) {
      throw new Error(response.errorMessage ?? 'Queued prompt failed.');
    }
    return response.outputText;
  }

  public getRuntimeObservability(): RuntimeObservabilityMetrics | null {
    return this.runtimeObservability;
  }

  public async getBackendObservability(): Promise<BackendObservability | null> {
    await this.ensureWorkerInitialized();
    const result = (await this.callWorker<
      Extract<WorkerRequestMessage, { kind: 'get-backend-observability' }>
    >({
      kind: 'get-backend-observability',
    })) as WorkerBackendObservabilityResult;
    this.transportObservability = result.transportObservability;
    return result.backendObservability;
  }

  private async ensureWorkerInitialized(): Promise<void> {
    if (this.worker == null) {
      const workerUrl =
        this.config.workerUrl ??
        new URL('./engine-runtime-worker-entry.js', import.meta.url).toString();
      this.worker = new Worker(workerUrl, { type: 'module' });
      this.worker.onmessage = (event: MessageEvent<WorkerResponseMessage>) => {
        this.handleWorkerMessage(event.data);
      };
    }

    if (!this.workerInitialized) {
      await this.callWorker<Extract<WorkerRequestMessage, { kind: 'init-module' }>>({
        kind: 'init-module',
        config: toWorkerSerializableConfig(this.config),
      });
      this.workerInitialized = true;
      this.transportObservability.executionMode = 'worker';
      this.transportObservability.workerBacked = true;
    }
  }

  private handleWorkerMessage(message: WorkerResponseMessage): void {
    if (message.kind === 'token') {
      const onToken = this.queuedTokenCallbacks.get(message.requestId);
      if (!onToken) {
        return;
      }
      try {
        onToken(message.text);
      } catch (error) {
        this.queuedTokenErrors.set(message.requestId, error);
        void this.cancelQueuedRequest(message.requestId);
      }
      return;
    }

    const pendingCall = this.pendingWorkerCalls.get(message.callId);
    if (!pendingCall) {
      return;
    }

    if (message.kind === 'load-progress') {
      pendingCall.onProgress?.(message.progressPct);
      return;
    }

    this.pendingWorkerCalls.delete(message.callId);
    if (message.kind === 'resolve') {
      pendingCall.resolve(message.value);
      return;
    }

    const error =
      message.errorName === 'AbortError'
        ? createAbortError(message.message)
        : Object.assign(new Error(message.message), {
            name: message.errorName ?? 'Error',
          });
    pendingCall.reject(error);
  }

  private async callWorkerLoadBuffer(
    buffer: Uint8Array,
    destFileName: string
  ): Promise<string> {
    const result = (await this.callWorker<
      Extract<WorkerRequestMessage, { kind: 'load-model-buffer' }>
    >({
      kind: 'load-model-buffer',
      buffer,
      destFileName,
    })) as WorkerLoadModelResult;
    this.lastModelLoadInfo = result.modelLoadInfo;
    this.transportObservability = result.transportObservability;
    return result.modelPath;
  }

  private async callWorker<T extends WorkerRequestMessage>(
    message: WithoutCallId<T>,
    onProgress?: (pct: number) => void
  ): Promise<unknown> {
    const callId = this.nextCallId++;
    return this.callWorkerWithId(callId, message, onProgress);
  }

  private async callWorkerWithId<T extends WorkerRequestMessage>(
    callId: number,
    message: WithoutCallId<T>,
    onProgress?: (pct: number) => void
  ): Promise<unknown> {
    if (this.worker == null) {
      throw new Error('Worker runtime is not available.');
    }

    const request = {
      ...message,
      callId,
    } as T;

    const transferables: Transferable[] = [];
    if (request.kind === 'load-model-buffer') {
      transferables.push(request.buffer.buffer);
    }

    return new Promise<unknown>((resolve, reject) => {
      this.pendingWorkerCalls.set(callId, {
        resolve,
        reject,
        onProgress,
      });
      this.worker!.postMessage(request, transferables);
    });
  }
}
