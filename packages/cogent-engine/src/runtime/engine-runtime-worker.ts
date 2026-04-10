import { CogentConfig } from '../cogent-config.js';
import {
  BackendObservability,
  EngineExecutionMode,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  ModelLoadInfo,
  PromptOptions,
  RuntimeAggregateObservabilityMetrics,
  TransportObservability,
} from '../types.js';
import { EngineRuntime } from './engine-runtime.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  WorkerLoadModelResult,
  WorkerRunQueuedRequestResult,
  WorkerBackendObservabilityResult,
} from './engine-runtime-worker-protocol.js';
import { createAbortError } from './runtime-shared.js';
import {
  createDefaultTransportObservability,
  PendingWorkerCall,
  QueuedRequestCompletionState,
  toTransferableChunkBuffer,
  toWorkerSerializableConfig,
  WithoutCallId,
} from './worker-runtime-shared.js';

export class WorkerEngineRuntime implements EngineRuntime {
  private worker: Worker | null = null;
  private workerInitialized = false;
  private nextCallId = 1;
  private readonly pendingWorkerCalls = new Map<number, PendingWorkerCall>();
  private readonly pendingStreamChunkAcks = new Map<
    number,
    { resolve: () => void; reject: (error: unknown) => void }
  >();
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
  private readonly queuedSignalAbortListeners = new Map<GenerateRequestId, () => void>();
  private readonly activeQueuedRequestRuns = new Set<GenerateRequestId>();
  private readonly queuedRequestCompletions = new Map<
    GenerateRequestId,
    QueuedRequestCompletionState
  >();
  private runtimeAggregateObservability: RuntimeAggregateObservabilityMetrics | null = null;
  private lastModelLoadInfo: ModelLoadInfo | null = null;
  private transportObservability: TransportObservability = createDefaultTransportObservability();

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

    const callId = this.nextCallId++;
    const result = (await this.callWorkerWithAbort<
      Extract<WorkerRequestMessage, { kind: 'load-model-url' }>
    >(
      callId,
      {
        kind: 'load-model-url',
        url,
        destFileName,
      },
      signal,
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

    const callId = this.nextCallId++;
    const result = (await this.callWorkerWithAbort<
      Extract<WorkerRequestMessage, { kind: 'load-model-file' }>
    >(
      callId,
      {
        kind: 'load-model-file',
        file,
        destFileName,
      },
      signal,
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
    await this.ensureWorkerInitialized();
    if (options.signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    const callId = this.nextCallId++;
    const reader = stream.getReader();
    const loadPromise = this.callWorkerWithAbort<
      Extract<WorkerRequestMessage, { kind: 'load-model-stream-start' }>
    >(
      callId,
      {
        kind: 'load-model-stream-start',
        destFileName,
        expectedBytes: options.expectedBytes,
      },
      options.signal,
      options.onProgress
    ) as Promise<WorkerLoadModelResult>;

    try {
      while (true) {
        if (options.signal?.aborted) {
          throw createAbortError('Model load aborted.');
        }
        const { done, value } = await Promise.race([
          reader.read(),
          loadPromise.then(() => ({ done: true, value: undefined as Uint8Array | undefined })),
        ]);
        if (done) {
          break;
        }
        if (value == null || value.byteLength === 0) {
          continue;
        }
        await this.sendStreamChunk(callId, value);
      }

      this.postWorkerMessage({
        kind: 'load-model-stream-end',
        callId,
      });

      const result = await loadPromise;
      this.lastModelLoadInfo = result.modelLoadInfo;
      this.transportObservability = result.transportObservability;
      return result.modelPath;
    } catch (error) {
      this.sendModelLoadCancel(callId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw error;
      }
      if (options.signal?.aborted) {
        throw createAbortError('Model load aborted.');
      }
      throw error;
    } finally {
      reader.releaseLock();
    }
  }

  public loadModelFromBuffer(buffer: Uint8Array, destFileName = 'model.gguf'): string {
    throw new Error(
      'loadModelFromBuffer() is not available synchronously in worker runtime. Use loadModelFromFile() or loadModelFromUrl().'
    );
  }

  public async loadModelFromFileShards(
    files: File[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    await this.ensureWorkerInitialized();
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    const callId = this.nextCallId++;
    const result = (await this.callWorkerWithAbort<
      Extract<WorkerRequestMessage, { kind: 'load-model-file-shards' }>
    >(
      callId,
      {
        kind: 'load-model-file-shards',
        files,
      },
      signal,
      onProgress
    )) as WorkerLoadModelResult;
    this.lastModelLoadInfo = result.modelLoadInfo;
    this.transportObservability = result.transportObservability;
    return result.modelPath;
  }

  public async loadModelFromUrls(
    urls: string[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    await this.ensureWorkerInitialized();
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    const callId = this.nextCallId++;
    const result = (await this.callWorkerWithAbort<
      Extract<WorkerRequestMessage, { kind: 'load-model-urls' }>
    >(
      callId,
      {
        kind: 'load-model-urls',
        urls,
      },
      signal,
      onProgress
    )) as WorkerLoadModelResult;
    this.lastModelLoadInfo = result.modelLoadInfo;
    this.transportObservability = result.transportObservability;
    return result.modelPath;
  }

  public async initEngine(
    modelPath: string,
    config?: InferenceInitConfig
  ): Promise<void> {
    await this.ensureWorkerInitialized();
    this.resetQueuedRequestLifecycleState();
    await this.callWorker<Extract<WorkerRequestMessage, { kind: 'init-engine' }>>({
      kind: 'init-engine',
      modelPath,
      config,
    });
  }

  public close(): void {
    if (this.worker != null) {
      this.worker.terminate();
    }
    this.resetWorkerState(new Error('Worker runtime was closed.'));
  }

  private detachQueuedRequestSignal(requestId: GenerateRequestId): void {
    const signal = this.queuedSignals.get(requestId);
    const abortListener = this.queuedSignalAbortListeners.get(requestId);
    if (signal != null && abortListener != null) {
      signal.removeEventListener('abort', abortListener);
    }
    this.queuedSignals.delete(requestId);
    this.queuedSignalAbortListeners.delete(requestId);
  }

  private releaseQueuedRequestExecutionState(requestId: GenerateRequestId): void {
    this.queuedTokenCallbacks.delete(requestId);
    this.queuedTokenErrors.delete(requestId);
    this.detachQueuedRequestSignal(requestId);
  }

  private cleanupConsumedCompletionState(requestId: GenerateRequestId): void {
    const completion = this.queuedRequestCompletions.get(requestId);
    if (
      completion != null &&
      completion.settled &&
      completion.consumed &&
      completion.waiterCount === 0
    ) {
      this.queuedRequestCompletions.delete(requestId);
    }
  }

  private startQueuedRequestCompletion(requestId: GenerateRequestId): void {
    if (this.queuedRequestCompletions.has(requestId)) {
      return;
    }

    this.activeQueuedRequestRuns.add(requestId);
    const promise = this.callWorker<
      Extract<WorkerRequestMessage, { kind: 'run-queued-request' }>
    >({
      kind: 'run-queued-request',
      requestId,
    }) as Promise<WorkerRunQueuedRequestResult>;
    void promise.catch(() => {});

    const completionState: QueuedRequestCompletionState = {
      promise,
      settled: false,
      consumed: false,
      waiterCount: 0,
      callbackError: undefined,
    };
    this.queuedRequestCompletions.set(requestId, completionState);

    const observedCompletion = promise.then(
      (result) => {
        this.runtimeAggregateObservability = result.runtimeAggregateObservability;
        this.transportObservability = result.transportObservability;
      },
      () => undefined
    );
    void observedCompletion.catch(() => {});

    observedCompletion
      .finally(() => {
        completionState.settled = true;
        completionState.callbackError = this.queuedTokenErrors.get(requestId);
        this.releaseQueuedRequestExecutionState(requestId);
        this.activeQueuedRequestRuns.delete(requestId);
        this.cleanupConsumedCompletionState(requestId);
      });
  }

  private attachQueuedRequestSignal(
    requestId: GenerateRequestId,
    signal?: AbortSignal
  ): void {
    if (signal == null) {
      return;
    }
    const abortListener = () => {
      void this.cancelQueuedRequest(requestId);
    };
    this.queuedSignals.set(requestId, signal);
    this.queuedSignalAbortListeners.set(requestId, abortListener);
    signal.addEventListener('abort', abortListener, { once: true });
  }

  private resetQueuedRequestLifecycleState(): void {
    this.queuedTokenCallbacks.clear();
    this.pendingQueuedTokenCallbacks.clear();
    this.queuedTokenErrors.clear();
    for (const requestId of this.queuedSignals.keys()) {
      this.detachQueuedRequestSignal(requestId);
    }
    this.activeQueuedRequestRuns.clear();
    this.queuedRequestCompletions.clear();
    this.runtimeAggregateObservability = null;
  }

  public async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
    await this.ensureWorkerInitialized();
    const cancelled = (await this.callWorker<Extract<WorkerRequestMessage, { kind: 'cancel-request' }>>({
      kind: 'cancel-request',
      requestId,
    })) as boolean;
    if (cancelled && !this.activeQueuedRequestRuns.has(requestId)) {
      this.releaseQueuedRequestExecutionState(requestId);
    }
    return cancelled;
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

    let requestId: GenerateRequestId;
    try {
      requestId = (await this.callWorkerWithId<
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
    } catch (error) {
      this.pendingQueuedTokenCallbacks.delete(callId);
      throw error;
    }

    const pendingCallback = this.pendingQueuedTokenCallbacks.get(callId);
    this.pendingQueuedTokenCallbacks.delete(callId);
    if (pendingCallback) {
      this.queuedTokenCallbacks.set(requestId, pendingCallback);
    }
    this.attachQueuedRequestSignal(requestId, signal);
    this.startQueuedRequestCompletion(requestId);
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

    const completionState = this.queuedRequestCompletions.get(requestId);
    if (completionState == null) {
      throw new Error(`Queued request ${requestId} is not available.`);
    }
    completionState.consumed = true;
    completionState.waiterCount += 1;
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
      const result = await completionState.promise;
      this.runtimeAggregateObservability = result.runtimeAggregateObservability;
      this.transportObservability = result.transportObservability;

      const tokenError = completionState.callbackError;
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
      completionState.waiterCount = Math.max(0, completionState.waiterCount - 1);
      this.cleanupConsumedCompletionState(requestId);
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

  public getRuntimeAggregateObservability(): RuntimeAggregateObservabilityMetrics | null {
    return this.runtimeAggregateObservability;
  }

  public getRuntimeObservability(): RuntimeAggregateObservabilityMetrics | null {
    return this.getRuntimeAggregateObservability();
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

  private resetWorkerState(error: unknown): void {
    this.worker = null;
    this.workerInitialized = false;
    for (const call of this.pendingWorkerCalls.values()) {
      call.reject(error);
    }
    this.pendingWorkerCalls.clear();
    for (const pendingAck of this.pendingStreamChunkAcks.values()) {
      pendingAck.reject(error);
    }
    this.pendingStreamChunkAcks.clear();
    this.queuedTokenCallbacks.clear();
    this.pendingQueuedTokenCallbacks.clear();
    this.queuedTokenErrors.clear();
    for (const requestId of this.queuedSignals.keys()) {
      this.detachQueuedRequestSignal(requestId);
    }
    this.activeQueuedRequestRuns.clear();
    this.queuedRequestCompletions.clear();
    this.runtimeAggregateObservability = null;
    this.lastModelLoadInfo = null;
    this.transportObservability = createDefaultTransportObservability();
  }

  private failWorker(error: unknown): void {
    if (this.worker != null) {
      this.worker.onmessage = null;
      this.worker.onerror = null;
      this.worker.onmessageerror = null;
      this.worker.terminate();
    }
    this.resetWorkerState(error);
  }

  private postWorkerMessage(
    message: WorkerRequestMessage,
    transferables: Transferable[] = []
  ): void {
    if (this.worker == null) {
      throw new Error('Worker runtime is not available.');
    }
    this.worker.postMessage(message, transferables);
  }

  private sendModelLoadCancel(callId: number): void {
    if (this.worker == null) {
      return;
    }
    this.postWorkerMessage({
      kind: 'cancel-model-load',
      callId,
    });
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
      this.worker.onerror = (event: ErrorEvent) => {
        this.failWorker(new Error(event.message || 'Worker runtime crashed.'));
      };
      this.worker.onmessageerror = () => {
        this.failWorker(new Error('Worker runtime failed to deserialize a message.'));
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

    if (message.kind === 'load-stream-ack') {
      const pendingAck = this.pendingStreamChunkAcks.get(message.callId);
      if (!pendingAck) {
        return;
      }
      this.pendingStreamChunkAcks.delete(message.callId);
      pendingAck.resolve();
      return;
    }

    const pendingCall = this.pendingWorkerCalls.get(message.callId);
    if (message.kind === 'load-progress') {
      pendingCall?.onProgress?.(message.progressPct);
      return;
    }

    const error =
      message.kind !== 'reject'
        ? null
        : message.errorName === 'AbortError'
          ? createAbortError(message.message)
          : Object.assign(new Error(message.message), {
              name: message.errorName ?? 'Error',
            });

    if (error != null) {
      const pendingAck = this.pendingStreamChunkAcks.get(message.callId);
      if (pendingAck != null) {
        this.pendingStreamChunkAcks.delete(message.callId);
        pendingAck.reject(error);
      }
    }

    if (!pendingCall) {
      return;
    }

    this.pendingWorkerCalls.delete(message.callId);
    if (message.kind === 'resolve') {
      pendingCall.resolve(message.value);
      return;
    }

    pendingCall.reject(error ?? new Error('Worker call failed.'));
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

    return new Promise<unknown>((resolve, reject) => {
      this.pendingWorkerCalls.set(callId, {
        resolve,
        reject,
        onProgress,
      });
      this.postWorkerMessage(request);
    });
  }

  private async callWorkerWithAbort<T extends WorkerRequestMessage>(
    callId: number,
    message: WithoutCallId<T>,
    signal?: AbortSignal,
    onProgress?: (pct: number) => void
  ): Promise<unknown> {
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    const abortListener =
      signal == null
        ? null
        : () => {
            this.sendModelLoadCancel(callId);
          };
    if (abortListener != null) {
      signal?.addEventListener('abort', abortListener, { once: true });
    }

    try {
      return await this.callWorkerWithId(callId, message, onProgress);
    } finally {
      if (abortListener != null) {
        signal?.removeEventListener('abort', abortListener);
      }
    }
  }

  private async sendStreamChunk(
    callId: number,
    chunk: Uint8Array
  ): Promise<void> {
    if (this.pendingStreamChunkAcks.has(callId)) {
      throw new Error(`Load stream ${callId} already has a pending chunk acknowledgement.`);
    }

    const transferableChunk = toTransferableChunkBuffer(chunk);

    return new Promise<void>((resolve, reject) => {
      this.pendingStreamChunkAcks.set(callId, { resolve, reject });
      try {
        this.postWorkerMessage(
          {
            kind: 'load-model-stream-chunk',
            callId,
            chunk: transferableChunk,
          },
          [transferableChunk]
        );
      } catch (error) {
        this.pendingStreamChunkAcks.delete(callId);
        reject(error);
      }
    });
  }
}
