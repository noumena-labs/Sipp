import { CogentConfig } from '../cogent-config.js';
import {
  BackendObservability,
  EngineExecutionMode,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  ModelBundleDescriptor,
  ModelLoadInfo,
  PromptOptions,
  PreparedModelBundle,
  PrepareModelBundleOptions,
  RuntimeAggregateObservabilityMetrics,
  TransportObservability,
} from '../types.js';
import { EngineRuntime } from './engine-runtime.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  WorkerLoadModelResult,
  WorkerPrepareModelBundleResult,
  WorkerRunQueuedRequestResult,
  WorkerBackendObservabilityResult,
  WorkerRuntimeMetadata,
} from './engine-runtime-worker-protocol.js';
import { createAbortError } from '../utils/abort.js';
import {
  createDefaultTransportObservability,
  countOccurrences,
  PendingWorkerCall,
  normalizeOptionalString,
  toTransferableMediaBuffers,
  toTransferableChunkBuffer,
  toWorkerSerializableConfig,
  WithoutCallId,
} from './worker-runtime-utils.js';
import { RequestTracker } from './request-tracker.js';

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
  private readonly tracker = new RequestTracker<WorkerRunQueuedRequestResult>();
  private lastWorkerTerminalError: unknown = null;
  private cachedRuntimeMetadata: WorkerRuntimeMetadata | null = null;
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

  public loadModelFromBuffer(_buffer: Uint8Array, _destFileName = 'model.gguf'): string {
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

  public async prepareModelBundle(
    descriptor: ModelBundleDescriptor,
    options?: PrepareModelBundleOptions
  ): Promise<PreparedModelBundle> {
    await this.ensureWorkerInitialized();
    if (options?.signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    const callId = this.nextCallId++;
    const result = (await this.callWorkerWithAbort<
      Extract<WorkerRequestMessage, { kind: 'prepare-model-bundle' }>
    >(
      callId,
      {
        kind: 'prepare-model-bundle',
        descriptor,
      },
      options?.signal
    )) as WorkerPrepareModelBundleResult;
    this.lastModelLoadInfo = result.bundle.modelLoadInfo;
    this.transportObservability = result.transportObservability;
    return result.bundle;
  }

  public async initEngine(
    modelPathOrBundle: string | PreparedModelBundle,
    config?: InferenceInitConfig
  ): Promise<void> {
    await this.ensureWorkerInitialized();
    this.resetQueuedRequestLifecycleState();
    this.cachedRuntimeMetadata = null;
    const modelPath =
      typeof modelPathOrBundle === 'string' ? modelPathOrBundle : modelPathOrBundle.modelPath;
    const effectiveConfig =
      typeof modelPathOrBundle === 'string' ||
      this.hasExplicitProjectorPath(config) ||
      modelPathOrBundle.multimodalProjectorPath == null
        ? config
        : {
            ...config,
            multimodalProjectorPath: modelPathOrBundle.multimodalProjectorPath,
          };
    const metadata = (await this.callWorker<
      Extract<WorkerRequestMessage, { kind: 'init-engine' }>
    >({
      kind: 'init-engine',
      modelPath,
      config: effectiveConfig,
    })) as WorkerRuntimeMetadata;
    this.cachedRuntimeMetadata = this.normalizeRuntimeMetadata(metadata);
  }

  private hasExplicitProjectorPath(config: InferenceInitConfig | undefined): boolean {
    return (
      typeof config?.multimodalProjectorPath === 'string' &&
      config.multimodalProjectorPath.trim().length > 0
    );
  }

  public close(): void {
    if (this.worker != null) {
      this.worker.terminate();
    }
    this.resetWorkerState(new Error('Worker runtime was closed.'));
  }

  private releaseTokenState(requestId: GenerateRequestId): void {
    this.queuedTokenCallbacks.delete(requestId);
    this.queuedTokenErrors.delete(requestId);
  }

  private finalizeRequest(
    requestId: GenerateRequestId,
    options: { deleteCompletion?: boolean } = {}
  ): void {
    this.releaseTokenState(requestId);
    this.tracker.finalize(requestId, options);
  }

  private ensureTracked(requestId: GenerateRequestId) {
    return this.tracker.track(requestId);
  }

  private settleQueuedRequestCompletion(
    requestId: GenerateRequestId,
    result: WorkerRunQueuedRequestResult
  ): void {
    const tracked = this.ensureTracked(requestId);
    if (tracked.settled) {
      return;
    }

    this.runtimeAggregateObservability = result.runtimeAggregateObservability;
    this.transportObservability = result.transportObservability;
    tracked.callbackError = this.queuedTokenErrors.get(requestId);
    this.tracker.resolve(requestId, result);
    this.finalizeRequest(requestId, {
      deleteCompletion: result.response.cancelled && !tracked.consumed,
    });
  }

  private rejectQueuedRequestCompletion(
    requestId: GenerateRequestId,
    error: unknown,
    options: { deleteCompletion?: boolean } = {}
  ): void {
    const tracked = this.tracker.get(requestId);
    if (tracked == null || tracked.settled) {
      return;
    }
    tracked.callbackError = this.queuedTokenErrors.get(requestId);
    this.tracker.reject(requestId, error);
    this.finalizeRequest(requestId, options);
  }

  private rejectAllTrackedRequests(error: unknown): void {
    for (const requestId of this.tracker.allTrackedIds()) {
      const tracked = this.tracker.get(requestId);
      if (tracked != null && !tracked.settled) {
        tracked.callbackError = this.queuedTokenErrors.get(requestId);
      }
      this.releaseTokenState(requestId);
    }
    this.tracker.rejectAll(error);
  }

  private attachSignal(
    requestId: GenerateRequestId,
    signal?: AbortSignal
  ): void {
    if (signal == null) {
      return;
    }
    this.tracker.attachSignal(requestId, signal, () => {
      void this.cancelQueuedRequest(requestId);
    });
  }

  private resetQueuedRequestLifecycleState(): void {
    this.rejectAllTrackedRequests(new Error('Queued request lifecycle was reset.'));
    this.queuedTokenCallbacks.clear();
    this.pendingQueuedTokenCallbacks.clear();
    this.queuedTokenErrors.clear();
    this.runtimeAggregateObservability = null;
  }

  public async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
    await this.ensureWorkerInitialized();
    const cancelled = (await this.callWorker<Extract<WorkerRequestMessage, { kind: 'cancel-request' }>>({
      kind: 'cancel-request',
      requestId,
    })) as boolean;
    if (cancelled && !this.tracker.hasActive(requestId)) {
      this.finalizeRequest(requestId, { deleteCompletion: true });
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
    const media = typeof options === 'object' ? options.media : undefined;
    const grammar = typeof options === 'object' ? options.grammar : undefined;
    const callId = this.nextCallId++;
    this.pendingQueuedTokenCallbacks.set(callId, onToken);

    const hasMedia = media != null && media.length > 0;
    const promptFormat =
      typeof options === 'number'
        ? undefined
        : options.promptFormat;
    if (hasMedia) {
      const marker = this.getMediaMarker();
      if (!marker) {
        this.pendingQueuedTokenCallbacks.delete(callId);
        throw new Error('Media prompts require cached media marker metadata.');
      }
      const markerCount = countOccurrences(promptText, marker);
      if (markerCount !== media.length) {
        this.pendingQueuedTokenCallbacks.delete(callId);
        throw new Error(
          `Prompt contains ${markerCount} media marker(s) but ${media.length} media attachment(s) were provided.`
        );
      }
    }

    const transferableMedia = hasMedia ? toTransferableMediaBuffers(media!) : undefined;

    let requestId: GenerateRequestId;
    try {
      requestId = (await this.callWorkerWithId<
        Extract<WorkerRequestMessage, { kind: 'queue-prompt' | 'queue-prompt-with-media' }>
      >(
        callId,
        hasMedia
          ? {
              kind: 'queue-prompt-with-media',
              contextKey,
              promptText,
              options: {
                nTokens: typeof options === 'number' ? options : options.nTokens,
                promptFormat,
                media: transferableMedia,
                grammar,
              },
            }
          : {
              kind: 'queue-prompt',
              contextKey,
              promptText,
              options: {
                nTokens: typeof options === 'number' ? options : options.nTokens,
                promptFormat,
                grammar,
              },
            },
        undefined,
        hasMedia ? transferableMedia ?? [] : []
      )) as GenerateRequestId;
    } catch (error) {
      this.pendingQueuedTokenCallbacks.delete(callId);
      throw error;
    }

    const pendingCallback = this.pendingQueuedTokenCallbacks.get(callId);
    this.pendingQueuedTokenCallbacks.delete(callId);
    if (pendingCallback) {
      this.queuedTokenCallbacks.set(requestId, pendingCallback);
    }
    this.attachSignal(requestId, signal);
    this.ensureTracked(requestId);
    return requestId;
  }

  public getChatTemplate(): string | null {
    return this.cachedRuntimeMetadata?.chatTemplate ?? null;
  }

  public getMediaMarker(): string | null {
    return this.cachedRuntimeMetadata?.mediaMarker ?? null;
  }

  public async runQueuedRequest(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    await this.ensureWorkerInitialized();
    if (options?.signal?.aborted) {
      await this.cancelQueuedRequest(requestId);
      throw createAbortError('Queued request cancelled.');
    }

    const tracked = this.tracker.get(requestId);
    if (tracked == null) {
      if (!this.workerInitialized || this.worker == null) {
        throw this.lastWorkerTerminalError ?? new Error('Worker runtime was closed.');
      }
      throw new Error(`Queued request ${requestId} is not available.`);
    }
    tracked.consumed = true;
    tracked.waiterCount += 1;
    const signal = options?.signal;
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
      const result = await tracked.promise;
      this.runtimeAggregateObservability = result.runtimeAggregateObservability;
      this.transportObservability = result.transportObservability;

      const tokenError = tracked.callbackError;
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
      tracked.waiterCount = Math.max(0, tracked.waiterCount - 1);
      this.tracker.cleanupIfConsumed(requestId);
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
    this.lastWorkerTerminalError = error;
    for (const call of this.pendingWorkerCalls.values()) {
      call.reject(error);
    }
    this.pendingWorkerCalls.clear();
    for (const pendingAck of this.pendingStreamChunkAcks.values()) {
      pendingAck.reject(error);
    }
    this.pendingStreamChunkAcks.clear();
    this.rejectAllTrackedRequests(error);
    this.queuedTokenCallbacks.clear();
    this.pendingQueuedTokenCallbacks.clear();
    this.queuedTokenErrors.clear();
    this.cachedRuntimeMetadata = null;
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
      this.lastWorkerTerminalError = null;
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

    if (message.kind === 'request-complete') {
      this.settleQueuedRequestCompletion(message.requestId, message.result);
      return;
    }

    if (message.kind === 'request-failed') {
      this.runtimeAggregateObservability = message.runtimeAggregateObservability;
      this.transportObservability = message.transportObservability;
      const error =
        message.errorName === 'AbortError'
          ? createAbortError(message.message)
          : Object.assign(new Error(message.message), {
              name: message.errorName ?? 'Error',
            });
      this.rejectQueuedRequestCompletion(message.requestId, error);
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
    onProgress?: (pct: number) => void,
    transferables: Transferable[] = []
  ): Promise<unknown> {
    const callId = this.nextCallId++;
    return this.callWorkerWithId(callId, message, onProgress, transferables);
  }

  private async callWorkerWithId<T extends WorkerRequestMessage>(
    callId: number,
    message: WithoutCallId<T>,
    onProgress?: (pct: number) => void,
    transferables: Transferable[] = []
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
      this.postWorkerMessage(request, transferables);
    });
  }

  private async callWorkerWithAbort<T extends WorkerRequestMessage>(
    callId: number,
    message: WithoutCallId<T>,
    signal?: AbortSignal,
    onProgress?: (pct: number) => void,
    transferables: Transferable[] = []
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
      return await this.callWorkerWithId(callId, message, onProgress, transferables);
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

  private normalizeRuntimeMetadata(
    metadata: WorkerRuntimeMetadata | null | undefined
  ): WorkerRuntimeMetadata {
    return {
      chatTemplate: normalizeOptionalString(metadata?.chatTemplate),
      mediaMarker: normalizeOptionalString(metadata?.mediaMarker),
    };
  }
}
