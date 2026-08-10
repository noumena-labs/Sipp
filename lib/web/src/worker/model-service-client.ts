import type { SippClientOptions } from '../engine/browser-client.js';
import { resolveOptimizedPackageAssetUrl } from '../engine/runtime-assets.js';
import { ObservabilityController } from '../models/observability-controller.js';
import { observabilitySnapshotToEngineState } from '../models/observability-controller.js';
import { SharedTokenRingReader } from '../runtime/shared-token-ring.js';
import { createAbortError } from '../utils/abort.js';
import { AsyncSerialQueue } from '../utils/async-queue.js';
import {
  hostTaskScheduler,
  type ScheduledTask,
  type TaskScheduler,
} from '../utils/task-scheduler.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  type WorkerQueryOptions,
  type WorkerRuntimeConfig,
} from './model-service-protocol.js';
import {
  QueryError,
  type EngineEvent,
  type EngineState,
  type ObservabilityEvent,
  type ObservabilitySnapshot,
  type ModelInfo,
  type ModelAddOptions,
  type ModelAddSource,
  type ModelLifecycleService,
  type ModelLoadOptions,
  type EmbedOptions,
  type AudioResult,
  type EmbeddingResult,
  type GenerationResult,
  type ListenOptions,
  type ChatInput,
  type InternalTextRequestOptions,
  type QueryInput,
  type QueryOptions,
  type SpeakOptions,
  type TokenBatch,
  type TokenEmissionStats,
} from '../models/types.js';

interface PendingWorkerCall {
  readonly incarnation: number;
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  onProgress?: ModelLoadOptions['onProgress'];
  tokenBatchSink?: (batch: TokenBatch) => void;
  /** Set once the Worker claims a native request ID for this call's stream. */
  nativeRequestId?: number;
  /** First failure thrown by the caller's token sink, surfaced on settle. */
  tokenSinkError: unknown;
  tokenSinkFailed: boolean;
  /** Allows the shutdown acknowledgement to settle after presentation stops. */
  allowDuringRetirement: boolean;
}

interface PendingCallFinalization {
  readonly failed: boolean;
  readonly error: unknown;
}

interface WorkerInstance {
  readonly worker: Worker;
  readonly incarnation: number;
  acceptPresentation: boolean;
}

/** @internal Test seams. Not part of the public client surface. */
export interface WorkerModelServiceClientInternals {
  readonly tasks?: TaskScheduler;
}

interface WorkerCallOptions {
  signal?: AbortSignal;
  onProgress?: ModelLoadOptions['onProgress'];
  tokenBatchSink?: (batch: TokenBatch) => void;
  emitTokens?: boolean;
  allowDuringRetirement?: boolean;
}

interface PendingTokenRecord {
  readonly streamId: number;
  readonly sequenceStart: number;
  readonly frameCount: number;
  readonly byteCount: number;
  readonly text: string;
}

// Distributes over the union so each variant keeps its own shape; a plain
// Omit over the union would collapse it to the shared members only.
type WithoutCallId<T> = T extends unknown ? Omit<T, 'callId'> : never;

/** Any operational request, minus the callId the client assigns. */
type WorkerCallRequest = WithoutCallId<Extract<WorkerRequestMessage, { callId: number }>>;
const WORKER_SHUTDOWN_BUDGET_MS = 1_000;

export function getOptimizedDefaultWorkerUrl(importerUrl: string = import.meta.url): string | null {
  return resolveOptimizedPackageAssetUrl('dist/esm/worker/model-service-entry.js', importerUrl);
}

function toWorkerRuntimeConfig(config: SippClientOptions): WorkerRuntimeConfig {
  if (typeof config.moduleOptions?.locateFile === 'function') {
    throw new Error(
      'Worker mode does not support moduleOptions.locateFile. Provide explicit moduleUrl/wasmUrl instead.'
    );
  }

  if (config.moduleOptions != null && typeof structuredClone === 'function') {
    try {
      structuredClone(config.moduleOptions);
    } catch (error) {
      throw new Error(
        'Worker mode only supports structured-cloneable moduleOptions.',
        { cause: error }
      );
    }
  }

  return {
    moduleUrl: config.moduleUrl,
    wasmUrl: config.wasmUrl,
    wasmThreading: config.wasmThreading,
    moduleOptions: config.moduleOptions,
    maxModelBytes: config.maxModelBytes,
    storageRoot: config.storageRoot,
    browserCache: config.browserCache,
    trustedOrigins: config.trustedOrigins,
  };
}

function toWorkerQueryOptions(
  options: QueryOptions = {},
  emitTokens: boolean
): WorkerQueryOptions {
  return {
    contextKey: options.contextKey,
    maxTokens: options.maxTokens,
    temperature: options.temperature,
    topP: options.topP,
    sampling: options.sampling,
    stop: options.stop,
    grammar: options.grammar,
    emitTokens,
  };
}

export class WorkerModelServiceClient implements ModelLifecycleService {
  private worker: WorkerInstance | null = null;
  private nextWorkerIncarnation = 1;
  private activeRuntimeIncarnation: number | null = null;
  private nextCallId = 1;
  private closed = false;
  private readonly lifecycleOperations = new AsyncSerialQueue();
  private currentSnapshot: ModelInfo | null = null;
  private readonly observability = new ObservabilityController();
  private readonly engineEventListeners = new Set<(event: EngineEvent) => void>();
  private readonly pendingCalls = new Map<number, PendingWorkerCall>();
  private workerConfig: WorkerRuntimeConfig | null = null;
  private tokenRingReader: SharedTokenRingReader | null = null;
  private tokenRingDrainTask: ScheduledTask | null = null;
  private activeTokenCallCount = 0;
  private readonly callIdByNativeRequestId = new Map<number, number>();
  private readonly pendingTokenRecordsByNativeRequestId = new Map<number, PendingTokenRecord[]>();
  private readonly streamStatsByCallId = new Map<number, TokenEmissionStats>();

  private readonly tasks: TaskScheduler;

  constructor(
    private readonly config: SippClientOptions = {},
    internals: WorkerModelServiceClientInternals = {}
  ) {
    this.tasks = internals.tasks ?? hostTaskScheduler;
  }

  public async add(
    source: ModelAddSource,
    options: ModelAddOptions = {}
  ): Promise<ModelInfo> {
    this.assertOpen();
    return await this.lifecycleOperations.run(async () =>
      (await this.callWorker(
        {
          kind: 'models-install',
          source,
        },
        {
          signal: options.signal,
          onProgress: options.onProgress,
        }
      )) as ModelInfo
    );
  }

  public async load(modelId: string, options: ModelLoadOptions = {}): Promise<ModelInfo> {
    this.assertOpen();
    return await this.lifecycleOperations.run(async () => {
      await this.retireWorker();
      const instance = this.ensureWorker();
      try {
        const result = (await this.callWorkerOn(
          instance,
          {
            kind: 'models-load',
            modelId,
            options: {
              backend: options.backend,
              observability: options.observability,
              runtime: options.runtime,
            },
          },
          {
            signal: options.signal,
            onProgress: options.onProgress,
          }
        )) as ModelInfo;
        this.activeRuntimeIncarnation = instance.incarnation;
        this.currentSnapshot = result;
        return result;
      } catch (error) {
        this.destroyWorker(instance, error);
        throw error;
      }
    });
  }

  public current(): ModelInfo | null {
    this.assertOpen();
    return this.currentSnapshot;
  }

  public async list(): Promise<ModelInfo[]> {
    this.assertOpen();
    return await this.lifecycleOperations.run(async () => {
      const models = (await this.callWorker({
        kind: 'models-list',
      })) as ModelInfo[];
      if (this.activeRuntimeIncarnation === this.worker?.incarnation) {
        this.currentSnapshot = models.find((model) => model.loaded) ?? null;
      }
      return models;
    });
  }

  public async remove(id: string): Promise<void> {
    this.assertOpen();
    await this.lifecycleOperations.run(async () => {
      const current = (await this.callWorker({
        kind: 'models-remove',
        id,
      })) as ModelInfo | null;
      if (this.activeRuntimeIncarnation === this.worker?.incarnation) {
        this.currentSnapshot = current;
      }
    });
  }

  public async unload(): Promise<void> {
    this.assertOpen();
    await this.lifecycleOperations.run(async () => {
      await this.retireWorker();
    });
  }

  public async runQuery(
    input: QueryInput,
    options: InternalTextRequestOptions
  ): Promise<GenerationResult> {
    this.assertOpen();
    const emitTokens = options.tokenBatchSink != null;
    return (await this.callRuntimeWorker(
      {
        kind: 'query',
        input,
        options: toWorkerQueryOptions(options, emitTokens),
      },
      {
        signal: options.signal,
        tokenBatchSink: options.tokenBatchSink,
        emitTokens,
      }
    )) as GenerationResult;
  }

  public async runChat(
    input: ChatInput,
    options: InternalTextRequestOptions
  ): Promise<GenerationResult> {
    this.assertOpen();
    const emitTokens = options.tokenBatchSink != null;
    return (await this.callRuntimeWorker(
      {
        kind: 'chat',
        input,
        options: toWorkerQueryOptions(options, emitTokens),
      },
      {
        signal: options.signal,
        tokenBatchSink: options.tokenBatchSink,
        emitTokens,
      }
    )) as GenerationResult;
  }

  public async runEmbedding(
    input: string,
    options: EmbedOptions
  ): Promise<EmbeddingResult> {
    this.assertOpen();
    return (await this.callRuntimeWorker(
      {
        kind: 'embed',
        input,
        options: {
          normalize: options.normalize,
          contextKey: options.contextKey,
        },
      },
      {
        signal: options.signal,
      }
    )) as EmbeddingResult;
  }

  public async runListen(
    audio: Uint8Array,
    options: ListenOptions
  ): Promise<GenerationResult> {
    this.assertOpen();
    return (await this.callRuntimeWorker(
      {
        kind: 'listen',
        audio,
        options: {
          language: options.language,
          maxTokens: options.maxTokens,
        },
      },
      { signal: options.signal }
    )) as GenerationResult;
  }

  public async runSpeak(text: string, options: SpeakOptions): Promise<AudioResult> {
    this.assertOpen();
    return (await this.callRuntimeWorker(
      {
        kind: 'speak',
        text,
        options: {
          language: options.language,
          speakerAudio: options.speakerAudio,
          maxDurationMs: options.maxDurationMs,
        },
      },
      { signal: options.signal }
    )) as AudioResult;
  }

  public currentObservability(): ObservabilitySnapshot {
    this.assertOpen();
    return this.observability.current();
  }

  public subscribeObservability(listener: (event: ObservabilityEvent) => void): () => void {
    this.assertOpen();
    return this.observability.subscribe(listener);
  }

  public state(): EngineState {
    this.assertOpen();
    return this.snapshotState();
  }

  public subscribeEvents(listener: (event: EngineEvent) => void): () => void {
    this.assertOpen();
    this.engineEventListeners.add(listener);
    return () => {
      this.engineEventListeners.delete(listener);
    };
  }

  public async close(): Promise<void> {
    if (this.closed) {
      return;
    }
    this.closed = true;
    try {
      await this.lifecycleOperations.run(async () => {
        await this.retireWorker();
      });
    } finally {
      this.clearRuntimeProjection();
      this.observability.markClosed();
      this.emitEngineEvent({ type: 'closed' });
    }
  }

  private assertOpen(): void {
    if (this.closed) {
      throw new QueryError('ENGINE_CLOSED', 'SippClient is closed.');
    }
  }

  private getWorkerConfig(): WorkerRuntimeConfig {
    this.workerConfig ??= toWorkerRuntimeConfig(this.config);
    return this.workerConfig;
  }

  private scheduleTokenRingDrain(): void {
    if (
      this.tokenRingDrainTask != null ||
      this.activeTokenCallCount === 0 ||
      this.tokenRingReader == null
    ) {
      return;
    }
    this.tokenRingDrainTask = this.tasks.frame(() => {
      this.tokenRingDrainTask = null;
      try {
        this.drainTokenRing();
      } finally {
        this.scheduleTokenRingDrain();
      }
    });
  }

  private drainTokenRing(): void {
    const reader = this.tokenRingReader;
    if (reader == null) {
      return;
    }
    reader.drain((streamId, sequenceStart, frameCount, byteCount, text) => {
      const record: PendingTokenRecord = {
        streamId,
        sequenceStart,
        frameCount,
        byteCount,
        text,
      };
      const callId = this.callIdByNativeRequestId.get(streamId);
      if (callId == null) {
        this.bufferPendingTokenRecord(record);
        return;
      }
      this.deliverTokenBatch(callId, record);
    });
  }

  private bufferPendingTokenRecord(record: PendingTokenRecord): void {
    const records = this.pendingTokenRecordsByNativeRequestId.get(record.streamId);
    if (records == null) {
      this.pendingTokenRecordsByNativeRequestId.set(record.streamId, [record]);
      return;
    }
    records.push(record);
  }

  private flushPendingTokenRecords(nativeRequestId: number, callId: number): void {
    const records = this.pendingTokenRecordsByNativeRequestId.get(nativeRequestId);
    if (records == null) {
      return;
    }
    this.pendingTokenRecordsByNativeRequestId.delete(nativeRequestId);
    for (const record of records) {
      this.deliverTokenBatch(callId, record);
    }
  }

  private deliverTokenBatch(callId: number, record: PendingTokenRecord): void {
    if (record.text.length === 0) {
      return;
    }
    const pending = this.pendingCalls.get(callId);
    if (pending?.tokenBatchSink == null) {
      return;
    }
    const stats = this.streamStatsByCallId.get(callId) ?? {
      framesSent: 0,
      bytesSent: 0,
      batchesSent: 0,
    };
    stats.framesSent += record.frameCount;
    stats.bytesSent += record.byteCount;
    stats.batchesSent += 1;
    this.streamStatsByCallId.set(callId, stats);
    try {
      pending.tokenBatchSink({
        requestId: String(record.streamId),
        streamId: record.streamId,
        sequenceStart: record.sequenceStart,
        text: record.text,
        frameCount: record.frameCount,
        byteCount: record.byteCount,
        stats: { ...stats },
      });
    } catch (error) {
      this.failTokenSink(callId, pending, error);
    }
  }

  /**
   * Records a caller token-sink failure and cancels the request that produced
   * it. The failure is reported when the call settles: letting it escape the
   * drain loop would skip rescheduling and strand the caller's promise.
   */
  private failTokenSink(
    callId: number,
    pending: PendingWorkerCall,
    error: unknown
  ): void {
    if (!pending.tokenSinkFailed) {
      pending.tokenSinkFailed = true;
      pending.tokenSinkError = error;
    }
    pending.tokenBatchSink = undefined;
    const instance = this.worker;
    if (instance != null && instance.incarnation === pending.incarnation) {
      this.postWorkerCancellation(instance, callId);
    }
  }

  /** Posts a best-effort cancellation without letting a retired Worker throw. */
  private postWorkerCancellation(instance: WorkerInstance, callId: number): void {
    if (this.worker !== instance) {
      return;
    }
    try {
      instance.worker.postMessage({
        kind: 'cancel',
        targetCallId: callId,
      } satisfies WorkerRequestMessage);
    } catch {
      // The request will still settle through the Worker response or retirement.
    }
  }

  private forgetStreamingCall(callId: number): void {
    const nativeRequestId = this.pendingCalls.get(callId)?.nativeRequestId;
    if (nativeRequestId != null) {
      this.callIdByNativeRequestId.delete(nativeRequestId);
      this.pendingTokenRecordsByNativeRequestId.delete(nativeRequestId);
    }
    this.streamStatsByCallId.delete(callId);
  }

  private ensureWorker(): WorkerInstance {
    if (this.worker != null) {
      return this.worker;
    }
    // Resolve the config before spawning, so an invalid config fails the
    // caller instead of leaving an unconfigurable Worker behind.
    const config = this.getWorkerConfig();
    const optimizedWorkerUrl = getOptimizedDefaultWorkerUrl();
    const worker =
      this.config.workerUrl == null
        ? optimizedWorkerUrl == null
          ? new Worker(new URL('./model-service-entry.js', import.meta.url), { type: 'module' })
          : new Worker(optimizedWorkerUrl, { type: 'module' })
        : new Worker(this.config.workerUrl, { type: 'module' });
    const instance: WorkerInstance = {
      worker,
      incarnation: this.nextWorkerIncarnation++,
      acceptPresentation: true,
    };
    this.worker = instance;
    worker.onmessage = (event: MessageEvent<WorkerResponseMessage>) => {
      this.handleWorkerMessage(instance, event.data);
    };
    worker.onerror = (event: ErrorEvent) => {
      this.failWorker(
        instance,
        event.error instanceof Error
          ? event.error
          : new Error(event.message || 'Worker runtime crashed.')
      );
    };
    worker.onmessageerror = () => {
      this.failWorker(instance, new Error('Worker runtime failed to deserialize a message.'));
    };
    // Configure the Worker before any operational request. Messages are
    // delivered in order, so every later request observes this service.
    try {
      worker.postMessage({ kind: 'initialize', config } satisfies WorkerRequestMessage);
    } catch (error) {
      // An unconfigurable Worker must not stay installed.
      this.destroyWorker(instance, error);
      throw error;
    }
    return instance;
  }

  private failWorker(instance: WorkerInstance, error: unknown): void {
    if (this.worker !== instance) {
      return;
    }
    this.destroyWorker(instance, error);
    this.observability.emit('error', {
      state: 'error',
      model: null,
      query: null,
    });
    this.emitEngineEvent({ type: 'state', state: this.snapshotState() });
  }

  private destroyWorker(instance: WorkerInstance, error: unknown): void {
    instance.worker.onmessage = null;
    instance.worker.onerror = null;
    instance.worker.onmessageerror = null;
    instance.worker.terminate();
    if (this.worker === instance) {
      this.worker = null;
      this.clearRuntimeProjection();
    }
    for (const [callId, pending] of this.pendingCalls) {
      if (pending.incarnation === instance.incarnation) {
        // reject() finalizes, which removes the entry.
        pending.reject(error);
      }
    }
  }

  /**
   * Stops delivering tokens from the current Worker. Called as soon as
   * retirement begins so nothing is presented during the shutdown budget,
   * which can be a full second before the Worker is destroyed.
   */
  private stopPresentation(): void {
    this.tokenRingDrainTask?.cancel();
    this.tokenRingDrainTask = null;
    this.tokenRingReader = null;
    this.callIdByNativeRequestId.clear();
    this.pendingTokenRecordsByNativeRequestId.clear();
  }

  private clearRuntimeProjection(): void {
    this.stopPresentation();
    this.activeTokenCallCount = 0;
    this.streamStatsByCallId.clear();
    this.activeRuntimeIncarnation = null;
    this.currentSnapshot = null;
    this.observability.update({
      state: 'idle',
      model: null,
      query: null,
      runtime: null,
      profile: null,
    });
  }

  private async retireWorker(): Promise<void> {
    const instance = this.worker;
    this.activeRuntimeIncarnation = null;
    this.currentSnapshot = null;
    if (instance == null) {
      this.clearRuntimeProjection();
      return;
    }

    instance.acceptPresentation = false;
    this.stopPresentation();
    const retirementError = new QueryError('ENGINE_CLOSED', 'Worker runtime was retired.');
    try {
      await this.awaitWorkerShutdown(instance);
    } finally {
      this.destroyWorker(instance, retirementError);
    }
  }

  private async awaitWorkerShutdown(instance: WorkerInstance): Promise<void> {
    await new Promise<void>((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) {
          return;
        }
        settled = true;
        budget.cancel();
        resolve();
      };
      const budget = this.tasks.delay(finish, WORKER_SHUTDOWN_BUDGET_MS);
      void this.callWorkerOn(
        instance,
        { kind: 'shutdown' },
        { allowDuringRetirement: true }
      ).then(finish, finish);
    });
  }

  private callWorker(
    message: WorkerCallRequest,
    options: WorkerCallOptions = {}
  ): Promise<unknown> {
    return this.callWorkerOn(this.ensureWorker(), message, options);
  }

  private callRuntimeWorker(
    message: WorkerCallRequest,
    options: WorkerCallOptions = {}
  ): Promise<unknown> {
    const instance = this.worker;
    if (
      instance == null ||
      this.activeRuntimeIncarnation !== instance.incarnation
    ) {
      throw new QueryError('MODEL_NOT_READY', 'No local model is active.');
    }
    return this.callWorkerOn(instance, message, options);
  }

  private callWorkerOn(
    instance: WorkerInstance,
    message: WorkerCallRequest,
    options: WorkerCallOptions = {}
  ): Promise<unknown> {
    if (options.signal?.aborted) {
      throw createAbortError('Operation aborted.');
    }
    if (this.worker !== instance) {
      throw new QueryError('ENGINE_CLOSED', 'Worker runtime was retired.');
    }

    const callId = this.nextCallId++;
    const request = { ...message, callId } as WorkerRequestMessage;

    let cleanup = (): void => {};
    if (options.signal != null) {
      const abortListener = () => {
        this.postWorkerCancellation(instance, callId);
      };
      options.signal.addEventListener('abort', abortListener, { once: true });
      cleanup = () => {
        options.signal?.removeEventListener('abort', abortListener);
      };
    }

    if (options.emitTokens === true) {
      this.activeTokenCallCount += 1;
      this.scheduleTokenRingDrain();
    }

    return new Promise<unknown>((resolve, reject) => {
      // Must never throw: the caller's promise is settled immediately after.
      const finalize = (): PendingCallFinalization => {
        let drainFailed = false;
        let drainError: unknown;
        if (options.emitTokens === true) {
          try {
            this.drainTokenRing();
          } catch (error) {
            // A transport failure must reject the call without preventing the
            // remaining bookkeeping from running.
            drainFailed = true;
            drainError = error;
          }
          this.forgetStreamingCall(callId);
          this.activeTokenCallCount = Math.max(0, this.activeTokenCallCount - 1);
          if (this.activeTokenCallCount === 0) {
            this.pendingTokenRecordsByNativeRequestId.clear();
          }
        }
        cleanup();
        const pending = this.pendingCalls.get(callId);
        this.pendingCalls.delete(callId);
        if (pending?.tokenSinkFailed === true) {
          return { failed: true, error: pending.tokenSinkError };
        }
        return { failed: drainFailed, error: drainError };
      };
      this.pendingCalls.set(callId, {
        incarnation: instance.incarnation,
        resolve: (value) => {
          const finalization = finalize();
          if (finalization.failed) {
            reject(finalization.error);
            return;
          }
          resolve(value);
        },
        reject: (error) => {
          // A sink failure is the caller's own error and outranks the
          // cancellation it triggered.
          const finalization = finalize();
          reject(finalization.failed ? finalization.error : error);
        },
        onProgress: options.onProgress,
        tokenBatchSink: options.tokenBatchSink,
        tokenSinkError: undefined,
        tokenSinkFailed: false,
        allowDuringRetirement: options.allowDuringRetirement === true,
      });
      try {
        instance.worker.postMessage(request);
      } catch (error) {
        finalize();
        reject(error);
      }
    });
  }

  private handleWorkerMessage(instance: WorkerInstance, message: WorkerResponseMessage): void {
    if (this.worker !== instance) {
      return;
    }
    if (message.kind === 'load-progress') {
      const pending = this.pendingCalls.get(message.callId);
      if (instance.acceptPresentation && pending?.incarnation === instance.incarnation) {
        pending.onProgress?.(message.progress);
      }
      return;
    }

    if (message.kind === 'token-ring-ready') {
      if (!instance.acceptPresentation) {
        return;
      }
      this.tokenRingReader = new SharedTokenRingReader(message.descriptor);
      this.scheduleTokenRingDrain();
      return;
    }

    if (message.kind === 'token-ring-claim') {
      if (!instance.acceptPresentation) {
        return;
      }
      this.callIdByNativeRequestId.set(message.nativeRequestId, message.callId);
      const claiming = this.pendingCalls.get(message.callId);
      if (claiming != null) {
        claiming.nativeRequestId = message.nativeRequestId;
      }
      this.flushPendingTokenRecords(message.nativeRequestId, message.callId);
      this.scheduleTokenRingDrain();
      return;
    }

    if (message.kind === 'token-batch') {
      const pending = this.pendingCalls.get(message.callId);
      if (instance.acceptPresentation && pending?.incarnation === instance.incarnation) {
        try {
          pending.tokenBatchSink?.(message.batch);
        } catch (error) {
          this.failTokenSink(message.callId, pending, error);
        }
      }
      return;
    }

    if (message.kind === 'observability-event') {
      if (!instance.acceptPresentation) {
        return;
      }
      this.observability.ingest(message.event);
      const model = message.event.snapshot.model;
      this.currentSnapshot = model?.loaded === true ? model : null;
      return;
    }

    if (message.kind === 'engine-event') {
      if (instance.acceptPresentation) {
        this.emitEngineEvent(message.event);
      }
      return;
    }

    const pending = this.pendingCalls.get(message.callId);
    if (pending == null || pending.incarnation !== instance.incarnation) {
      return;
    }
    if (!instance.acceptPresentation && !pending.allowDuringRetirement) {
      return;
    }

    if (message.kind === 'resolve') {
      pending.resolve(message.value);
      return;
    }

    pending.reject(this.deserializeError(message));
  }

  private deserializeError(
    message: Extract<WorkerResponseMessage, { kind: 'reject' }>
  ): unknown {
    const error = this.reviveWorkerError(message);
    error.stack = message.errorStack ?? error.stack;
    if (message.cleanupFailures != null && message.cleanupFailures.length > 0) {
      Object.defineProperty(error, 'cleanupFailures', {
        configurable: true,
        enumerable: false,
        value: new AggregateError(
          message.cleanupFailures.map((failure) => new Error(failure)),
          'Cleanup failed after the primary operation error.'
        ),
      });
    }
    return error;
  }

  private reviveWorkerError(
    message: Extract<WorkerResponseMessage, { kind: 'reject' }>
  ): Error {
    if (message.queryErrorCode != null) {
      return new QueryError(message.queryErrorCode, message.message);
    }
    if (message.errorName === 'AbortError') {
      return new DOMException(message.message, 'AbortError') as unknown as Error;
    }
    return Object.assign(new Error(message.message), {
      name: message.errorName ?? 'Error',
    });
  }

  private emitEngineEvent(event: EngineEvent): void {
    for (const listener of this.engineEventListeners) {
      listener(event);
    }
  }

  private snapshotState(): EngineState {
    return observabilitySnapshotToEngineState(this.observability.current());
  }
}
