import type { CogentClientOptions } from '../engine/browser-client.js';
import {
  resolveOptimizedPackageAssetUrl,
  resolveRuntimeUrls,
  supportsWasmPthreads,
  type WasmThreadingMode,
} from '../engine/runtime-assets.js';
import { ObservabilityController } from '../models/observability-controller.js';
import { observabilitySnapshotToEngineState } from '../models/observability-controller.js';
import { SharedTokenRingReader } from '../runtime/shared-token-ring.js';
import { createAbortError } from '../utils/abort.js';
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
  type ModelLifecycleService,
  type ModelLoadOptions,
  type ModelSource,
  type EmbedOptions,
  type EmbeddingResult,
  type GenerationResult,
  type ChatInput,
  type InternalTextRequestOptions,
  type QueryInput,
  type QueryOptions,
  type TokenBatch,
} from '../models/types.js';

interface PendingWorkerCall {
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  onProgress?: ModelLoadOptions['onProgress'];
  tokenBatchSink?: (batch: TokenBatch) => void;
}

interface WorkerCallOptions {
  signal?: AbortSignal;
  onProgress?: ModelLoadOptions['onProgress'];
  tokenBatchSink?: (batch: TokenBatch) => void;
  emitTokens?: boolean;
}

interface PendingTokenRecord {
  readonly streamId: number;
  readonly sequenceStart: number;
  readonly frameCount: number;
  readonly byteCount: number;
  readonly text: string;
  readonly drainMs: number;
}

type RequestWithCallId = Extract<WorkerRequestMessage, { callId: number }>;
type WithoutCallId<T> = T extends { callId: number } ? Omit<T, 'callId'> : never;

export function getOptimizedDefaultWorkerUrl(importerUrl: string = import.meta.url): string | null {
  return resolveOptimizedPackageAssetUrl('dist/esm/worker/model-service-entry.js', importerUrl);
}

function toWorkerRuntimeConfig(config: CogentClientOptions): WorkerRuntimeConfig {
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

  const hasRuntimeUrlOverride =
    config.moduleUrl != null ||
    config.wasmUrl != null ||
    config.pthreadModuleUrl != null ||
    config.pthreadWasmUrl != null;
  const runtimeUrls =
    !hasRuntimeUrlOverride
      ? null
      : resolveRuntimeUrls(config);
  const wasmThreading =
    runtimeUrls?.threading ??
    config.wasmThreading ??
    defaultWorkerThreadingMode();

  return {
    moduleUrl: runtimeUrls?.moduleUrl,
    wasmUrl: runtimeUrls?.wasmUrl,
    wasmThreading,
    moduleOptions: config.moduleOptions,
    maxModelBytes: config.maxModelBytes,
    browserCache: config.browserCache,
    trustedOrigins: config.trustedOrigins,
  };
}

function defaultWorkerThreadingMode(): WasmThreadingMode {
  return supportsWasmPthreads() ? 'pthread' : 'single-thread';
}

function toWorkerQueryOptions(
  options: QueryOptions = {},
  emitTokens: boolean
): WorkerQueryOptions {
  return {
    session: options.session,
    maxTokens: options.maxTokens,
    temperature: options.temperature,
    topP: options.topP,
    stop: options.stop,
    grammar: options.grammar,
    emitTokens,
  };
}

export class WorkerModelServiceClient implements ModelLifecycleService {
  private worker: Worker | null = null;
  private nextCallId = 1;
  private closed = false;
  private currentSnapshot: ModelInfo | null = null;
  private readonly observability = new ObservabilityController();
  private readonly engineEventListeners = new Set<(event: EngineEvent) => void>();
  private readonly pendingCalls = new Map<number, PendingWorkerCall>();
  private readonly workerConfig: WorkerRuntimeConfig;
  private tokenRingReader: SharedTokenRingReader | null = null;
  private tokenRingDrainScheduled = false;
  private activeTokenCallCount = 0;
  private readonly callIdByNativeRequestId = new Map<number, number>();
  private readonly pendingTokenRecordsByNativeRequestId = new Map<number, PendingTokenRecord[]>();
  private readonly streamStatsByCallId = new Map<
    number,
    {
      framesSent: number;
      bytesSent: number;
      batchesSent: number;
      drainMs: number;
      drainCalls: number;
    }
  >();

  constructor(private readonly config: CogentClientOptions = {}) {
    this.workerConfig = toWorkerRuntimeConfig(config);
  }

  public async load(source: ModelSource, options: ModelLoadOptions = {}): Promise<ModelInfo> {
    this.assertOpen();
    const result = (await this.callWorker(
      {
        kind: 'models-load',
        config: this.workerConfig,
        source,
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
    this.currentSnapshot = result.loaded ? result : null;
    return result;
  }

  public current(): ModelInfo | null {
    this.assertOpen();
    return this.currentSnapshot;
  }

  public async list(): Promise<ModelInfo[]> {
    this.assertOpen();
    const models = (await this.callWorker({
      kind: 'models-list',
      config: this.workerConfig,
    })) as ModelInfo[];
    this.currentSnapshot = models.find((model) => model.loaded) ?? null;
    return models;
  }

  public async remove(id: string): Promise<void> {
    this.assertOpen();
    const current = (await this.callWorker({
      kind: 'models-remove',
      config: this.workerConfig,
      id,
    })) as ModelInfo | null;
    this.currentSnapshot = current;
  }

  public async unload(): Promise<void> {
    this.assertOpen();
    await this.callWorker({
      kind: 'models-unload',
      config: this.workerConfig,
    });
    this.currentSnapshot = null;
  }

  public async runQuery(
    input: QueryInput,
    options: InternalTextRequestOptions
  ): Promise<GenerationResult> {
    this.assertOpen();
    const emitTokens = options.tokenBatchSink != null;
    return (await this.callWorker(
      {
        kind: 'query',
        config: this.workerConfig,
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
    return (await this.callWorker(
      {
        kind: 'chat',
        config: this.workerConfig,
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
    return (await this.callWorker(
      {
        kind: 'embed',
        config: this.workerConfig,
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

  public close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.tokenRingReader = null;
    this.tokenRingDrainScheduled = false;
    this.activeTokenCallCount = 0;
    this.callIdByNativeRequestId.clear();
    this.pendingTokenRecordsByNativeRequestId.clear();
    this.streamStatsByCallId.clear();
    for (const pending of this.pendingCalls.values()) {
      pending.reject(new QueryError('ENGINE_CLOSED', 'CogentClient is closed.'));
    }
    this.pendingCalls.clear();

    if (this.worker == null) {
      this.currentSnapshot = null;
      this.observability.markClosed();
      this.emitEngineEvent({ type: 'closed' });
      return;
    }

    this.worker.terminate();
    this.worker = null;
    this.currentSnapshot = null;
    this.observability.markClosed();
    this.emitEngineEvent({ type: 'closed' });
  }

  private assertOpen(): void {
    if (this.closed) {
      throw new QueryError('ENGINE_CLOSED', 'CogentClient is closed.');
    }
  }

  private scheduleTokenRingDrain(): void {
    if (
      this.tokenRingDrainScheduled ||
      this.activeTokenCallCount === 0 ||
      this.tokenRingReader == null
    ) {
      return;
    }
    this.tokenRingDrainScheduled = true;
    const drain = () => {
      this.tokenRingDrainScheduled = false;
      this.drainTokenRing();
      this.scheduleTokenRingDrain();
    };
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(drain);
      return;
    }
    setTimeout(drain, 16);
  }

  private drainTokenRing(): void {
    const reader = this.tokenRingReader;
    if (reader == null) {
      return;
    }
    reader.drain((streamId, sequenceStart, frameCount, byteCount, text) => {
      const callId = this.callIdByNativeRequestId.get(streamId);
      const recordDrainStart = performance.now();
      const drainMs = performance.now() - recordDrainStart;
      if (callId == null) {
        this.bufferPendingTokenRecord({
          streamId,
          sequenceStart,
          frameCount,
          byteCount,
          text,
          drainMs,
        });
        return;
      }
      this.deliverTokenBatch(
        callId,
        streamId,
        sequenceStart,
        frameCount,
        byteCount,
        text,
        drainMs
      );
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
      this.deliverTokenRecord(callId, record);
    }
  }

  private deliverTokenRecord(callId: number, record: PendingTokenRecord): void {
    this.deliverTokenBatch(
      callId,
      record.streamId,
      record.sequenceStart,
      record.frameCount,
      record.byteCount,
      record.text,
      record.drainMs
    );
  }

  private deliverTokenBatch(
    callId: number,
    streamId: number,
    sequenceStart: number,
    frameCount: number,
    byteCount: number,
    text: string,
    drainMs: number
  ): void {
    if (text.length === 0) {
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
      drainMs: 0,
      drainCalls: 0,
    };
    stats.framesSent += frameCount;
    stats.bytesSent += byteCount;
    stats.batchesSent += 1;
    stats.drainMs += drainMs;
    stats.drainCalls += 1;
    this.streamStatsByCallId.set(callId, stats);
    pending.tokenBatchSink({
      requestId: String(streamId),
      streamId,
      sequenceStart,
      text,
      frameCount,
      byteCount,
      stats: { ...stats },
    });
  }

  private forgetStreamingCall(callId: number): void {
    for (const [nativeRequestId, mappedCallId] of this.callIdByNativeRequestId) {
      if (mappedCallId === callId) {
        this.callIdByNativeRequestId.delete(nativeRequestId);
        this.pendingTokenRecordsByNativeRequestId.delete(nativeRequestId);
      }
    }
    this.streamStatsByCallId.delete(callId);
  }

  private ensureWorker(): Worker {
    if (this.worker != null) {
      return this.worker;
    }
    const optimizedWorkerUrl = getOptimizedDefaultWorkerUrl();
    this.worker =
      this.config.workerUrl == null
        ? optimizedWorkerUrl == null
          ? new Worker(new URL('./model-service-entry.js', import.meta.url), { type: 'module' })
          : new Worker(optimizedWorkerUrl, { type: 'module' })
        : new Worker(this.config.workerUrl, { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<WorkerResponseMessage>) => {
      this.handleWorkerMessage(event.data);
    };
    this.worker.onerror = (event: ErrorEvent) => {
      this.failWorker(new Error(event.message || 'Worker runtime crashed.'));
    };
    this.worker.onmessageerror = () => {
      this.failWorker(new Error('Worker runtime failed to deserialize a message.'));
    };
    return this.worker;
  }

  private failWorker(error: unknown): void {
    if (this.worker != null) {
      this.worker.onmessage = null;
      this.worker.onerror = null;
      this.worker.onmessageerror = null;
      this.worker.terminate();
      this.worker = null;
    }
    this.tokenRingReader = null;
    this.tokenRingDrainScheduled = false;
    this.activeTokenCallCount = 0;
    this.callIdByNativeRequestId.clear();
    this.pendingTokenRecordsByNativeRequestId.clear();
    this.streamStatsByCallId.clear();
    for (const pending of this.pendingCalls.values()) {
      pending.reject(error);
    }
    this.pendingCalls.clear();
    this.currentSnapshot = null;
    this.observability.emit('error', {
      state: 'error',
      model: null,
      query: null,
    });
    this.emitEngineEvent({ type: 'state', state: this.snapshotState() });
  }

  private postWorkerMessage(message: WorkerRequestMessage): void {
    this.ensureWorker().postMessage(message);
  }

  private callWorker<T extends RequestWithCallId>(
    message: WithoutCallId<T>,
    options: WorkerCallOptions = {}
  ): Promise<unknown> {
    if (options.signal?.aborted) {
      throw createAbortError('Operation aborted.');
    }

    const callId = this.nextCallId++;
    const request = {
      ...message,
      callId,
    } as T;

    let cleanup = (): void => {};
    if (options.signal != null) {
      const abortListener = () => {
        this.postWorkerMessage({
          kind: 'cancel',
          targetCallId: callId,
        });
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
      const finalize = () => {
        if (options.emitTokens === true) {
          this.drainTokenRing();
          this.forgetStreamingCall(callId);
          this.activeTokenCallCount = Math.max(0, this.activeTokenCallCount - 1);
          if (this.activeTokenCallCount === 0) {
            this.pendingTokenRecordsByNativeRequestId.clear();
          }
        }
        cleanup();
        this.pendingCalls.delete(callId);
      };
      this.pendingCalls.set(callId, {
        resolve: (value) => {
          finalize();
          resolve(value);
        },
        reject: (error) => {
          finalize();
          reject(error);
        },
        onProgress: options.onProgress,
        tokenBatchSink: options.tokenBatchSink,
      });
      try {
        this.postWorkerMessage(request);
      } catch (error) {
        finalize();
        this.pendingCalls.delete(callId);
        reject(error);
      }
    });
  }

  private handleWorkerMessage(message: WorkerResponseMessage): void {
    if (message.kind === 'load-progress') {
      this.pendingCalls.get(message.callId)?.onProgress?.(message.progress);
      return;
    }

    if (message.kind === 'token-ring-ready') {
      this.tokenRingReader = new SharedTokenRingReader(message.descriptor);
      this.scheduleTokenRingDrain();
      return;
    }

    if (message.kind === 'token-ring-claim') {
      this.callIdByNativeRequestId.set(message.nativeRequestId, message.callId);
      this.flushPendingTokenRecords(message.nativeRequestId, message.callId);
      this.scheduleTokenRingDrain();
      return;
    }

    if (message.kind === 'token-batch') {
      this.pendingCalls.get(message.callId)?.tokenBatchSink?.(message.batch);
      return;
    }

    if (message.kind === 'observability-event') {
      this.observability.ingest(message.event);
      this.currentSnapshot =
        message.event.snapshot.state === 'closed' ? null : message.event.snapshot.model;
      return;
    }

    if (message.kind === 'engine-event') {
      this.emitEngineEvent(message.event);
      return;
    }

    const pending = this.pendingCalls.get(message.callId);
    if (pending == null) {
      return;
    }

    if (message.kind === 'resolve') {
      pending.resolve(message.value);
      return;
    }

    pending.reject(this.deserializeError(message));
  }

  private deserializeError(message: Extract<WorkerResponseMessage, { kind: 'reject' }>): unknown {
    if (message.queryErrorCode != null) {
      return new QueryError(message.queryErrorCode, message.message);
    }
    if (message.errorName === 'AbortError') {
      return new DOMException(message.message, 'AbortError');
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
