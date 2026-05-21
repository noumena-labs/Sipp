import type { CogentConfig } from '../engine/engine-options.js';
import { resolveRuntimeUrls } from '../engine/runtime-assets.js';
import { resolveOptimizedPackageAssetUrl } from '../runtime/package-assets.js';
import { ObservabilityController } from '../models/observability-controller.js';
import {
  StreamingRingReader,
  createStreamingRingBuffer,
  DEFAULT_STREAMING_RING_CAPACITY,
} from '../runtime/streaming-ring.js';
import { createAbortError } from '../utils/abort.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  type WorkerChatOptions,
  type WorkerQueryOptions,
  type WorkerSerializableCogentConfig,
} from './model-service-protocol.js';
import {
  QueryError,
  type EngineEvent,
  type EngineState,
  type ObservabilityEvent,
  type ObservabilitySnapshot,
  type ModelInfo,
  type ModelLoadOptions,
  type ModelSource,
  type RequestResult,
  type ChatInput,
  type ChatOptions,
  type QueryInput,
  type QueryOptions,
  type TokenBatch,
  type TokenFlushMode,
} from '../models/types.js';
import { observabilitySnapshotToEngineState } from '../models/engine-protocol-adapter.js';
import type { ModelLifecycleService } from '../models/contract.js';

interface PendingWorkerCall {
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  onProgress?: ModelLoadOptions['onProgress'];
  onTokens?: QueryOptions['onTokens'] | ChatOptions['onTokens'];
  tokenFlush?: TokenFlushMode;
}

interface WorkerCallOptions {
  signal?: AbortSignal;
  onProgress?: ModelLoadOptions['onProgress'];
  onTokens?: QueryOptions['onTokens'] | ChatOptions['onTokens'];
  tokenFlush?: TokenFlushMode;
}

type RequestWithCallId = Extract<WorkerRequestMessage, { callId: number }>;
type WithoutCallId<T> = T extends { callId: number } ? Omit<T, 'callId'> : never;
const TOKEN_FLUSH_DRAIN_BUDGET = 64;

function utf8ByteLength(text: string): number {
  return new TextEncoder().encode(text).byteLength;
}

export function getOptimizedDefaultWorkerUrl(importerUrl: string = import.meta.url): string | null {
  return resolveOptimizedPackageAssetUrl('dist/esm/worker/model-service-entry.js', importerUrl);
}

function toWorkerSerializableConfig(config: CogentConfig): WorkerSerializableCogentConfig {
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

  const runtimeUrls =
    config.moduleUrl == null && config.wasmUrl == null
      ? null
      : resolveRuntimeUrls(config);

  return {
    moduleUrl: runtimeUrls?.moduleUrl,
    wasmUrl: runtimeUrls?.wasmUrl,
    moduleOptions: config.moduleOptions,
    maxModelBytes: config.maxModelBytes,
    trustedOrigins: config.trustedOrigins,
  };
}

function toWorkerQueryOptions(options: QueryOptions = {}): WorkerQueryOptions {
  return {
    session: options.session,
    maxTokens: options.maxTokens,
    grammar: options.grammar,
    tokenFlush: options.onTokens == null ? undefined : options.tokenFlush ?? 'token',
    // Carry the caller's streaming intent across the worker boundary.  When
    // false, the worker leaves engine emission_mode at NONE — required to get
    // a real native baseline TPS comparison from a worker-backed engine.
    streaming: options.onTokens != null,
  };
}

function toWorkerChatOptions(options: ChatOptions = {}): WorkerChatOptions {
  return {
    session: options.session,
    maxTokens: options.maxTokens,
    grammar: options.grammar,
    tokenFlush: options.onTokens == null ? undefined : options.tokenFlush ?? 'token',
    streaming: options.onTokens != null,
  };
}

function toWorkerModelLoadOptions(options: ModelLoadOptions = {}): ModelLoadOptions {
  return {
    observability: options.observability,
    runtime: options.runtime,
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
  private readonly workerConfig: WorkerSerializableCogentConfig;
  // SAB streaming ring, lazily allocated when first needed.  Null when
  // COOP/COEP is missing; streaming requests will error in that case.
  private streamingRingBuffer: SharedArrayBuffer | null = null;
  private streamingRingReader: StreamingRingReader | null = null;
  private streamingActiveCount = 0;
  // Native request id → worker callId, populated by `streaming-claim`.
  private readonly callIdByNativeRequestId = new Map<number, number>();
  private readonly streamStatsByCallId = new Map<
    number,
    { framesSent: number; bytesSent: number; batchesSent: number }
  >();

  constructor(private readonly config: CogentConfig = {}) {
    this.workerConfig = toWorkerSerializableConfig(config);
  }

  public async load(source: ModelSource, options: ModelLoadOptions = {}): Promise<ModelInfo> {
    this.assertOpen();
    const result = (await this.callWorker(
      {
        kind: 'models-load',
        config: this.workerConfig,
        source,
        options: toWorkerModelLoadOptions(options),
      },
      {
        signal: options.signal,
        onProgress: options.onProgress,
      }
    )) as ModelInfo;
    this.currentSnapshot = result.loaded ? result : null;
    return result;
  }

  public currentModel(): ModelInfo | null {
    this.assertOpen();
    return this.currentSnapshot;
  }

  public current(): ModelInfo | null {
    return this.currentModel();
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

  public async query(input: QueryInput, options: QueryOptions = {}): Promise<RequestResult> {
    this.assertOpen();
    return (await this.callWorker(
      {
        kind: 'query',
        config: this.workerConfig,
        input,
        options: toWorkerQueryOptions(options),
      },
      {
        signal: options.signal,
        onTokens: options.onTokens,
        tokenFlush: options.onTokens == null ? undefined : options.tokenFlush ?? 'token',
      }
    )) as RequestResult;
  }

  public async queryResult(input: QueryInput, options: QueryOptions = {}): Promise<RequestResult> {
    this.assertOpen();
    return (await this.callWorker(
      {
        kind: 'query-result',
        config: this.workerConfig,
        input,
        options: toWorkerQueryOptions(options),
      },
      {
        signal: options.signal,
        onTokens: options.onTokens,
        tokenFlush: options.onTokens == null ? undefined : options.tokenFlush ?? 'token',
      }
    )) as RequestResult;
  }

  public async chat(input: ChatInput, options: ChatOptions = {}): Promise<RequestResult> {
    this.assertOpen();
    return (await this.callWorker(
      {
        kind: 'chat',
        config: this.workerConfig,
        input,
        options: toWorkerChatOptions(options),
      },
      {
        signal: options.signal,
        onTokens: options.onTokens,
        tokenFlush: options.onTokens == null ? undefined : options.tokenFlush ?? 'token',
      }
    )) as RequestResult;
  }

  public async chatResult(input: ChatInput, options: ChatOptions = {}): Promise<RequestResult> {
    this.assertOpen();
    return (await this.callWorker(
      {
        kind: 'chat-result',
        config: this.workerConfig,
        input,
        options: toWorkerChatOptions(options),
      },
      {
        signal: options.signal,
        onTokens: options.onTokens,
        tokenFlush: options.onTokens == null ? undefined : options.tokenFlush ?? 'token',
      }
    )) as RequestResult;
  }

  public currentObservability(): ObservabilitySnapshot {
    this.assertOpen();
    return this.observability.current();
  }

  public subscribeObservability(listener: (event: ObservabilityEvent) => void): () => void {
    this.assertOpen();
    return this.observability.subscribe(listener);
  }

  public currentState(): EngineState {
    this.assertOpen();
    return this.currentStateUnsafe();
  }

  public state(): EngineState {
    return this.currentState();
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
    // Tear down the streaming poll so it doesn't keep `this` reachable.
    this.streamingActiveCount = 0;
    this.streamingRingReader = null;
    this.streamingRingBuffer = null;
    this.callIdByNativeRequestId.clear();
    this.streamStatsByCallId.clear();
    for (const pending of this.pendingCalls.values()) {
      pending.reject(new QueryError('ENGINE_CLOSED', 'CogentEngine is closed.'));
    }
    this.pendingCalls.clear();

    if (this.worker == null) {
      this.currentSnapshot = null;
      this.observability.markClosed();
      this.emitEngineEvent({ type: 'closed' });
      return;
    }

    try {
      this.postWorkerMessage({
        kind: 'close',
        callId: this.nextCallId++,
      });
    } catch {
      // The worker is being terminated locally; close notification is best-effort.
    } finally {
      this.worker.terminate();
      this.worker = null;
      this.currentSnapshot = null;
      this.observability.markClosed();
      this.emitEngineEvent({ type: 'closed' });
    }
  }

  private assertOpen(): void {
    if (this.closed) {
      throw new QueryError('ENGINE_CLOSED', 'CogentEngine is closed.');
    }
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
    // Tell the worker about the SAB ring (or null) before any operation.
    this.ensureStreamingRing();
    this.worker.postMessage({
      kind: 'streaming-init',
      ringBuffer: this.streamingRingBuffer,
    } satisfies WorkerRequestMessage);
    return this.worker;
  }

  private ensureStreamingRing(): void {
    if (this.streamingRingBuffer != null) {
      return;
    }
    if (typeof SharedArrayBuffer === 'undefined') {
      return;
    }
    try {
      this.streamingRingBuffer = createStreamingRingBuffer(
        DEFAULT_STREAMING_RING_CAPACITY
      );
      this.streamingRingReader = new StreamingRingReader(this.streamingRingBuffer);
    } catch {
      this.streamingRingBuffer = null;
      this.streamingRingReader = null;
    }
  }

  private assertWorkerStreamingSupported(): void {
    this.ensureStreamingRing();
    if (this.streamingRingBuffer == null || this.streamingRingReader == null) {
      throw new QueryError(
        'STREAMING_UNAVAILABLE',
        'Worker streaming requires SharedArrayBuffer. Enable cross-origin isolation or run without onTokens.'
      );
    }
  }

  private registerStreamingCall(): void {
    this.streamingActiveCount += 1;
  }

  private unregisterStreamingCall(): void {
    if (this.streamingActiveCount > 0) {
      this.streamingActiveCount -= 1;
    }
  }



  // Drains the SAB ring and consolidation tokens into batches per request.
  // Invoked from the 'streaming-tick' macrotask and one final time from
  // the call finalizer to capture tail tokens.
  private drainStreamingRing(): void {
    const reader = this.streamingRingReader;
    if (reader == null) {
      return;
    }
    const batches = new Map<number, { nativeRequestId: number; texts: string[]; firstSequence: number }>();
    const maxMessages = this.hasTokenFlushCall() ? TOKEN_FLUSH_DRAIN_BUDGET : undefined;
    for (const { requestId, sequence, text } of reader.drain(maxMessages)) {
      const callId = this.callIdByNativeRequestId.get(requestId);
      if (callId == null) continue;
      const pending = this.pendingCalls.get(callId);
      if (pending?.tokenFlush === 'token') {
        this.deliverStreamingBatch(callId, requestId, sequence, [text]);
        continue;
      }
      let batch = batches.get(callId);
      if (batch == null) {
        batch = { nativeRequestId: requestId, texts: [], firstSequence: sequence };
        batches.set(callId, batch);
      }
      batch.texts.push(text);
    }

    for (const [callId, tokenBatch] of batches) {
      this.deliverStreamingBatch(
        callId,
        tokenBatch.nativeRequestId,
        tokenBatch.firstSequence,
        tokenBatch.texts
      );
    }
  }

  private hasTokenFlushCall(): boolean {
    for (const pending of this.pendingCalls.values()) {
      if (pending.tokenFlush === 'token') {
        return true;
      }
    }
    return false;
  }

  private deliverStreamingBatch(
    callId: number,
    nativeRequestId: number,
    sequenceStart: number,
    texts: string[]
  ): void {
    const pending = this.pendingCalls.get(callId);
    if (pending == null || texts.length === 0) {
      return;
    }
    const text = texts.join('');
    const byteCount = utf8ByteLength(text);
    const stats = this.streamStatsByCallId.get(callId) ?? {
      framesSent: 0,
      bytesSent: 0,
      batchesSent: 0,
    };
    stats.framesSent += texts.length;
    stats.bytesSent += byteCount;
    stats.batchesSent += 1;
    this.streamStatsByCallId.set(callId, stats);
    const batch: TokenBatch = {
      requestId: String(nativeRequestId),
      streamId: nativeRequestId,
      sequenceStart,
      text,
      frameCount: texts.length,
      byteCount,
      stats: {
        ...stats,
        framesDropped: 0,
      },
    };
    try {
      pending.onTokens?.(batch);
    } catch {
      /* user error */
    }
  }

  private failWorker(error: unknown): void {
    if (this.worker != null) {
      this.worker.onmessage = null;
      this.worker.onerror = null;
      this.worker.onmessageerror = null;
      this.worker.terminate();
      this.worker = null;
    }
    // Reset streaming state; the next worker spawn allocates a fresh ring.
    this.streamingActiveCount = 0;
    this.streamingRingReader = null;
    this.streamingRingBuffer = null;
    this.callIdByNativeRequestId.clear();
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
    this.emitEngineEvent({ type: 'state', state: this.currentStateUnsafe() });
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

    const isStreaming = options.onTokens != null;
    if (isStreaming) {
      this.assertWorkerStreamingSupported();
      this.registerStreamingCall();
    }

    return new Promise<unknown>((resolve, reject) => {
      const finalize = () => {
        if (isStreaming) {
          // Drain one last time to catch tokens that arrived just before or
          // along with the final resolution message.
          this.drainStreamingRing();
          for (const [nativeId, mappedCallId] of this.callIdByNativeRequestId) {
            if (mappedCallId === callId) {
              this.streamingRingReader?.forgetRequest(nativeId);
              this.callIdByNativeRequestId.delete(nativeId);
              break;
            }
          }
          this.streamStatsByCallId.delete(callId);
          this.unregisterStreamingCall();
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
        onTokens: options.onTokens,
        tokenFlush: isStreaming ? options.tokenFlush ?? 'token' : undefined,
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

    if (message.kind === 'streaming-tick') {
      this.drainStreamingRing();
      return;
    }

    if (message.kind === 'streaming-claim') {
      this.callIdByNativeRequestId.set(message.nativeRequestId, message.callId);
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

  private currentStateUnsafe(): EngineState {
    return observabilitySnapshotToEngineState(this.observability.current());
  }
}
