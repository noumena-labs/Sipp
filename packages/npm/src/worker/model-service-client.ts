import type { CogentClientOptions } from '../engine/browser-client.js';
import {
  resolveOptimizedPackageAssetUrl,
  resolveRuntimeUrls,
} from '../engine/runtime-assets.js';
import { ObservabilityController } from '../models/observability-controller.js';
import { observabilitySnapshotToEngineState } from '../models/observability-controller.js';
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

  return {
    moduleUrl: runtimeUrls?.moduleUrl,
    wasmUrl: runtimeUrls?.wasmUrl,
    wasmThreading: runtimeUrls?.threading ?? config.wasmThreading,
    moduleOptions: config.moduleOptions,
    maxModelBytes: config.maxModelBytes,
    trustedOrigins: config.trustedOrigins,
  };
}

function toWorkerQueryOptions(
  options: QueryOptions = {},
  emitTokens: boolean
): WorkerQueryOptions {
  return {
    session: options.session,
    maxTokens: options.maxTokens,
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

    return new Promise<unknown>((resolve, reject) => {
      const finalize = () => {
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
