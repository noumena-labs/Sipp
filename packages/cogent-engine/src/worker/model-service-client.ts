import type { CogentConfig } from '../cogent-config.js';
import { ObservabilityController } from '../model-management/observability-controller.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  type WorkerSerializableCogentConfig,
} from './model-service-protocol.js';
import {
  QueryError,
  type ObservabilityEvent,
  type ObservabilitySnapshot,
  type ModelInfo,
  type ModelLoadOptions,
  type ModelSource,
  type QueryInput,
  type QueryOptions,
} from '../model-management/model-types.js';
import type { ModelLifecycleService } from '../model-management/model-service-contract.js';

interface PendingWorkerCall {
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  onProgress?: ModelLoadOptions['onProgress'];
  onToken?: QueryOptions['onToken'];
}

type RequestWithCallId = Extract<WorkerRequestMessage, { callId: number }>;
type WithoutCallId<T> = T extends { callId: number } ? Omit<T, 'callId'> : never;

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
  };
}

function toWorkerQueryOptions(options: QueryOptions = {}): QueryOptions {
  return {
    session: options.session,
    maxTokens: options.maxTokens,
    format: options.format,
    grammar: options.grammar,
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
  private readonly pendingCalls = new Map<number, PendingWorkerCall>();
  private readonly workerConfig: WorkerSerializableCogentConfig;

  constructor(private readonly config: CogentConfig = {}) {
    this.workerConfig = toWorkerSerializableConfig(config);
  }

  public async load(source: ModelSource, options: ModelLoadOptions = {}): Promise<ModelInfo> {
    this.assertOpen();
    const result = (await this.callWorkerWithAbort(
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

  public async query(input: QueryInput, options: QueryOptions = {}): Promise<string> {
    this.assertOpen();
    return (await this.callWorkerWithAbort(
      {
        kind: 'query',
        config: this.workerConfig,
        input,
        options: toWorkerQueryOptions(options),
      },
      {
        signal: options.signal,
        onToken: options.onToken,
      }
    )) as string;
  }

  public async applyChatTemplate(
    messages: Array<{ role: string; content: string }>,
    addAssistant: boolean
  ): Promise<string> {
    this.assertOpen();
    return (await this.callWorker({
      kind: 'apply-chat-template',
      config: this.workerConfig,
      messages,
      addAssistant,
    })) as string;
  }

  public getChatTemplate(): string | null {
    this.assertOpen();
    return this.currentSnapshot?.chatTemplate ?? null;
  }

  public getBosText(): string {
    this.assertOpen();
    return this.currentSnapshot?.bosText ?? '';
  }

  public getEosText(): string {
    this.assertOpen();
    return this.currentSnapshot?.eosText ?? '';
  }

  public getMediaMarker(): string | null {
    this.assertOpen();
    return this.currentSnapshot?.mediaMarker ?? null;
  }

  public currentObservability(): ObservabilitySnapshot {
    this.assertOpen();
    return this.observability.current();
  }

  public subscribeObservability(listener: (event: ObservabilityEvent) => void): () => void {
    this.assertOpen();
    return this.observability.subscribe(listener);
  }

  public close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    for (const pending of this.pendingCalls.values()) {
      pending.reject(new QueryError('ENGINE_CLOSED', 'CogentEngine is closed.'));
    }
    this.pendingCalls.clear();

    if (this.worker == null) {
      this.currentSnapshot = null;
      this.observability.markClosed();
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
    const workerUrl =
      this.config.workerUrl ??
      new URL('./model-service-entry.js', import.meta.url).toString();
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
  }

  private postWorkerMessage(message: WorkerRequestMessage): void {
    this.ensureWorker().postMessage(message);
  }

  private callWorker<T extends RequestWithCallId>(message: WithoutCallId<T>): Promise<unknown> {
    const callId = this.nextCallId++;
    const request = {
      ...message,
      callId,
    } as T;

    return new Promise<unknown>((resolve, reject) => {
      this.pendingCalls.set(callId, { resolve, reject });
      this.postWorkerMessage(request);
    });
  }

  private callWorkerWithAbort<T extends RequestWithCallId>(
    message: WithoutCallId<T>,
    options: {
      signal?: AbortSignal;
      onProgress?: ModelLoadOptions['onProgress'];
      onToken?: QueryOptions['onToken'];
    }
  ): Promise<unknown> {
    if (options.signal?.aborted) {
      throw new DOMException('Operation aborted.', 'AbortError');
    }

    const callId = this.nextCallId++;
    const request = {
      ...message,
      callId,
    } as T;

    const abortListener =
      options.signal == null
        ? null
        : () => {
            this.postWorkerMessage({
              kind: 'cancel',
              targetCallId: callId,
            });
          };

    if (abortListener != null) {
      options.signal?.addEventListener('abort', abortListener, { once: true });
    }

    return new Promise<unknown>((resolve, reject) => {
      this.pendingCalls.set(callId, {
        resolve: (value) => {
          if (abortListener != null) {
            options.signal?.removeEventListener('abort', abortListener);
          }
          resolve(value);
        },
        reject: (error) => {
          if (abortListener != null) {
            options.signal?.removeEventListener('abort', abortListener);
          }
          reject(error);
        },
        onProgress: options.onProgress,
        onToken: options.onToken,
      });
      this.postWorkerMessage(request);
    });
  }

  private handleWorkerMessage(message: WorkerResponseMessage): void {
    if (message.kind === 'load-progress') {
      this.pendingCalls.get(message.callId)?.onProgress?.(message.progress);
      return;
    }

    if (message.kind === 'token') {
      this.pendingCalls.get(message.callId)?.onToken?.(message.text);
      return;
    }

    if (message.kind === 'observability-event') {
      this.observability.ingest(message.event);
      this.currentSnapshot =
        message.event.snapshot.state === 'closed' ? null : message.event.snapshot.model;
      return;
    }

    const pending = this.pendingCalls.get(message.callId);
    if (pending == null) {
      return;
    }
    this.pendingCalls.delete(message.callId);

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
}
