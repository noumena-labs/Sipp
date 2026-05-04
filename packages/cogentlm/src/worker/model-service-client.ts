import type { CogentConfig } from '../cogent-config.js';
import { resolveRuntimeUrls } from '../runtime-assets.js';
import { resolveOptimizedPackageAssetUrl } from '../runtime/package-assets.js';
import { ObservabilityController } from '../model-management/observability-controller.js';
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
  type ObservabilityEvent,
  type ObservabilitySnapshot,
  type ModelInfo,
  type ModelLoadOptions,
  type ModelSource,
  type ChatInput,
  type ChatOptions,
  type QueryInput,
  type QueryOptions,
} from '../model-management/model-types.js';
import type { ModelLifecycleService } from '../model-management/model-service-contract.js';

interface PendingWorkerCall {
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  onProgress?: ModelLoadOptions['onProgress'];
  onToken?: QueryOptions['onToken'] | ChatOptions['onToken'];
}

interface WorkerCallOptions {
  signal?: AbortSignal;
  onProgress?: ModelLoadOptions['onProgress'];
  onToken?: QueryOptions['onToken'] | ChatOptions['onToken'];
}

type RequestWithCallId = Extract<WorkerRequestMessage, { callId: number }>;
type WithoutCallId<T> = T extends { callId: number } ? Omit<T, 'callId'> : never;

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
  };
}

function toWorkerChatOptions(options: ChatOptions = {}): WorkerChatOptions {
  return {
    session: options.session,
    maxTokens: options.maxTokens,
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
    return (await this.callWorker(
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

  public async chat(input: ChatInput, options: ChatOptions = {}): Promise<string> {
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
        onToken: options.onToken,
      }
    )) as string;
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
      this.pendingCalls.set(callId, {
        resolve: (value) => {
          cleanup();
          resolve(value);
        },
        reject: (error) => {
          cleanup();
          reject(error);
        },
        onProgress: options.onProgress,
        onToken: options.onToken,
      });
      try {
        this.postWorkerMessage(request);
      } catch (error) {
        cleanup();
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
