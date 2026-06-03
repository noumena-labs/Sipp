import { ModelService } from '../models/model-service.js';
import { AssetStore, type BrowserCachePolicyOptions } from '../models/asset-store.js';
import { createBrowserEmbeddingRun, createBrowserTextRun } from '../models/token-queue.js';
import {
  QueryError,
  type CogentClient as CogentClientShape,
  type BrowserEmbeddingRun,
  type BrowserTextRun,
  type ChatInput,
  type ChatOptions,
  type EmbedOptions,
  type EndpointRef,
  type EngineEvent,
  type EngineObservability,
  type EngineState,
  type ModelLifecycleService,
  type ModelInfo,
  type ModelLoadOptions,
  type ModelSource,
  type QueryInput,
  type QueryOptions,
  type RemoteGatewayConfig,
} from '../models/types.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import { WorkerModelServiceClient } from '../worker/model-service-client.js';
import type { BackendObservability } from './inference-types.js';
import {
  RemoteGatewayRegistry,
  runRemoteChat,
  runRemoteEmbedding,
  runRemoteQuery,
} from './remote-gateway.js';

export interface EngineModuleOptions {
  locateFile?: (path: string, prefix?: string) => string;
  [key: string]: unknown;
}

export interface CogentClientOptions {
  moduleUrl?: string;
  wasmUrl?: string;
  pthreadModuleUrl?: string;
  pthreadWasmUrl?: string;
  wasmThreading?: 'single-thread' | 'pthread';
  moduleOptions?: EngineModuleOptions;
  maxModelBytes?: number;
  /** Override browser OPFS split thresholds for large GGUF model files. */
  browserCache?: BrowserCachePolicyOptions;
  trustedOrigins?: string[];
  executionMode?: 'auto' | 'worker' | 'main-thread';
  workerUrl?: string;
}

export interface BrowserGgufIngestSmokeResult {
  available: boolean;
  layoutForLargeFile: 'single-file' | 'split-gguf' | null;
  plannedShardCount: number | null;
  streamedShardCount: number;
  streamedBytes: number;
  error: string | null;
}

export interface BrowserRustEngineSmokeResult {
  available: boolean;
  abiVersion: number;
  engineId: number | null;
  error: string | null;
}

export interface BrowserRuntimeSmokeResult {
  rustEngine: BrowserRustEngineSmokeResult;
  ggufIngest: BrowserGgufIngestSmokeResult;
  backend: BackendObservability | null;
  webgpuReady: boolean;
}

function shouldUseWorker(config: CogentClientOptions): boolean {
  if (config.executionMode === 'main-thread') {
    return false;
  }
  if (config.executionMode === 'worker') {
    return true;
  }

  return (
    typeof window !== 'undefined' &&
    typeof document !== 'undefined' &&
    typeof Worker !== 'undefined'
  );
}

/**
 * Browser application client that owns one local model lifecycle service.
 */
export class CogentClient implements CogentClientShape {
  public readonly observability: EngineObservability;
  #service: ModelLifecycleService;
  #remotes = new RemoteGatewayRegistry();
  #closed = false;

  public constructor(options: CogentClientOptions = {}) {
    this.#service = shouldUseWorker(options)
      ? new WorkerModelServiceClient(options)
      : new ModelService(
        new MainThreadEngineRuntime(options),
        undefined,
        new AssetStore(undefined, options.browserCache)
      );
    this.observability = {
      current: () => {
        this.assertOpen();
        return this.#service.currentObservability();
      },
      subscribe: (listener) => {
        this.assertOpen();
        return this.#service.subscribeObservability(listener);
      },
    };
  }

  public static async browserRuntimeSmoke(
    options: CogentClientOptions = {}
  ): Promise<BrowserRuntimeSmokeResult> {
    const runtime = new MainThreadEngineRuntime({
      ...options,
      executionMode: 'main-thread',
    });
    try {
      return await runtime.runBrowserRuntimeSmoke();
    } finally {
      runtime.close();
    }
  }

  /**
   * Load a local model and make it the current local endpoint.
   */
  public addLocal(source: ModelSource, options?: ModelLoadOptions): Promise<ModelInfo> {
    this.assertOpen();
    return this.#service.load(source, options);
  }

  /**
   * Return the currently loaded local model, if one is active.
   */
  public currentLocal(): ModelInfo | null {
    this.assertOpen();
    return this.#service.current();
  }

  /**
   * List installed local models.
   */
  public listLocal(): Promise<ModelInfo[]> {
    this.assertOpen();
    return this.#service.list();
  }

  /**
   * Remove an installed local model by id.
   */
  public async removeLocal(id: string): Promise<void> {
    this.assertOpen();
    await this.#service.remove(id);
  }

  public addRemote(id: string, config: RemoteGatewayConfig): EndpointRef {
    this.assertOpen();
    return this.#remotes.add(id, config);
  }

  public updateRemote(id: string, config: RemoteGatewayConfig): EndpointRef {
    this.assertOpen();
    return this.#remotes.update(id, config);
  }

  public query(input: QueryInput, options: QueryOptions = {}): BrowserTextRun {
    this.assertOpen();
    const remote = this.#remotes.get(options.endpoint);
    if (remote != null) {
      return createBrowserTextRun(options, (tokenBatchSink, signal) =>
        runRemoteQuery(remote, input, options, tokenBatchSink, signal)
      );
    }
    const localOptions = localQueryOptions(options);
    return createBrowserTextRun(localOptions, (tokenBatchSink, signal) =>
      this.#service.runQuery(input, { ...localOptions, signal, tokenBatchSink })
    );
  }

  public chat(input: ChatInput, options: ChatOptions = {}): BrowserTextRun {
    this.assertOpen();
    const remote = this.#remotes.get(options.endpoint);
    if (remote != null) {
      return createBrowserTextRun(options, (tokenBatchSink, signal) =>
        runRemoteChat(remote, input, options, tokenBatchSink, signal)
      );
    }
    const localOptions = localQueryOptions(options);
    return createBrowserTextRun(localOptions, (tokenBatchSink, signal) =>
      this.#service.runChat(input, { ...localOptions, signal, tokenBatchSink })
    );
  }

  public embed(input: string, options: EmbedOptions = {}): BrowserEmbeddingRun {
    this.assertOpen();
    const remote = this.#remotes.get(options.endpoint);
    if (remote != null) {
      return createBrowserEmbeddingRun(options.signal, (signal) =>
        runRemoteEmbedding(remote, input, options, signal)
      );
    }
    const localOptions = localEmbedOptions(options);
    return createBrowserEmbeddingRun(localOptions.signal, (signal) =>
      this.#service.runEmbedding(input, { ...localOptions, signal })
    );
  }

  public state(): EngineState {
    this.assertOpen();
    return this.#service.state();
  }

  public subscribeEvents(listener: (event: EngineEvent) => void): () => void {
    this.assertOpen();
    return this.#service.subscribeEvents(listener);
  }

  public async close(): Promise<void> {
    this.assertOpen();
    this.#closed = true;
    await this.#service.close();
  }

  private assertOpen(): void {
    if (this.#closed) {
      throw new QueryError('ENGINE_CLOSED', 'CogentClient is closed.');
    }
  }
}

function localQueryOptions(options: QueryOptions): QueryOptions {
  const { endpoint: _endpoint, gatewayOptions: _gatewayOptions, ...localOptions } = options;
  return localOptions;
}

function localEmbedOptions(options: EmbedOptions): EmbedOptions {
  const { endpoint: _endpoint, gatewayOptions: _gatewayOptions, ...localOptions } = options;
  return localOptions;
}
