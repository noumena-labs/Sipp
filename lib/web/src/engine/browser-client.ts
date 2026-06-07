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
  type EndpointDescriptor,
  type EndpointRef,
  type EngineEvent,
  type EngineObservability,
  type EngineState,
  type ModelLifecycleService,
  type ModelInfo,
  type QueryInput,
  type QueryOptions,
  type ProviderEndpointDescriptor,
} from '../models/types.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import { WorkerModelServiceClient } from '../worker/model-service-client.js';
import type { BackendObservability } from './inference-types.js';
import {
  GatewayEndpointRegistry,
  runGatewayChat,
  runGatewayEmbedding,
  runGatewayQuery,
} from './gateway-endpoint.js';
import {
  ProviderEndpointRegistry,
  runProviderChat,
  runProviderEmbedding,
  runProviderQuery,
} from './provider-endpoint.js';

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
  #gatewayEndpoints = new GatewayEndpointRegistry();
  #providers = new ProviderEndpointRegistry();
  #localEndpoint: EndpointRef | null = null;
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
   * Registers or replaces an endpoint after its descriptor is validated.
   *
   * Replacing an endpoint with a different kind invalidates prior references
   * for the same id.
   */
  public async add(id: string, descriptor: EndpointDescriptor): Promise<EndpointRef> {
    this.assertOpen();
    const normalizedId = normalizeEndpointId(id, 'endpoint id');
    assertEndpointDescriptor(descriptor);
    if (descriptor.kind === 'local') {
      await this.#service.load(descriptor.source, descriptor.options);
      this.#gatewayEndpoints.remove(normalizedId);
      this.#providers.remove(normalizedId);
      const endpoint = { kind: 'local', id: normalizedId } as const;
      this.#localEndpoint = endpoint;
      return endpoint;
    }
    if (descriptor.kind === 'gateway') {
      const endpoint = this.#gatewayEndpoints.prepare(normalizedId, descriptor);
      await this.removeLocalEndpoint(normalizedId);
      this.#providers.remove(normalizedId);
      return this.#gatewayEndpoints.commit(endpoint);
    }
    const provider = this.#providers.prepare(
      normalizedId,
      descriptor as ProviderEndpointDescriptor
    );
    await this.removeLocalEndpoint(normalizedId);
    this.#gatewayEndpoints.remove(normalizedId);
    return this.#providers.commit(provider);
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

  public query(input: QueryInput, options: QueryOptions = {}): BrowserTextRun {
    this.assertOpen();
    const endpoint = this.#gatewayEndpoints.get(options.endpoint);
    if (endpoint != null) {
      return createBrowserTextRun(options, (tokenBatchSink, signal) =>
        runGatewayQuery(endpoint, input, options, tokenBatchSink, signal)
      );
    }
    const provider = this.#providers.get(options.endpoint);
    if (provider != null) {
      return createBrowserTextRun(options, (tokenBatchSink, signal) =>
        runProviderQuery(provider, input, options, tokenBatchSink, signal)
      );
    }
    this.ensureLocalEndpoint(options.endpoint);
    const localOptions = localQueryOptions(options);
    return createBrowserTextRun(localOptions, (tokenBatchSink, signal) =>
      this.#service.runQuery(input, { ...localOptions, signal, tokenBatchSink })
    );
  }

  public chat(input: ChatInput, options: ChatOptions = {}): BrowserTextRun {
    this.assertOpen();
    const endpoint = this.#gatewayEndpoints.get(options.endpoint);
    if (endpoint != null) {
      return createBrowserTextRun(options, (tokenBatchSink, signal) =>
        runGatewayChat(endpoint, input, options, tokenBatchSink, signal)
      );
    }
    const provider = this.#providers.get(options.endpoint);
    if (provider != null) {
      return createBrowserTextRun(options, (tokenBatchSink, signal) =>
        runProviderChat(provider, input, options, tokenBatchSink, signal)
      );
    }
    this.ensureLocalEndpoint(options.endpoint);
    const localOptions = localQueryOptions(options);
    return createBrowserTextRun(localOptions, (tokenBatchSink, signal) =>
      this.#service.runChat(input, { ...localOptions, signal, tokenBatchSink })
    );
  }

  public embed(input: string, options: EmbedOptions = {}): BrowserEmbeddingRun {
    this.assertOpen();
    const endpoint = this.#gatewayEndpoints.get(options.endpoint);
    if (endpoint != null) {
      return createBrowserEmbeddingRun(options.signal, (signal) =>
        runGatewayEmbedding(endpoint, input, options, signal)
      );
    }
    const provider = this.#providers.get(options.endpoint);
    if (provider != null) {
      return createBrowserEmbeddingRun(options.signal, (signal) =>
        runProviderEmbedding(provider, input, options, signal)
      );
    }
    this.ensureLocalEndpoint(options.endpoint);
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

  private async removeLocalEndpoint(id: string): Promise<void> {
    if (this.#localEndpoint?.id === id) {
      await this.#service.unload();
      this.#localEndpoint = null;
    }
  }

  private ensureLocalEndpoint(endpoint: EndpointRef | undefined): void {
    if (endpoint == null) {
      return;
    }
    if (endpoint.kind !== 'local') {
      throw new QueryError('MODEL_NOT_FOUND', `${endpoint.kind} endpoint not found: ${endpoint.id}`);
    }
    if (this.#localEndpoint == null || this.#localEndpoint.id !== endpoint.id) {
      throw new QueryError('MODEL_NOT_FOUND', `local endpoint not found: ${endpoint.id}`);
    }
  }
}

function localQueryOptions(options: QueryOptions): QueryOptions {
  rejectLocalEndpointProviderOptions(options.endpointOptions, options.providerOptions);
  const {
    endpoint: _endpoint,
    endpointOptions: _endpointOptions,
    providerOptions: _providerOptions,
    ...localOptions
  } = options;
  return localOptions;
}

function localEmbedOptions(options: EmbedOptions): EmbedOptions {
  rejectLocalEndpointProviderOptions(options.endpointOptions, options.providerOptions);
  const {
    endpoint: _endpoint,
    endpointOptions: _endpointOptions,
    providerOptions: _providerOptions,
    ...localOptions
  } = options;
  return localOptions;
}

function rejectLocalEndpointProviderOptions(endpointOptions: unknown, providerOptions: unknown): void {
  if (endpointOptions != null) {
    throw new QueryError('UNSUPPORTED_OPERATION', 'endpointOptions are not valid for local endpoints');
  }
  if (providerOptions != null) {
    throw new QueryError('UNSUPPORTED_OPERATION', 'providerOptions are not valid for local endpoints');
  }
}

function normalizeEndpointId(value: unknown, name: string): string {
  if (typeof value !== 'string') {
    throw new QueryError('QUERY_FAILED', `${name} must be a string`);
  }
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    throw new QueryError('QUERY_FAILED', `${name} must not be empty`);
  }
  if (trimmed !== value) {
    throw new QueryError('QUERY_FAILED', `${name} must not contain surrounding whitespace`);
  }
  return value;
}

function assertEndpointDescriptor(value: unknown): asserts value is EndpointDescriptor {
  if (typeof value !== 'object' || value == null || Array.isArray(value)) {
    throw new QueryError('QUERY_FAILED', 'endpoint descriptor must be an object');
  }
  const kind = (value as { readonly kind?: unknown }).kind;
  if (kind !== 'local' && kind !== 'gateway' && kind !== 'provider') {
    throw new QueryError(
      'QUERY_FAILED',
      'endpoint descriptor kind must be local, gateway, or provider'
    );
  }
}
