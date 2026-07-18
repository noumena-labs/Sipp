import { ModelService } from '../models/model-service.js';
import { AssetStore, type BrowserCachePolicyOptions } from '../models/asset-store.js';
import { ModelRegistryStore } from '../models/model-registry-store.js';
import { createBrowserEmbeddingRun, createBrowserTextRun } from '../models/token-queue.js';
import {
  QueryError,
  type SippClient as SippClientShape,
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
  type ManagedModel,
  ENDPOINT_DESCRIPTOR_PAYLOAD,
  type ModelLifecycleService,
  type ModelAddOptions,
  type ModelStore,
  type QueryInput,
  type QueryOptions,
  ENDPOINT_REF_PAYLOAD,
} from '../models/types.js';
import { resolveUrl } from '../utils/url.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import { WorkerModelServiceClient } from '../worker/model-service-client.js';
import { FileSystemStorage } from './file-system-storage.js';
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

export interface SippClientOptions {
  moduleUrl?: string;
  wasmUrl?: string;
  pthreadModuleUrl?: string;
  pthreadWasmUrl?: string;
  wasmThreading?: 'single-thread' | 'pthread';
  moduleOptions?: EngineModuleOptions;
  maxModelBytes?: number;
  /** Browser OPFS directory used for the managed model registry and cached assets. */
  storageRoot?: string;
  /** Override browser OPFS split thresholds for large GGUF model files. */
  browserCache?: BrowserCachePolicyOptions;
  trustedOrigins?: string[];
  executionMode?: 'auto' | 'worker' | 'main-thread';
  workerUrl?: string;
}

class BrowserModelStore implements ModelStore {
  public constructor(
    private readonly service: ModelLifecycleService,
    private readonly assertOpen: () => void
  ) {}

  public async add(
    sources: readonly (File | string | URL)[],
    options: ModelAddOptions = {}
  ): Promise<ManagedModel> {
    this.assertOpen();
    if (sources.length === 0) {
      throw new QueryError('INVALID_MODEL_SOURCE', 'Model sources must not be empty.');
    }

    const files: File[] = [];
    const urls: string[] = [];
    for (const source of sources) {
      if (typeof source !== 'string' && !(source instanceof URL)) {
        files.push(source);
        continue;
      }
      let url: URL;
      try {
        url = source instanceof URL ? source : resolveUrl(source, 'model source');
      } catch (cause) {
        throw new QueryError('INVALID_MODEL_SOURCE', 'Model URL is invalid.', { cause });
      }
      if (url.protocol !== 'http:' && url.protocol !== 'https:') {
        throw new QueryError(
          'INVALID_MODEL_SOURCE',
          `Model URL scheme must be HTTP or HTTPS, not ${url.protocol}`
        );
      }
      urls.push(url.href);
    }
    if (files.length > 0 && urls.length > 0) {
      throw new QueryError(
        'INVALID_MODEL_SOURCE',
        'Browser files and remote URLs cannot be added together.'
      );
    }
    const source = files.length > 0
      ? { kind: 'local' as const, files }
      : { kind: 'remote' as const, urls };
    return managedModel(await this.service.add(source, options));
  }

  public async list(): Promise<ManagedModel[]> {
    this.assertOpen();
    return (await this.service.list()).map(managedModel);
  }

  public async remove(modelId: string): Promise<void> {
    this.assertOpen();
    await this.service.remove(modelId);
  }
}

function shouldUseWorker(config: SippClientOptions): boolean {
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
export class SippClient implements SippClientShape {
  public readonly observability: EngineObservability;
  public readonly models: ModelStore;
  #service: ModelLifecycleService;
  #gatewayEndpoints = new GatewayEndpointRegistry();
  #providers = new ProviderEndpointRegistry();
  #localEndpointId: string | null = null;
  #closed = false;

  public constructor(options: SippClientOptions = {}) {
    if (shouldUseWorker(options)) {
      this.#service = new WorkerModelServiceClient(options);
    } else {
      const storage = new FileSystemStorage(options.storageRoot);
      this.#service = new ModelService(
        new MainThreadEngineRuntime(options),
        new ModelRegistryStore(storage),
        new AssetStore(storage, options.browserCache)
      );
    }
    this.models = new BrowserModelStore(this.#service, () => this.assertOpen());
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

  /**
   * Registers or replaces an endpoint after its descriptor is validated.
   */
  public async add(id: string, descriptor: EndpointDescriptor): Promise<EndpointRef> {
    this.assertOpen();
    const normalizedId = normalizeEndpointId(id, 'endpoint id');
    assertEndpointDescriptor(descriptor);
    const payload = descriptor[ENDPOINT_DESCRIPTOR_PAYLOAD];
    if (payload.kind === 'local') {
      await this.#service.load(payload.modelId, payload.options);
      this.#gatewayEndpoints.remove(normalizedId);
      this.#providers.remove(normalizedId);
      this.#localEndpointId = normalizedId;
      return { [ENDPOINT_REF_PAYLOAD]: { kind: 'local', id: normalizedId } };
    }
    if (payload.kind === 'gateway') {
      const endpoint = this.#gatewayEndpoints.prepare(normalizedId, payload.options);
      await this.removeLocalEndpoint(normalizedId);
      this.#providers.remove(normalizedId);
      return this.#gatewayEndpoints.commit(endpoint);
    }
    const provider = this.#providers.prepare(normalizedId, payload.options);
    await this.removeLocalEndpoint(normalizedId);
    this.#gatewayEndpoints.remove(normalizedId);
    return this.#providers.commit(provider);
  }

  public async remove(id: string): Promise<void> {
    this.assertOpen();
    const normalizedId = normalizeEndpointId(id, 'endpoint id');
    if (this.#localEndpointId === normalizedId) {
      await this.removeLocalEndpoint(normalizedId);
      return;
    }
    const removed = this.#gatewayEndpoints.remove(normalizedId)
      || this.#providers.remove(normalizedId);
    if (!removed) {
      throw new QueryError('MODEL_NOT_FOUND', `endpoint not found: ${normalizedId}`);
    }
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
      throw new QueryError('ENGINE_CLOSED', 'SippClient is closed.');
    }
  }

  private async removeLocalEndpoint(id: string): Promise<void> {
    if (this.#localEndpointId === id) {
      await this.#service.unload();
      this.#localEndpointId = null;
    }
  }

  private ensureLocalEndpoint(endpoint: EndpointRef | undefined): void {
    if (endpoint == null) {
      return;
    }
    const { id, kind } = endpoint[ENDPOINT_REF_PAYLOAD];
    if (kind !== 'local') {
      throw new QueryError('MODEL_NOT_FOUND', `${kind} endpoint not found: ${id}`);
    }
    if (this.#localEndpointId !== id) {
      throw new QueryError('MODEL_NOT_FOUND', `local endpoint not found: ${id}`);
    }
  }
}

function managedModel(model: {
  id: string;
  name: string;
  bytes: number;
  modality: ManagedModel['modality'];
  status: ManagedModel['status'];
}): ManagedModel {
  return {
    id: model.id,
    name: model.name,
    bytes: model.bytes,
    modality: model.modality,
    status: model.status,
  };
}

function localQueryOptions(options: QueryOptions): QueryOptions {
  rejectLocalExtra(options.extra);
  const {
    endpoint: _endpoint,
    extra: _extra,
    ...localOptions
  } = options;
  return localOptions;
}

function localEmbedOptions(options: EmbedOptions): EmbedOptions {
  rejectLocalExtra(options.extra);
  const {
    endpoint: _endpoint,
    extra: _extra,
    ...localOptions
  } = options;
  return localOptions;
}

function rejectLocalExtra(extra: unknown): void {
  if (extra != null) {
    throw new QueryError('UNSUPPORTED_OPERATION', 'extra is not valid for local endpoints');
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
  if (
    typeof value !== 'object' ||
    value == null ||
    Array.isArray(value) ||
    !(ENDPOINT_DESCRIPTOR_PAYLOAD in value)
  ) {
    throw new QueryError(
      'QUERY_FAILED',
      'endpoint descriptors must be created by EndpointDescriptor'
    );
  }
}
