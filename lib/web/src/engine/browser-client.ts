import type { BrowserCachePolicyOptions } from '../models/asset-store.js';
import {
  createBrowserAudioRun,
  createBrowserEmbeddingRun,
  createBrowserTextRun,
} from '../models/token-queue.js';
import {
  QueryError,
  type SippClient as SippClientShape,
  type BrowserAudioRun,
  type BrowserEmbeddingRun,
  type BrowserTextRun,
  type ChatInput,
  type ChatOptions,
  type EmbedOptions,
  type Endpoint,
  type EndpointRef,
  type EngineEvent,
  type EngineObservability,
  type EngineState,
  type ManagedModel,
  type ListenOptions,
  ENDPOINT_PAYLOAD,
  type ModelLifecycleService,
  type ModelAddOptions,
  type ModelStore,
  type QueryInput,
  type QueryOptions,
  type SpeakOptions,
  ENDPOINT_REF_PAYLOAD,
} from '../models/types.js';
import { resolveUrl } from '../utils/url.js';
import { AsyncSerialQueue } from '../utils/async-queue.js';
import { WorkerModelServiceClient } from '../worker/model-service-client.js';
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
  wasmThreading?: 'single-thread' | 'pthread';
  moduleOptions?: EngineModuleOptions;
  maxModelBytes?: number;
  /** Browser OPFS directory used for the managed model registry and cached assets. */
  storageRoot?: string;
  /** Override browser OPFS split thresholds for large GGUF model files. */
  browserCache?: BrowserCachePolicyOptions;
  trustedOrigins?: readonly string[];
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
  #endpointOperations = new AsyncSerialQueue();

  public constructor(options: SippClientOptions = {}) {
    this.#service = new WorkerModelServiceClient(options);
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
   * Registers or replaces an endpoint.
   */
  public async add(id: string, endpoint: Endpoint): Promise<EndpointRef> {
    this.assertOpen();
    const normalizedId = normalizeEndpointId(id, 'endpoint id');
    const payload = endpoint[ENDPOINT_PAYLOAD];
    if (payload.kind === 'local') {
      return await this.#endpointOperations.run(async () => {
        this.#localEndpointId = null;
        this.#gatewayEndpoints.remove(normalizedId);
        this.#providers.remove(normalizedId);
        await this.#service.load(payload.modelId, payload.options);
        this.#localEndpointId = normalizedId;
        return { [ENDPOINT_REF_PAYLOAD]: { kind: 'local', id: normalizedId } };
      });
    }
    if (payload.kind === 'gateway') {
      const prepared = this.#gatewayEndpoints.prepare(normalizedId, payload.options);
      return await this.#endpointOperations.run(async () => {
        await this.unpublishLocalEndpoint(normalizedId);
        this.#providers.remove(normalizedId);
        return this.#gatewayEndpoints.commit(prepared);
      });
    }
    const prepared = this.#providers.prepare(normalizedId, payload.options);
    return await this.#endpointOperations.run(async () => {
      await this.unpublishLocalEndpoint(normalizedId);
      this.#gatewayEndpoints.remove(normalizedId);
      return this.#providers.commit(prepared);
    });
  }

  public async remove(id: string): Promise<void> {
    this.assertOpen();
    const normalizedId = normalizeEndpointId(id, 'endpoint id');
    await this.#endpointOperations.run(async () => {
      if (this.#localEndpointId === normalizedId) {
        await this.unpublishLocalEndpoint(normalizedId);
        return;
      }
      const removed = this.#gatewayEndpoints.remove(normalizedId)
        || this.#providers.remove(normalizedId);
      if (!removed) {
        throw new QueryError('MODEL_NOT_FOUND', `endpoint not found: ${normalizedId}`);
      }
    });
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

  public listen(audio: Uint8Array, options: ListenOptions = {}): BrowserTextRun {
    this.assertOpen();
    this.ensureLocalEndpoint(options.endpoint);
    const { endpoint: _endpoint, ...localOptions } = options;
    return createBrowserTextRun(localOptions, (_tokenBatchSink, signal) =>
      this.#service.runListen(audio, { ...localOptions, signal })
    );
  }

  public speak(text: string, options: SpeakOptions = {}): BrowserAudioRun {
    this.assertOpen();
    this.ensureLocalEndpoint(options.endpoint);
    const { endpoint: _endpoint, ...localOptions } = options;
    return createBrowserAudioRun(localOptions.signal, (signal) =>
      this.#service.runSpeak(text, { ...localOptions, signal })
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
    // A queued add(...) can still publish a local endpoint before it observes
    // the closed flag, so clear the route after the queue drains.
    await this.#endpointOperations.idle();
    this.#localEndpointId = null;
    await this.#service.close();
  }

  private assertOpen(): void {
    if (this.#closed) {
      throw new QueryError('ENGINE_CLOSED', 'SippClient is closed.');
    }
  }

  private async unpublishLocalEndpoint(id: string): Promise<void> {
    if (this.#localEndpointId === id) {
      this.#localEndpointId = null;
      await this.#service.unload();
    }
  }

  private ensureLocalEndpoint(endpoint: EndpointRef | undefined): void {
    if (endpoint == null) {
      if (this.#localEndpointId == null) {
        throw new QueryError('MODEL_NOT_FOUND', 'local endpoint not found');
      }
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
