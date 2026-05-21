import { ModelService } from '../models/model-service.js';
import type { ModelLifecycleService } from '../models/contract.js';
import { WorkerModelServiceClient } from '../worker/model-service-client.js';
import {
  QueryError,
  type ChatInput,
  type ChatOptions,
  type EngineEvent,
  type EngineState,
  type EngineObservability,
  type ObservabilityEvent,
  type ObservabilitySnapshot,
  type ModelInfo,
  type ModelLoadOptions,
  type ModelSource,
  type QueryInput,
  type QueryOptions,
  type RequestResult,
} from '../models/types.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import type { BrowserRuntimeSmokeResult } from '../runtime/browser-smoke-types.js';

export interface CogentEngineOptions {
  moduleUrl?: string;
  wasmUrl?: string;
  moduleOptions?: {
    locateFile?: (path: string, prefix?: string) => string;
    [key: string]: unknown;
  };
  trustedOrigins?: string[];
  executionMode?: 'auto' | 'worker' | 'main-thread';
  workerUrl?: string;
}

function shouldUseWorker(config: CogentEngineOptions): boolean {
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

interface CogentModelManager {
  load(source: ModelSource, options?: ModelLoadOptions): Promise<ModelInfo>;
  current(): ModelInfo | null;
  list(): Promise<ModelInfo[]>;
  remove(id: string): Promise<void>;
}

class RuntimeModelManager implements CogentModelManager {
  constructor(
    private readonly assertOpen: () => void,
    private readonly service: ModelLifecycleService
  ) { }

  public load(source: ModelSource, options?: ModelLoadOptions): Promise<ModelInfo> {
    this.assertOpen();
    return this.service.load(source, options);
  }

  public current(): ModelInfo | null {
    this.assertOpen();
    return this.service.currentModel();
  }

  public list(): Promise<ModelInfo[]> {
    this.assertOpen();
    return this.service.list();
  }

  public async remove(id: string): Promise<void> {
    this.assertOpen();
    await this.service.remove(id);
  }
}

class RuntimeObservability implements EngineObservability {
  constructor(
    private readonly assertOpen: () => void,
    private readonly service: ModelLifecycleService
  ) { }

  public current(): ObservabilitySnapshot {
    this.assertOpen();
    return this.service.currentObservability();
  }

  public subscribe(listener: (event: ObservabilityEvent) => void): () => void {
    this.assertOpen();
    return this.service.subscribeObservability(listener);
  }
}

export class CogentEngine {
  public readonly models: CogentModelManager;
  public readonly observability: EngineObservability;
  #service: ModelLifecycleService;
  #closed = false;

  private constructor(options: CogentEngineOptions = {}) {
    this.#service = shouldUseWorker(options)
      ? new WorkerModelServiceClient(options)
      : new ModelService(new MainThreadEngineRuntime(options));
    this.models = new RuntimeModelManager(() => this.assertOpen(), this.#service);
    this.observability = new RuntimeObservability(() => this.assertOpen(), this.#service);
  }

  public static async create(options: CogentEngineOptions = {}): Promise<CogentEngine> {
    return new CogentEngine(options);
  }

  public static async browserRuntimeSmoke(
    options: CogentEngineOptions = {}
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

  public async query(input: QueryInput, options?: QueryOptions): Promise<RequestResult> {
    this.assertOpen();
    return await this.#service.query(input, options);
  }

  public async queryResult(input: QueryInput, options?: QueryOptions): Promise<RequestResult> {
    this.assertOpen();
    return await this.#service.queryResult(input, options);
  }

  public async chat(input: ChatInput, options: ChatOptions = {}): Promise<RequestResult> {
    this.assertOpen();
    return await this.#service.chat(input, options);
  }

  public async chatResult(input: ChatInput, options: ChatOptions = {}): Promise<RequestResult> {
    this.assertOpen();
    return await this.#service.chatResult(input, options);
  }

  public state(): EngineState {
    this.assertOpen();
    return this.#service.currentState();
  }

  public subscribeEvents(listener: (event: EngineEvent) => void): () => void {
    this.assertOpen();
    return this.#service.subscribeEvents(listener);
  }

  public async close(): Promise<void> {
    if (this.#closed) {
      return;
    }
    this.#closed = true;
    await this.#service.close();
  }

  private assertOpen(): void {
    if (this.#closed) {
      throw new QueryError('ENGINE_CLOSED', 'CogentEngine is closed.');
    }
  }
}
