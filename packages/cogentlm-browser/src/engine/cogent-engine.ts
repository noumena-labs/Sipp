import { ModelService } from '../models/model-service.js';
import { WorkerModelServiceClient } from '../worker/model-service-client.js';
import {
  QueryError,
  type ChatInput,
  type ChatOptions,
  type EmbedOptions,
  type EmbeddingResult,
  type EngineEvent,
  type EngineState,
  type EngineObservability,
  type ModelLifecycleService,
  type ModelInfo,
  type ModelLoadOptions,
  type ModelSource,
  type QueryInput,
  type QueryOptions,
  type GenerationResult,
} from '../models/types.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import type { BackendObservability } from '../core/inference-types.js';

export interface EngineModuleOptions {
  locateFile?: (path: string, prefix?: string) => string;
  [key: string]: unknown;
}

export interface CogentEngineOptions {
  moduleUrl?: string;
  wasmUrl?: string;
  pthreadModuleUrl?: string;
  pthreadWasmUrl?: string;
  wasmThreading?: 'single-thread' | 'pthread';
  moduleOptions?: EngineModuleOptions;
  maxModelBytes?: number;
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

export class CogentEngine {
  public readonly models: CogentModelManager;
  public readonly observability: EngineObservability;
  #service: ModelLifecycleService;
  #closed = false;

  private constructor(options: CogentEngineOptions = {}) {
    this.#service = shouldUseWorker(options)
      ? new WorkerModelServiceClient(options)
      : new ModelService(new MainThreadEngineRuntime(options));
    this.models = {
      load: (source, loadOptions) => {
        this.assertOpen();
        return this.#service.load(source, loadOptions);
      },
      current: () => {
        this.assertOpen();
        return this.#service.current();
      },
      list: () => {
        this.assertOpen();
        return this.#service.list();
      },
      remove: async (id) => {
        this.assertOpen();
        await this.#service.remove(id);
      },
    };
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

  public async query(input: QueryInput, options?: QueryOptions): Promise<GenerationResult> {
    this.assertOpen();
    return await this.#service.query(input, options);
  }

  public async chat(input: ChatInput, options: ChatOptions = {}): Promise<GenerationResult> {
    this.assertOpen();
    return await this.#service.chat(input, options);
  }

  public async embed(input: string, options: EmbedOptions = {}): Promise<EmbeddingResult> {
    this.assertOpen();
    return await this.#service.embed(input, options);
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
