import { ModelService } from './model-management/model-service.js';
import type { ModelLifecycleService } from './model-management/model-service-contract.js';
import { WorkerModelServiceClient } from './model-management/worker-model-service-client.js';
import {
  QueryError,
  type ModelInfo,
  type ModelLoadOptions,
  type ModelSource,
  type QueryInput,
  type QueryOptions,
} from './model-management/model-types.js';
import { MainThreadEngineRuntime } from './runtime/engine-runtime-main-thread.js';

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
  ) {}

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

export class CogentEngine {
  public readonly models: CogentModelManager;
  #service: ModelLifecycleService;
  #closed = false;

  private constructor(config: CogentEngineOptions = {}) {
    this.#service = shouldUseWorker(config)
      ? new WorkerModelServiceClient(config)
      : new ModelService(new MainThreadEngineRuntime(config));
    this.models = new RuntimeModelManager(() => this.assertOpen(), this.#service);
  }

  public static async create(options: CogentEngineOptions = {}): Promise<CogentEngine> {
    return new CogentEngine(options);
  }

  public async query(input: QueryInput, options?: QueryOptions): Promise<string> {
    this.assertOpen();
    return await this.#service.query(input, options);
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
