import { CogentConfig } from './cogent-config.js';
import { MainThreadEngineRuntime } from './runtime/engine-runtime-main-thread.js';
import { EngineRuntime } from './runtime/engine-runtime.js';
import { WorkerEngineRuntime } from './runtime/engine-runtime-worker.js';
import {
  BackendInfo,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  ModelLoadInfo,
  PromptPerformanceStats,
  PromptOptions,
  TransportInfo,
} from './types.js';

function shouldUseWorker(config: CogentConfig): boolean {
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

export type { CogentConfig } from './cogent-config.js';

export class CogentEngine {
  private readonly runtime: EngineRuntime;

  constructor(config: CogentConfig = {}) {
    this.runtime = shouldUseWorker(config)
      ? new WorkerEngineRuntime(config)
      : new MainThreadEngineRuntime(config);
  }

  public getExecutionMode() {
    return this.runtime.getExecutionMode();
  }

  public getLastModelLoadInfo(): ModelLoadInfo | null {
    return this.runtime.getLastModelLoadInfo();
  }

  public getTransportInfo(): TransportInfo {
    return this.runtime.getTransportInfo();
  }

  public async initModule(): Promise<void> {
    await this.runtime.initModule();
  }

  public async loadModelFromUrl(
    url: string,
    destFileName = 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    return this.runtime.loadModelFromUrl(url, destFileName, onProgress, signal);
  }

  public async loadModelFromFile(
    file: File,
    destFileName: string = file.name || 'model.gguf',
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    return this.runtime.loadModelFromFile(file, destFileName, onProgress, signal);
  }

  public async loadModelFromReadableStream(
    stream: ReadableStream<Uint8Array>,
    destFileName = 'model.gguf',
    options: {
      expectedBytes?: number;
      onProgress?: (pct: number) => void;
      signal?: AbortSignal;
    } = {}
  ): Promise<string> {
    return this.runtime.loadModelFromReadableStream(stream, destFileName, options);
  }

  public loadModelFromBuffer(buffer: Uint8Array, destFileName = 'model.gguf'): string {
    return this.runtime.loadModelFromBuffer(buffer, destFileName);
  }

  public async initEngine(
    modelPath: string,
    config?: InferenceInitConfig
  ): Promise<void> {
    await this.runtime.initEngine(modelPath, config);
  }

  public close(): void {
    this.runtime.close();
  }

  public async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
    return this.runtime.cancelQueuedRequest(requestId);
  }

  public async queuePrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<GenerateRequestId> {
    return this.runtime.queuePrompt(contextKey, promptText, options);
  }

  public async runQueuedRequest(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    return this.runtime.runQueuedRequest(requestId, options);
  }

  public async submitPrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<string> {
    return this.runtime.submitPrompt(contextKey, promptText, options);
  }

  public getLastPromptPerformance(): PromptPerformanceStats | null {
    return this.runtime.getLastPromptPerformance();
  }

  public async getBackendInfo(): Promise<BackendInfo | null> {
    return this.runtime.getBackendInfo();
  }
}
