import { CogentConfig } from './cogent-config.js';
import { MainThreadEngineRuntime } from './runtime/engine-runtime-main-thread.js';
import { EngineRuntime } from './runtime/engine-runtime.js';
import { WorkerEngineRuntime } from './runtime/engine-runtime-worker.js';
import {
  BackendObservability,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  ModelBundleDescriptor,
  ModelLoadInfo,
  PromptOptions,
  PreparedModelBundle,
  PrepareModelBundleOptions,
  RuntimeAggregateObservabilityMetrics,
  TransportObservability,
} from './types.js';
import type { ChatTemplateMessage } from './wasm/wasm-bridge.js';

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

  public getTransportObservability(): TransportObservability {
    return this.runtime.getTransportObservability();
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

  public async loadModelFromFileShards(
    files: File[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    return this.runtime.loadModelFromFileShards(files, onProgress, signal);
  }

  public async loadModelFromUrls(
    urls: string[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string> {
    return this.runtime.loadModelFromUrls(urls, onProgress, signal);
  }

  public async initEngine(
    modelPathOrBundle: string | PreparedModelBundle,
    config?: InferenceInitConfig
  ): Promise<void> {
    await this.runtime.initEngine(modelPathOrBundle, config);
  }

  public async prepareModelBundle(
    descriptor: ModelBundleDescriptor,
    options?: PrepareModelBundleOptions
  ): Promise<PreparedModelBundle> {
    return this.runtime.prepareModelBundle(descriptor, options);
  }

  public close(): void {
    this.runtime.close();
  }

  public getChatTemplate(): string | null {
    return this.runtime.getChatTemplate();
  }

  public getMediaMarker(): string | null {
    return this.runtime.getMediaMarker();
  }

  public getBosText(): string {
    return this.runtime.getBosText();
  }

  public getEosText(): string {
    return this.runtime.getEosText();
  }

  public async applyChatTemplate(messages: ChatTemplateMessage[], addAssistant: boolean): Promise<string> {
    return this.runtime.applyChatTemplate(messages, addAssistant);
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

  public getRuntimeAggregateObservability(): RuntimeAggregateObservabilityMetrics | null {
    return this.runtime.getRuntimeAggregateObservability();
  }

  public getRuntimeObservability(): RuntimeAggregateObservabilityMetrics | null {
    return this.runtime.getRuntimeAggregateObservability();
  }

  public async getBackendObservability(): Promise<BackendObservability | null> {
    return this.runtime.getBackendObservability();
  }
}
