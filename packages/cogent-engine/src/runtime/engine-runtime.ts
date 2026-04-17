import { CogentConfig } from '../cogent-config.js';
import {
  BackendObservability,
  EngineExecutionMode,
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
} from '../types.js';

export type { CogentConfig };

export interface EngineRuntime {
  getExecutionMode(): EngineExecutionMode;
  getLastModelLoadInfo(): ModelLoadInfo | null;
  getTransportObservability(): TransportObservability;
  initModule(): Promise<void>;
  loadModelFromUrl(
    url: string,
    destFileName?: string,
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string>;
  loadModelFromFile(
    file: File,
    destFileName?: string,
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string>;
  loadModelFromReadableStream(
    stream: ReadableStream<Uint8Array>,
    destFileName?: string,
    options?: {
      expectedBytes?: number;
      onProgress?: (pct: number) => void;
      signal?: AbortSignal;
    }
  ): Promise<string>;
  loadModelFromBuffer(buffer: Uint8Array, destFileName?: string): string;
  loadModelFromFileShards(
    files: File[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string>;
  loadModelFromUrls(
    urls: string[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string>;
  prepareModelBundle(
    descriptor: ModelBundleDescriptor,
    options?: PrepareModelBundleOptions
  ): Promise<PreparedModelBundle>;
  initEngine(
    modelPathOrBundle: string | PreparedModelBundle,
    config?: InferenceInitConfig
  ): Promise<void>;
  close(): void;
  getChatTemplate(): string | null;
  getMediaMarker(): string | null;
  cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean>;
  queuePrompt(
    contextKey: string,
    promptText: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestId>;
  runQueuedRequest(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse>;
  submitPrompt(
    contextKey: string,
    promptText: string,
    options?: number | PromptOptions
  ): Promise<string>;
  getRuntimeAggregateObservability(): RuntimeAggregateObservabilityMetrics | null;
  getRuntimeObservability(): RuntimeAggregateObservabilityMetrics | null;
  getBackendObservability(): Promise<BackendObservability | null>;
}
