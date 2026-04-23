import { CogentConfig } from '../cogent-config.js';
import {
  BackendObservability,
  EngineExecutionMode,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  InternalBundleDescriptor,
  ModelLoadInfo,
  PromptOptions,
  StagedModelBundle,
  StageModelBundleOptions,
  RuntimeAggregateObservabilityMetrics,
  TransportObservability,
} from '../types.js';

export type { CogentConfig };

export interface EngineRuntime {
  getExecutionMode(): EngineExecutionMode;
  getStagedModelInfo(): ModelLoadInfo | null;
  getTransportObservability(): TransportObservability;
  initModule(): Promise<void>;
  stageModelUrl(
    url: string,
    destFileName?: string,
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string>;
  stageModelFile(
    file: File,
    destFileName?: string,
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string>;
  stageModelStream(
    stream: ReadableStream<Uint8Array>,
    destFileName?: string,
    options?: {
      expectedBytes?: number;
      onProgress?: (pct: number) => void;
      signal?: AbortSignal;
    }
  ): Promise<string>;
  stageModelBuffer(buffer: Uint8Array, destFileName?: string): string;
  stageModelFiles(
    files: File[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string>;
  stageModelUrls(
    urls: string[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<string>;
  stageModelBundle(
    descriptor: InternalBundleDescriptor,
    options?: StageModelBundleOptions
  ): Promise<StagedModelBundle>;
  loadRuntimeModel(
    modelPathOrBundle: string | StagedModelBundle,
    config?: InferenceInitConfig
  ): Promise<void>;
  close(): void;
  readChatTemplate(): string | null;
  readMediaMarker(): string | null;
  cancelQuery(requestId: GenerateRequestId): Promise<boolean>;
  enqueueQuery(
    contextKey: string,
    promptText: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestId>;
  awaitQuery(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse>;
  executeQuery(
    contextKey: string,
    promptText: string,
    options?: number | PromptOptions
  ): Promise<string>;
  getRuntimeAggregateObservability(): RuntimeAggregateObservabilityMetrics | null;
  getRuntimeObservability(): RuntimeAggregateObservabilityMetrics | null;
  getBackendObservability(): Promise<BackendObservability | null>;
}
