import { CogentConfig } from '../cogent-config.js';
import {
  BackendObservability,
  EngineExecutionMode,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  InternalBundleDescriptor,
  PromptOptions,
  RuntimeAggregateObservabilityMetrics,
  StagedModelBundle,
  StageModelBundleOptions,
  TransportObservability,
} from '../types.js';

export type { CogentConfig };

export interface EngineRuntime {
  getExecutionMode(): EngineExecutionMode;
  getTransportObservability(): TransportObservability;
  initModule(): Promise<void>;
  stageModelBundle(
    descriptor: InternalBundleDescriptor,
    options?: StageModelBundleOptions
  ): Promise<StagedModelBundle>;
  loadRuntimeModel(
    modelPathOrBundle: string | StagedModelBundle,
    config?: InferenceInitConfig
  ): Promise<void>;
  close(): void;
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
  getRuntimeObservability(): RuntimeAggregateObservabilityMetrics | null;
  getBackendObservability(): Promise<BackendObservability | null>;
}
