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
import type { ChatTemplateMessage } from '../wasm/wasm-bridge.js';

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
  getChatTemplate(): string | null;
  readMediaMarker(): string | null;
  /**
   * Returns the model's BOS token rendered as text, or '' if the model has
   * no BOS token. Used by the character-agent custom template builder to
   * emit the correct leading special token per model.
   */
  getBosText(): string;
  /** Returns the model's EOS token rendered as text, or '' if unavailable. */
  getEosText(): string;
  applyChatTemplate(messages: ChatTemplateMessage[], addAssistant: boolean): Promise<string>;
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
