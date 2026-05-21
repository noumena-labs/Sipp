import type {
  ModelInfo,
  ModelLoadOptions,
  EngineEvent,
  EngineState,
  ObservabilityEvent,
  ObservabilitySnapshot,
  RequestResult,
  ModelSource,
  ChatInput,
  ChatOptions,
  QueryInput,
  QueryOptions,
} from './types.js';

export interface ModelLifecycleService {
  load(source: ModelSource, options?: ModelLoadOptions): Promise<ModelInfo>;
  unload(): void | Promise<void>;
  current(): ModelInfo | null;
  currentModel(): ModelInfo | null;
  list(): Promise<ModelInfo[]>;
  remove(id: string): Promise<void>;
  query(input: QueryInput, options?: QueryOptions): Promise<RequestResult>;
  queryResult(input: QueryInput, options?: QueryOptions): Promise<RequestResult>;
  chat(input: ChatInput, options?: ChatOptions): Promise<RequestResult>;
  chatResult(input: ChatInput, options?: ChatOptions): Promise<RequestResult>;
  state(): EngineState;
  currentState(): EngineState;
  subscribeEvents(listener: (event: EngineEvent) => void): () => void;
  currentObservability(): ObservabilitySnapshot;
  subscribeObservability(listener: (event: ObservabilityEvent) => void): () => void;
  close(): void | Promise<void>;
}
