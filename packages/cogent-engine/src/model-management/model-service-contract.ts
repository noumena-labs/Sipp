import type {
  ModelInfo,
  ModelLoadOptions,
  ObservabilityEvent,
  ObservabilitySnapshot,
  ModelSource,
  QueryInput,
  QueryOptions,
} from './model-types.js';

export interface ModelLifecycleService {
  load(source: ModelSource, options?: ModelLoadOptions): Promise<ModelInfo>;
  currentModel(): ModelInfo | null;
  list(): Promise<ModelInfo[]>;
  remove(id: string): Promise<void>;
  query(input: QueryInput, options?: QueryOptions): Promise<string>;
  currentObservability(): ObservabilitySnapshot;
  subscribeObservability(listener: (event: ObservabilityEvent) => void): () => void;
  close(): void | Promise<void>;
}
