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
  applyChatTemplate(
    messages: Array<{ role: string; content: string }>,
    addAssistant: boolean
  ): Promise<string>;
  getChatTemplate(): string | null;
  getBosText(): string;
  getEosText(): string;
  getMediaMarker(): string | null;
  currentObservability(): ObservabilitySnapshot;
  subscribeObservability(listener: (event: ObservabilityEvent) => void): () => void;
  close(): void | Promise<void>;
}
