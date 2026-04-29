import type { CogentConfig } from '../cogent-config.js';
import type {
  ModelLoadOptions,
  ModelLoadProgress,
  ModelSource,
  ObservabilityEvent,
  QueryErrorCode,
  ChatInput,
  ChatOptions,
  QueryInput,
  QueryOptions,
} from '../model-management/model-types.js';

export interface WorkerSerializableCogentConfig {
  moduleUrl?: string;
  wasmUrl?: string;
  moduleOptions?: Record<string, unknown>;
  maxModelBytes?: number;
  trustedOrigins?: string[];
}

export type WorkerModelLoadOptions = Pick<ModelLoadOptions, 'observability' | 'runtime'>;
export type WorkerQueryOptions = Pick<QueryOptions, 'session' | 'maxTokens' | 'format' | 'grammar'>;
export type WorkerChatOptions = Pick<ChatOptions, 'session' | 'maxTokens' | 'grammar'>;

export type WorkerRequestMessage =
  | {
      kind: 'models-load';
      callId: number;
      config: WorkerSerializableCogentConfig;
      source: ModelSource;
      options: WorkerModelLoadOptions;
    }
  | {
      kind: 'models-list';
      callId: number;
      config: WorkerSerializableCogentConfig;
    }
  | {
      kind: 'models-remove';
      callId: number;
      config: WorkerSerializableCogentConfig;
      id: string;
    }
  | {
      kind: 'query';
      callId: number;
      config: WorkerSerializableCogentConfig;
      input: QueryInput;
      options: WorkerQueryOptions;
    }
  | {
      kind: 'chat';
      callId: number;
      config: WorkerSerializableCogentConfig;
      input: ChatInput;
      options: WorkerChatOptions;
    }
  | {
      kind: 'close';
      callId: number;
    }
  | {
      kind: 'cancel';
      targetCallId: number;
    };

export type WorkerResponseMessage =
  | {
      kind: 'resolve';
      callId: number;
      value?: unknown;
    }
  | {
      kind: 'reject';
      callId: number;
      message: string;
      errorName?: string;
      queryErrorCode?: QueryErrorCode;
    }
  | {
      kind: 'load-progress';
      callId: number;
      progress: ModelLoadProgress;
    }
  | {
      kind: 'token';
      callId: number;
      text: string;
    }
  | {
      kind: 'observability-event';
      event: ObservabilityEvent;
    };

export type WorkerServiceConfig = Pick<
  CogentConfig,
  'moduleUrl' | 'wasmUrl' | 'moduleOptions' | 'maxModelBytes' | 'trustedOrigins'
>;
