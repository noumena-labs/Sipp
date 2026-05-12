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
// `streaming` carries the caller's intent across the worker boundary because
// `onToken` itself can't be cloned through postMessage.  When false the worker
// must NOT inject its own onToken into the engine — otherwise the engine
// silently runs in StreamingBuffer/DirectCallback mode and TOKEN_EMISSION_NONE
// is unreachable from a worker-backed engine.
export type WorkerQueryOptions =
  Pick<QueryOptions, 'session' | 'maxTokens' | 'grammar'> & {
    streaming: boolean;
  };
export type WorkerChatOptions =
  Pick<ChatOptions, 'session' | 'maxTokens' | 'grammar'> & {
    streaming: boolean;
  };

export type WorkerRequestMessage =
  | {
      // Sent once on worker spawn.  Null when SAB is unavailable; worker
      // falls back to per-token postMessage in that case.
      kind: 'streaming-init';
      ringBuffer: SharedArrayBuffer | null;
    }
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
      // Maps native GenerateRequestId → worker callId.  Sent once per
      // streaming request on enqueue, before any ring records arrive.
      kind: 'streaming-claim';
      callId: number;
      nativeRequestId: number;
    }
  | {
      kind: 'observability-event';
      event: ObservabilityEvent;
    };

export type WorkerServiceConfig = Pick<
  CogentConfig,
  'moduleUrl' | 'wasmUrl' | 'moduleOptions' | 'maxModelBytes' | 'trustedOrigins'
>;
