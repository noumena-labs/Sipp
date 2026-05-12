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
// runs the engine in TOKEN_EMISSION_NONE; when true the worker writes tokens
// to the SAB ring for the main thread to drain.
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
      // Sent once on worker spawn.  Carries the SAB ring used for streaming;
      // null disables streaming (engine runs in NONE mode for that worker).
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
      // Maps native GenerateRequestId → worker callId.  Sent once per
      // streaming request on enqueue, before any ring records arrive.
      kind: 'streaming-claim';
      callId: number;
      nativeRequestId: number;
    }
  | {
      // Fired after the worker writes a batch of tokens into the SAB ring.
      // Main drains the ring in its onmessage handler, keeping the drain
      // out of the rendering phase so it doesn't compete with the app's
      // own rAF loops (e.g. Three.js).
      kind: 'streaming-tick';
    }
  | {
      kind: 'observability-event';
      event: ObservabilityEvent;
    };

export type WorkerServiceConfig = Pick<
  CogentConfig,
  'moduleUrl' | 'wasmUrl' | 'moduleOptions' | 'maxModelBytes' | 'trustedOrigins'
>;
