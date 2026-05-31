import type {
  ModelLoadOptions,
  ModelLoadProgress,
  ModelSource,
  EngineEvent,
  ObservabilityEvent,
  QueryErrorCode,
  ChatInput,
  EmbedOptions,
  QueryInput,
  QueryOptions,
  TokenDeliveryMode,
} from '../models/types.js';

export interface WorkerRuntimeConfig {
  moduleUrl?: string;
  wasmUrl?: string;
  wasmThreading?: 'single-thread' | 'pthread';
  moduleOptions?: Record<string, unknown>;
  maxModelBytes?: number;
  trustedOrigins?: string[];
}

// tokenDelivery carries the caller's intent across the worker boundary because
// token sinks can't be cloned through postMessage. The worker turns delivery
// back into a local sink that writes sanitized batches to the SAB ring.
export type WorkerQueryOptions =
  Pick<QueryOptions, 'session' | 'maxTokens' | 'grammar'> & {
    tokenDelivery: TokenDeliveryMode;
  };

export type WorkerRequestMessage =
  | {
      // Sent once on worker spawn. Carries the SAB ring used for token delivery.
      kind: 'token-init';
      ringBuffer: SharedArrayBuffer | null;
    }
  | {
      kind: 'models-load';
      callId: number;
      config: WorkerRuntimeConfig;
      source: ModelSource;
      options: Pick<ModelLoadOptions, 'backend' | 'observability' | 'runtime'>;
    }
  | {
      kind: 'models-list';
      callId: number;
      config: WorkerRuntimeConfig;
    }
  | {
      kind: 'models-remove';
      callId: number;
      config: WorkerRuntimeConfig;
      id: string;
    }
  | {
      kind: 'models-unload';
      callId: number;
      config: WorkerRuntimeConfig;
    }
  | {
      kind: 'query';
      callId: number;
      config: WorkerRuntimeConfig;
      input: QueryInput;
      options: WorkerQueryOptions;
    }
  | {
      kind: 'chat';
      callId: number;
      config: WorkerRuntimeConfig;
      input: ChatInput;
      options: WorkerQueryOptions;
    }
  | {
      kind: 'embed';
      callId: number;
      config: WorkerRuntimeConfig;
      input: string;
      options: Pick<EmbedOptions, 'normalize' | 'contextKey'>;
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
      // Maps native GenerateRequestId -> worker callId before ring records arrive.
      kind: 'token-claim';
      callId: number;
      nativeRequestId: number;
    }
  | {
      kind: 'observability-event';
      event: ObservabilityEvent;
    }
  | {
      kind: 'engine-event';
      event: EngineEvent;
    }
  | {
      // Pure signal from worker to main thread: bytes were written to the ring.
      kind: 'token-tick';
    };
