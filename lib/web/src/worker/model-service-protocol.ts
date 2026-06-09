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
  TokenBatch,
} from '../models/types.js';
import type { BrowserCachePolicyOptions } from '../models/asset-store.js';
import type { SharedTokenRingDescriptor } from '../runtime/shared-token-ring.js';

export interface WorkerRuntimeConfig {
  moduleUrl?: string;
  wasmUrl?: string;
  wasmThreading?: 'single-thread' | 'pthread';
  moduleOptions?: Record<string, unknown>;
  maxModelBytes?: number;
  browserCache?: BrowserCachePolicyOptions;
  trustedOrigins?: string[];
}

export type WorkerQueryOptions =
  Pick<
    QueryOptions,
    'contextKey' | 'maxTokens' | 'temperature' | 'topP' | 'stop' | 'grammar'
  > & {
    emitTokens: boolean;
  };

export type WorkerRequestMessage =
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
      kind: 'token-ring-ready';
      descriptor: SharedTokenRingDescriptor;
    }
  | {
      kind: 'token-ring-claim';
      callId: number;
      nativeRequestId: number;
    }
  | {
      kind: 'token-batch';
      callId: number;
      batch: TokenBatch;
    }
  | {
      kind: 'observability-event';
      event: ObservabilityEvent;
    }
  | {
      kind: 'engine-event';
      event: EngineEvent;
    };
