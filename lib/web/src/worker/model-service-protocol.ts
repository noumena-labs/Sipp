import type {
  ModelAddSource,
  ModelLoadOptions,
  ModelLoadProgress,
  EngineEvent,
  ObservabilityEvent,
  QueryErrorCode,
  ChatInput,
  EmbedOptions,
  ListenOptions,
  QueryInput,
  QueryOptions,
  SpeakOptions,
  TokenBatch,
} from '../models/types.js';
import type { BrowserCachePolicyOptions } from '../models/asset-store.js';
import type { SharedTokenRingDescriptor } from '../runtime/shared-token-ring.js';

export interface WorkerRuntimeConfig {
  readonly moduleUrl?: string;
  readonly wasmUrl?: string;
  readonly wasmThreading?: 'single-thread' | 'pthread';
  readonly moduleOptions?: Record<string, unknown>;
  readonly maxModelBytes?: number;
  readonly storageRoot?: string;
  readonly browserCache?: BrowserCachePolicyOptions;
  readonly trustedOrigins?: readonly string[];
}

export type WorkerQueryOptions =
  Pick<
    QueryOptions,
    | 'contextKey'
    | 'maxTokens'
    | 'temperature'
    | 'topP'
    | 'sampling'
    | 'stop'
    | 'grammar'
  > & {
    emitTokens: boolean;
  };

export type WorkerRequestMessage =
  /**
   * Configures the Worker's single model service. Sent once, before any other
   * request, so operational requests never carry runtime configuration.
   */
  | {
      kind: 'initialize';
      config: WorkerRuntimeConfig;
    }
  | {
      kind: 'models-install';
      callId: number;
      source: ModelAddSource;
    }
  | {
      kind: 'models-load';
      callId: number;
      modelId: string;
      options: Pick<ModelLoadOptions, 'backend' | 'observability' | 'runtime'>;
    }
  | {
      kind: 'models-list';
      callId: number;
    }
  | {
      kind: 'models-remove';
      callId: number;
      id: string;
    }
  | {
      kind: 'shutdown';
      callId: number;
    }
  | {
      kind: 'query';
      callId: number;
      input: QueryInput;
      options: WorkerQueryOptions;
    }
  | {
      kind: 'chat';
      callId: number;
      input: ChatInput;
      options: WorkerQueryOptions;
    }
  | {
      kind: 'embed';
      callId: number;
      input: string;
      options: Pick<EmbedOptions, 'normalize' | 'contextKey'>;
    }
  | {
      kind: 'listen';
      callId: number;
      audio: Uint8Array;
      options: Pick<ListenOptions, 'language' | 'maxTokens'>;
    }
  | {
      kind: 'speak';
      callId: number;
      text: string;
      options: Pick<SpeakOptions, 'language' | 'speakerAudio' | 'maxDurationMs'>;
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
      errorStack?: string;
      queryErrorCode?: QueryErrorCode;
      /** Messages of the AggregateError attached as `cleanupFailures`. */
      cleanupFailures?: readonly string[];
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
