import {
  BackendObservability,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  ModelBundleDescriptor,
  ModelLoadInfo,
  PromptOptions,
  PreparedModelBundle,
  RuntimeAggregateObservabilityMetrics,
  TransportObservability,
} from '../types.js';

export interface WorkerSerializableCogentConfig {
  moduleUrl?: string;
  wasmUrl?: string;
  moduleOptions?: Record<string, unknown>;
  maxModelBytes?: number;
  trustedOrigins?: string[];
  workerMaxBufferedTokens?: number;
  workerTokenFlushIntervalMs?: number;
  persistentModelCache?: {
    enabled?: boolean;
  };
  debugTokenTransport?: 'auto' | 'runtime-events';
}

export interface WorkerRuntimeMetadata {
  chatTemplate: string | null;
  mediaMarker: string | null;
}

export interface WorkerQueuedPromptOptions {
  nTokens?: number;
  promptFormat?: PromptOptions['promptFormat'];
  media?: ArrayBuffer[];
  grammar?: string;
}

export type WorkerRequestMessage =
  | {
      kind: 'init-module';
      callId: number;
      config: WorkerSerializableCogentConfig;
    }
  | {
      kind: 'load-model-url';
      callId: number;
      url: string;
      destFileName: string;
    }
  | {
      kind: 'load-model-file';
      callId: number;
      file: File;
      destFileName: string;
    }
  | {
      kind: 'load-model-file-shards';
      callId: number;
      files: File[];
    }
  | {
      kind: 'load-model-urls';
      callId: number;
      urls: string[];
    }
  | {
      kind: 'prepare-model-bundle';
      callId: number;
      descriptor: ModelBundleDescriptor;
    }
  | {
      kind: 'load-model-stream-start';
      callId: number;
      destFileName: string;
      expectedBytes?: number;
    }
  | {
      kind: 'load-model-stream-chunk';
      callId: number;
      chunk: ArrayBuffer;
    }
  | {
      kind: 'load-model-stream-end';
      callId: number;
    }
  | {
      kind: 'cancel-model-load';
      callId: number;
    }
  | {
      kind: 'init-engine';
      callId: number;
      modelPath: string;
      config?: InferenceInitConfig;
    }
  | {
      kind: 'queue-prompt';
      callId: number;
      contextKey: string;
      promptText: string;
      options: WorkerQueuedPromptOptions;
    }
  | {
      kind: 'queue-prompt-with-media';
      callId: number;
      contextKey: string;
      promptText: string;
      options: WorkerQueuedPromptOptions;
    }
  | {
      kind: 'cancel-request';
      callId: number;
      requestId: GenerateRequestId;
    }
  | {
      kind: 'get-backend-observability';
      callId: number;
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
    }
  | {
      kind: 'load-progress';
      callId: number;
      progressPct: number;
    }
  | {
      kind: 'load-stream-ack';
      callId: number;
    }
  | {
      kind: 'token';
      requestId: GenerateRequestId;
      text: string;
      bufferedTokenCount: number;
    }
  | {
      kind: 'request-complete';
      requestId: GenerateRequestId;
      result: WorkerRunQueuedRequestResult;
    }
  | {
      kind: 'request-failed';
      requestId: GenerateRequestId;
      message: string;
      errorName?: string;
      runtimeAggregateObservability: RuntimeAggregateObservabilityMetrics | null;
      transportObservability: TransportObservability;
    };

export interface WorkerLoadModelResult {
  modelPath: string;
  modelLoadInfo: ModelLoadInfo | null;
  transportObservability: TransportObservability;
}

export interface WorkerPrepareModelBundleResult {
  bundle: PreparedModelBundle;
  transportObservability: TransportObservability;
}

export interface WorkerRunQueuedRequestResult {
  response: GenerateResponse;
  runtimeAggregateObservability: RuntimeAggregateObservabilityMetrics | null;
  transportObservability: TransportObservability;
}

export interface WorkerBackendObservabilityResult {
  backendObservability: BackendObservability | null;
  transportObservability: TransportObservability;
}
