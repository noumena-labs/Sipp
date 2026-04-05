import {
  BackendInfo,
  GenerateRequestId,
  GenerateResponse,
  InferenceInitConfig,
  ModelLoadInfo,
  PromptPerformanceStats,
  PromptOptions,
  TransportInfo,
} from '../types.js';

export interface WorkerSerializableCogentConfig {
  moduleUrl?: string;
  wasmUrl?: string;
  moduleOptions?: Record<string, unknown>;
  maxModelBytes?: number;
  trustedOrigins?: string[];
  allowUnknownContentLength?: boolean;
  workerMaxBufferedTokens?: number;
  workerTokenFlushIntervalMs?: number;
  persistentModelCache?: {
    enabled?: boolean;
    namespace?: string;
    cacheLocalFiles?: boolean;
    maxEntryBytes?: number;
  };
}

export interface WorkerQueuedPromptOptions {
  nTokens?: number;
  promptFormat?: PromptOptions['promptFormat'];
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
      kind: 'load-model-buffer';
      callId: number;
      buffer: Uint8Array;
      destFileName: string;
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
      kind: 'run-queued-request';
      callId: number;
      requestId: GenerateRequestId;
    }
  | {
      kind: 'cancel-request';
      callId: number;
      requestId: GenerateRequestId;
    }
  | {
      kind: 'get-backend-info';
      callId: number;
    }
  | {
      kind: 'get-transport-info';
      callId: number;
    }
  | {
      kind: 'get-last-model-load-info';
      callId: number;
    }
  | {
      kind: 'close';
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
      kind: 'token';
      requestId: GenerateRequestId;
      text: string;
      bufferedTokenCount: number;
    };

export interface WorkerLoadModelResult {
  modelPath: string;
  modelLoadInfo: ModelLoadInfo | null;
  transportInfo: TransportInfo;
}

export interface WorkerRunQueuedRequestResult {
  response: GenerateResponse;
  lastPromptPerformance: PromptPerformanceStats | null;
  transportInfo: TransportInfo;
}

export interface WorkerBackendInfoResult {
  backendInfo: BackendInfo | null;
  transportInfo: TransportInfo;
}
