import { ModelService } from '../models/model-service.js';
import { QueryError } from '../models/types.js';
import { resolveRuntimeUrls } from '../engine/runtime-assets.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import { StreamingRingWriter } from '../runtime/streaming-ring.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  type WorkerRuntimeConfig,
} from './model-service-protocol.js';

let service: ModelService | null = null;
let serviceConfigFingerprint: string | null = null;
const activeCalls = new Map<number, AbortController>();
// SAB streaming ring writer; set on `streaming-init`. When null, streaming
// requests are rejected upstream.
let streamingRingWriter: StreamingRingWriter | null = null;
let streamingTickQueued = false;

type WorkerOperationRequest = Exclude<
  WorkerRequestMessage,
  { kind: 'cancel' } | { kind: 'streaming-init' }
>;

function buildServiceConfig(config: WorkerRuntimeConfig) {
  const runtimeUrls = resolveRuntimeUrls(config);

  return {
    moduleUrl: runtimeUrls.moduleUrl,
    wasmUrl: runtimeUrls.wasmUrl,
    wasmThreading: runtimeUrls.threading,
    moduleOptions: config.moduleOptions,
    maxModelBytes: config.maxModelBytes,
    trustedOrigins: config.trustedOrigins,
  };
}

// Direct runtime handle for installing the SAB ring writer after ensureService.
let runtime: MainThreadEngineRuntime | null = null;

function ensureService(config: WorkerRuntimeConfig): ModelService {
  const fingerprint = JSON.stringify(buildServiceConfig(config));
  if (service != null) {
    if (serviceConfigFingerprint !== fingerprint) {
      throw new Error('Worker model service was initialized with different runtime options.');
    }
    return service;
  }
  runtime = new MainThreadEngineRuntime({
    ...buildServiceConfig(config),
    executionMode: 'worker',
  });
  service = new ModelService(runtime);
  if (streamingRingWriter != null) {
    runtime.setStreamingRingWriter(streamingRingWriter);
  }
  runtime.setStreamingTickCallback(scheduleStreamingTick);
  service.subscribeObservability((event) => {
    post({ kind: 'observability-event', event });
  });
  service.subscribeEvents((event) => {
    post({ kind: 'engine-event', event });
  });
  serviceConfigFingerprint = fingerprint;
  return service;
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function post(message: WorkerResponseMessage): void {
  self.postMessage(message);
}

function scheduleStreamingTick(): void {
  if (streamingTickQueued) {
    return;
  }
  streamingTickQueued = true;
  self.setTimeout(() => {
    streamingTickQueued = false;
    post({ kind: 'streaming-tick' });
  }, 0);
}

function abortActiveCall(callId: number): void {
  activeCalls.get(callId)?.abort();
}

async function withAbortController<T>(
  callId: number,
  operation: (signal: AbortSignal) => Promise<T>
): Promise<T> {
  const abortController = new AbortController();
  activeCalls.set(callId, abortController);
  try {
    return await operation(abortController.signal);
  } finally {
    activeCalls.delete(callId);
  }
}

function postLoadProgress(callId: number): NonNullable<Parameters<ModelService['load']>[1]>['onProgress'] {
  return (progress) => {
    post({
      kind: 'load-progress',
      callId,
      progress,
    });
  };
}

// Wires the engine to emit tokens through the SAB ring and publishes a
// `streaming-claim` message so the main thread can map the native request id
// back to its call id.  When `streaming=false`, returns {} so the engine
// selects TOKEN_EMISSION_NONE.
function streamingOptionsFor(
  callId: number,
  streaming: boolean
): {
  streamTokens?: boolean;
  onRequestStarted?: (requestId: number) => void;
} {
  if (!streaming) {
    return {};
  }
  if (streamingRingWriter == null) {
    throw new QueryError(
      'STREAMING_UNAVAILABLE',
      'Worker streaming requires SharedArrayBuffer. Enable cross-origin isolation or run without streamTokens.'
    );
  }
  return {
    streamTokens: true,
    onRequestStarted: (requestId) =>
      post({ kind: 'streaming-claim', callId, nativeRequestId: requestId }),
  };
}

async function handleRequest(message: WorkerOperationRequest): Promise<unknown> {
  switch (message.kind) {
    case 'models-load':
      return await withAbortController(message.callId, (signal) =>
        ensureService(message.config).load(message.source, {
          ...message.options,
          signal,
          onProgress: postLoadProgress(message.callId),
        })
      );
    case 'models-list':
      return await ensureService(message.config).list();
    case 'models-unload':
      await ensureService(message.config).unload();
      return null;
    case 'models-remove': {
      const modelService = ensureService(message.config);
      await modelService.remove(message.id);
      return modelService.current();
    }
    case 'query':
      return await withAbortController(message.callId, (signal) =>
        ensureService(message.config).query(message.input, {
          ...message.options,
          signal,
          ...streamingOptionsFor(message.callId, message.options.streaming),
        }).response
      );
    case 'chat':
      return await withAbortController(message.callId, (signal) =>
        ensureService(message.config).chat(message.input, {
          ...message.options,
          signal,
          ...streamingOptionsFor(message.callId, message.options.streaming),
        }).response
      );
    case 'embed':
      return await withAbortController(message.callId, (signal) =>
        ensureService(message.config).embed(message.input, {
          ...message.options,
          signal,
        }).response
      );
  }
}

self.onmessage = async (event: MessageEvent<WorkerRequestMessage>) => {
  const message = event.data;
  if (message.kind === 'cancel') {
    abortActiveCall(message.targetCallId);
    return;
  }
  if (message.kind === 'streaming-init') {
    streamingRingWriter =
      message.ringBuffer != null
        ? new StreamingRingWriter(message.ringBuffer)
        : null;
    if (runtime != null) {
      runtime.setStreamingRingWriter(streamingRingWriter);
    }
    return;
  }

  try {
    const value = await handleRequest(message);
    post({
      kind: 'resolve',
      callId: message.callId,
      value,
    });
  } catch (error) {
    post({
      kind: 'reject',
      callId: message.callId,
      message: toErrorMessage(error),
      errorName: error instanceof Error ? error.name : undefined,
      queryErrorCode: error instanceof QueryError ? error.code : undefined,
    });
  }
};
