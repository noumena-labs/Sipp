import { ModelService } from '../models/model-service.js';
import { QueryError, type TokenBatch, type TokenDeliveryMode } from '../models/types.js';
import { resolveRuntimeUrls } from '../engine/runtime-assets.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import { TokenRingWriter } from '../runtime/token-ring.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  type WorkerRuntimeConfig,
} from './model-service-protocol.js';

let service: ModelService | null = null;
let serviceConfigFingerprint: string | null = null;
const activeCalls = new Map<number, AbortController>();
// SAB token ring writer; set on `token-init`. When null, token delivery
// requests are rejected upstream.
let tokenRingWriter: TokenRingWriter | null = null;
let tokenTickQueued = false;

type WorkerOperationRequest = Exclude<
  WorkerRequestMessage,
  { kind: 'cancel' } | { kind: 'token-init' }
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

function ensureService(config: WorkerRuntimeConfig): ModelService {
  const fingerprint = JSON.stringify(buildServiceConfig(config));
  if (service != null) {
    if (serviceConfigFingerprint !== fingerprint) {
      throw new Error('Worker model service was initialized with different runtime options.');
    }
    return service;
  }
  const runtime = new MainThreadEngineRuntime({
    ...buildServiceConfig(config),
    executionMode: 'worker',
  });
  service = new ModelService(runtime);
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

function scheduleTokenTick(): void {
  if (tokenTickQueued) {
    return;
  }
  tokenTickQueued = true;
  self.setTimeout(() => {
    tokenTickQueued = false;
    post({ kind: 'token-tick' });
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

// Wires a service-level token sink to the SAB ring and publishes a
// `token-claim` message so the main thread can map the native request id
// back to its call id. Chat and query use this same path, so chat boundary
// sanitization stays inside ModelService.
function tokenDeliveryOptionsFor(
  callId: number,
  tokenDelivery: TokenDeliveryMode
): {
  tokenDelivery: TokenDeliveryMode;
  tokenSink?: (batch: TokenBatch) => void;
  onRequestStarted?: (requestId: number) => void;
} {
  if (tokenDelivery === 'off') {
    return { tokenDelivery };
  }
  if (tokenRingWriter == null) {
    throw new QueryError(
      'TOKEN_DELIVERY_UNAVAILABLE',
      'Worker token delivery requires SharedArrayBuffer. Enable cross-origin isolation or run with tokenDelivery: "off".'
    );
  }
  return {
    tokenDelivery,
    tokenSink: (batch) => {
      if (batch.text.length > 0 && tokenRingWriter?.tryWriteString(batch.streamId, batch.text)) {
        scheduleTokenTick();
      }
    },
    onRequestStarted: (requestId) =>
      post({ kind: 'token-claim', callId, nativeRequestId: requestId }),
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
      return await withAbortController(message.callId, (signal) => {
        const delivery = tokenDeliveryOptionsFor(message.callId, message.options.tokenDelivery);
        return ensureService(message.config).runQuery(
          message.input,
          {
            ...message.options,
            signal,
            tokenDelivery: delivery.tokenDelivery,
            tokenSink: delivery.tokenSink,
            onRequestStarted: delivery.onRequestStarted,
          }
        );
      });
    case 'chat':
      return await withAbortController(message.callId, (signal) => {
        const delivery = tokenDeliveryOptionsFor(message.callId, message.options.tokenDelivery);
        return ensureService(message.config).runChat(
          message.input,
          {
            ...message.options,
            signal,
            tokenDelivery: delivery.tokenDelivery,
            tokenSink: delivery.tokenSink,
            onRequestStarted: delivery.onRequestStarted,
          }
        );
      });
    case 'embed':
      return await withAbortController(message.callId, (signal) =>
        ensureService(message.config).runEmbedding(message.input, {
          ...message.options,
          signal,
        })
      );
  }
}

self.onmessage = async (event: MessageEvent<WorkerRequestMessage>) => {
  const message = event.data;
  if (message.kind === 'cancel') {
    abortActiveCall(message.targetCallId);
    return;
  }
  if (message.kind === 'token-init') {
    tokenRingWriter =
      message.ringBuffer != null
        ? new TokenRingWriter(message.ringBuffer)
        : null;
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
