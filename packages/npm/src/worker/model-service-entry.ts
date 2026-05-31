import { ModelService } from '../models/model-service.js';
import { QueryError, type TokenBatch } from '../models/types.js';
import { resolveRuntimeUrls } from '../engine/runtime-assets.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  type WorkerRuntimeConfig,
} from './model-service-protocol.js';

let service: ModelService | null = null;
let serviceConfigFingerprint: string | null = null;
const activeCalls = new Map<number, AbortController>();

type WorkerOperationRequest = Exclude<
  WorkerRequestMessage,
  { kind: 'cancel' }
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

// Wires a service-level token batch sink to worker messages. Chat and query
// use this same path, so chat boundary sanitization stays inside ModelService.
function tokenEmissionOptionsFor(
  callId: number,
  emitTokens: boolean
): {
  tokenBatchSink?: (batch: TokenBatch) => void;
} {
  if (!emitTokens) {
    return {};
  }
  return {
    tokenBatchSink: (batch) => {
      if (batch.text.length > 0) {
        post({ kind: 'token-batch', callId, batch });
      }
    },
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
        const emission = tokenEmissionOptionsFor(message.callId, message.options.emitTokens);
        return ensureService(message.config).runQuery(
          message.input,
          {
            ...message.options,
            signal,
            tokenBatchSink: emission.tokenBatchSink,
          }
        );
      });
    case 'chat':
      return await withAbortController(message.callId, (signal) => {
        const emission = tokenEmissionOptionsFor(message.callId, message.options.emitTokens);
        return ensureService(message.config).runChat(
          message.input,
          {
            ...message.options,
            signal,
            tokenBatchSink: emission.tokenBatchSink,
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
