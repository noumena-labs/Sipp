import { ModelService } from '../models/model-service.js';
import { AssetStore } from '../models/asset-store.js';
import { QueryError, type TokenBatch } from '../models/types.js';
import { resolveRuntimeUrls } from '../engine/runtime-assets.js';
import { MainThreadEngineRuntime } from '../runtime/main-thread/engine-runtime.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  type WorkerRuntimeConfig,
} from './model-service-protocol.js';

let service: ModelService | null = null;
let runtime: MainThreadEngineRuntime | null = null;
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
    defaultBackendOverride: config.defaultBackendOverride,
    moduleOptions: config.moduleOptions,
    maxModelBytes: config.maxModelBytes,
    browserCache: config.browserCache,
    trustedOrigins: config.trustedOrigins,
  };
}

function ensureService(config: WorkerRuntimeConfig): ModelService {
  const serviceConfig = buildServiceConfig(config);
  const fingerprint = JSON.stringify(serviceConfig);
  if (service != null) {
    if (serviceConfigFingerprint !== fingerprint) {
      throw new Error('Worker model service was initialized with different runtime options.');
    }
    return service;
  }
  runtime = new MainThreadEngineRuntime(
    {
      moduleUrl: serviceConfig.moduleUrl,
      wasmUrl: serviceConfig.wasmUrl,
      wasmThreading: serviceConfig.wasmThreading,
      moduleOptions: serviceConfig.moduleOptions,
      maxModelBytes: serviceConfig.maxModelBytes,
      browserCache: serviceConfig.browserCache,
      trustedOrigins: serviceConfig.trustedOrigins,
      executionMode: 'worker',
    },
    {
      defaultBackendOverride: serviceConfig.defaultBackendOverride,
    }
  );
  service = new ModelService(runtime, undefined, new AssetStore(undefined, config.browserCache));
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

function postTokenRingReady(): boolean {
  const descriptor = runtime?.getSharedTokenRingDescriptor();
  if (
    descriptor == null ||
    typeof SharedArrayBuffer === 'undefined' ||
    !(descriptor.buffer instanceof SharedArrayBuffer)
  ) {
    return false;
  }
  post({ kind: 'token-ring-ready', descriptor });
  return true;
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

function tokenEmissionOptionsFor(
  callId: number,
  emitTokens: boolean,
  config: WorkerRuntimeConfig
): {
  emitTokens: boolean;
  onRequestStarted?: (requestId: number) => void;
  tokenBatchSink?: (batch: TokenBatch) => void;
} {
  if (!emitTokens) {
    return { emitTokens: false };
  }
  if (config.wasmThreading !== 'pthread') {
    return {
      emitTokens: true,
      tokenBatchSink: (batch) => post({ kind: 'token-batch', callId, batch }),
    };
  }
  if (!postTokenRingReady()) {
    throw new QueryError(
      'STREAMING_UNAVAILABLE',
      'Pthread worker token streaming requires shared wasm memory. Serve the page with cross-origin isolation.'
    );
  }
  return {
    emitTokens: true,
    onRequestStarted: (requestId) =>
      post({ kind: 'token-ring-claim', callId, nativeRequestId: requestId }),
  };
}

async function handleRequest(message: WorkerOperationRequest): Promise<unknown> {
  switch (message.kind) {
    case 'models-load': {
      const result = await withAbortController(message.callId, (signal) =>
        ensureService(message.config).load(message.source, {
          ...message.options,
          signal,
          onProgress: postLoadProgress(message.callId),
        })
      );
      postTokenRingReady();
      return result;
    }
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
        const modelService = ensureService(message.config);
        const emission = tokenEmissionOptionsFor(
          message.callId,
          message.options.emitTokens,
          message.config
        );
        return modelService.runQuery(
          message.input,
          {
            ...message.options,
            signal,
            emitTokens: emission.emitTokens,
            onRequestStarted: emission.onRequestStarted,
            tokenBatchSink: emission.tokenBatchSink,
          }
        );
      });
    case 'chat':
      return await withAbortController(message.callId, (signal) => {
        const modelService = ensureService(message.config);
        const emission = tokenEmissionOptionsFor(
          message.callId,
          message.options.emitTokens,
          message.config
        );
        return modelService.runChat(
          message.input,
          {
            ...message.options,
            signal,
            emitTokens: emission.emitTokens,
            onRequestStarted: emission.onRequestStarted,
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
