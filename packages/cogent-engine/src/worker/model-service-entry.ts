import { ModelService } from '../model-management/model-service.js';
import { QueryError } from '../model-management/model-types.js';
import { MainThreadEngineRuntime } from '../runtime/engine-runtime-main-thread.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  type WorkerSerializableCogentConfig,
  type WorkerServiceConfig,
} from './model-service-protocol.js';

let service: ModelService | null = null;
let serviceConfigFingerprint: string | null = null;
let unsubscribeObservability: (() => void) | null = null;
const activeCalls = new Map<number, AbortController>();

function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(',')}]`;
  }
  if (value != null && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => `${JSON.stringify(key)}:${stableJson(entry)}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
}

function buildServiceConfig(config: WorkerSerializableCogentConfig): WorkerServiceConfig {
  const bundledRuntimeUrls =
    config.moduleUrl == null && config.wasmUrl == null
      ? {
          moduleUrl: new URL('../../wasm/cogent-engine-wasm.js', import.meta.url).toString(),
          wasmUrl: new URL('../../wasm/cogent-engine-wasm.wasm', import.meta.url).toString(),
        }
      : null;

  return {
    moduleUrl: config.moduleUrl ?? bundledRuntimeUrls?.moduleUrl,
    wasmUrl: config.wasmUrl ?? bundledRuntimeUrls?.wasmUrl,
    moduleOptions: config.moduleOptions,
    maxModelBytes: config.maxModelBytes,
    trustedOrigins: config.trustedOrigins,
  };
}

function ensureService(config: WorkerSerializableCogentConfig): ModelService {
  const fingerprint = stableJson(buildServiceConfig(config));
  if (service != null) {
    if (serviceConfigFingerprint !== fingerprint) {
      throw new Error('Worker model service was initialized with different runtime options.');
    }
    return service;
  }

  service = new ModelService(
    new MainThreadEngineRuntime({
      ...buildServiceConfig(config),
      executionMode: 'worker',
    })
  );
  unsubscribeObservability = service.subscribeObservability((event) => {
    post({
      kind: 'observability-event',
      event,
    });
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

async function handleRequest(message: WorkerRequestMessage): Promise<unknown> {
  switch (message.kind) {
    case 'models-load':
      return await withAbortController(message.callId, (signal) =>
        ensureService(message.config).load(message.source, {
          ...message.options,
          signal,
          onProgress: (progress) => {
            post({
              kind: 'load-progress',
              callId: message.callId,
              progress,
            });
          },
        })
      );
    case 'models-list':
      return await ensureService(message.config).list();
    case 'models-remove': {
      const modelService = ensureService(message.config);
      await modelService.remove(message.id);
      return modelService.currentModel();
    }
    case 'query':
      return await withAbortController(message.callId, (signal) =>
        ensureService(message.config).query(message.input, {
          ...message.options,
          signal,
          onToken: (token) => {
            post({
              kind: 'token',
              callId: message.callId,
              text: token,
            });
          },
        })
      );
    case 'chat':
      return await withAbortController(message.callId, (signal) =>
        ensureService(message.config).chat(message.input, {
          ...message.options,
          signal,
          onToken: (token) => {
            post({
              kind: 'token',
              callId: message.callId,
              text: token,
            });
          },
        })
      );
    case 'close':
      for (const callId of activeCalls.keys()) {
        abortActiveCall(callId);
      }
      service?.close();
      unsubscribeObservability?.();
      unsubscribeObservability = null;
      service = null;
      serviceConfigFingerprint = null;
      return undefined;
    case 'cancel':
      abortActiveCall(message.targetCallId);
      return undefined;
    default:
      throw new Error('Unknown worker request kind.');
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
