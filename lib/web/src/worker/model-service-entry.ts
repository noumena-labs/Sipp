import { ModelService } from '../models/model-service.js';
import { AssetStore } from '../models/asset-store.js';
import { ModelRegistryStore } from '../models/model-registry-store.js';
import {
  QueryError,
  type AudioResult,
  type GenerationResult,
  type InternalTextRequestOptions,
  type TokenBatch,
} from '../models/types.js';
import { FileSystemStorage } from '../engine/file-system-storage.js';
import { resolveRuntimeAssetSelection } from '../engine/runtime-assets.js';
import { WasmEngineRuntime } from '../runtime/wasm/engine-runtime.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  type WorkerRuntimeConfig,
} from './model-service-protocol.js';

let service: ModelService | null = null;
let runtime: WasmEngineRuntime | null = null;
let serviceInitialization: Promise<ModelService> | null = null;
const activeCalls = new Map<number, AbortController>();
const activeOperations = new Map<number, Promise<void>>();
let shuttingDown = false;

type WorkerOperationRequest = Exclude<
  WorkerRequestMessage,
  { kind: 'initialize' | 'cancel' | 'shutdown' }
>;

async function createService(config: WorkerRuntimeConfig): Promise<ModelService> {
  const runtimeAssets = await resolveRuntimeAssetSelection(config);
  const storage = new FileSystemStorage(config.storageRoot);
  runtime = new WasmEngineRuntime(
    {
      moduleUrl: runtimeAssets.moduleUrl,
      wasmUrl: runtimeAssets.wasmUrl,
      wasmThreading: runtimeAssets.threading,
      moduleOptions: config.moduleOptions,
      maxModelBytes: config.maxModelBytes,
      storageRoot: config.storageRoot,
      browserCache: config.browserCache,
      trustedOrigins: config.trustedOrigins,
    },
    { backendConstraint: runtimeAssets.backendConstraint }
  );
  const initialized = new ModelService(
    runtime,
    new ModelRegistryStore(storage),
    new AssetStore(storage, config.browserCache)
  );
  initialized.subscribeObservability((event) => {
    post({ kind: 'observability-event', event });
  });
  initialized.subscribeEvents((event) => {
    post({ kind: 'engine-event', event });
  });
  service = initialized;
  return initialized;
}

function initializeService(config: WorkerRuntimeConfig): void {
  // One Worker configures one service. A failed initialization keeps its
  // rejected promise so every later operation reports the original cause
  // instead of a misleading "not initialized".
  serviceInitialization ??= createService(config).catch((error) => {
    service = null;
    runtime = null;
    throw error;
  });
  void serviceInitialization.catch(() => {});
}

async function requireService(): Promise<ModelService> {
  if (serviceInitialization == null) {
    throw new QueryError(
      'ENGINE_CLOSED',
      'Worker model service received a request before it was initialized.'
    );
  }
  return await serviceInitialization;
}

/**
 * Extracts the cleanup failures attached by `attachCleanupFailures`. Messages
 * cross the boundary explicitly rather than relying on browser-specific
 * structured-clone support for AggregateError.
 */
function cleanupFailureMessages(error: unknown): readonly string[] | undefined {
  const failures = (error as { cleanupFailures?: AggregateError } | null)
    ?.cleanupFailures;
  if (failures == null) {
    return undefined;
  }
  return failures.errors.map((failure: unknown) => toErrorMessage(failure));
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function post(message: WorkerResponseMessage, transfer: Transferable[] = []): void {
  self.postMessage(message, { transfer });
}

function postResolved(message: WorkerOperationRequest, value: unknown): void {
  if (message.kind === 'speak') {
    const audio = value as AudioResult;
    post(
      {
        kind: 'resolve',
        callId: message.callId,
        value: audio,
      },
      [audio.audio.buffer]
    );
    return;
  }
  post({
    kind: 'resolve',
    callId: message.callId,
    value,
  });
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
  emitTokens: boolean
): {
  emitTokens: boolean;
  onRequestStarted?: (requestId: number) => void;
  tokenBatchSink?: (batch: TokenBatch) => void;
} {
  if (!emitTokens) {
    return { emitTokens: false };
  }
  if (runtime?.getWasmThreadingMode() !== 'pthread') {
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

/**
 * Runs a token-emitting text request. Query and chat differ only in which
 * service method consumes the shared abort, emission, and streaming setup.
 */
async function runTextRequest(
  message: Extract<WorkerOperationRequest, { kind: 'query' | 'chat' }>,
  run: (
    modelService: ModelService,
    options: InternalTextRequestOptions
  ) => Promise<GenerationResult>
): Promise<GenerationResult> {
  return await withAbortController(message.callId, async (signal) => {
    const modelService = await requireService();
    const emission = tokenEmissionOptionsFor(message.callId, message.options.emitTokens);
    return await run(modelService, {
      ...message.options,
      signal,
      emitTokens: emission.emitTokens,
      onRequestStarted: emission.onRequestStarted,
      tokenBatchSink: emission.tokenBatchSink,
    });
  });
}

async function handleRequest(message: WorkerOperationRequest): Promise<unknown> {
  switch (message.kind) {
    case 'models-install':
      return await withAbortController(message.callId, async (signal) =>
        (await requireService()).add(message.source, {
          signal,
          onProgress: postLoadProgress(message.callId),
        })
      );
    case 'models-load': {
      const result = await withAbortController(message.callId, async (signal) =>
        (await requireService()).load(message.modelId, {
          ...message.options,
          signal,
          onProgress: postLoadProgress(message.callId),
        })
      );
      postTokenRingReady();
      return result;
    }
    case 'models-list':
      return await (await requireService()).list();
    case 'models-remove': {
      const modelService = await requireService();
      await modelService.remove(message.id);
      return modelService.current();
    }
    case 'query':
      return await runTextRequest(message, (modelService, options) =>
        modelService.runQuery(message.input, options)
      );
    case 'chat':
      return await runTextRequest(message, (modelService, options) =>
        modelService.runChat(message.input, options)
      );
    case 'embed':
      return await withAbortController(message.callId, async (signal) =>
        (await requireService()).runEmbedding(message.input, {
          ...message.options,
          signal,
        })
      );
    case 'listen':
      return await withAbortController(message.callId, async (signal) =>
        (await requireService()).runListen(message.audio, {
          ...message.options,
          signal,
        })
      );
    case 'speak':
      return await withAbortController(message.callId, async (signal) =>
        (await requireService()).runSpeak(message.text, {
          ...message.options,
          signal,
        })
      );
  }
}

function postRejected(callId: number, error: unknown): void {
  post({
    kind: 'reject',
    callId,
    message: toErrorMessage(error),
    errorName: error instanceof Error ? error.name : undefined,
    errorStack: error instanceof Error ? error.stack : undefined,
    queryErrorCode: error instanceof QueryError ? error.code : undefined,
    cleanupFailures: cleanupFailureMessages(error),
  });
}

async function processOperation(message: WorkerOperationRequest): Promise<void> {
  try {
    const value = await handleRequest(message);
    postResolved(message, value);
  } catch (error) {
    postRejected(message.callId, error);
  }
}

async function shutDown(message: Extract<WorkerRequestMessage, { kind: 'shutdown' }>): Promise<void> {
  shuttingDown = true;
  for (const controller of activeCalls.values()) {
    controller.abort();
  }
  try {
    await Promise.allSettled(activeOperations.values());
    await service?.close();
    service = null;
    runtime = null;
    serviceInitialization = null;
    post({ kind: 'resolve', callId: message.callId });
  } catch (error) {
    postRejected(message.callId, error);
  }
}

self.onmessage = (event: MessageEvent<WorkerRequestMessage>) => {
  const message = event.data;
  if (message.kind === 'initialize') {
    initializeService(message.config);
    return;
  }
  if (message.kind === 'cancel') {
    abortActiveCall(message.targetCallId);
    return;
  }
  if (message.kind === 'shutdown') {
    void shutDown(message);
    return;
  }
  if (shuttingDown) {
    postRejected(
      message.callId,
      new QueryError('ENGINE_CLOSED', 'Worker runtime is shutting down.')
    );
    return;
  }

  const operation = processOperation(message);
  activeOperations.set(message.callId, operation);
  void operation.finally(() => {
    activeOperations.delete(message.callId);
  });
};
