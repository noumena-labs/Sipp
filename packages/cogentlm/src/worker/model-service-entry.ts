import { ModelService } from '../model-management/model-service.js';
import { QueryError } from '../model-management/model-types.js';
import { getDefaultRuntimeUrls } from '../runtime-assets.js';
import { MainThreadEngineRuntime } from '../runtime/engine-runtime-main-thread.js';
import { StreamingRingWriter } from '../runtime/streaming-ring.js';
import { stableJson } from '../utils/stable-json.js';
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
// SAB streaming ring writer; set on `streaming-init`.  Null when SAB is
// unavailable — tokens then fall back to per-token postMessage.
let streamingRingWriter: StreamingRingWriter | null = null;

type WorkerOperationRequest = Exclude<
  WorkerRequestMessage,
  { kind: 'cancel' } | { kind: 'streaming-init' }
>;

function buildServiceConfig(config: WorkerSerializableCogentConfig): WorkerServiceConfig {
  const bundledRuntimeUrls =
    config.moduleUrl == null && config.wasmUrl == null
      ? getDefaultRuntimeUrls()
      : null;

  return {
    moduleUrl: config.moduleUrl ?? bundledRuntimeUrls?.moduleUrl,
    wasmUrl: config.wasmUrl ?? bundledRuntimeUrls?.wasmUrl,
    moduleOptions: config.moduleOptions,
    maxModelBytes: config.maxModelBytes,
    trustedOrigins: config.trustedOrigins,
  };
}

// Direct runtime handle for installing the SAB ring writer after ensureService.
let runtime: MainThreadEngineRuntime | null = null;

function ensureService(config: WorkerSerializableCogentConfig): ModelService {
  const fingerprint = stableJson(buildServiceConfig(config));
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
  unsubscribeObservability = service.subscribeObservability((event) => {
    post({ kind: 'observability-event', event });
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

// Fallback path used by StreamingBuffer mode only when the SAB ring isn't
// wired (or overflows).  Native StreamingBuffer drains write to the ring
// directly via the scheduler's `_ce_yield_drain` hook — this closure is
// only invoked by the DirectCallback fallback path.
function postToken(callId: number): (token: string) => void {
  return (token) => {
    if (
      streamingRingWriter != null &&
      streamingRingWriter.tryWriteString(callId, token)
    ) {
      return;
    }
    post({
      kind: 'token',
      callId,
      text: token,
    });
  };
}

async function runLoad(message: Extract<WorkerOperationRequest, { kind: 'models-load' }>): Promise<unknown> {
  return await withAbortController(message.callId, (signal) =>
    ensureService(message.config).load(message.source, {
      ...message.options,
      signal,
      onProgress: postLoadProgress(message.callId),
    })
  );
}

// Sends `streaming-claim` so main can map ring records (carrying native
// request id) to its callId.  Fires synchronously on enqueue, before any
// token bytes are produced.
//
// `streaming=false` MUST NOT inject an onToken — the engine selects
// TOKEN_EMISSION_NONE only when onToken is null.  Injecting postToken anyway
// would force StreamingBuffer/DirectCallback mode in worker runs even when
// the caller explicitly opted out of streaming.
function streamingOptionsFor(
  callId: number,
  streaming: boolean
): {
  onToken?: (token: string) => void;
  __internalRequestStarted?: (requestId: number) => void;
} {
  if (!streaming) {
    return {};
  }
  return {
    onToken: postToken(callId),
    __internalRequestStarted: (requestId) =>
      post({ kind: 'streaming-claim', callId, nativeRequestId: requestId }),
  };
}

async function runQuery(message: Extract<WorkerOperationRequest, { kind: 'query' }>): Promise<string> {
  return await withAbortController(message.callId, (signal) =>
    ensureService(message.config).query(message.input, {
      ...message.options,
      signal,
      ...streamingOptionsFor(message.callId, message.options.streaming),
    })
  );
}

async function runChat(message: Extract<WorkerOperationRequest, { kind: 'chat' }>): Promise<string> {
  return await withAbortController(message.callId, (signal) =>
    ensureService(message.config).chat(message.input, {
      ...message.options,
      signal,
      ...streamingOptionsFor(message.callId, message.options.streaming),
    })
  );
}

async function handleRequest(message: WorkerOperationRequest): Promise<unknown> {
  switch (message.kind) {
    case 'models-load':
      return await runLoad(message);
    case 'models-list':
      return await ensureService(message.config).list();
    case 'models-remove': {
      const modelService = ensureService(message.config);
      await modelService.remove(message.id);
      return modelService.currentModel();
    }
    case 'query':
      return await runQuery(message);
    case 'chat':
      return await runChat(message);
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
