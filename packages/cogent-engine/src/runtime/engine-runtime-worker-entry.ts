import { CogentConfig } from '../cogent-config.js';
import { MainThreadEngineRuntime } from './engine-runtime-main-thread.js';
import {
  GenerateRequestId,
  TransportObservability,
} from '../types.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  WorkerSerializableCogentConfig,
  WorkerLoadModelResult,
  WorkerRunQueuedRequestResult,
  WorkerBackendObservabilityResult,
} from './engine-runtime-worker-protocol.js';

interface BufferedTokenState {
  text: string;
  tokenCount: number;
  timer: number | null;
}

interface ActiveModelLoadState {
  abortController: AbortController;
  streamController: ReadableStreamDefaultController<Uint8Array> | null;
}

const DEFAULT_MAX_BUFFERED_TOKENS = 8;
const DEFAULT_FLUSH_INTERVAL_MS = 16;

let engine: MainThreadEngineRuntime | null = null;
const requestAbortControllers = new Map<GenerateRequestId, AbortController>();
const bufferedTokens = new Map<GenerateRequestId, BufferedTokenState>();
const activeModelLoads = new Map<number, ActiveModelLoadState>();

const transportObservability: TransportObservability = {
  executionMode: 'worker',
  workerBacked: true,
  enabled: false,
  bufferedTokenLimit: DEFAULT_MAX_BUFFERED_TOKENS,
  flushIntervalMs: DEFAULT_FLUSH_INTERVAL_MS,
  flushCount: 0,
  coalescedTokenCount: 0,
  maxObservedBufferedTokenCount: 0,
};

function cloneTransportObservability(): TransportObservability {
  return { ...transportObservability };
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function createAbortError(message = 'The operation was aborted.'): Error {
  if (typeof DOMException === 'function') {
    return new DOMException(message, 'AbortError');
  }
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

function ensureEngine(): MainThreadEngineRuntime {
  if (engine == null) {
    throw new Error('Worker runtime is not initialized.');
  }
  return engine;
}

function postLoadProgress(callId: number, progressPct: number): void {
  const progressMessage: WorkerResponseMessage = {
    kind: 'load-progress',
    callId,
    progressPct,
  };
  self.postMessage(progressMessage);
}

function releaseModelLoad(callId: number): void {
  activeModelLoads.delete(callId);
}

function abortModelLoad(callId: number): void {
  const loadState = activeModelLoads.get(callId);
  if (loadState == null) {
    return;
  }
  loadState.abortController.abort();
  if (loadState.streamController != null) {
    loadState.streamController.error(createAbortError('Model load aborted.'));
    loadState.streamController = null;
  }
}

function flushBufferedTokens(requestId: GenerateRequestId): void {
  const state = bufferedTokens.get(requestId);
  if (state == null || state.text.length === 0) {
    return;
  }

  if (state.timer != null) {
    clearTimeout(state.timer);
    state.timer = null;
  }

  const payload: WorkerResponseMessage = {
    kind: 'token',
    requestId,
    text: state.text,
    bufferedTokenCount: state.tokenCount,
  };
  self.postMessage(payload);
  if (transportObservability.enabled) {
    transportObservability.flushCount += 1;
    transportObservability.coalescedTokenCount += state.tokenCount;
    transportObservability.maxObservedBufferedTokenCount = Math.max(
      transportObservability.maxObservedBufferedTokenCount,
      state.tokenCount
    );
  }
  state.text = '';
  state.tokenCount = 0;
}

function bufferTokenPiece(requestId: GenerateRequestId, token: string): void {
  let state = bufferedTokens.get(requestId);
  if (state == null) {
    state = {
      text: '',
      tokenCount: 0,
      timer: null,
    };
    bufferedTokens.set(requestId, state);
  }

  state.text += token;
  state.tokenCount += 1;

  if (state.tokenCount >= transportObservability.bufferedTokenLimit) {
    flushBufferedTokens(requestId);
    return;
  }

  if (state.timer == null) {
    state.timer = self.setTimeout(() => {
      flushBufferedTokens(requestId);
    }, transportObservability.flushIntervalMs);
  }
}

function releaseRequestResources(requestId: GenerateRequestId): void {
  const state = bufferedTokens.get(requestId);
  if (state?.timer != null) {
    clearTimeout(state.timer);
  }
  bufferedTokens.delete(requestId);
  requestAbortControllers.delete(requestId);
}

function buildEngineConfig(config: WorkerSerializableCogentConfig): CogentConfig {
  transportObservability.bufferedTokenLimit =
    config.workerMaxBufferedTokens ?? DEFAULT_MAX_BUFFERED_TOKENS;
  transportObservability.flushIntervalMs =
    config.workerTokenFlushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS;

  return {
    ...config,
    executionMode: 'main-thread',
  };
}

async function handleInitModule(
  message: Extract<WorkerRequestMessage, { kind: 'init-module' }>
): Promise<void> {
  if (engine == null) {
    engine = new MainThreadEngineRuntime(buildEngineConfig(message.config));
  }
  await engine.initModule();
}

async function handleLoadModelUrl(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-url' }>
): Promise<WorkerLoadModelResult> {
  const runtime = ensureEngine();
  const abortController = new AbortController();
  activeModelLoads.set(message.callId, {
    abortController,
    streamController: null,
  });
  try {
    const modelPath = await runtime.loadModelFromUrl(
      message.url,
      message.destFileName,
      (progressPct) => {
        postLoadProgress(message.callId, progressPct);
      },
      abortController.signal
    );

    return {
      modelPath,
      modelLoadInfo: runtime.getLastModelLoadInfo(),
      transportObservability: cloneTransportObservability(),
    };
  } finally {
    releaseModelLoad(message.callId);
  }
}

async function handleLoadModelFile(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-file' }>
): Promise<WorkerLoadModelResult> {
  const runtime = ensureEngine();
  const abortController = new AbortController();
  activeModelLoads.set(message.callId, {
    abortController,
    streamController: null,
  });
  try {
    const modelPath = await runtime.loadModelFromFile(
      message.file,
      message.destFileName,
      (progressPct) => {
        postLoadProgress(message.callId, progressPct);
      },
      abortController.signal
    );

    return {
      modelPath,
      modelLoadInfo: runtime.getLastModelLoadInfo(),
      transportObservability: cloneTransportObservability(),
    };
  } finally {
    releaseModelLoad(message.callId);
  }
}

async function handleLoadModelStreamStart(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-stream-start' }>
): Promise<WorkerLoadModelResult> {
  const runtime = ensureEngine();
  const abortController = new AbortController();
  const loadState: ActiveModelLoadState = {
    abortController,
    streamController: null,
  };
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      loadState.streamController = controller;
    },
  });

  activeModelLoads.set(message.callId, loadState);

  try {
    const modelPath = await runtime.loadModelFromReadableStream(
      stream,
      message.destFileName,
      {
        expectedBytes: message.expectedBytes,
        signal: abortController.signal,
        onProgress: (progressPct) => {
          postLoadProgress(message.callId, progressPct);
        },
      }
    );

    return {
      modelPath,
      modelLoadInfo: runtime.getLastModelLoadInfo(),
      transportObservability: cloneTransportObservability(),
    };
  } finally {
    releaseModelLoad(message.callId);
  }
}

function handleLoadModelStreamChunk(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-stream-chunk' }>
): void {
  const loadState = activeModelLoads.get(message.callId);
  if (loadState?.streamController == null) {
    throw new Error(`No active model stream for call ${message.callId}.`);
  }

  loadState.streamController.enqueue(new Uint8Array(message.chunk));
  const response: WorkerResponseMessage = {
    kind: 'load-stream-ack',
    callId: message.callId,
  };
  self.postMessage(response);
}

function handleLoadModelStreamEnd(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-stream-end' }>
): void {
  const loadState = activeModelLoads.get(message.callId);
  if (loadState?.streamController == null) {
    return;
  }

  loadState.streamController.close();
  loadState.streamController = null;
}

function handleCancelModelLoad(
  message: Extract<WorkerRequestMessage, { kind: 'cancel-model-load' }>
): void {
  abortModelLoad(message.callId);
}

async function handleQueuePrompt(
  message: Extract<WorkerRequestMessage, { kind: 'queue-prompt' }>
): Promise<GenerateRequestId> {
  const runtime = ensureEngine();
  const abortController = new AbortController();
  const requestId = await runtime.queuePrompt(
    message.contextKey,
    message.promptText,
    {
      nTokens: message.options.nTokens,
      promptFormat: message.options.promptFormat,
      signal: abortController.signal,
      onToken: (token) => {
        bufferTokenPiece(requestId, token);
      },
    }
  );
  requestAbortControllers.set(requestId, abortController);
  return requestId;
}

async function handleRunQueuedRequest(
  message: Extract<WorkerRequestMessage, { kind: 'run-queued-request' }>
): Promise<WorkerRunQueuedRequestResult> {
  const runtime = ensureEngine();
  try {
    const response = await runtime.runQueuedRequest(message.requestId);
    flushBufferedTokens(message.requestId);
    return {
      response,
      runtimeObservability: runtime.getRuntimeObservability(),
      transportObservability: cloneTransportObservability(),
    };
  } finally {
    releaseRequestResources(message.requestId);
  }
}

async function handleCancelRequest(
  message: Extract<WorkerRequestMessage, { kind: 'cancel-request' }>
): Promise<boolean> {
  const runtime = ensureEngine();
  requestAbortControllers.get(message.requestId)?.abort();
  const cancelled = await runtime.cancelQueuedRequest(message.requestId);
  return cancelled;
}

async function handleGetBackendObservability(): Promise<WorkerBackendObservabilityResult> {
  const runtime = ensureEngine();
  return {
    backendObservability: await runtime.getBackendObservability(),
    transportObservability: cloneTransportObservability(),
  };
}

async function handleGetTransportObservability(): Promise<TransportObservability> {
  return cloneTransportObservability();
}

async function handleGetLastModelLoadInfo() {
  return ensureEngine().getLastModelLoadInfo();
}

async function handleClose(): Promise<void> {
  for (const callId of activeModelLoads.keys()) {
    abortModelLoad(callId);
  }
  activeModelLoads.clear();
  if (engine != null) {
    engine.close();
  }
  engine = null;
  for (const requestId of bufferedTokens.keys()) {
    releaseRequestResources(requestId);
  }
  for (const requestId of requestAbortControllers.keys()) {
    releaseRequestResources(requestId);
  }
}

self.onmessage = async (event: MessageEvent<WorkerRequestMessage>) => {
  const message = event.data;
  try {
    let value: unknown;
    switch (message.kind) {
      case 'init-module':
        value = await handleInitModule(message);
        break;
      case 'load-model-url':
        value = await handleLoadModelUrl(message);
        break;
      case 'load-model-file':
        value = await handleLoadModelFile(message);
        break;
      case 'load-model-stream-start':
        value = await handleLoadModelStreamStart(message);
        break;
      case 'load-model-stream-chunk':
        handleLoadModelStreamChunk(message);
        return;
      case 'load-model-stream-end':
        handleLoadModelStreamEnd(message);
        return;
      case 'cancel-model-load':
        handleCancelModelLoad(message);
        return;
      case 'init-engine':
        value = await ensureEngine().initEngine(message.modelPath, message.config);
        transportObservability.enabled = message.config?.enableRuntimeObservability === true
          || message.config?.enableBackendProfiling === true;
        break;
      case 'queue-prompt':
        value = await handleQueuePrompt(message);
        break;
      case 'run-queued-request':
        value = await handleRunQueuedRequest(message);
        break;
      case 'cancel-request':
        value = await handleCancelRequest(message);
        break;
      case 'get-backend-observability':
        value = await handleGetBackendObservability();
        break;
      case 'get-transport-observability':
        value = await handleGetTransportObservability();
        break;
      case 'get-last-model-load-info':
        value = await handleGetLastModelLoadInfo();
        break;
      case 'close':
        value = await handleClose();
        break;
      default:
        throw new Error('Unknown worker request kind.');
    }

    const response: WorkerResponseMessage = {
      kind: 'resolve',
      callId: message.callId,
      value,
    };
    self.postMessage(response);
  } catch (error) {
    const response: WorkerResponseMessage = {
      kind: 'reject',
      callId: message.callId,
      message: toErrorMessage(error),
      errorName: error instanceof Error ? error.name : undefined,
    };
    self.postMessage(response);
  }
};
