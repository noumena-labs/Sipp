import { CogentConfig } from '../cogent-config.js';
import { MainThreadEngineRuntime } from './engine-runtime-main-thread.js';
import {
  GenerateRequestId,
  TransportInfo,
} from '../types.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  WorkerSerializableCogentConfig,
  WorkerLoadModelResult,
  WorkerRunQueuedRequestResult,
  WorkerBackendInfoResult,
} from './engine-runtime-worker-protocol.js';

interface BufferedTokenState {
  text: string;
  tokenCount: number;
  timer: number | null;
}

const DEFAULT_MAX_BUFFERED_TOKENS = 8;
const DEFAULT_FLUSH_INTERVAL_MS = 16;

let engine: MainThreadEngineRuntime | null = null;
let workerConfig: WorkerSerializableCogentConfig | null = null;
const requestAbortControllers = new Map<GenerateRequestId, AbortController>();
const bufferedTokens = new Map<GenerateRequestId, BufferedTokenState>();

const transportInfo: TransportInfo = {
  executionMode: 'worker',
  workerBacked: true,
  backpressureEnabled: true,
  maxBufferedTokenCount: DEFAULT_MAX_BUFFERED_TOKENS,
  flushIntervalMs: DEFAULT_FLUSH_INTERVAL_MS,
  flushCount: 0,
  coalescedTokenCount: 0,
  maxObservedBufferedTokenCount: 0,
};

function cloneTransportInfo(): TransportInfo {
  return { ...transportInfo };
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function ensureEngine(): MainThreadEngineRuntime {
  if (engine == null) {
    throw new Error('Worker runtime is not initialized.');
  }
  return engine;
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
  transportInfo.flushCount += 1;
  transportInfo.coalescedTokenCount += state.tokenCount;
  transportInfo.maxObservedBufferedTokenCount = Math.max(
    transportInfo.maxObservedBufferedTokenCount,
    state.tokenCount
  );
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

  if (state.tokenCount >= transportInfo.maxBufferedTokenCount) {
    flushBufferedTokens(requestId);
    return;
  }

  if (state.timer == null) {
    state.timer = self.setTimeout(() => {
      flushBufferedTokens(requestId);
    }, transportInfo.flushIntervalMs);
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
  transportInfo.maxBufferedTokenCount =
    config.workerMaxBufferedTokens ?? DEFAULT_MAX_BUFFERED_TOKENS;
  transportInfo.flushIntervalMs =
    config.workerTokenFlushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS;

  return {
    ...config,
    executionMode: 'main-thread',
  };
}

async function handleInitModule(
  message: Extract<WorkerRequestMessage, { kind: 'init-module' }>
): Promise<void> {
  workerConfig = message.config;
  if (engine == null) {
    engine = new MainThreadEngineRuntime(buildEngineConfig(message.config));
  }
  await engine.initModule();
}

async function handleLoadModelUrl(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-url' }>
): Promise<WorkerLoadModelResult> {
  const runtime = ensureEngine();
  const modelPath = await runtime.loadModelFromUrl(
    message.url,
    message.destFileName,
    (progressPct) => {
      const progressMessage: WorkerResponseMessage = {
        kind: 'load-progress',
        callId: message.callId,
        progressPct,
      };
      self.postMessage(progressMessage);
    }
  );

  return {
    modelPath,
    modelLoadInfo: runtime.getLastModelLoadInfo(),
    transportInfo: cloneTransportInfo(),
  };
}

async function handleLoadModelFile(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-file' }>
): Promise<WorkerLoadModelResult> {
  const runtime = ensureEngine();
  const modelPath = await runtime.loadModelFromFile(
    message.file,
    message.destFileName,
    (progressPct) => {
      const progressMessage: WorkerResponseMessage = {
        kind: 'load-progress',
        callId: message.callId,
        progressPct,
      };
      self.postMessage(progressMessage);
    }
  );

  return {
    modelPath,
    modelLoadInfo: runtime.getLastModelLoadInfo(),
    transportInfo: cloneTransportInfo(),
  };
}

async function handleLoadModelBuffer(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-buffer' }>
): Promise<WorkerLoadModelResult> {
  const runtime = ensureEngine();
  const modelPath = runtime.loadModelFromBuffer(
    message.buffer,
    message.destFileName
  );
  return {
    modelPath,
    modelLoadInfo: runtime.getLastModelLoadInfo(),
    transportInfo: cloneTransportInfo(),
  };
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
      lastPromptPerformance: runtime.getLastPromptPerformance(),
      transportInfo: cloneTransportInfo(),
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

async function handleGetBackendInfo(): Promise<WorkerBackendInfoResult> {
  const runtime = ensureEngine();
  return {
    backendInfo: await runtime.getBackendInfo(),
    transportInfo: cloneTransportInfo(),
  };
}

async function handleGetTransportInfo(): Promise<TransportInfo> {
  return cloneTransportInfo();
}

async function handleGetLastModelLoadInfo() {
  return ensureEngine().getLastModelLoadInfo();
}

async function handleClose(): Promise<void> {
  if (engine != null) {
    engine.close();
  }
  engine = null;
  workerConfig = null;
  for (const requestId of bufferedTokens.keys()) {
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
      case 'load-model-buffer':
        value = await handleLoadModelBuffer(message);
        break;
      case 'init-engine':
        value = await ensureEngine().initEngine(message.modelPath, message.config);
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
      case 'get-backend-info':
        value = await handleGetBackendInfo();
        break;
      case 'get-transport-info':
        value = await handleGetTransportInfo();
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
