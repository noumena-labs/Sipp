import {
  GenerateRequestId,
} from '../types.js';
import {
  WorkerRequestMessage,
  WorkerResponseMessage,
  WorkerLoadModelResult,
  WorkerRunQueuedRequestResult,
  WorkerBackendObservabilityResult,
} from './engine-runtime-worker-protocol.js';
import { WorkerEntryState } from './worker-entry-state.js';

const state = new WorkerEntryState();

async function handleInitModule(
  message: Extract<WorkerRequestMessage, { kind: 'init-module' }>
): Promise<void> {
  await state.initModule(message.config);
}

async function handleLoadModelUrl(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-url' }>
): Promise<WorkerLoadModelResult> {
  const runtime = state.ensureEngine();
  const signal = state.beginModelLoad(message.callId);
  try {
    const modelPath = await runtime.loadModelFromUrl(
      message.url,
      message.destFileName,
      (progressPct) => {
        state.postLoadProgress(message.callId, progressPct);
      },
      signal
    );

    return {
      modelPath,
      modelLoadInfo: runtime.getLastModelLoadInfo(),
      transportObservability: state.cloneTransportObservability(),
    };
  } finally {
    state.releaseModelLoad(message.callId);
  }
}

async function handleLoadModelFile(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-file' }>
): Promise<WorkerLoadModelResult> {
  const runtime = state.ensureEngine();
  const signal = state.beginModelLoad(message.callId);
  try {
    const modelPath = await runtime.loadModelFromFile(
      message.file,
      message.destFileName,
      (progressPct) => {
        state.postLoadProgress(message.callId, progressPct);
      },
      signal
    );

    return {
      modelPath,
      modelLoadInfo: runtime.getLastModelLoadInfo(),
      transportObservability: state.cloneTransportObservability(),
    };
  } finally {
    state.releaseModelLoad(message.callId);
  }
}

async function handleLoadModelFileShards(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-file-shards' }>
): Promise<WorkerLoadModelResult> {
  const runtime = state.ensureEngine();
  const signal = state.beginModelLoad(message.callId);
  try {
    const modelPath = await runtime.loadModelFromFileShards(
      message.files,
      (progressPct) => {
        state.postLoadProgress(message.callId, progressPct);
      },
      signal
    );

    return {
      modelPath,
      modelLoadInfo: runtime.getLastModelLoadInfo(),
      transportObservability: state.cloneTransportObservability(),
    };
  } finally {
    state.releaseModelLoad(message.callId);
  }
}

async function handleLoadModelUrls(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-urls' }>
): Promise<WorkerLoadModelResult> {
  const runtime = state.ensureEngine();
  const signal = state.beginModelLoad(message.callId);
  try {
    const modelPath = await runtime.loadModelFromUrls(
      message.urls,
      (progressPct) => {
        state.postLoadProgress(message.callId, progressPct);
      },
      signal
    );

    return {
      modelPath,
      modelLoadInfo: runtime.getLastModelLoadInfo(),
      transportObservability: state.cloneTransportObservability(),
    };
  } finally {
    state.releaseModelLoad(message.callId);
  }
}

async function handleLoadModelStreamStart(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-stream-start' }>
): Promise<WorkerLoadModelResult> {
  const runtime = state.ensureEngine();
  const { signal, stream } = state.beginStreamModelLoad(message.callId);

  try {
    const modelPath = await runtime.loadModelFromReadableStream(
      stream,
      message.destFileName,
      {
        expectedBytes: message.expectedBytes,
        signal,
        onProgress: (progressPct) => {
          state.postLoadProgress(message.callId, progressPct);
        },
      }
    );

    return {
      modelPath,
      modelLoadInfo: runtime.getLastModelLoadInfo(),
      transportObservability: state.cloneTransportObservability(),
    };
  } finally {
    state.releaseModelLoad(message.callId);
  }
}

function handleLoadModelStreamChunk(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-stream-chunk' }>
): void {
  state.enqueueStreamChunk(message.callId, message.chunk);
}

function handleLoadModelStreamEnd(
  message: Extract<WorkerRequestMessage, { kind: 'load-model-stream-end' }>
): void {
  state.closeStreamModelLoad(message.callId);
}

function handleCancelModelLoad(
  message: Extract<WorkerRequestMessage, { kind: 'cancel-model-load' }>
): void {
  state.abortModelLoad(message.callId);
}

async function handleQueuePrompt(
  message: Extract<WorkerRequestMessage, { kind: 'queue-prompt' }>
): Promise<GenerateRequestId> {
  const runtime = state.ensureEngine();
  const abortController = new AbortController();
  const requestId = await runtime.queuePrompt(
    message.contextKey,
    message.promptText,
    {
      nTokens: message.options.nTokens,
      promptFormat: message.options.promptFormat,
      signal: abortController.signal,
      onToken: (token) => {
        state.bufferTokenPiece(requestId, token);
      },
    }
  );
  state.rememberRequestAbortController(requestId, abortController);
  return requestId;
}

async function handleRunQueuedRequest(
  message: Extract<WorkerRequestMessage, { kind: 'run-queued-request' }>
): Promise<WorkerRunQueuedRequestResult> {
  const runtime = state.ensureEngine();
  state.markRequestRunning(message.requestId);
  try {
    const response = await runtime.runQueuedRequest(message.requestId);
    state.flushBufferedTokens(message.requestId);
    return {
      response,
      runtimeAggregateObservability: runtime.getRuntimeAggregateObservability(),
      transportObservability: state.cloneTransportObservability(),
    };
  } finally {
    state.releaseRequestResources(message.requestId);
    state.unmarkRequestRunning(message.requestId);
  }
}

async function handleCancelRequest(
  message: Extract<WorkerRequestMessage, { kind: 'cancel-request' }>
): Promise<boolean> {
  const runtime = state.ensureEngine();
  state.abortQueuedRequest(message.requestId);
  const cancelled = await runtime.cancelQueuedRequest(message.requestId);
  if (cancelled && !state.isRequestRunning(message.requestId)) {
    state.releaseRequestResources(message.requestId);
  }
  return cancelled;
}

async function handleGetBackendObservability(): Promise<WorkerBackendObservabilityResult> {
  const runtime = state.ensureEngine();
  return {
    backendObservability: await runtime.getBackendObservability(),
    transportObservability: state.cloneTransportObservability(),
  };
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
      case 'load-model-file-shards':
        value = await handleLoadModelFileShards(message);
        break;
      case 'load-model-urls':
        value = await handleLoadModelUrls(message);
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
        state.abortAllModelLoads();
        state.releaseAllRequestResources();
        value = await state.ensureEngine().initEngine(message.modelPath, message.config);
        state.setRuntimeObservabilityEnabled(
          message.config?.enableRuntimeObservability === true ||
            message.config?.enableBackendProfiling === true
        );
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
      message: state.toErrorMessage(error),
      errorName: error instanceof Error ? error.name : undefined,
    };
    self.postMessage(response);
  }
};
