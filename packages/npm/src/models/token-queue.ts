import type {
  BrowserEmbeddingRun,
  BrowserTextRun,
  BrowserTokenBatches,
  EmbeddingResult,
  GenerationResult,
  TokenBatch,
} from './types.js';
import { createLinkedAbortController } from '../utils/abort.js';

const TOKEN_QUEUE_CAPACITY = 256;

class BoundedTokenBatchQueue implements BrowserTokenBatches, AsyncIterator<TokenBatch> {
  private readonly items: Array<TokenBatch | undefined> = [];
  private readonly waiters: Array<(result: IteratorResult<TokenBatch>) => void> = [];
  private readIndex = 0;
  private closed = false;

  public push(batch: TokenBatch): void {
    if (this.closed) {
      return;
    }
    const waiter = this.waiters.shift();
    if (waiter != null) {
      waiter({ done: false, value: batch });
      return;
    }
    const pendingCount = this.items.length - this.readIndex;
    if (pendingCount >= TOKEN_QUEUE_CAPACITY) {
      const lastIndex = this.items.length - 1;
      this.items[lastIndex] = mergeTokenBatches(this.items[lastIndex]!, batch);
      return;
    }
    this.items.push(batch);
  }

  public close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    while (this.waiters.length > 0) {
      this.waiters.shift()?.({ done: true, value: undefined });
    }
  }

  public next(): Promise<IteratorResult<TokenBatch>> {
    const item = this.items[this.readIndex];
    if (item != null) {
      this.items[this.readIndex] = undefined;
      this.readIndex += 1;
      if (this.readIndex > 64 && this.readIndex * 2 >= this.items.length) {
        this.items.splice(0, this.readIndex);
        this.readIndex = 0;
      }
      return Promise.resolve({ done: false, value: item });
    }
    if (this.closed) {
      return Promise.resolve({ done: true, value: undefined });
    }
    return new Promise((resolve) => {
      this.waiters.push(resolve);
    });
  }

  public [Symbol.asyncIterator](): AsyncIterator<TokenBatch> {
    return this;
  }
}

/**
 * Create a browser text run with an exact coalescing token queue.
 */
export function createBrowserTextRun(
  options: { signal?: AbortSignal; emitTokens?: boolean },
  responseFactory: (
    tokenBatchSink: ((batch: TokenBatch) => void) | undefined,
    signal: AbortSignal
  ) => Promise<GenerationResult>
): BrowserTextRun {
  const linkedAbort = createLinkedAbortController(options.signal);
  const queue = new BoundedTokenBatchQueue();
  const response = responseFactory(
    options.emitTokens === true ? (batch) => queue.push(batch) : undefined,
    linkedAbort.signal
  ).finally(() => {
    queue.close();
    linkedAbort.dispose();
  });
  return {
    response,
    tokens: queue,
    cancel: () => {
      linkedAbort.controller.abort();
    },
  };
}

function mergeTokenBatches(left: TokenBatch, right: TokenBatch): TokenBatch {
  return {
    requestId: left.requestId,
    streamId: left.streamId,
    sequenceStart: left.sequenceStart,
    text: left.text + right.text,
    frameCount: left.frameCount + right.frameCount,
    byteCount: left.byteCount + right.byteCount,
    stats: right.stats,
  };
}

/**
 * Create a browser embedding run that owns cancellation for its response promise.
 */
export function createBrowserEmbeddingRun(
  signal: AbortSignal | undefined,
  responseFactory: (signal: AbortSignal) => Promise<EmbeddingResult>
): BrowserEmbeddingRun {
  const linkedAbort = createLinkedAbortController(signal);
  const response = responseFactory(linkedAbort.signal).finally(() => {
    linkedAbort.dispose();
  });
  return {
    response,
    cancel: () => {
      linkedAbort.controller.abort();
    },
  };
}
