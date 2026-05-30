import type { BrowserEmbeddingRun, BrowserTextRun, EmbeddingResult, GenerationResult, TokenBatch } from './types.js';
import { createLinkedAbortController } from '../utils/abort.js';

const TOKEN_QUEUE_CAPACITY = 256;

class BoundedTokenBatchQueue implements AsyncIterable<TokenBatch>, AsyncIterator<TokenBatch> {
  private readonly items: TokenBatch[] = [];
  private readonly waiters: Array<(result: IteratorResult<TokenBatch>) => void> = [];
  private closed = false;
  private pendingDroppedFrames = 0;

  public push(batch: TokenBatch): void {
    if (this.closed) {
      return;
    }
    if (this.pendingDroppedFrames > 0) {
      batch = {
        ...batch,
        stats: {
          ...batch.stats,
          framesDropped: batch.stats.framesDropped + this.pendingDroppedFrames,
        },
      };
      this.pendingDroppedFrames = 0;
    }
    const waiter = this.waiters.shift();
    if (waiter != null) {
      waiter({ done: false, value: batch });
      return;
    }
    if (this.items.length >= TOKEN_QUEUE_CAPACITY) {
      this.pendingDroppedFrames += batch.frameCount;
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
    const item = this.items.shift();
    if (item != null) {
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
 * Create a browser text run with a bounded best-effort token queue.
 */
export function createBrowserTextRun(
  options: { signal?: AbortSignal; streamTokens?: boolean },
  responseFactory: (
    emitTokens: ((batch: TokenBatch) => void) | undefined,
    signal: AbortSignal
  ) => Promise<GenerationResult>
): BrowserTextRun {
  const linkedAbort = createLinkedAbortController(options.signal);
  const queue = new BoundedTokenBatchQueue();
  const response = responseFactory(
    options.streamTokens === true ? (batch) => queue.push(batch) : undefined,
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
