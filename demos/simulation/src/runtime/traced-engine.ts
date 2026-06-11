import type {
  BrowserTextRun,
  BrowserTokenBatches,
  ChatInput,
  ChatMessage,
  ChatOptions,
  CogentClient,
  GenerationResult,
  QueryInput,
  QueryOptions,
  TokenBatch,
} from '@noumena-labs/cogentlm';
import type { CharacterRuntimeClient } from '@noumena-labs/cogentlm/character';
import type { DirectorRuntimeClient } from '@noumena-labs/cogentlm/director';
import type { BrainDefinition, BrainQueryType, BrainQueryStatus, BrainActivityStore } from './brain-activity-store.js';
import type { SimulationBus } from './bus.js';

export function createTracedBrainClient(
  client: CogentClient,
  store: BrainActivityStore,
  bus: SimulationBus,
  brain: BrainDefinition
): CharacterRuntimeClient & DirectorRuntimeClient {
  return new TracedBrainClient(client, store, bus, brain);
}

class TracedBrainClient implements CharacterRuntimeClient, DirectorRuntimeClient {
  public currentLocal(): ReturnType<CogentClient['currentLocal']> {
    return this.client.currentLocal();
  }

  public constructor(
    private readonly client: CogentClient,
    private readonly store: BrainActivityStore,
    private readonly bus: SimulationBus,
    private readonly brain: BrainDefinition
  ) { }

  public chat(input: ChatInput, options: ChatOptions = {}): BrowserTextRun {
    const messages = getChatMessages(input);
    const prompts = extractPromptSections(messages);
    const requestContextKey = options.contextKey ?? 'default';
    const directorTaskName = this.brain.kind === 'director' ? parseDirectorTaskName(requestContextKey) : null;
    const queryType = classifyQueryType(this.brain.kind, directorTaskName);
    const queryId = this.store.beginQuery({
      brainId: this.brain.id,
      queryType,
      ...(directorTaskName ? { queryName: directorTaskName } : {}),
      contextKey: requestContextKey,
      systemPrompt: prompts.systemPrompt,
      userPrompt: prompts.userPrompt,
      renderedPrompt: renderMessagesForTrace(messages),
      grammar: options.grammar ?? null,
    });

    const run = this.client.chat(input, {
      ...options,
      emitTokens: true,
    });
    return this.traceRun(run, queryId);
  }

  public query(input: QueryInput, options: QueryOptions = {}): BrowserTextRun {
    const promptText = typeof input === 'string' ? input : input.prompt;

    const requestContextKey = options.contextKey ?? 'default';
    const directorTaskName = this.brain.kind === 'director' ? parseDirectorTaskName(requestContextKey) : null;
    const queryType = classifyQueryType(this.brain.kind, directorTaskName);
    const queryId = this.store.beginQuery({
      brainId: this.brain.id,
      queryType,
      ...(directorTaskName ? { queryName: directorTaskName } : {}),
      contextKey: requestContextKey,
      systemPrompt: null,
      userPrompt: null,
      renderedPrompt: promptText,
      grammar: options.grammar ?? null,
    });

    const run = this.client.query(input, {
      ...options,
      emitTokens: true,
    });
    return this.traceRun(run, queryId);
  }

  private traceRun(run: BrowserTextRun, queryId: string): BrowserTextRun {
    const tokenQueue = new TokenBatchQueue();
    const tokenDrain = (async () => {
      for await (const batch of run.tokens) {
        this.recordTokens(queryId, batch);
        tokenQueue.push(batch);
      }
      tokenQueue.close();
    })().catch((error) => {
      tokenQueue.fail(error);
    });

    const response = run.response.then(
      async (result) => {
        await tokenDrain;
        this.finishSuccessfulQuery(queryId, result);
        return result;
      },
      (error) => {
        this.finishFailedQuery(queryId, error);
        throw error;
      }
    );

    return {
      response,
      tokens: tokenQueue,
      cancel: (reason?: unknown) => run.cancel(reason),
    };
  }

  private recordTokens(queryId: string, batch: TokenBatch): void {
    const tokens = batch.text.length === 0 ? [] : [batch.text];
    this.store.appendResponse(queryId, tokens);
    this.bus.emit({
      kind: 'agent-token',
      tick: 0, // Tick is not strictly needed for live UI updates, but part of schema
      agentId: this.brain.id,
      queryId,
      tokens,
    });
  }

  private finishSuccessfulQuery(queryId: string, result: GenerationResult): void {
    this.store.finishQuery(queryId, {
      status: 'completed',
      responseText: result.text,
      observability: result.stats,
      errorMessage: null,
    });
  }

  private finishFailedQuery(queryId: string, error: unknown): void {
    this.store.finishQuery(queryId, {
      status: classifyErrorStatus(error),
      errorMessage: asErrorMessage(error),
    });
  }
}

function getChatMessages(input: ChatInput): readonly ChatMessage[] {
  return isChatObjectInput(input) ? input.messages : input;
}

function isChatObjectInput(
  input: ChatInput
): input is { messages: readonly ChatMessage[]; media?: Uint8Array[] } {
  return !Array.isArray(input);
}

class TokenBatchQueue implements BrowserTokenBatches, AsyncIterator<TokenBatch> {
  private readonly items: TokenBatch[] = [];
  private readonly waiters: Array<{
    resolve: (result: IteratorResult<TokenBatch>) => void;
    reject: (reason?: unknown) => void;
  }> = [];
  private closed = false;
  private failed: unknown = null;

  public push(batch: TokenBatch): void {
    if (this.closed || this.failed != null) {
      return;
    }
    const waiter = this.waiters.shift();
    if (waiter != null) {
      waiter.resolve({ done: false, value: batch });
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
      this.waiters.shift()?.resolve({ done: true, value: undefined });
    }
  }

  public fail(error: unknown): void {
    if (this.closed || this.failed != null) {
      return;
    }
    this.failed = error;
    while (this.waiters.length > 0) {
      this.waiters.shift()?.reject(error);
    }
  }

  public next(): Promise<IteratorResult<TokenBatch>> {
    const item = this.items.shift();
    if (item != null) {
      return Promise.resolve({ done: false, value: item });
    }
    if (this.failed != null) {
      return Promise.reject(this.failed);
    }
    if (this.closed) {
      return Promise.resolve({ done: true, value: undefined });
    }
    return new Promise((resolve, reject) => {
      this.waiters.push({ resolve, reject });
    });
  }

  public [Symbol.asyncIterator](): AsyncIterator<TokenBatch> {
    return this;
  }
}

function extractPromptSections(messages: readonly ChatMessage[]): {
  systemPrompt: string;
  userPrompt: string;
} {
  const systemPrompt = messages
    .filter((message) => message.role === 'system')
    .map((message) => message.content.trim())
    .filter((message) => message.length > 0)
    .join('\n\n');
  const userPrompt = [...messages]
    .reverse()
    .find((message) => message.role === 'user')
    ?.content.trim() ?? '';
  return { systemPrompt, userPrompt };
}

function renderMessagesForTrace(messages: readonly ChatMessage[]): string {
  return messages.map((message) => `${message.role}: ${message.content}`).join('\n\n');
}

function parseDirectorTaskName(contextKey: string): string | null {
  const parts = contextKey.split(':').map((part) => part.trim()).filter((part) => part.length > 0);
  return parts.length > 0 ? parts[parts.length - 1]! : null;
}

function classifyQueryType(
  brainKind: BrainDefinition['kind'],
  directorTaskName: string | null
): BrainQueryType {
  if (brainKind === 'agent') {
    return 'decision';
  }
  if (directorTaskName?.includes('narrate')) {
    return 'narration';
  }
  return 'referee';
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function classifyErrorStatus(error: unknown): Exclude<BrainQueryStatus, 'idle' | 'running'> {
  if (!isAbortError(error)) {
    return asErrorMessage(error).toLowerCase().includes('timed out') ? 'timed_out' : 'failed';
  }
  return asErrorMessage(error).toLowerCase().includes('timed out') ? 'timed_out' : 'aborted';
}

function asErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
