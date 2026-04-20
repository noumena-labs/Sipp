//////////////////////////////////////////////////////////////////////////////
//
// character-agent.ts
//
// - High-level character turn loop that ties an engine instance to a
//   CharacterConfig: builds the system prompt, tracks memory, compiles the
//   grammar once, and exposes a single `chat()` async iterator that emits
//   prose/action events as they arrive.
//
//////////////////////////////////////////////////////////////////////////////

import type { GenerateRequestId, GenerateResponse, PromptOptions } from '../core/inference-types.js';
import { ActionBus, type CharacterEvent } from './action-bus.js';
import { compileActionGrammar } from './action-grammar.js';
import { StreamingActionParser, type ParsedEvent } from './action-parser.js';
import {
  DEFAULT_MEMORY_MAX_TURNS,
  resolveMaxMemoryTurns,
  type CharacterConfig,
} from './character-config.js';
import { renderSystemPrompt } from './persona.js';

/**
 * Minimal shape of the engine the agent needs. Defined structurally so tests
 * can supply a fake without pulling in the full CogentEngine class.
 */
export interface CharacterAgentEngine {
  queuePrompt(
    contextKey: string,
    promptText: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestId>;
  runQueuedRequest(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse>;
  cancelQueuedRequest?(requestId: GenerateRequestId): Promise<boolean>;
}

export interface CharacterAgentOptions {
  /**
   * Maximum tokens the model may emit per turn. Defaults to 256, which is a
   * reasonable upper bound for conversational replies.
   */
  readonly maxOutputTokens?: number;
  /**
   * Prebuilt ActionBus. When omitted, the agent creates one internally.
   * Injecting a shared bus lets multiple consumers observe the same agent.
   */
  readonly bus?: ActionBus;
}

export interface ChatTurn {
  readonly role: 'user' | 'assistant';
  readonly content: string;
}

/**
 * Turn-level chat event yielded by {@link CharacterAgent.chat}. Mirrors the
 * ActionBus event shape so consumers can choose either transport.
 */
export type ChatEvent = CharacterEvent;

/**
 * A character-driven conversation agent. Pair one with a CogentEngine and a
 * CharacterConfig to get a grammar-constrained, memory-aware chat loop.
 */
export class CharacterAgent {
  private readonly engine: CharacterAgentEngine;
  private readonly config: CharacterConfig;
  private readonly maxOutputTokens: number;
  private readonly systemPrompt: string;
  private readonly grammarSource: string;
  private readonly memoryLimitTurns: number;
  private readonly turnHistory: ChatTurn[] = [];
  private readonly eventBus: ActionBus;

  public constructor(
    engine: CharacterAgentEngine,
    config: CharacterConfig,
    options: CharacterAgentOptions = {}
  ) {
    this.engine = engine;
    this.config = config;
    this.maxOutputTokens = options.maxOutputTokens ?? 256;
    this.eventBus = options.bus ?? new ActionBus();
    this.systemPrompt = renderSystemPrompt(config.persona, config.actions);
    this.grammarSource = compileActionGrammar(config.actions);
    this.memoryLimitTurns = Math.max(
      0,
      resolveMaxMemoryTurns(config) ?? DEFAULT_MEMORY_MAX_TURNS
    );
  }

  /** Exposes the event bus for imperative subscribers (VRM bindings, logs). */
  public get bus(): ActionBus {
    return this.eventBus;
  }

  /** Read-only snapshot of the sliding-window memory. */
  public getMemory(): readonly ChatTurn[] {
    return this.turnHistory.slice();
  }

  /** Clears the sliding-window memory. Does not reset the engine's KV cache. */
  public clearMemory(): void {
    this.turnHistory.length = 0;
  }

  /** Compiled GBNF source — exposed for inspection and tests. */
  public getGrammarSource(): string {
    return this.grammarSource;
  }

  /** Final rendered system prompt — exposed for inspection and tests. */
  public getSystemPrompt(): string {
    return this.systemPrompt;
  }

  /**
   * Processes a single user turn. Returns an async iterator that yields
   * `ChatEvent`s as they arrive, terminating with a `turn-end` event.
   *
   * The iterator is backed by a small internal queue so upstream token
   * callbacks never block on a slow consumer — if the consumer falls
   * behind, events buffer in memory rather than back-pressuring decode.
   */
  public chat(userMessage: string, options: { signal?: AbortSignal } = {}): AsyncIterable<ChatEvent> {
    const trimmed = userMessage ?? '';
    const queue = new AsyncEventQueue<ChatEvent>();
    const parser = new StreamingActionParser();

    const emit = (event: ChatEvent): void => {
      queue.push(event);
      this.eventBus.emit(event);
    };

    const turnPrompt = this.buildTurnPrompt(trimmed);

    emit({ kind: 'turn-start', userMessage: trimmed });

    // Drive the engine in a detached async task so the async iterator can
    // begin delivering buffered events to the consumer immediately.
    void this.runTurn(turnPrompt, trimmed, parser, emit, queue, options.signal);

    return queue;
  }

  /** Builds the raw prompt text passed to `queuePrompt`. */
  private buildTurnPrompt(userMessage: string): string {
    const lines: string[] = [this.systemPrompt, ''];
    for (const turn of this.turnHistory) {
      lines.push(`${turn.role === 'user' ? 'User' : this.config.persona.name}: ${turn.content}`);
    }
    lines.push(`User: ${userMessage}`);
    lines.push(`${this.config.persona.name}:`);
    return lines.join('\n');
  }

  private async runTurn(
    promptText: string,
    userMessage: string,
    parser: StreamingActionParser,
    emit: (event: ChatEvent) => void,
    queue: AsyncEventQueue<ChatEvent>,
    signal: AbortSignal | undefined
  ): Promise<void> {
    let accumulatedText = '';
    let requestId: GenerateRequestId = 0;
    let cancelled = false;
    let errorMessage: string | undefined;

    const onToken = (token: string): void => {
      if (token.length === 0) {
        return;
      }
      accumulatedText += token;
      const events = parser.consume(token);
      for (const event of events) {
        emit(event);
      }
    };

    try {
      const promptOptions: PromptOptions = {
        nTokens: this.maxOutputTokens,
        grammar: this.grammarSource,
        onToken,
        signal,
      };
      requestId = await this.engine.queuePrompt(this.config.id, promptText, promptOptions);
      const response = await this.engine.runQueuedRequest(requestId, { signal });
      cancelled = response.cancelled;
      if (response.failed && response.errorMessage) {
        errorMessage = response.errorMessage;
      }
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : String(error);
      if (signal?.aborted) {
        cancelled = true;
      }
      // Best-effort cancel in case the runtime still holds the request.
      if (requestId !== 0 && this.engine.cancelQueuedRequest) {
        try {
          await this.engine.cancelQueuedRequest(requestId);
        } catch {
          // Swallow cancel errors — they are secondary to the original one.
        }
      }
    }

    // Drain any buffered prose or action still sitting in the parser.
    const trailing = parser.flush();
    for (const event of trailing) {
      emit(event);
    }

    if (!cancelled && !errorMessage) {
      this.pushTurnToMemory({ role: 'user', content: userMessage });
      this.pushTurnToMemory({ role: 'assistant', content: accumulatedText });
    }

    const endEvent = {
      kind: 'turn-end' as const,
      finalText: accumulatedText,
      cancelled,
      ...(errorMessage ? { errorMessage } : {}),
    };
    emit(endEvent);
    queue.close();
  }

  private pushTurnToMemory(turn: ChatTurn): void {
    this.turnHistory.push(turn);
    // Each "turn" in config counts a user+assistant pair; prune oldest pairs.
    const maxEntries = this.memoryLimitTurns * 2;
    while (this.turnHistory.length > maxEntries) {
      this.turnHistory.shift();
    }
  }
}

/**
 * Small async FIFO queue. Consumers await items via `for await`; producers
 * push items synchronously. `close()` marks the stream done and unblocks any
 * pending consumer.
 *
 * The implementation is purposely minimal and never drops events. For a
 * chat turn the expected queue depth is small (bounded by how many tokens
 * the model emits before the consumer re-enters the await).
 */
class AsyncEventQueue<T> implements AsyncIterable<T> {
  private readonly pendingValues: T[] = [];
  private readonly pendingResolvers: Array<(result: IteratorResult<T>) => void> = [];
  private closed = false;

  public push(value: T): void {
    if (this.closed) {
      return;
    }
    const resolver = this.pendingResolvers.shift();
    if (resolver) {
      resolver({ value, done: false });
      return;
    }
    this.pendingValues.push(value);
  }

  public close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    while (this.pendingResolvers.length > 0) {
      const resolver = this.pendingResolvers.shift();
      resolver?.({ value: undefined as never, done: true });
    }
  }

  public [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: (): Promise<IteratorResult<T>> => {
        if (this.pendingValues.length > 0) {
          const value = this.pendingValues.shift() as T;
          return Promise.resolve({ value, done: false });
        }
        if (this.closed) {
          return Promise.resolve({ value: undefined as never, done: true });
        }
        return new Promise<IteratorResult<T>>((resolve) => {
          this.pendingResolvers.push(resolve);
        });
      },
      return: (): Promise<IteratorResult<T>> => {
        this.close();
        return Promise.resolve({ value: undefined as never, done: true });
      },
    };
  }
}
