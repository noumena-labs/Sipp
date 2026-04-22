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

import type {
  ChatMessage,
  GenerateRequestId,
  GenerateResponse,
  PromptOptions,
} from '../core/inference-types.js';
import { ActionBus, type CharacterEvent } from './action-bus.js';
import { compileActionGrammar } from './action-grammar.js';
import { StreamingActionParser, type ParsedEvent } from './action-parser.js';
import {
  DEFAULT_MEMORY_MAX_TURNS,
  resolveMaxMemoryTurns,
  type CharacterConfig,
} from './character-config.js';
import { buildAppliedChatTemplateContext } from './chat-template-metadata.js';
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
  /** Returns the Jinja chat_template string embedded in the GGUF, or null. */
  getChatTemplate?(): string | null;
  /** Returns the model's BOS token rendered as text (may be empty). */
  getBosText?(): string;
  /** Returns the model's EOS token rendered as text (may be empty). */
  getEosText?(): string;
  applyChatTemplate(messages: Array<{ role: string; content: string }>, addAssistant: boolean): Promise<string>;
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

interface ResolvedPromptContext {
  readonly promptText: string;
  readonly boundaryMarkers: readonly string[];
  readonly templateSource: string | null;
}

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
  private turnSequence = 0;

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

    const emit = (event: ChatEvent): void => {
      queue.push(event);
      this.eventBus.emit(event);
    };

    emit({ kind: 'turn-start', userMessage: trimmed });

    // Drive the engine in a detached async task so the async iterator can
    // begin delivering buffered events to the consumer immediately.
    void (async () => {
      const parser = new StreamingActionParser(this.config.actions);
      const turnMessages = this.buildTurnMessages(trimmed);
      try {
        const promptContext = await this.buildPromptContext(turnMessages);
        await this.runTurn(promptContext, turnMessages, trimmed, parser, emit, queue, options.signal);
      } catch (error) {
        emit({
          kind: 'turn-end',
          finalText: '',
          cancelled: false,
          errorMessage: error instanceof Error ? error.message : String(error),
        });
        queue.close();
      }
    })();

    return queue;
  }

  /**
   * Builds the ordered ChatMessage[] for the current turn. This sequence is
   * rendered through the model's embedded chat template so the character
   * system prompt and turn history follow the model's native chat contract.
   */
  private buildTurnMessages(userMessage: string): ChatMessage[] {
    const messages: ChatMessage[] = [{ role: 'system', content: this.systemPrompt }];
    for (const turn of this.turnHistory) {
      messages.push({ role: turn.role, content: turn.content });
    }
    messages.push({ role: 'user', content: userMessage });
    return messages;
  }

  /**
   * Renders the full conversation through the model's embedded chat template
   * and derives assistant boundaries from that applied template output.
   */
  private async buildPromptContext(messages: ChatMessage[]): Promise<ResolvedPromptContext> {
    return buildAppliedChatTemplateContext(this.engine, messages);
  }

  private async runTurn(
    promptContext: ResolvedPromptContext,
    messages: ChatMessage[],
    userMessage: string,
    parser: StreamingActionParser,
    emit: (event: ChatEvent) => void,
    queue: AsyncEventQueue<ChatEvent>,
    signal: AbortSignal | undefined
  ): Promise<void> {
    let assistantProse = '';
    let rawOutputText = '';
    let pendingText = '';
    let requestId: GenerateRequestId = 0;
    let cancelled = false;
    let errorMessage: string | undefined;
    let reachedBoundary = false;
    const contextKey = this.nextTurnContextKey();
    const promptText = promptContext.promptText;
    const boundaryMarkers = promptContext.boundaryMarkers;

    console.info('[CharacterAgent] queuePrompt input', {
      characterId: this.config.id,
      contextKey,
      templateSourcePreview: promptContext.templateSource?.slice(0, 160) ?? null,
      userMessage,
      messages,
      promptTextPreview: promptText.slice(0, 400),
      promptTextByteLength: promptText.length,
      maxOutputTokens: this.maxOutputTokens,
      boundaryMarkers,
      // Temporarily disable action grammar while debugging prompt quality so
      // we can compare unconstrained model responses 1:1 against the
      // grammar-constrained path.
      grammar: this.grammarSource,
      //grammar: undefined,
    });

    const emitParsedText = (text: string): void => {
      if (text.length === 0) {
        return;
      }
      const events = parser.consume(text);
      for (const event of events) {
        if (event.kind === 'prose') {
          assistantProse += event.text;
        }
        emit(event);
      }
    };

    const requestBoundaryStop = (): void => {
      if (requestId === 0 || signal?.aborted || !this.engine.cancelQueuedRequest) {
        return;
      }
      void this.engine.cancelQueuedRequest(requestId).catch(() => {
        // Best-effort only; generation may already be terminating naturally.
      });
    };

    const onToken = (token: string): void => {
      if (token.length === 0 || reachedBoundary) {
        return;
      }
      rawOutputText += token;
      pendingText += token;

      const split = splitOnAssistantBoundary(pendingText, boundaryMarkers);
      emitParsedText(split.safeText);
      pendingText = split.trailingText;

      if (split.hitBoundary) {
        pendingText = '';
        reachedBoundary = true;
        requestBoundaryStop();
      }
    };

    try {
      const promptOptions: PromptOptions = {
        nTokens: this.maxOutputTokens,
        promptFormat: 'raw',
        grammar: this.grammarSource,
        onToken,
        signal,
      };
      requestId = await this.engine.queuePrompt(contextKey, promptText, promptOptions);
      const response = await this.engine.runQueuedRequest(requestId, { signal });
      rawOutputText = response.outputText;
      cancelled = response.cancelled && !reachedBoundary;
      if (response.failed && response.errorMessage) {
        errorMessage = response.errorMessage;
      }
    } catch (error) {
      if (reachedBoundary && !signal?.aborted) {
        cancelled = false;
      } else {
        errorMessage = error instanceof Error ? error.message : String(error);
      }
      if (signal?.aborted && !reachedBoundary) {
        cancelled = true;
      }
      // Best-effort cancel in case the runtime still holds the request.
      if (requestId !== 0 && this.engine.cancelQueuedRequest && !reachedBoundary) {
        try {
          await this.engine.cancelQueuedRequest(requestId);
        } catch {
          // Swallow cancel errors — they are secondary to the original one.
        }
      }
    }

    if (!reachedBoundary) {
      emitParsedText(trimTrailingBoundaryPrefix(pendingText, boundaryMarkers));
    }

    const finalText = this.buildFinalText(rawOutputText, boundaryMarkers).trim();

    console.info('[CharacterAgent] raw LLM output', {
      characterId: this.config.id,
      contextKey,
      requestId,
      cancelled,
      errorMessage,
      reachedBoundary,
      rawOutputText,
      parsedProseText: finalText,
    });

    if (!cancelled && !errorMessage) {
      this.pushTurnToMemory({ role: 'user', content: userMessage });
      this.pushTurnToMemory({ role: 'assistant', content: finalText });
    }

    const endEvent = {
      kind: 'turn-end' as const,
      finalText,
      cancelled,
      ...(errorMessage ? { errorMessage } : {}),
    };
    emit(endEvent);
    queue.close();
  }

  private nextTurnContextKey(): string {
    this.turnSequence += 1;
    return `${this.config.id}::turn-${this.turnSequence}`;
  }

  private buildFinalText(rawOutputText: string, boundaryMarkers: readonly string[]): string {
    const parser = new StreamingActionParser(this.config.actions);
    const cleaned = sanitizeAssistantMemoryText(rawOutputText, boundaryMarkers);
    const events = parser.consume(cleaned);
    const trailing = parser.flush();
    let prose = '';
    for (const event of [...events, ...trailing]) {
      if (event.kind === 'prose') {
        prose += event.text;
      }
    }
    return prose;
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

function splitOnAssistantBoundary(
  text: string,
  boundaryMarkers: readonly string[]
): { safeText: string; trailingText: string; hitBoundary: boolean } {
  let earliestIndex = -1;
  let matchedMarker = '';

  for (const marker of boundaryMarkers) {
    if (marker.length === 0) {
      continue;
    }
    const index = text.indexOf(marker);
    if (index >= 0 && (earliestIndex < 0 || index < earliestIndex)) {
      earliestIndex = index;
      matchedMarker = marker;
    }
  }

  if (earliestIndex >= 0) {
    return {
      safeText: text.slice(0, earliestIndex),
      trailingText: text.slice(earliestIndex + matchedMarker.length),
      hitBoundary: true,
    };
  }

  let safeLength = text.length;
  for (const marker of boundaryMarkers) {
    if (marker.length <= 1) {
      continue;
    }
    const overlap = longestSuffixPrefixOverlap(text, marker);
    safeLength = Math.min(safeLength, text.length - overlap);
  }

  return {
    safeText: text.slice(0, safeLength),
    trailingText: text.slice(safeLength),
    hitBoundary: false,
  };
}

function trimTrailingBoundaryPrefix(
  text: string,
  boundaryMarkers: readonly string[]
): string {
  let out = text;
  let changed = true;
  while (changed && out.length > 0) {
    changed = false;
    for (const marker of boundaryMarkers) {
      if (marker.length === 0) {
        continue;
      }
      if (marker.startsWith(out)) {
        out = '';
        changed = true;
        break;
      }
    }
  }
  return out;
}

function sanitizeAssistantMemoryText(
  text: string,
  boundaryMarkers: readonly string[]
): string {
  const split = splitOnAssistantBoundary(text, boundaryMarkers);
  return split.safeText;
}

function longestSuffixPrefixOverlap(source: string, marker: string): number {
  const maxLength = Math.min(source.length, marker.length - 1);
  for (let length = maxLength; length > 0; length -= 1) {
    if (source.endsWith(marker.slice(0, length))) {
      return length;
    }
  }
  return 0;
}
