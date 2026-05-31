//////////////////////////////////////////////////////////////////////////////
//
// character-agent.ts
//
// - High-level character runtime that ties a chat client to a
//   CharacterConfig: builds the system prompt, tracks memory, compiles the
//   grammar once, and exposes a single `chat()` async iterator that emits
//   prose/action events as they arrive.
//
//////////////////////////////////////////////////////////////////////////////

import type {
  BrowserTextRun,
  ChatInput,
  ChatOptions,
} from '../models/types.js';
import type { ChatMessage } from '../engine/inference-types.js';
import { sliceUndeliveredSuffix } from '../engine/chat-boundary-sanitizer.js';
import { CharacterEventBus, type CharacterEvent } from './action-bus.js';
import { compileActionGrammar } from './action-grammar.js';
import { IncrementalActionParser, type ParsedEvent } from './action-parser.js';
import { compileChoiceGrammar, parseChoiceOutput } from './choice-grammar.js';
import {
  resolveMaxMemoryTurns,
  type CharacterConfig,
} from './character-config.js';
import { summarizeActionCues } from './action-schema.js';
import { renderSystemPrompt } from './persona.js';
import { createTimedAbortController } from '../utils/abort.js';

export type RunStatus =
  | 'ok'
  | 'aborted'
  | 'timed_out'
  | 'failed'
  | 'invalid_request'
  | 'invalid_response';

/** Minimal chat client required by character runtimes. */
export interface CharacterRuntimeClient {
  chat(input: ChatInput, options?: ChatOptions): BrowserTextRun;
}

export interface CharacterRuntimeOptions {
  /**
   * Maximum tokens the model may emit per turn. Defaults to 256, which is a
   * reasonable upper bound for conversational replies.
   */
  readonly maxOutputTokens?: number;
  /**
   * Prebuilt CharacterEventBus. When omitted, the runtime creates one internally.
   * Injecting a shared bus lets multiple consumers observe the same character.
   */
  readonly bus?: CharacterEventBus;
  /** Stable client context key prefix. Defaults to `character:${config.id}`. */
  readonly contextKey?: string;
}

export interface ChatTurn {
  readonly role: 'user' | 'assistant';
  readonly content: string;
}

export interface CharacterChooseResult {
  readonly selection: string | null;
  readonly status: RunStatus;
  readonly errorMessage?: string;
  readonly rawText: string;
}

export interface CharacterChooseOptions {
  readonly choices: readonly string[];
  readonly signal?: AbortSignal;
  readonly timeoutMs?: number;
  readonly maxOutputTokens?: number;
}

/**
 * Turn-level chat event yielded by {@link CharacterRuntime.chat}. Mirrors the
 * CharacterEventBus event shape so consumers can choose either transport.
 */
export type ChatEvent = CharacterEvent;

interface InFlightTurn {
  readonly controller: AbortController;
  readonly done: Promise<void>;
}

/**
 * A character-driven conversation runtime. Pair one with a CogentClient and a
 * CharacterConfig to get a grammar-constrained, memory-aware chat loop.
 */
export class CharacterRuntime {
  private readonly client: CharacterRuntimeClient;
  private readonly config: CharacterConfig;
  private readonly maxOutputTokens: number;
  private readonly contextKey: string;
  private readonly systemPrompt: string;
  private readonly grammarSource: string;
  private readonly memoryLimitTurns: number;
  private readonly canonicalCueLabelsByActionId: ReadonlyMap<string, string>;
  private readonly turnHistory: ChatTurn[] = [];
  private readonly eventBus: CharacterEventBus;
  private currentTurn: InFlightTurn | undefined;

  public constructor(
    client: CharacterRuntimeClient,
    config: CharacterConfig,
    options: CharacterRuntimeOptions = {}
  ) {
    this.client = client;
    this.config = config;
    this.maxOutputTokens = options.maxOutputTokens ?? 256;
    this.contextKey = options.contextKey ?? `character:${config.id}`;
    this.eventBus = options.bus ?? new CharacterEventBus();
    this.systemPrompt = renderSystemPrompt(config.persona, config.actions);
    this.grammarSource = compileActionGrammar(config.actions);
    this.canonicalCueLabelsByActionId = new Map(
      summarizeActionCues(config.actions).map((cue) => [cue.id, cue.label])
    );
    this.memoryLimitTurns = Math.max(0, resolveMaxMemoryTurns(config));
  }

  /** Exposes the event bus for imperative subscribers (VRM bindings, logs). */
  public get bus(): CharacterEventBus {
    return this.eventBus;
  }

  /** Read-only snapshot of the sliding-window memory. */
  public getMemory(): readonly ChatTurn[] {
    return this.turnHistory.slice();
  }

  /** Clears the sliding-window memory. Does not reset the client's KV cache. */
  public clearMemory(): void {
    this.turnHistory.length = 0;
  }

  public async choose(
    userMessage: string,
    options: CharacterChooseOptions
  ): Promise<CharacterChooseResult> {
    let grammar: string;
    try {
      grammar = compileChoiceGrammar(options.choices);
    } catch (error) {
      return {
        selection: null,
        status: 'invalid_request',
        errorMessage: error instanceof Error ? error.message : String(error),
        rawText: '',
      };
    }
    const choicePrompt = renderChoicePrompt(userMessage, options.choices);
    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: choicePrompt },
    ];

    const abort = createTimedAbortController(options.signal, options.timeoutMs);
    if (abort.signal.aborted) {
      const status = abort.timedOut() ? 'timed_out' : 'aborted';
      abort.dispose();
      return {
        selection: null,
        status,
        errorMessage: status === 'timed_out' ? 'Choice timed out.' : 'Choice aborted.',
        rawText: '',
      };
    }
    const chatOptions: ChatOptions = {
      session: `${this.contextKey}:choose`,
      maxTokens: options.maxOutputTokens ?? 24,
      signal: abort.signal,
    };
    const contextKey = `${this.contextKey}:choose`;

    logChoiceQuery({
      phase: 'request',
      contextKey,
      systemPrompt: this.systemPrompt,
      userPrompt: choicePrompt,
      grammar,
      choices: options.choices,
    });

    try {
      const result = await this.client.chat(messages, {
        ...chatOptions,
        grammar,
      }).response;
      const rawText = result.text;
      if (abort.signal.aborted) {
        const status = abort.timedOut() ? 'timed_out' : 'aborted';
        return {
          selection: null,
          status,
          errorMessage: status === 'timed_out' ? 'Choice timed out.' : 'Choice aborted.',
          rawText: '',
        };
      }
      const selection = parseChoiceOutput(rawText, options.choices);
      if (selection == null) {
        logChoiceQuery({
          phase: 'response',
          contextKey,
          rawText,
          selection: null,
          status: 'invalid_response',
          errorMessage: 'choice output did not match any available option',
        });
        return {
          selection: null,
          status: 'invalid_response',
          errorMessage: 'choice output did not match any available option',
          rawText,
        };
      }
      return {
        selection,
        status: 'ok',
        rawText,
      };
    } catch (error) {
      const cancelled = abort.signal.aborted;
      const status = cancelled
        ? abort.timedOut()
          ? 'timed_out'
          : 'aborted'
        : 'failed';
      const errorMessage = status === 'timed_out'
        ? 'Choice timed out.'
        : status === 'aborted'
          ? 'Choice aborted.'
          : error instanceof Error ? error.message : String(error);
      logChoiceQuery({
        phase: 'response',
        contextKey,
        rawText: '',
        selection: null,
        status,
        errorMessage,
      });
      return {
        selection: null,
        status,
        errorMessage,
        rawText: '',
      };
    } finally {
      abort.dispose();
    }
  }

  /**
   * Processes a single user turn. Returns an async iterator that yields
   * `ChatEvent`s as they arrive, terminating with a `turn-end` event.
   *
   * The iterator is backed by a small internal queue so upstream token
   * emission never blocks on a slow consumer — if the consumer falls behind,
   * events buffer in memory rather than back-pressuring decode.
   */
  public chat(userMessage: string, options: { signal?: AbortSignal } = {}): AsyncIterable<ChatEvent> {
    const trimmed = userMessage;
    const controller = new AbortController();
    const queue = new AsyncEventQueue<ChatEvent>(() => controller.abort());

    const emit = (event: ChatEvent): void => {
      queue.push(event);
      this.eventBus.emit(event);
    };

    const detachSignalForwarder = forwardAbortSignal(options.signal, controller);
    const previousTurn = this.currentTurn;

    let inFlightTurn: InFlightTurn;
    const done = this.executeTurn(trimmed, emit, controller.signal, previousTurn).finally(() => {
      detachSignalForwarder();
      if (this.currentTurn === inFlightTurn) {
        this.currentTurn = undefined;
      }
      queue.close();
    });
    inFlightTurn = {
      controller,
      done,
    };
    this.currentTurn = inFlightTurn;
    void done;

    return queue;
  }

  private async executeTurn(
    userMessage: string,
    emit: (event: ChatEvent) => void,
    signal: AbortSignal,
    previousTurn: InFlightTurn | undefined
  ): Promise<void> {
    if (previousTurn) {
      previousTurn.controller.abort();
      try {
        await Promise.race([
          previousTurn.done,
          new Promise<void>((resolve) => setTimeout(resolve, 1000)),
        ]);
      } catch {
        // A prior turn already surfaced its own terminal event.
      }
    }

    emit({ kind: 'turn-start', userMessage });

    if (signal.aborted) {
      emit({ kind: 'turn-end', finalText: '', status: 'aborted' });
      return;
    }

    const parser = new IncrementalActionParser(this.config.actions);
    const turnMessages = this.buildTurnMessages(userMessage);
    try {
      if (signal.aborted) {
        emit({ kind: 'turn-end', finalText: '', status: 'aborted' });
        return;
      }
      await this.runTurn(turnMessages, userMessage, parser, emit, signal);
    } catch (error) {
      emit({
        kind: 'turn-end',
        finalText: '',
        status: signal.aborted ? 'aborted' : 'failed',
        ...(signal.aborted
          ? {}
          : { errorMessage: error instanceof Error ? error.message : String(error) }),
      });
    }
  }

  /**
   * Builds the ordered ChatMessage[] for the current turn. This sequence is
   * rendered through the model's embedded chat template so the character
   * system prompt and turn history follow the model's native chat contract.
   */
  private buildTurnMessages(userMessage: string): ChatMessage[] {
    const messages: ChatMessage[] = [{ role: 'system', content: this.systemPrompt }];
    for (const example of this.config.persona.dialogExamples ?? []) {
      messages.push({ role: 'user', content: example.user });
      messages.push({ role: 'assistant', content: example.assistant });
    }
    for (const turn of this.turnHistory) {
      messages.push({ role: turn.role, content: turn.content });
    }
    messages.push({ role: 'user', content: userMessage });
    return messages;
  }

  private async runTurn(
    messages: ChatMessage[],
    userMessage: string,
    parser: IncrementalActionParser,
    emit: (event: ChatEvent) => void,
    signal: AbortSignal
  ): Promise<void> {
    let deliveredOutputText = '';
    let proseText = '';
    let memoryText = '';
    let status: RunStatus = 'ok';
    let errorMessage: string | undefined;
    const contextKey = `${this.contextKey}:chat`;

    const recordParsedEvents = (events: readonly ParsedEvent[]): void => {
      for (const event of events) {
        if (event.kind === 'prose') {
          proseText += event.text;
          memoryText += event.text;
        } else {
          memoryText += this.renderCanonicalActionCue(event.id, event.raw);
        }
        emit(event);
      }
    };

    const flushParsedText = (): void => {
      recordParsedEvents(parser.flush());
    };

    const consumeTokens = (batch: { text: string }): void => {
      if (batch.text.length === 0) {
        return;
      }
      const text = batch.text;
      deliveredOutputText += text;
      recordParsedEvents(parser.consume(text));
    };

    try {
      const queryOptions: ChatOptions = {
        session: contextKey,
        maxTokens: this.maxOutputTokens,
        emitTokens: true,
        signal,
      };
      const run = this.client.chat(messages, {
        ...queryOptions,
        grammar: this.grammarSource,
      });
      const response = run.response;
      for await (const batch of run.tokens) {
        consumeTokens(batch);
      }
      const result = await response;
      const rawText = result.text;
      if (signal.aborted) {
        status = 'aborted';
      } else {
        const unseenOutputSuffix = sliceUndeliveredSuffix(deliveredOutputText, rawText);
        if (unseenOutputSuffix.length > 0) {
          deliveredOutputText += unseenOutputSuffix;
          recordParsedEvents(parser.consume(unseenOutputSuffix));
        }
      }
    } catch (error) {
      status = signal.aborted ? 'aborted' : 'failed';
      if (!signal.aborted) {
        errorMessage = error instanceof Error ? error.message : String(error);
      }
    }

    flushParsedText();

    const finalText = stripExactUserMessageEcho(proseText, userMessage).trim();
    const sanitizedMemoryText = stripExactUserMessageEcho(memoryText, userMessage).trim();

    if (status === 'ok' && !errorMessage && sanitizedMemoryText.length > 0) {
      this.pushTurnToMemory({ role: 'user', content: userMessage });
      this.pushTurnToMemory({ role: 'assistant', content: sanitizedMemoryText });
    }

    const endEvent = {
      kind: 'turn-end' as const,
      finalText,
      status,
      ...(errorMessage ? { errorMessage } : {}),
    };
    emit(endEvent);
  }

  private renderCanonicalActionCue(id: string, rawCue: string): string {
    const label = this.canonicalCueLabelsByActionId.get(id);
    return label == null ? rawCue : `[${label}]`;
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

  public constructor(private readonly onReturn?: () => void) { }

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
        this.onReturn?.();
        this.close();
        return Promise.resolve({ value: undefined as never, done: true });
      },
    };
  }
}

function stripExactUserMessageEcho(text: string, userMessage: string): string {
  const source = userMessage.trim();
  if (source.length === 0) {
    return text;
  }

  const escaped = escapeRegExp(source).replace(/\s+/g, '\\s+');
  const exactEcho = new RegExp(`^\\s*${escaped}\\s*$`);
  const echoedLinePrefix = new RegExp(`^\\s*${escaped}\\s*(?:\\r?\\n\\s*)+`);

  let out = text;
  if (exactEcho.test(out)) {
    return '';
  }
  const prefixMatch = out.match(echoedLinePrefix);
  if (prefixMatch) {
    out = out.slice(prefixMatch[0].length).trimStart();
  }
  return out;
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function forwardAbortSignal(
  source: AbortSignal | undefined,
  controller: AbortController
): () => void {
  if (!source) {
    return () => { };
  }
  if (source.aborted) {
    controller.abort();
    return () => { };
  }

  const onAbort = (): void => {
    controller.abort();
  };
  source.addEventListener('abort', onAbort, { once: true });
  return () => {
    source.removeEventListener('abort', onAbort);
  };
}

function renderChoicePrompt(userMessage: string, choices: readonly string[]): string {
  const normalizedChoices = choices.map((choice) => choice.trim());
  return [
    userMessage.trim(),
    '',
    'Choose exactly one of the following options and output only that option text:',
    ...normalizedChoices.map((choice) => `- ${choice}`),
  ].join('\n');
}

function logChoiceQuery(args: {
  phase: 'request' | 'response';
  contextKey: string;
  systemPrompt?: string;
  userPrompt?: string;
  grammar?: string;
  choices?: readonly string[];
  rawText?: string;
  selection?: string | null;
  status?: CharacterChooseResult['status'];
  errorMessage?: string;
}): void {
  if (!isPromptTraceEnabled()) return;
  if (args.phase === 'request') {
    console.groupCollapsed(`[CharacterRuntime.choose] -> ${args.contextKey}`);
    console.log('systemPrompt', args.systemPrompt ?? '');
    console.log('userPrompt', args.userPrompt ?? '');
    console.log('choices', args.choices ?? []);
    console.log('grammar', args.grammar ?? '');
    console.groupEnd();
    return;
  }
  console.groupCollapsed(`[CharacterRuntime.choose] <- ${args.contextKey}`);
  console.log('rawText', args.rawText ?? '');
  console.log('selection', args.selection ?? null);
  console.log('status', args.status ?? 'ok');
  if (args.errorMessage) {
    console.warn('error', args.errorMessage);
  }
  console.groupEnd();
}

function isPromptTraceEnabled(): boolean {
  return (globalThis as { COGENT_TRACE_PROMPTS?: boolean }).COGENT_TRACE_PROMPTS === true;
}
