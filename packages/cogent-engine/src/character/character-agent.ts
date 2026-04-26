//////////////////////////////////////////////////////////////////////////////
//
// character-agent.ts
//
// - High-level character runtime that ties an engine instance to a
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
import { CharacterEventBus, type CharacterEvent } from './action-bus.js';
import { compileActionGrammar } from './action-grammar.js';
import { StreamingActionParser, type ParsedEvent } from './action-parser.js';
import { compileChoiceGrammar, parseChoiceOutput } from './choice-grammar.js';
import {
  DEFAULT_MEMORY_MAX_TURNS,
  resolveMaxMemoryTurns,
  type CharacterConfig,
} from './character-config.js';
import { summarizeActionCues } from './action-schema.js';
import {
  ChatTemplatePromptRuntime,
  sanitizeAssistantText,
  StreamingBoundaryTextSanitizer,
} from '../core/chat-template-boundaries.js';
import { renderSystemPrompt } from './persona.js';
import { createTimedAbortController, waitForAbort } from '../utils/abort.js';
import type { RunStatus } from '../core/run-status.js';

export interface CharacterRuntimeEngine {
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
  applyChatTemplate(
    messages: Array<{ role: string; content: string }>,
    addAssistant: boolean
  ): Promise<string>;
  getChatTemplate?(): string | null;
  getEosText?(): string;
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
  /** Stable engine context key prefix. Defaults to `character:${config.id}`. */
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

interface ResolvedPromptContext {
  readonly promptText: string;
  readonly boundaryMarkers: readonly string[];
  readonly templateSource: string | null;
}

interface InFlightTurn {
  readonly controller: AbortController;
  readonly done: Promise<void>;
}

/**
 * A character-driven conversation runtime. Pair one with a CogentEngine and a
 * CharacterConfig to get a grammar-constrained, memory-aware chat loop.
 */
export class CharacterRuntime {
  private readonly engine: CharacterRuntimeEngine;
  private readonly config: CharacterConfig;
  private readonly maxOutputTokens: number;
  private readonly contextKey: string;
  private readonly systemPrompt: string;
  private readonly grammarSource: string;
  private readonly promptRuntime: ChatTemplatePromptRuntime;
  private readonly memoryLimitTurns: number;
  private readonly canonicalCueLabelsByActionId: ReadonlyMap<string, string>;
  private readonly turnHistory: ChatTurn[] = [];
  private readonly eventBus: CharacterEventBus;
  private currentTurn: InFlightTurn | undefined;

  public constructor(
    engine: CharacterRuntimeEngine,
    config: CharacterConfig,
    options: CharacterRuntimeOptions = {}
  ) {
    this.engine = engine;
    this.config = config;
    this.maxOutputTokens = options.maxOutputTokens ?? 256;
    this.contextKey = options.contextKey ?? `character:${config.id}`;
    this.eventBus = options.bus ?? new CharacterEventBus();
    this.promptRuntime = new ChatTemplatePromptRuntime(engine);
    this.systemPrompt = renderSystemPrompt(config.persona, config.actions);
    this.grammarSource = compileActionGrammar(config.actions);
    this.canonicalCueLabelsByActionId = new Map(
      summarizeActionCues(config.actions).map((cue) => [cue.id, cue.label])
    );
    this.memoryLimitTurns = Math.max(
      0,
      resolveMaxMemoryTurns(config) ?? DEFAULT_MEMORY_MAX_TURNS
    );
  }

  /** Exposes the event bus for imperative subscribers (VRM bindings, logs). */
  public get bus(): CharacterEventBus {
    return this.eventBus;
  }

  public getConfig(): CharacterConfig {
    return this.config;
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

    let promptText: string;
    let boundaryMarkers: readonly string[] = [];
    try {
      const promptContext = await this.promptRuntime.render(messages);
      promptText = promptContext.promptText;
      boundaryMarkers = promptContext.boundaryMarkers;
    } catch (error) {
      return {
        selection: null,
        status: options.signal?.aborted === true ? 'aborted' : 'failed',
        errorMessage: error instanceof Error ? error.message : String(error),
        rawText: '',
      };
    }

    const abort = createTimedAbortController(options.signal, options.timeoutMs);
    const promptOptions: PromptOptions = {
      nTokens: options.maxOutputTokens ?? 24,
      promptFormat: 'raw',
      grammar,
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

    let requestId = 0;
    try {
      requestId = await this.engine.queuePrompt(contextKey, promptText, promptOptions);
      const response = await Promise.race([
        this.engine.runQueuedRequest(requestId, { signal: abort.signal }),
        waitForAbort(abort.signal, {
          timedOut: abort.timedOut,
          timeoutMessage: 'Choice timed out.',
          abortMessage: 'Choice aborted.',
        }),
      ]);
      const rawText = response.outputText ?? '';
      const parseText = sanitizeAssistantText(rawText, boundaryMarkers);
      if (response.cancelled) {
        const status = abort.timedOut() ? 'timed_out' : 'aborted';
        const errorMessage = status === 'timed_out' ? 'Choice timed out.' : 'Choice aborted.';
        logChoiceQuery({
          phase: 'response',
          contextKey,
          rawText,
          selection: null,
          status,
          errorMessage,
        });
        return {
          selection: null,
          status,
          errorMessage,
          rawText,
        };
      }
      if (response.failed) {
        logChoiceQuery({
          phase: 'response',
          contextKey,
          rawText,
          selection: null,
          status: 'failed',
          errorMessage: response.errorMessage ?? 'generation failed',
        });
        return {
          selection: null,
          status: 'failed',
          errorMessage: response.errorMessage ?? 'generation failed',
          rawText,
        };
      }
      const selection = parseChoiceOutput(parseText, options.choices);
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
      if (requestId !== 0 && cancelled && this.engine.cancelQueuedRequest) {
        void this.engine.cancelQueuedRequest(requestId).catch(() => undefined);
      } else if (requestId !== 0 && !cancelled && this.engine.cancelQueuedRequest) {
        try {
          await this.engine.cancelQueuedRequest(requestId);
        } catch {
          // Swallow cancel errors.
        }
      }
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
   * callbacks never block on a slow consumer — if the consumer falls
   * behind, events buffer in memory rather than back-pressuring decode.
   */
  public chat(userMessage: string, options: { signal?: AbortSignal } = {}): AsyncIterable<ChatEvent> {
    const trimmed = userMessage ?? '';
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
        await previousTurn.done;
      } catch {
        // A prior turn already surfaced its own terminal event.
      }
    }

    emit({ kind: 'turn-start', userMessage });

    if (signal.aborted) {
      emit({ kind: 'turn-end', finalText: '', status: 'aborted' });
      return;
    }

    const parser = new StreamingActionParser(this.config.actions);
    const turnMessages = this.buildTurnMessages(userMessage);
    try {
      const promptContext = await this.buildPromptContext(turnMessages);
      if (signal.aborted) {
        emit({ kind: 'turn-end', finalText: '', status: 'aborted' });
        return;
      }
      await this.runTurn(promptContext, userMessage, parser, emit, signal);
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

  /**
   * Renders the full conversation through the model's embedded chat template
   * and derives assistant boundaries from that applied template output.
   */
  private async buildPromptContext(messages: ChatMessage[]): Promise<ResolvedPromptContext> {
    return this.promptRuntime.render(messages);
  }

  private async runTurn(
    promptContext: ResolvedPromptContext,
    userMessage: string,
    parser: StreamingActionParser,
    emit: (event: ChatEvent) => void,
    signal: AbortSignal
  ): Promise<void> {
    let streamedOutputText = '';
    let proseText = '';
    let memoryText = '';
    let requestId: GenerateRequestId = 0;
    let status: RunStatus = 'ok';
    let errorMessage: string | undefined;
    const contextKey = `${this.contextKey}:chat`;
    const promptText = promptContext.promptText;
    const outputSanitizer = new StreamingBoundaryTextSanitizer(promptContext.boundaryMarkers);

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

    const emitParsedText = (text: string): void => {
      if (text.length === 0) {
        return;
      }
      recordParsedEvents(parser.consume(text));
    };

    const flushParsedText = (): void => {
      recordParsedEvents(parser.flush());
    };

    const requestBoundaryStop = (): void => {
      if (requestId === 0 || signal.aborted || !this.engine.cancelQueuedRequest) {
        return;
      }
      void this.engine.cancelQueuedRequest(requestId).catch(() => {
        // Best-effort only; generation may already be terminating naturally.
      });
    };

    const consumeOutputText = (text: string): void => {
      if (text.length === 0 || outputSanitizer.reachedBoundary) {
        return;
      }
      streamedOutputText += text;
      const result = outputSanitizer.consume(text);
      emitParsedText(result.safeText);

      if (result.hitBoundary) {
        requestBoundaryStop();
      }
    };

    const onToken = (token: string): void => {
      consumeOutputText(token);
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
      const unseenOutputSuffix = sliceUnstreamedSuffix(streamedOutputText, response.outputText);
      if (!outputSanitizer.reachedBoundary && unseenOutputSuffix.length > 0) {
        consumeOutputText(unseenOutputSuffix);
      }
      if (response.cancelled && !outputSanitizer.reachedBoundary) {
        status = 'aborted';
      }
      if (response.failed && response.errorMessage) {
        status = 'failed';
        errorMessage = response.errorMessage;
      }
    } catch (error) {
      if (outputSanitizer.reachedBoundary && !signal.aborted) {
        status = 'ok';
      } else {
        status = signal.aborted ? 'aborted' : 'failed';
        if (!signal.aborted) {
          errorMessage = error instanceof Error ? error.message : String(error);
        }
      }
      if (signal.aborted && !outputSanitizer.reachedBoundary) {
        status = 'aborted';
      }
      // Best-effort cancel in case the runtime still holds the request.
      if (requestId !== 0 && this.engine.cancelQueuedRequest && !outputSanitizer.reachedBoundary) {
        try {
          await this.engine.cancelQueuedRequest(requestId);
        } catch {
          // Swallow cancel errors — they are secondary to the original one.
        }
      }
    }

    emitParsedText(outputSanitizer.flush());
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

  public constructor(private readonly onReturn?: () => void) {}

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
    return () => {};
  }
  if (source.aborted) {
    controller.abort();
    return () => {};
  }

  const onAbort = (): void => {
    controller.abort();
  };
  source.addEventListener('abort', onAbort, { once: true });
  return () => {
    source.removeEventListener('abort', onAbort);
  };
}

function sliceUnstreamedSuffix(streamedOutputText: string, finalOutputText: string): string {
  if (streamedOutputText.length === 0) {
    return finalOutputText;
  }
  if (!finalOutputText.startsWith(streamedOutputText)) {
    return '';
  }
  return finalOutputText.slice(streamedOutputText.length);
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
