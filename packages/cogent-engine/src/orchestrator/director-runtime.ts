//////////////////////////////////////////////////////////////////////////////
//
// director-runtime.ts
//
// - Thin runtime wrapper over the core engine.
// - Given a parsed `DirectorConfig`, it builds a stable system prompt,
//   renders query-specific payload prompts, compiles response grammars, and
//   returns validated JSON values.
//
//////////////////////////////////////////////////////////////////////////////

import type { CharacterAgentEngine } from '../character/character-agent.js';
import type { ChatMessage, PromptOptions } from '../core/inference-types.js';
import { renderDirectorSystemPrompt, renderDirectorUserMessage } from './director-prompt.js';
import { compileResponseGrammar } from './response-grammar.js';
import { validateResponseValue } from './response-schema.js';
import { createTimedAbortController, waitForAbort } from '../utils/abort.js';
import type {
  DirectorConfig,
  DirectorQueryOptions,
  DirectorQueryPayload,
  DirectorQueryResult,
  DirectorRuntimeOptions,
  JsonValue,
} from './director-types.js';

export class DirectorRuntime {
  private readonly engine: CharacterAgentEngine;
  private readonly config: DirectorConfig;
  private readonly maxOutputTokens: number;
  private readonly contextKey: string;
  private readonly systemPrompt: string;
  private readonly grammarCache = new Map<string, string>();

  public constructor(
    engine: CharacterAgentEngine,
    config: DirectorConfig,
    options: DirectorRuntimeOptions = {}
  ) {
    this.engine = engine;
    this.config = config;
    this.maxOutputTokens = options.maxOutputTokens ?? 256;
    this.contextKey = options.contextKey ?? `director:${config.id}`;
    this.systemPrompt = renderDirectorSystemPrompt(config);
  }

  public getConfig(): DirectorConfig {
    return this.config;
  }

  public getSystemPrompt(): string {
    return this.systemPrompt;
  }

  public getGrammarSource(queryName: string): string {
    const cached = this.grammarCache.get(queryName);
    if (cached) {
      return cached;
    }
    const query = this.requireQuery(queryName);
    const grammar = compileResponseGrammar(query.response);
    this.grammarCache.set(queryName, grammar);
    return grammar;
  }

  public async query(
    queryName: string,
    payload: DirectorQueryPayload,
    options: DirectorQueryOptions = {}
  ): Promise<DirectorQueryResult> {
    const query = this.requireQuery(queryName);
    const grammar = this.getGrammarSource(queryName);
    const userText = renderDirectorUserMessage(this.config, queryName, query, payload);

    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: userText },
    ];

    let promptText: string;
    try {
      promptText = await this.engine.applyChatTemplate(messages, true);
    } catch (error) {
      return {
        data: null,
        status: options.signal?.aborted === true ? 'aborted' : 'failed',
        errorMessage: error instanceof Error ? error.message : String(error),
        rawText: '',
      };
    }

    const abort = createTimedAbortController(options.signal, options.timeoutMs);
    const promptOptions: PromptOptions = {
      nTokens: this.maxOutputTokens,
      promptFormat: 'raw',
      grammar,
      signal: abort.signal,
    };

    logDirectorQuery({
      phase: 'request',
      queryName,
      contextKey: this.contextKey,
      systemPrompt: this.systemPrompt,
      userPrompt: userText,
      grammar,
    });

    let requestId = 0;
    try {
      requestId = await this.engine.queuePrompt(this.contextKey, promptText, promptOptions);
      const response = await Promise.race([
        this.engine.runQueuedRequest(requestId, { signal: abort.signal }),
        waitForAbort(abort.signal, {
          timedOut: abort.timedOut,
          timeoutMessage: 'Director query timed out.',
          abortMessage: 'Director query aborted.',
        }),
      ]);
      const rawText = response.outputText ?? '';
      if (response.cancelled) {
        const status = abort.timedOut() ? 'timed_out' : 'aborted';
        const errorMessage = status === 'timed_out' ? 'Director query timed out.' : 'Director query aborted.';
        logDirectorQuery({
          phase: 'response',
          queryName,
          contextKey: this.contextKey,
          rawText,
          parsed: null,
          status,
          errorMessage,
        });
        return { data: null, status, errorMessage, rawText };
      }
      if (response.failed) {
        logDirectorQuery({
          phase: 'response',
          queryName,
          contextKey: this.contextKey,
          rawText,
          parsed: null,
          status: 'failed',
          errorMessage: response.errorMessage ?? 'generation failed',
        });
        return {
          data: null,
          status: 'failed',
          errorMessage: response.errorMessage ?? 'generation failed',
          rawText,
        };
      }
      const parsed = parseJsonValue(rawText);
      if (parsed == null) {
        logDirectorQuery({
          phase: 'response',
          queryName,
          contextKey: this.contextKey,
          rawText,
          parsed: null,
          status: 'invalid_response',
          errorMessage: 'response was not valid JSON',
        });
        return {
          data: null,
          status: 'invalid_response',
          errorMessage: 'response was not valid JSON',
          rawText,
        };
      }
      const validationError = validateResponseValue(parsed, query.response);
      if (validationError) {
        logDirectorQuery({
          phase: 'response',
          queryName,
          contextKey: this.contextKey,
          rawText,
          parsed,
          status: 'invalid_response',
          errorMessage: validationError,
        });
        return {
          data: null,
          status: 'invalid_response',
          errorMessage: validationError,
          rawText,
        };
      }
      logDirectorQuery({
        phase: 'response',
        queryName,
        contextKey: this.contextKey,
        rawText,
        parsed,
      });
      return { data: parsed, status: 'ok', rawText };
    } catch (error) {
      const cancelled = abort.signal.aborted;
      if (requestId !== 0 && cancelled && this.engine.cancelQueuedRequest) {
        void this.engine.cancelQueuedRequest(requestId).catch(() => undefined);
      } else if (requestId !== 0 && !cancelled && this.engine.cancelQueuedRequest) {
        try {
          await this.engine.cancelQueuedRequest(requestId);
        } catch {
          // Swallow; the original error is more useful.
        }
      }
      const status = cancelled
        ? abort.timedOut()
          ? 'timed_out'
          : 'aborted'
        : 'failed';
      const errorMessage = status === 'timed_out'
        ? 'Director query timed out.'
        : status === 'aborted'
          ? 'Director query aborted.'
          : error instanceof Error ? error.message : String(error);
      logDirectorQuery({
        phase: 'response',
        queryName,
        contextKey: this.contextKey,
        rawText: '',
        parsed: null,
        status,
        errorMessage,
      });
      return {
        data: null,
        status,
        errorMessage,
        rawText: '',
      };
    } finally {
      abort.dispose();
    }
  }

  private requireQuery(queryName: string) {
    const query = this.config.queries[queryName];
    if (!query) {
      throw new Error(`director query ${JSON.stringify(queryName)} is not defined in ${this.config.id}.`);
    }
    return query;
  }
}

function parseJsonValue(raw: string): JsonValue | null {
  const trimmed = raw.trim();
  if (trimmed.length === 0) {
    return null;
  }
  try {
    return JSON.parse(trimmed) as JsonValue;
  } catch {
    return null;
  }
}

function logDirectorQuery(args: {
  phase: 'request' | 'response';
  queryName: string;
  contextKey: string;
  systemPrompt?: string;
  userPrompt?: string;
  grammar?: string;
  rawText?: string;
  parsed?: JsonValue | null;
  status?: DirectorQueryResult['status'];
  errorMessage?: string;
}): void {
  if (!isPromptTraceEnabled()) return;
  if (args.phase === 'request') {
    console.groupCollapsed(`[DirectorRuntime] ${args.queryName} -> ${args.contextKey}`);
    console.log('systemPrompt', args.systemPrompt ?? '');
    console.log('userPrompt', args.userPrompt ?? '');
    console.log('grammar', args.grammar ?? '');
    console.groupEnd();
    return;
  }
  console.groupCollapsed(`[DirectorRuntime] ${args.queryName} <- ${args.contextKey}`);
  console.log('rawText', args.rawText ?? '');
  console.log('parsed', args.parsed ?? null);
  console.log('status', args.status ?? 'ok');
  if (args.errorMessage) {
    console.warn('error', args.errorMessage);
  }
  console.groupEnd();
}

function isPromptTraceEnabled(): boolean {
  return (globalThis as { COGENT_TRACE_PROMPTS?: boolean }).COGENT_TRACE_PROMPTS === true;
}
