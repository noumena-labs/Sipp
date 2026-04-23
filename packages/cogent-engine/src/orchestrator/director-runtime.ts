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
import type {
  DirectorConfig,
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
    options: { signal?: AbortSignal } = {}
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
        cancelled: options.signal?.aborted === true,
        errorMessage: error instanceof Error ? error.message : String(error),
        rawText: '',
      };
    }

    const promptOptions: PromptOptions = {
      nTokens: this.maxOutputTokens,
      promptFormat: 'raw',
      grammar,
      ...(options.signal ? { signal: options.signal } : {}),
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
      const response = await this.engine.runQueuedRequest(
        requestId,
        options.signal ? { signal: options.signal } : {}
      );
      const rawText = response.outputText ?? '';
      if (response.cancelled) {
        logDirectorQuery({
          phase: 'response',
          queryName,
          contextKey: this.contextKey,
          rawText,
          parsed: null,
          cancelled: true,
        });
        return { data: null, cancelled: true, rawText };
      }
      if (response.failed) {
        logDirectorQuery({
          phase: 'response',
          queryName,
          contextKey: this.contextKey,
          rawText,
          parsed: null,
          errorMessage: response.errorMessage ?? 'generation failed',
        });
        return {
          data: null,
          cancelled: false,
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
          errorMessage: 'response was not valid JSON',
        });
        return {
          data: null,
          cancelled: false,
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
          errorMessage: validationError,
        });
        return {
          data: null,
          cancelled: false,
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
      return { data: parsed, cancelled: false, rawText };
    } catch (error) {
      const cancelled = options.signal?.aborted === true;
      if (requestId !== 0 && !cancelled && this.engine.cancelQueuedRequest) {
        try {
          await this.engine.cancelQueuedRequest(requestId);
        } catch {
          // Swallow; the original error is more useful.
        }
      }
      logDirectorQuery({
        phase: 'response',
        queryName,
        contextKey: this.contextKey,
        rawText: '',
        parsed: null,
        cancelled,
        errorMessage: error instanceof Error ? error.message : String(error),
      });
      return {
        data: null,
        cancelled,
        errorMessage: error instanceof Error ? error.message : String(error),
        rawText: '',
      };
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
  cancelled?: boolean;
  errorMessage?: string;
}): void {
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
  console.log('cancelled', args.cancelled ?? false);
  if (args.errorMessage) {
    console.warn('error', args.errorMessage);
  }
  console.groupEnd();
}
