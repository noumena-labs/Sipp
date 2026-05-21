//////////////////////////////////////////////////////////////////////////////
//
// director-runtime.ts
//
// - Thin runtime wrapper over the core engine.
// - Given a parsed `DirectorConfig`, it builds a stable system prompt,
//   renders task-specific prompts, constrains selection shapes with literal
//   grammars, and parses shape-driven text outputs.
//
//////////////////////////////////////////////////////////////////////////////

import type {
  ChatInput,
  ChatOptions,
  ModelInfo,
  RequestResult,
} from '../models/types.js';
import type { ChatMessage } from '../types.js';
import { createTimedAbortController } from '../utils/abort.js';
import {
  compileDirectorOutputGrammar,
  DirectorOutputError,
  type ResolvedDirectorChoices,
  parseDirectorOutput,
  resolveDirectorChoices,
} from './director-output.js';
import { renderDirectorSystemPrompt, renderDirectorUserMessage } from './director-prompt.js';
import type {
  DirectorConfig,
  DirectorRunRequest,
  DirectorRunResult,
  DirectorRuntimeOptions,
  DirectorTaskConfig,
  DirectorTaskPrompt,
} from './director-types.js';

export interface DirectorRuntimeEngine {
  chat(input: ChatInput, options?: ChatOptions): Promise<RequestResult>;
  models?: {
    current(): Pick<ModelInfo, 'mediaMarker'> | null;
  };
}

export class DirectorRuntime {
  private readonly engine: DirectorRuntimeEngine;
  private readonly config: DirectorConfig;
  private readonly maxOutputTokens: number;
  private readonly contextKey: string;
  private readonly systemPrompt: string;

  public constructor(
    engine: DirectorRuntimeEngine,
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

  public getTaskGrammar<TPayload = unknown>(
    taskName: string,
    request: DirectorRunRequest<TPayload> = {}
  ): string | undefined {
    const task = this.requireTask(taskName);
    const resolved = resolveDirectorChoices(task.output, request);
    return compileDirectorOutputGrammar(task.output, resolved);
  }

  public getTaskPrompt<TPayload = unknown>(
    taskName: string,
    request: DirectorRunRequest<TPayload> = {}
  ): DirectorTaskPrompt {
    const task = this.requireTask(taskName);
    const resolved = resolveDirectorChoices(task.output, request);
    const grammar = compileDirectorOutputGrammar(task.output, resolved);
    const rendered = renderDirectorUserMessage(
      this.config,
      taskName,
      task,
      request,
      resolved,
      this.getMediaMarker()
    );
    return {
      systemPrompt: this.systemPrompt,
      userPrompt: rendered.text,
      media: rendered.media,
      ...(grammar ? { grammar } : {}),
    };
  }

  public async run<TPayload = unknown>(
    taskName: string,
    request: DirectorRunRequest<TPayload> = {}
  ): Promise<DirectorRunResult<TPayload>> {
    let task: DirectorTaskConfig;
    let resolved: ResolvedDirectorChoices<TPayload> = {};
    let grammar: string | undefined;
    let userText = '';
    let media: readonly Uint8Array[] = [];

    try {
      task = this.requireTask(taskName);
      resolved = resolveDirectorChoices(task.output, request);
      grammar = compileDirectorOutputGrammar(task.output, resolved);
      const rendered = renderDirectorUserMessage(
        this.config,
        taskName,
        task,
        request,
        resolved,
        this.getMediaMarker()
      );
      userText = rendered.text;
      media = rendered.media;
    } catch (error) {
      return failedPreflightResult(error);
    }

    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: userText },
    ];

    const abort = createTimedAbortController(request.signal, request.timeoutMs);
    if (abort.signal.aborted) {
      const status = abort.timedOut() ? 'timed_out' : 'aborted';
      abort.dispose();
      return {
        status,
        text: '',
        selections: [],
        rawText: '',
        errorMessage: status === 'timed_out' ? 'Director task timed out.' : 'Director task aborted.',
      };
    }
    const contextKey = this.getTaskContextKey(taskName);
    const queryOptions: ChatOptions = {
      session: contextKey,
      maxTokens: request.maxOutputTokens ?? defaultTokenBudget(task.output.shape, this.maxOutputTokens),
      signal: abort.signal,
    };

    logDirectorRun({
      phase: 'request',
      taskName,
      contextKey,
      systemPrompt: this.systemPrompt,
      userPrompt: userText,
      grammar,
    });

    try {
      const result = await this.engine.chat(
        media.length > 0 ? { messages, media: [...media] } : messages,
        {
          ...queryOptions,
          grammar,
        }
      );
      const rawText = result.text;
      if (abort.signal.aborted) {
        const status = abort.timedOut() ? 'timed_out' : 'aborted';
        return {
          status,
          text: '',
          selections: [],
          rawText: '',
          errorMessage: status === 'timed_out' ? 'Director task timed out.' : 'Director task aborted.',
        };
      }
      const parsed = parseDirectorOutput(rawText, task.output, resolved);
      logDirectorRun({ phase: 'response', taskName, contextKey, rawText, status: 'ok' });
      return {
        status: 'ok',
        text: parsed.text,
        selections: parsed.selections,
        rawText,
      };
    } catch (error) {
      const cancelled = abort.signal.aborted;
      const status = classifyCaughtStatus(error, cancelled, abort.timedOut());
      const errorMessage = status === 'timed_out'
        ? 'Director task timed out.'
        : status === 'aborted'
          ? 'Director task aborted.'
          : error instanceof Error ? error.message : String(error);
      logDirectorRun({
        phase: 'response',
        taskName,
        contextKey,
        rawText: '',
        status,
        errorMessage,
      });
      return {
        status,
        text: '',
        selections: [],
        errorMessage,
        rawText: '',
      };
    } finally {
      abort.dispose();
    }
  }

  private requireTask(taskName: string): DirectorTaskConfig {
    const task = this.config.tasks[taskName];
    if (!task) {
      throw new Error(`director task ${JSON.stringify(taskName)} is not defined in ${this.config.id}.`);
    }
    return task;
  }

  private getMediaMarker(): string | null {
    return this.engine.models?.current()?.mediaMarker ?? null;
  }

  private getTaskContextKey(taskName: string): string {
    return `${this.contextKey}:${taskName}`;
  }
}

function failedPreflightResult<TPayload>(error: unknown): DirectorRunResult<TPayload> {
  return {
    status: 'invalid_request',
    text: '',
    selections: [],
    errorMessage: error instanceof Error ? error.message : String(error),
    rawText: '',
  };
}

function classifyCaughtStatus(
  error: unknown,
  cancelled: boolean,
  timedOut: boolean
): DirectorRunResult['status'] {
  if (cancelled) {
    return timedOut ? 'timed_out' : 'aborted';
  }
  return error instanceof DirectorOutputError ? 'invalid_response' : 'failed';
}

function defaultTokenBudget(
  shape: DirectorTaskConfig['output']['shape'],
  fallback: number
): number {
  switch (shape) {
    case 'select_one':
      return 8;
    case 'select_many':
      return 48;
    case 'select_slots':
      return 64;
    case 'text':
    case 'text_with_directives':
      return fallback;
  }
}

function logDirectorRun(args: {
  phase: 'request' | 'response';
  taskName: string;
  contextKey: string;
  systemPrompt?: string;
  userPrompt?: string;
  grammar?: string;
  rawText?: string;
  status?: DirectorRunResult['status'];
  errorMessage?: string;
}): void {
  if (!isPromptTraceEnabled()) return;
  if (args.phase === 'request') {
    console.groupCollapsed(`[DirectorRuntime] ${args.taskName} -> ${args.contextKey}`);
    console.log('systemPrompt', args.systemPrompt ?? '');
    console.log('userPrompt', args.userPrompt ?? '');
    console.log('grammar', args.grammar ?? '');
    console.groupEnd();
    return;
  }
  console.groupCollapsed(`[DirectorRuntime] ${args.taskName} <- ${args.contextKey}`);
  console.log('rawText', args.rawText ?? '');
  console.log('status', args.status ?? 'ok');
  if (args.errorMessage) {
    console.warn('error', args.errorMessage);
  }
  console.groupEnd();
}

function isPromptTraceEnabled(): boolean {
  return (globalThis as { COGENT_TRACE_PROMPTS?: boolean }).COGENT_TRACE_PROMPTS === true;
}
