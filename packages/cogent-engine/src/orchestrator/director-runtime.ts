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
  ChatMessage,
  GenerateRequestId,
  GenerateResponse,
  PromptOptions,
} from '../core/inference-types.js';
import { ChatTemplatePromptRuntime, sanitizeAssistantText } from '../core/chat-template-boundaries.js';
import { createTimedAbortController, waitForAbort } from '../utils/abort.js';
import {
  compileDirectorOutputGrammar,
  DirectorOutputError,
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
  getMediaMarker?(): string | null;
}

export class DirectorRuntime {
  private readonly engine: DirectorRuntimeEngine;
  private readonly config: DirectorConfig;
  private readonly maxOutputTokens: number;
  private readonly contextKey: string;
  private readonly systemPrompt: string;
  private readonly promptRuntime: ChatTemplatePromptRuntime;

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
    this.promptRuntime = new ChatTemplatePromptRuntime(engine);
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
    let grammar: string | undefined;
    let userText = '';
    let media: readonly Uint8Array[] = [];

    try {
      task = this.requireTask(taskName);
      const taskPrompt = this.getTaskPrompt(taskName, request);
      grammar = taskPrompt.grammar;
      userText = taskPrompt.userPrompt;
      media = taskPrompt.media;
    } catch (error) {
      return failedResult(error);
    }

    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: userText },
    ];

    let promptText: string;
    let boundaryMarkers: readonly string[];
    try {
      const promptContext = await this.promptRuntime.render(messages);
      promptText = promptContext.promptText;
      boundaryMarkers = promptContext.boundaryMarkers;
    } catch (error) {
      return {
        status: request.signal?.aborted === true ? 'aborted' : 'failed',
        text: '',
        selections: [],
        errorMessage: error instanceof Error ? error.message : String(error),
        rawText: '',
      };
    }

    const abort = createTimedAbortController(request.signal, request.timeoutMs);
    const promptOptions: PromptOptions = {
      nTokens: request.maxOutputTokens ?? defaultTokenBudget(task.output.shape, this.maxOutputTokens),
      promptFormat: 'raw',
      signal: abort.signal,
      ...(grammar ? { grammar } : {}),
      ...(media.length > 0 ? { media: [...media] } : {}),
    };
    const contextKey = this.getTaskContextKey(taskName);

    logDirectorRun({
      phase: 'request',
      taskName,
      contextKey,
      systemPrompt: this.systemPrompt,
      userPrompt: userText,
      grammar,
    });

    let requestId = 0;
    let rawText = '';
    try {
      requestId = await this.engine.queuePrompt(contextKey, promptText, promptOptions);
      const response = await Promise.race([
        this.engine.runQueuedRequest(requestId, { signal: abort.signal }),
        waitForAbort(abort.signal, {
          timedOut: abort.timedOut,
          timeoutMessage: 'Director task timed out.',
          abortMessage: 'Director task aborted.',
        }),
      ]);
      rawText = response.outputText ?? '';
      const parseText = sanitizeAssistantText(rawText, boundaryMarkers);
      if (response.cancelled) {
        const status = abort.timedOut() ? 'timed_out' : 'aborted';
        const errorMessage = status === 'timed_out'
          ? 'Director task timed out.'
          : 'Director task aborted.';
        logDirectorRun({
          phase: 'response',
          taskName,
          contextKey,
          rawText,
          status,
          errorMessage,
        });
        return { status, text: '', selections: [], errorMessage, rawText };
      }
      if (response.failed) {
        const errorMessage = response.errorMessage ?? 'generation failed';
        logDirectorRun({
          phase: 'response',
          taskName,
          contextKey,
          rawText,
          status: 'failed',
          errorMessage,
        });
        return { status: 'failed', text: '', selections: [], errorMessage, rawText };
      }

      const resolved = resolveDirectorChoices(task.output, request);
      const parsed = parseDirectorOutput(parseText, task.output, resolved);
      logDirectorRun({ phase: 'response', taskName, contextKey, rawText, status: 'ok' });
      return {
        status: 'ok',
        text: parsed.text,
        selections: parsed.selections,
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
          // Swallow; the original error is more useful.
        }
      }
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
        rawText,
        status,
        errorMessage,
      });
      return {
        status,
        text: '',
        selections: [],
        errorMessage,
        rawText,
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
    return this.engine.getMediaMarker?.() ?? null;
  }

  private getTaskContextKey(taskName: string): string {
    return `${this.contextKey}:${taskName}`;
  }
}

function failedResult<TPayload>(error: unknown): DirectorRunResult<TPayload> {
  return {
    status: error instanceof DirectorOutputError ? 'invalid_response' : 'failed',
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
