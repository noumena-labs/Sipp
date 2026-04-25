import type {
  GenerateRequestId,
  GenerateResponse,
  PromptOptions,
  RequestObservabilityMetrics,
} from '@noumena-labs/cogent-engine';
import type { CogentEngine } from '@noumena-labs/cogent-engine';
import type { CharacterAgentEngine } from '@noumena-labs/cogent-engine/character';
import type { BrainDefinition, BrainQueryType, BrainQueryStatus, BrainActivityStore } from './brain-activity-store.js';

interface AppliedTemplateSnapshot {
  readonly promptText: string;
  readonly messages: ReadonlyArray<{ role: string; content: string }>;
}

export function createTracedBrainEngine(
  engine: CogentEngine,
  store: BrainActivityStore,
  brain: BrainDefinition
): CharacterAgentEngine {
  return new TracedBrainEngine(engine, store, brain);
}

class TracedBrainEngine implements CharacterAgentEngine {
  private lastAppliedTemplate: AppliedTemplateSnapshot | null = null;

  public constructor(
    private readonly engine: CogentEngine,
    private readonly store: BrainActivityStore,
    private readonly brain: BrainDefinition
  ) {}

  public async queuePrompt(
    contextKey: string,
    promptText: string,
    options: number | PromptOptions = 128
  ): Promise<GenerateRequestId> {
    const template = this.lastAppliedTemplate?.promptText === promptText ? this.lastAppliedTemplate : null;
    this.lastAppliedTemplate = null;

    const prompts = extractPromptSections(template?.messages ?? []);
    const directorTaskName = this.brain.kind === 'director' ? parseDirectorTaskName(prompts.userPrompt) : null;
    const queryType = classifyQueryType(this.brain.kind, directorTaskName);
    const responseSanitizer = createTraceResponseSanitizer();
    const queryId = this.store.beginQuery({
      brainId: this.brain.id,
      queryType,
      ...(directorTaskName ? { queryName: directorTaskName } : {}),
      contextKey,
      systemPrompt: prompts.systemPrompt,
      userPrompt: prompts.userPrompt,
      renderedPrompt: promptText,
      grammar: typeof options === 'object' ? options.grammar : null,
    });

    try {
      const requestId = await this.engine.queuePrompt(
        contextKey,
        promptText,
        withStreamingTap(options, (chunk) => {
          const safeChunk = responseSanitizer.consume(chunk);
          if (safeChunk.length > 0) {
            this.store.appendResponse(queryId, safeChunk);
          }
        })
      );
      this.store.attachRequestId(queryId, requestId);
      return requestId;
    } catch (error) {
      this.store.finishQuery(queryId, {
        status: classifyErrorStatus(error),
        errorMessage: asErrorMessage(error),
      });
      throw error;
    }
  }

  public async runQueuedRequest(
    requestId: GenerateRequestId,
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    const queryId = this.store.getQueryIdForRequest(requestId);
    try {
      const response = await this.engine.runQueuedRequest(requestId, options);
      if (queryId) {
        this.store.finishQuery(queryId, {
          status: classifyResponseStatus(response),
          responseText: sanitizeTraceResponseText(response.outputText),
          errorMessage: response.errorMessage ?? null,
          requestObservability: getObservability(response),
        });
      }
      return response;
    } catch (error) {
      if (queryId) {
        this.store.finishQuery(queryId, {
          status: classifyErrorStatus(error),
          errorMessage: asErrorMessage(error),
        });
      }
      throw error;
    }
  }

  public async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
    return this.engine.cancelQueuedRequest(requestId);
  }

  public getChatTemplate(): string | null {
    return this.engine.getChatTemplate();
  }

  public getBosText(): string {
    return this.engine.getBosText();
  }

  public getEosText(): string {
    return this.engine.getEosText();
  }

  public getMediaMarker(): string | null {
    return this.engine.getMediaMarker();
  }

  public async applyChatTemplate(
    messages: Array<{ role: string; content: string }>,
    addAssistant: boolean
  ): Promise<string> {
    const promptText = await this.engine.applyChatTemplate(messages, addAssistant);
    this.lastAppliedTemplate = addAssistant ? { promptText, messages } : null;
    return promptText;
  }
}

function withStreamingTap(
  options: number | PromptOptions,
  onChunk: (chunk: string) => void
): PromptOptions {
  if (typeof options === 'number') {
    return {
      nTokens: options,
      onToken: onChunk,
    };
  }

  const upstream = options.onToken;
  return {
    ...options,
    onToken: (chunk: string) => {
      onChunk(chunk);
      upstream?.(chunk);
    },
  };
}

function extractPromptSections(messages: ReadonlyArray<{ role: string; content: string }>): {
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

function parseDirectorTaskName(userPrompt: string): string | null {
  const match = userPrompt.match(/^Task:\s*([^\n]+)/m);
  return match?.[1]?.trim() ?? null;
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

function classifyResponseStatus(response: GenerateResponse): Exclude<BrainQueryStatus, 'idle' | 'running'> {
  if (response.cancelled) {
    return response.errorMessage?.toLowerCase().includes('timed out') ? 'timed_out' : 'cancelled';
  }
  if (response.failed) {
    return response.errorMessage?.toLowerCase().includes('timed out') ? 'timed_out' : 'failed';
  }
  return 'completed';
}

function classifyErrorStatus(error: unknown): Exclude<BrainQueryStatus, 'idle' | 'running'> {
  if (!isAbortError(error)) {
    return asErrorMessage(error).toLowerCase().includes('timed out') ? 'timed_out' : 'failed';
  }
  return asErrorMessage(error).toLowerCase().includes('timed out') ? 'timed_out' : 'cancelled';
}

function getObservability(response: GenerateResponse): RequestObservabilityMetrics | null {
  return response.requestObservability ?? response.runtimeObservability ?? null;
}

function asErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

const TRACE_RESPONSE_BOUNDARY_MARKERS = [
  '<system-reminder>',
  '</system-reminder>',
  '<|im_start|>system',
  '<|im_start|>user',
  '<|im_start|>assistant',
  '<|start_header_id|>system<|end_header_id|>',
  '<|start_header_id|>user<|end_header_id|>',
  '<|start_header_id|>assistant<|end_header_id|>',
  '<｜System｜>',
  '<｜User｜>',
  '<｜Assistant｜>',
  '<system>',
  '<user>',
  '<assistant>',
] as const;

function sanitizeTraceResponseText(rawText: string): string {
  const sanitizer = createTraceResponseSanitizer();
  return sanitizer.consume(rawText).trim();
}

function createTraceResponseSanitizer(): { consume(chunk: string): string } {
  let pending = '';
  let stopped = false;

  return {
    consume(chunk: string): string {
      if (stopped || chunk.length === 0) {
        return '';
      }
      pending += chunk;
      const split = splitOnTraceBoundary(pending);
      pending = split.trailingText;
      if (split.hitBoundary) {
        pending = '';
        stopped = true;
      }
      return split.safeText;
    },
  };
}

function splitOnTraceBoundary(text: string): { safeText: string; trailingText: string; hitBoundary: boolean } {
  let earliestIndex = -1;
  let matchedMarker = '';

  for (const marker of TRACE_RESPONSE_BOUNDARY_MARKERS) {
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
  for (const marker of TRACE_RESPONSE_BOUNDARY_MARKERS) {
    const overlap = longestTraceSuffixPrefixOverlap(text, marker);
    safeLength = Math.min(safeLength, text.length - overlap);
  }

  return {
    safeText: text.slice(0, safeLength),
    trailingText: text.slice(safeLength),
    hitBoundary: false,
  };
}

function longestTraceSuffixPrefixOverlap(source: string, marker: string): number {
  const maxLength = Math.min(source.length, marker.length - 1);
  for (let length = maxLength; length > 0; length -= 1) {
    if (source.endsWith(marker.slice(0, length))) {
      return length;
    }
  }
  return 0;
}
