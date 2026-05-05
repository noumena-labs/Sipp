import type { ChatInput, ChatOptions, CogentEngine, QueryInput, QueryOptions } from '@noumena-labs/cogentlm';
import type { CharacterRuntimeEngine } from '@noumena-labs/cogentlm/character';
import type { DirectorRuntimeEngine } from '@noumena-labs/cogentlm/director';
import type { BrainDefinition, BrainQueryType, BrainQueryStatus, BrainActivityStore } from './brain-activity-store.js';

interface TracedChatMessage {
  role: string;
  content: string;
}

export function createTracedBrainEngine(
  engine: CogentEngine,
  store: BrainActivityStore,
  brain: BrainDefinition
): CharacterRuntimeEngine & DirectorRuntimeEngine {
  return new TracedBrainEngine(engine, store, brain);
}

class TracedBrainEngine implements CharacterRuntimeEngine, DirectorRuntimeEngine {
  public readonly models = {
    current: () => this.engine.models.current(),
  };

  public constructor(
    private readonly engine: CogentEngine,
    private readonly store: BrainActivityStore,
    private readonly brain: BrainDefinition
  ) { }

  public async chat(input: ChatInput, options: ChatOptions = {}): Promise<string> {
    const messages = Array.isArray(input) ? input : input.messages;
    const prompts = extractPromptSections(messages);
    const directorTaskName = this.brain.kind === 'director' ? parseDirectorTaskName(options.session ?? '') : null;
    const queryType = classifyQueryType(this.brain.kind, directorTaskName);
    const queryId = this.store.beginQuery({
      brainId: this.brain.id,
      queryType,
      ...(directorTaskName ? { queryName: directorTaskName } : {}),
      contextKey: options.session ?? 'default',
      systemPrompt: prompts.systemPrompt,
      userPrompt: prompts.userPrompt,
      renderedPrompt: renderMessagesForTrace(messages),
      grammar: options.grammar ?? null,
    });

    try {
      const response = await this.engine.chat(
        input,
        withStreamingTap(options, (chunk) => {
          this.store.appendResponse(queryId, chunk);
        })
      );

      this.store.finishQuery(queryId, {
        status: 'completed',
        responseText: response,
        errorMessage: null,
      });
      return response;
    } catch (error) {
      this.store.finishQuery(queryId, {
        status: classifyErrorStatus(error),
        errorMessage: asErrorMessage(error),
      });
      throw error;
    }
  }

  public async query(input: QueryInput, options: QueryOptions = {}): Promise<string> {
    const promptText = typeof input === 'string' ? input : input.prompt;

    const directorTaskName = this.brain.kind === 'director' ? parseDirectorTaskName(options.session ?? '') : null;
    const queryType = classifyQueryType(this.brain.kind, directorTaskName);
    const queryId = this.store.beginQuery({
      brainId: this.brain.id,
      queryType,
      ...(directorTaskName ? { queryName: directorTaskName } : {}),
      contextKey: options.session ?? 'default',
      systemPrompt: null,
      userPrompt: null,
      renderedPrompt: promptText,
      grammar: options.grammar ?? null,
    });

    try {
      const response = await this.engine.query(
        input,
        withStreamingTap(options, (chunk) => {
          this.store.appendResponse(queryId, chunk);
        })
      );

      this.store.finishQuery(queryId, {
        status: 'completed',
        responseText: response,
        errorMessage: null,
      });
      return response;
    } catch (error) {
      this.store.finishQuery(queryId, {
        status: classifyErrorStatus(error),
        errorMessage: asErrorMessage(error),
      });
      throw error;
    }
  }
}

function withStreamingTap(
  options: QueryOptions,
  onChunk: (chunk: string) => void
): QueryOptions {
  const upstream = options.onToken;
  return {
    ...options,
    onToken: (chunk: string) => {
      onChunk(chunk);
      upstream?.(chunk);
    },
  };
}

function extractPromptSections(messages: readonly TracedChatMessage[]): {
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

function renderMessagesForTrace(messages: readonly TracedChatMessage[]): string {
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
