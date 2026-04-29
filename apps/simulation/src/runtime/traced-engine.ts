import type { CogentEngine, QueryInput, QueryOptions } from '@noumena-labs/cogent-engine';
import type { CharacterRuntimeEngine } from '@noumena-labs/cogent-engine/character';
import type { DirectorRuntimeEngine } from '@noumena-labs/cogent-engine/director';
import type { BrainDefinition, BrainQueryType, BrainQueryStatus, BrainActivityStore } from './brain-activity-store.js';

interface TracedChatMessage {
  role: string;
  content: string;
}

interface AppliedTemplateSnapshot {
  readonly promptText: string;
  readonly messages: readonly TracedChatMessage[];
}

export function createTracedBrainEngine(
  engine: CogentEngine,
  store: BrainActivityStore,
  brain: BrainDefinition
): CharacterRuntimeEngine & DirectorRuntimeEngine {
  return new TracedBrainEngine(engine, store, brain);
}

class TracedBrainEngine implements CharacterRuntimeEngine, DirectorRuntimeEngine {
  private readonly appliedTemplatesByPrompt = new Map<string, AppliedTemplateSnapshot[]>();

  public constructor(
    private readonly engine: CogentEngine,
    private readonly store: BrainActivityStore,
    private readonly brain: BrainDefinition
  ) { }

  public async query(input: QueryInput, options: QueryOptions = {}): Promise<string> {
    const promptText = typeof input === 'string' ? input : input.prompt;
    const template = this.takeAppliedTemplate(promptText ?? '');

    const prompts = extractPromptSections(template?.messages ?? []);
    const directorTaskName = this.brain.kind === 'director' ? parseDirectorTaskName(options.session ?? '') : null;
    const queryType = classifyQueryType(this.brain.kind, directorTaskName);
    const queryId = this.store.beginQuery({
      brainId: this.brain.id,
      queryType,
      ...(directorTaskName ? { queryName: directorTaskName } : {}),
      contextKey: options.session ?? 'default',
      systemPrompt: prompts.systemPrompt,
      userPrompt: prompts.userPrompt,
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

  public getMediaMarker(): string | null {
    return (this.engine as any).getMediaMarker?.() ?? null;
  }

  public async applyChatTemplate(
    messages: TracedChatMessage[],
    addAssistant: boolean
  ): Promise<string> {
    const promptText = await this.engine.applyChatTemplate(messages, addAssistant);
    if (addAssistant) {
      this.rememberAppliedTemplate({ promptText, messages });
    }
    return promptText;
  }

  private rememberAppliedTemplate(snapshot: AppliedTemplateSnapshot): void {
    const entries = this.appliedTemplatesByPrompt.get(snapshot.promptText) ?? [];
    entries.push(snapshot);
    this.appliedTemplatesByPrompt.set(snapshot.promptText, entries);
  }

  private takeAppliedTemplate(promptText: string): AppliedTemplateSnapshot | null {
    const entries = this.appliedTemplatesByPrompt.get(promptText);
    const snapshot = entries?.shift() ?? null;
    if (entries && entries.length === 0) {
      this.appliedTemplatesByPrompt.delete(promptText);
    }
    return snapshot;
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
