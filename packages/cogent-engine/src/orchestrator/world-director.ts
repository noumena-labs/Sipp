//////////////////////////////////////////////////////////////////////////////
//
// world-director.ts
//
// - Global "god" LLM role. Called by the orchestrator in two situations:
//     1. Conflict resolution — contested pick_up / use targets.
//     2. Tick narration — every N ticks (configurable) to emit a short
//        authorial note even when nothing is contested.
//
//   Stateless like SimulationAgent. Uses its own context key so its KV
//   cache is isolated from agent contexts.
//
//////////////////////////////////////////////////////////////////////////////

import type { CharacterAgentEngine } from '../character/character-agent.js';
import type { ChatMessage, PromptOptions } from '../core/inference-types.js';
import {
  getDirectorGrammar,
  parseDirectorOutput,
} from './director-grammar.js';
import type {
  DirectorDecision,
  WorldConflict,
  WorldSnapshot,
} from './simulation-types.js';

export interface WorldDirectorOptions {
  readonly maxOutputTokens?: number;
  readonly contextKey?: string;
}

export interface DirectorQueryResult {
  readonly decision: DirectorDecision;
  readonly cancelled: boolean;
  readonly errorMessage?: string;
  readonly rawText: string;
}

export class WorldDirector {
  private readonly engine: CharacterAgentEngine;
  private readonly grammarSource: string;
  private readonly maxOutputTokens: number;
  private readonly contextKey: string;

  public constructor(
    orchestratorId: string,
    engine: CharacterAgentEngine,
    options: WorldDirectorOptions = {}
  ) {
    this.engine = engine;
    this.maxOutputTokens = options.maxOutputTokens ?? 192;
    this.contextKey = options.contextKey ?? `director:${orchestratorId}`;
    this.grammarSource = getDirectorGrammar();
  }

  public getGrammarSource(): string {
    return this.grammarSource;
  }

  public async narrate(
    snapshot: WorldSnapshot,
    options: { signal?: AbortSignal } = {}
  ): Promise<DirectorQueryResult> {
    return this.run(buildNarrationPrompt(snapshot), options, deterministicNarration(snapshot));
  }

  public async resolveConflicts(
    snapshot: WorldSnapshot,
    conflicts: readonly WorldConflict[],
    options: { signal?: AbortSignal } = {}
  ): Promise<DirectorQueryResult> {
    const fallback = deterministicConflictResolution(conflicts);
    if (conflicts.length === 0) {
      return { decision: fallback, cancelled: false, rawText: '' };
    }
    return this.run(buildConflictPrompt(snapshot, conflicts), options, fallback);
  }

  private async run(
    userText: string,
    options: { signal?: AbortSignal },
    fallback: DirectorDecision
  ): Promise<DirectorQueryResult> {
    const messages: ChatMessage[] = [
      { role: 'system', content: DIRECTOR_SYSTEM_PROMPT },
      { role: 'user', content: userText },
    ];
    let promptText: string;
    try {
      promptText = await this.engine.applyChatTemplate(messages, true);
    } catch (error) {
      return {
        decision: fallback,
        cancelled: options.signal?.aborted === true,
        errorMessage: error instanceof Error ? error.message : String(error),
        rawText: '',
      };
    }
    const promptOptions: PromptOptions = {
      nTokens: this.maxOutputTokens,
      promptFormat: 'raw',
      grammar: this.grammarSource,
      ...(options.signal ? { signal: options.signal } : {}),
    };
    let requestId = 0;
    try {
      requestId = await this.engine.queuePrompt(this.contextKey, promptText, promptOptions);
      const response = await this.engine.runQueuedRequest(
        requestId,
        options.signal ? { signal: options.signal } : {}
      );
      const rawText = response.outputText ?? '';
      if (response.cancelled) {
        return { decision: fallback, cancelled: true, rawText };
      }
      if (response.failed) {
        return {
          decision: fallback,
          cancelled: false,
          errorMessage: response.errorMessage ?? 'generation failed',
          rawText,
        };
      }
      const parsed = parseDirectorOutput(rawText);
      if (!parsed) {
        return { decision: fallback, cancelled: false, rawText };
      }
      return { decision: parsed, cancelled: false, rawText };
    } catch (error) {
      const cancelled = options.signal?.aborted === true;
      if (requestId !== 0 && !cancelled && this.engine.cancelQueuedRequest) {
        try {
          await this.engine.cancelQueuedRequest(requestId);
        } catch {
          // swallow
        }
      }
      return {
        decision: fallback,
        cancelled,
        errorMessage: error instanceof Error ? error.message : String(error),
        rawText: '',
      };
    }
  }
}

const DIRECTOR_SYSTEM_PROMPT = `You are the Director of a small 2D courtyard world. You are not a
character in it; you are the authorial voice deciding global narration and
breaking ties when two agents want the same thing.

Respond with a single JSON object only:

{ "note": "<very short authorial line>",
  "resolutions": [
    { "objectId": "<id>", "winnerAgentId": "<id>"|null, "note": "<short>" }
  ]
}

Rules:
- "note" is always required. Keep it under 200 characters.
- "resolutions" is required when the user message lists conflicts, and each
  listed objectId must appear exactly once.
- When no conflicts are listed, omit "resolutions" entirely.
- "winnerAgentId" must be one of the contender ids, or null to deny all.
- No prose, no code fences, no extra keys.`;

function buildNarrationPrompt(snapshot: WorldSnapshot): string {
  const lines: string[] = [];
  lines.push(`tick: ${snapshot.tick}`);
  lines.push('agents:');
  for (const a of snapshot.agents) {
    lines.push(
      `  - ${a.id} "${a.name}" pos=(${a.position.x.toFixed(2)}, ${a.position.z.toFixed(2)}) holding=${a.holding ?? 'nothing'} emotion=${a.emotion ?? 'none'} status="${a.status}"`
    );
  }
  lines.push('objects:');
  for (const o of snapshot.objects) {
    lines.push(
      `  - ${o.id} (${o.kind}) pos=(${o.position.x.toFixed(2)}, ${o.position.z.toFixed(2)})${o.heldBy ? ` held_by=${o.heldBy}` : ''}`
    );
  }
  lines.push('');
  lines.push(
    'No conflicts this tick. Emit a one-line authorial note (a short narration or a gentle nudge).'
  );
  return lines.join('\n');
}

function buildConflictPrompt(
  snapshot: WorldSnapshot,
  conflicts: readonly WorldConflict[]
): string {
  const lines: string[] = [];
  lines.push(`tick: ${snapshot.tick}`);
  lines.push('agents:');
  for (const a of snapshot.agents) {
    lines.push(
      `  - ${a.id} "${a.name}" pos=(${a.position.x.toFixed(2)}, ${a.position.z.toFixed(2)}) holding=${a.holding ?? 'nothing'} emotion=${a.emotion ?? 'none'}`
    );
  }
  lines.push('objects:');
  for (const o of snapshot.objects) {
    lines.push(
      `  - ${o.id} (${o.kind}) pos=(${o.position.x.toFixed(2)}, ${o.position.z.toFixed(2)})${o.heldBy ? ` held_by=${o.heldBy}` : ''}`
    );
  }
  lines.push('conflicts:');
  for (const c of conflicts) {
    lines.push(
      `  - object=${c.objectId} contenders=[${c.contenderAgentIds.join(', ')}]`
    );
  }
  lines.push('');
  lines.push('Resolve each conflict. Pick one winner per object (or null to deny all).');
  return lines.join('\n');
}

function deterministicConflictResolution(
  conflicts: readonly WorldConflict[]
): DirectorDecision {
  const resolutions = conflicts.map((c) => ({
    objectId: c.objectId,
    winnerAgentId: c.contenderAgentIds[0] ?? null,
    note: 'first-come tie break',
  }));
  return { note: 'Director fell back to deterministic tie-break.', resolutions };
}

function deterministicNarration(snapshot: WorldSnapshot): DirectorDecision {
  return { note: `Tick ${snapshot.tick}: the courtyard carries on.`, resolutions: [] };
}
