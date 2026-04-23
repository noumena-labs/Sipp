//////////////////////////////////////////////////////////////////////////////
//
// simulation-agent.ts
//
// - Per-agent LLM wrapper used by the orchestrator to produce an intent
//   given the current perception. Stateless: every query builds a fresh
//   prompt from the character persona + current snapshot slice.
//
//   Uses the same structural CharacterAgentEngine interface as the
//   character subpath so the shared CogentEngine instance works unchanged.
//
//////////////////////////////////////////////////////////////////////////////

import type { CharacterAgentEngine } from '../character/character-agent.js';
import type { CharacterConfig } from '../character/character-config.js';
import type { ChatMessage, PromptOptions } from '../core/inference-types.js';
import { renderSystemPrompt } from '../character/persona.js';
import {
  defaultAgentOutput,
  getAgentGrammar,
  parseAgentOutput,
  type AgentOutput,
} from './agent-grammar.js';
import { assertCharacterActionsMatchSimulation } from './simulation-character-actions.js';
import type { AgentPerception } from './simulation-types.js';

export interface SimulationAgentOptions {
  readonly maxOutputTokens?: number;
  /** Override for `agent:<id>` engine context key. */
  readonly contextKey?: string;
}

export interface SimulationAgentQueryResult {
  readonly output: AgentOutput;
  readonly cancelled: boolean;
  readonly errorMessage?: string;
  readonly rawText: string;
}

/**
 * SimulationAgent — wraps a CharacterConfig + engine for just-in-time
 * intent queries. Does not own a timer; the orchestrator drives it.
 */
export class SimulationAgent {
  private readonly engine: CharacterAgentEngine;
  private readonly config: CharacterConfig;
  private readonly systemPrompt: string;
  private readonly grammarSource: string;
  private readonly maxOutputTokens: number;
  private readonly contextKey: string;

  public readonly agentId: string;

  public constructor(
    agentId: string,
    engine: CharacterAgentEngine,
    config: CharacterConfig,
    options: SimulationAgentOptions = {}
  ) {
    assertCharacterActionsMatchSimulation(config);
    this.agentId = agentId;
    this.engine = engine;
    this.config = config;
    this.maxOutputTokens = options.maxOutputTokens ?? 192;
    this.contextKey = options.contextKey ?? `agent:${agentId}`;
    this.systemPrompt = buildSystemPrompt(config);
    this.grammarSource = getAgentGrammar();
  }

  public get characterConfig(): CharacterConfig {
    return this.config;
  }

  public getGrammarSource(): string {
    return this.grammarSource;
  }

  public getSystemPrompt(): string {
    return this.systemPrompt;
  }

  /**
   * Runs one query with the given perception. Always resolves — on parse or
   * runtime failure a default `{ wait, confused }` output is returned with
   * `errorMessage` set.
   */
  public async query(
    perception: AgentPerception,
    options: { signal?: AbortSignal } = {}
  ): Promise<SimulationAgentQueryResult> {
    const userText = renderPerceptionMessage(perception);
    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: userText },
    ];

    let promptText: string;
    try {
      promptText = await this.engine.applyChatTemplate(messages, true);
    } catch (error) {
      return {
        output: defaultAgentOutput('prompt-failed'),
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
        return {
          output: defaultAgentOutput('cancelled'),
          cancelled: true,
          rawText,
        };
      }
      if (response.failed) {
        return {
          output: defaultAgentOutput('failed'),
          cancelled: false,
          errorMessage: response.errorMessage ?? 'generation failed',
          rawText,
        };
      }
      const parsed = parseAgentOutput(rawText);
      if (!parsed) {
        return {
          output: defaultAgentOutput('invalid-output'),
          cancelled: false,
          rawText,
        };
      }
      return { output: parsed, cancelled: false, rawText };
    } catch (error) {
      const cancelled = options.signal?.aborted === true;
      if (requestId !== 0 && !cancelled && this.engine.cancelQueuedRequest) {
        try {
          await this.engine.cancelQueuedRequest(requestId);
        } catch {
          // Swallow — original error is more useful.
        }
      }
      return {
        output: defaultAgentOutput(cancelled ? 'cancelled' : 'error'),
        cancelled,
        errorMessage: error instanceof Error ? error.message : String(error),
        rawText: '',
      };
    }
  }
}

function buildSystemPrompt(config: CharacterConfig): string {
  const persona = renderSystemPrompt(config.persona, config.actions);
  return `${persona}

You are an autonomous agent living in a small 2D courtyard world. Every tick
you receive a perception summary and must respond with one JSON object and
nothing else:

{ "intent": { "kind": "<one-of>", ...fields }, "status": "<very short>" }

Allowed intent kinds and their required fields:
- wait:            { "kind": "wait",            "emotion": "<emoji>", "reason": "<short>" }
- wander:          { "kind": "wander",          "emotion": "<emoji>" }
- move_to:         { "kind": "move_to",         "target": { "x": <num>, "z": <num> }, "emotion": "<emoji>" }
- approach_agent:  { "kind": "approach_agent",  "agentId": "<id>", "emotion": "<emoji>" }
- pick_up:         { "kind": "pick_up",         "objectId": "<id>", "emotion": "<emoji>" }
- drop:            { "kind": "drop",            "emotion": "<emoji>" }
- use:             { "kind": "use",             "objectId": "<id>", "emotion": "<emoji>" }

"emotion" must be exactly one of: thinking, curious, happy, confused, alert,
frustrated, sleepy, celebrate.

Keep "status" under 80 characters. Do not add prose, code fences, or extra
keys. Output exactly one JSON object.`;
}

function renderPerceptionMessage(perception: AgentPerception): string {
  const { self, nearbyAgents, nearbyObjects, tick, bounds, directorNote } = perception;
  const lines: string[] = [];
  lines.push(`tick: ${tick}`);
  lines.push(
    `world: square bounds, half-extent ${bounds.halfExtent.toFixed(1)} (from -${bounds.halfExtent.toFixed(1)} to +${bounds.halfExtent.toFixed(1)} on x and z)`
  );
  lines.push(`you: id=${self.id} name=${self.name}`);
  lines.push(
    `  position=(${self.position.x.toFixed(2)}, ${self.position.z.toFixed(2)}) heading=${self.heading.toFixed(2)}rad`
  );
  lines.push(`  holding=${self.holding ?? 'nothing'} status="${self.status}"`);
  lines.push(`  last_emotion=${self.emotion ?? 'none'}`);

  if (nearbyAgents.length === 0) {
    lines.push('nearby_agents: (none)');
  } else {
    lines.push('nearby_agents:');
    for (const a of nearbyAgents) {
      lines.push(
        `  - id=${a.id} name=${a.name} distance=${a.distance.toFixed(2)} holding=${a.holding ?? 'nothing'} emotion=${a.emotion ?? 'none'} status="${a.status}"`
      );
    }
  }
  if (nearbyObjects.length === 0) {
    lines.push('nearby_objects: (none)');
  } else {
    lines.push('nearby_objects:');
    for (const o of nearbyObjects) {
      const owner = o.heldBy ? ` held_by=${o.heldBy}` : '';
      const contested = o.contested ? ' contested=true' : '';
      lines.push(
        `  - id=${o.id} kind=${o.kind} distance=${o.distance.toFixed(2)}${owner}${contested}`
      );
    }
  }
  if (directorNote) {
    lines.push(`director_note: "${directorNote}"`);
  }
  lines.push('');
  lines.push('Respond with one JSON object only.');
  return lines.join('\n');
}
