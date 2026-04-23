import {
  createCharacterFromConfigUrl,
  type CharacterAgent,
  type CharacterConfig,
} from 'cogent-engine/character';
import type { CharacterAgentEngine } from 'cogent-engine/character';
import type { AgentIntent, AgentPerception } from './types.js';

export interface SimulationAgentChooserOptions {
  readonly maxChoiceOutputTokens?: number;
}

export interface SimulationAgentChoiceResult {
  readonly intent: AgentIntent;
  readonly cancelled: boolean;
  readonly errorMessage?: string;
}

export class SimulationAgentChooser {
  private readonly agent: CharacterAgent;
  private readonly config: CharacterConfig;
  private readonly actionNames: readonly string[];
  private readonly defaultEmotion: string;
  private readonly maxChoiceOutputTokens: number;

  public readonly agentId: string;

  public constructor(
    agentId: string,
    agent: CharacterAgent,
    config: CharacterConfig,
    options: SimulationAgentChooserOptions = {}
  ) {
    this.agentId = agentId;
    this.agent = agent;
    this.config = config;
    this.actionNames = config.actions.actions.map((action) => action.name);
    this.defaultEmotion = this.actionNames[0] ?? 'idle';
    this.maxChoiceOutputTokens = options.maxChoiceOutputTokens ?? 24;
  }

  public get characterConfig(): CharacterConfig {
    return this.config;
  }

  public async query(
    perception: AgentPerception,
    options: { signal?: AbortSignal } = {}
  ): Promise<SimulationAgentChoiceResult> {
    const signal = options.signal;
    const emotionChoice = await this.askEmotion(perception, signal);
    if (emotionChoice.cancelled) {
      return {
        intent: { kind: 'wait', emotion: this.defaultEmotion, reason: 'cancelled' },
        cancelled: true,
        ...(emotionChoice.errorMessage ? { errorMessage: emotionChoice.errorMessage } : {}),
      };
    }
    const emotion = emotionChoice.choice ?? this.defaultEmotion;

    const intentKindChoice = await this.agent.choose(renderIntentKindPrompt(perception), {
      choices: buildIntentKindChoices(perception),
      signal,
      maxOutputTokens: this.maxChoiceOutputTokens,
    });
    if (intentKindChoice.cancelled) {
      return {
        intent: { kind: 'wait', emotion, reason: 'cancelled' },
        cancelled: true,
        ...(intentKindChoice.errorMessage ? { errorMessage: intentKindChoice.errorMessage } : {}),
      };
    }

    const kind = intentKindChoice.choice ?? 'wait';
    const intent = await this.resolveIntent(kind, perception, emotion, signal);
    return {
      intent,
      cancelled: false,
      ...(intentKindChoice.errorMessage ? { errorMessage: intentKindChoice.errorMessage } : {}),
    };
  }

  private async askEmotion(
    perception: AgentPerception,
    signal: AbortSignal | undefined
  ) {
    return this.agent.choose(renderEmotionPrompt(perception), {
      choices: this.actionNames,
      signal,
      maxOutputTokens: this.maxChoiceOutputTokens,
    });
  }

  private async resolveIntent(
    kind: string,
    perception: AgentPerception,
    emotion: string,
    signal: AbortSignal | undefined
  ): Promise<AgentIntent> {
    switch (kind) {
      case 'wander':
        return { kind: 'wander', emotion };
      case 'drop':
        return { kind: 'drop', emotion };
      case 'wait':
        return { kind: 'wait', emotion, reason: 'pausing' };
      case 'approach_agent': {
        const agentIds = perception.nearbyAgents.map((agent) => agent.id);
        if (agentIds.length === 0) {
          return { kind: 'wait', emotion, reason: 'no-agent-visible' };
        }
        const choice = await this.agent.choose(renderTargetAgentPrompt(perception), {
          choices: agentIds,
          signal,
          maxOutputTokens: this.maxChoiceOutputTokens,
        });
        return choice.choice
          ? { kind: 'approach_agent', agentId: choice.choice, emotion }
          : { kind: 'wait', emotion, reason: 'no-agent-choice' };
      }
      case 'pick_up': {
        const objectIds = perception.nearbyObjects
          .filter((object) => object.heldBy == null)
          .map((object) => object.id);
        if (objectIds.length === 0) {
          return { kind: 'wait', emotion, reason: 'no-object-visible' };
        }
        const choice = await this.agent.choose(renderTargetObjectPrompt(perception, 'pick up'), {
          choices: objectIds,
          signal,
          maxOutputTokens: this.maxChoiceOutputTokens,
        });
        return choice.choice
          ? { kind: 'pick_up', objectId: choice.choice, emotion }
          : { kind: 'wait', emotion, reason: 'no-object-choice' };
      }
      case 'use': {
        const objectIds = perception.nearbyObjects.map((object) => object.id);
        if (objectIds.length === 0) {
          return { kind: 'wait', emotion, reason: 'no-object-visible' };
        }
        const choice = await this.agent.choose(renderTargetObjectPrompt(perception, 'use'), {
          choices: objectIds,
          signal,
          maxOutputTokens: this.maxChoiceOutputTokens,
        });
        return choice.choice
          ? { kind: 'use', objectId: choice.choice, emotion }
          : { kind: 'wait', emotion, reason: 'no-object-choice' };
      }
      case 'move_to_object': {
        const objectIds = perception.nearbyObjects.map((object) => object.id);
        if (objectIds.length === 0) {
          return { kind: 'wait', emotion, reason: 'no-object-visible' };
        }
        const choice = await this.agent.choose(renderTargetObjectPrompt(perception, 'move toward'), {
          choices: objectIds,
          signal,
          maxOutputTokens: this.maxChoiceOutputTokens,
        });
        const object = perception.nearbyObjects.find((entry) => entry.id === choice.choice);
        return object
          ? {
              kind: 'move_to',
              target: projectAhead(perception.self.position, object.direction, object.distance),
              emotion,
            }
          : { kind: 'wait', emotion, reason: 'no-object-choice' };
      }
      case 'move_to_agent': {
        const agentIds = perception.nearbyAgents.map((agent) => agent.id);
        if (agentIds.length === 0) {
          return { kind: 'wait', emotion, reason: 'no-agent-visible' };
        }
        const choice = await this.agent.choose(renderTargetAgentPrompt(perception), {
          choices: agentIds,
          signal,
          maxOutputTokens: this.maxChoiceOutputTokens,
        });
        const agent = perception.nearbyAgents.find((entry) => entry.id === choice.choice);
        return agent
          ? {
              kind: 'move_to',
              target: projectAhead(perception.self.position, agent.direction, agent.distance),
              emotion,
            }
          : { kind: 'wait', emotion, reason: 'no-agent-choice' };
      }
      default:
        return { kind: 'wait', emotion, reason: 'invalid-choice' };
    }
  }
}

export interface CreateSimulationAgentChooserFromConfigUrlOptions {
  readonly agentId: string;
  readonly configUrl: string;
  readonly engine: CharacterAgentEngine;
  readonly chooserOptions?: SimulationAgentChooserOptions;
  readonly fetch?: typeof globalThis.fetch;
  readonly signal?: AbortSignal;
}

export async function createSimulationAgentChooserFromConfigUrl(
  options: CreateSimulationAgentChooserFromConfigUrlOptions
): Promise<{ agent: SimulationAgentChooser; config: CharacterConfig }> {
  const { agent: characterAgent, config } = await createCharacterFromConfigUrl({
    configUrl: options.configUrl,
    engine: options.engine,
    fetch: options.fetch,
    signal: options.signal,
  });
  return {
    agent: new SimulationAgentChooser(options.agentId, characterAgent, config, options.chooserOptions),
    config,
  };
}

function buildIntentKindChoices(perception: AgentPerception): readonly string[] {
  const choices = ['wait', 'wander'];
  if (perception.nearbyAgents.length > 0) {
    choices.push('approach_agent', 'move_to_agent');
  }
  if (perception.nearbyObjects.length > 0) {
    choices.push('move_to_object', 'use');
    if (perception.nearbyObjects.some((object) => object.heldBy == null)) {
      choices.push('pick_up');
    }
  }
  if (perception.self.holding) {
    choices.push('drop');
  }
  return choices;
}

function renderIntentKindPrompt(perception: AgentPerception): string {
  return [
    renderPerceptionSummary(perception),
    '',
    'Given the current scene, choose the single next high-level action you most want to take.',
  ].join('\n');
}

function renderEmotionPrompt(perception: AgentPerception): string {
  return [
    renderPerceptionSummary(perception),
    '',
    'Choose the one expression that best matches your next move and current mood.',
  ].join('\n');
}

function renderTargetAgentPrompt(perception: AgentPerception): string {
  return [
    renderPerceptionSummary(perception),
    '',
    'Choose the one nearby agent you want to focus on right now.',
  ].join('\n');
}

function renderTargetObjectPrompt(perception: AgentPerception, verb: string): string {
  return [
    renderPerceptionSummary(perception),
    '',
    `Choose the one nearby object you want to ${verb} right now.`,
  ].join('\n');
}

function renderPerceptionSummary(perception: AgentPerception): string {
  const { self, nearbyAgents, nearbyObjects, tick, directorNote } = perception;
  const lines: string[] = [];
  lines.push(`tick ${tick}`);
  lines.push(
    `you are at (${self.position.x.toFixed(1)}, ${self.position.z.toFixed(1)}) holding ${self.holding ?? 'nothing'} and feeling ${self.emotion ?? 'neutral'}`
  );
  if (self.status) {
    lines.push(`your current status: ${self.status}`);
  }
  if (nearbyAgents.length > 0) {
    lines.push('nearby agents:');
    for (const agent of nearbyAgents) {
      lines.push(
        `- ${agent.id}: ${agent.name}, distance ${agent.distance.toFixed(1)}, holding ${agent.holding ?? 'nothing'}, emotion ${agent.emotion ?? 'none'}, status ${agent.status || 'none'}`
      );
    }
  }
  if (nearbyObjects.length > 0) {
    lines.push('nearby objects:');
    for (const object of nearbyObjects) {
      lines.push(
        `- ${object.id}: ${object.kind}, distance ${object.distance.toFixed(1)}, held by ${object.heldBy ?? 'nobody'}, contested ${object.contested ? 'yes' : 'no'}`
      );
    }
  }
  if (directorNote) {
    lines.push(`director note: ${directorNote}`);
  }
  return lines.join('\n');
}

function projectAhead(origin: { x: number; z: number }, direction: { x: number; z: number }, distance: number) {
  const travel = Math.min(Math.max(distance, 0.5), 4);
  return {
    x: origin.x + direction.x * travel,
    z: origin.z + direction.z * travel,
  };
}
