import {
  createCharacterFromConfigUrl,
  type CharacterAgent,
  type CharacterConfig,
} from 'cogent-engine/character';
import type { CharacterAgentEngine } from 'cogent-engine/character';
import { buildDecisionContext } from './decision-context.js';
import type { AgentGoal, AgentPerception, DecisionContext } from './types.js';

export interface SimulationAgentChooserOptions {
  readonly maxChoiceOutputTokens?: number;
}

export interface SimulationAgentChoiceResult {
  readonly goal: AgentGoal;
  readonly cancelled: boolean;
  readonly errorMessage?: string;
}

export class SimulationAgentChooser {
  private readonly agent: CharacterAgent;
  private readonly config: CharacterConfig;
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
    this.maxChoiceOutputTokens = options.maxChoiceOutputTokens ?? 24;
  }

  public get characterConfig(): CharacterConfig {
    return this.config;
  }

  public async query(
    perception: AgentPerception,
    options: { signal?: AbortSignal; timeoutMs?: number } = {}
  ): Promise<SimulationAgentChoiceResult> {
    const decision = buildDecisionContext(perception);
    const choiceResult = await this.agent.choose(decision.prompt, {
      choices: decision.options.map((option) => option.label),
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      maxOutputTokens: this.maxChoiceOutputTokens,
    });

    if (choiceResult.cancelled) {
      return {
        goal: { kind: 'wait', label: 'wait' },
        cancelled: true,
        ...(choiceResult.errorMessage ? { errorMessage: choiceResult.errorMessage } : {}),
      };
    }

    const chosen = findOptionByLabel(decision, choiceResult.choice) ?? findOptionByLabel(decision, 'wait');
    return {
      goal: chosen?.goal ?? { kind: 'wait', label: 'wait' },
      cancelled: false,
      ...(choiceResult.errorMessage ? { errorMessage: choiceResult.errorMessage } : {}),
    };
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

function findOptionByLabel(
  context: DecisionContext,
  label: string | null
): { label: string; goal: AgentGoal } | undefined {
  if (!label) return undefined;
  return context.options.find((option) => option.label === label);
}
