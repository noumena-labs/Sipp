import {
  createCharacterFromConfigUrl,
  type CharacterRuntime,
  type CharacterRuntimeEngine,
  type CharacterConfig,
  type CharacterChooseResult,
} from '@noumena-labs/cogentlm-browser/character';
import { buildDecisionContext } from './decision-context.js';
import type { AgentGoal, AgentPerception, DecisionContext } from './types.js';

export interface SimulationAgentChooserOptions {
  readonly maxDecisionOutputTokens?: number;
}

export interface SimulationAgentDecisionResult {
  readonly goal: AgentGoal | null;
  readonly status: CharacterChooseResult['status'];
  readonly errorMessage?: string;
  readonly rawText: string;
}

export class SimulationAgentChooser {
  private readonly character: CharacterRuntime;
  private readonly config: CharacterConfig;
  private readonly maxDecisionOutputTokens: number;

  public readonly agentId: string;

  public constructor(
    agentId: string,
    character: CharacterRuntime,
    config: CharacterConfig,
    options: SimulationAgentChooserOptions = {}
  ) {
    this.agentId = agentId;
    if (config.id !== agentId) {
      throw new Error(`character config id ${JSON.stringify(config.id)} must match simulation agent id ${JSON.stringify(agentId)}.`);
    }
    this.character = character;
    this.config = config;
    this.maxDecisionOutputTokens = options.maxDecisionOutputTokens ?? 24;
  }

  public get characterConfig(): CharacterConfig {
    return this.config;
  }

  public async query(
    perception: AgentPerception,
    options: { signal?: AbortSignal; timeoutMs?: number } = {}
  ): Promise<SimulationAgentDecisionResult> {
    const decision = buildDecisionContext(perception);
    const chooseResult = await this.character.choose(decision.prompt, {
      choices: decision.options.map((option) => option.label),
      signal: options.signal,
      timeoutMs: options.timeoutMs,
      maxOutputTokens: this.maxDecisionOutputTokens,
    });

    if (chooseResult.status !== 'ok') {
      return {
        goal: null,
        status: chooseResult.status,
        ...(chooseResult.errorMessage ? { errorMessage: chooseResult.errorMessage } : {}),
        rawText: chooseResult.rawText,
      };
    }

    const chosen = findOptionByLabel(decision, chooseResult.selection);
    if (!chosen) {
      return {
        goal: null,
        status: 'invalid_response',
        errorMessage: chooseResult.errorMessage ?? 'choice output did not match any available option',
        rawText: chooseResult.rawText,
      };
    }
    return {
      goal: chosen.goal,
      status: 'ok',
      rawText: chooseResult.rawText,
      ...(chooseResult.errorMessage ? { errorMessage: chooseResult.errorMessage } : {}),
    };
  }
}

export interface CreateSimulationAgentChooserFromConfigUrlOptions {
  readonly agentId: string;
  readonly configUrl: string;
  readonly engine: CharacterRuntimeEngine;
  readonly chooserOptions?: SimulationAgentChooserOptions;
  readonly fetch?: typeof globalThis.fetch;
  readonly signal?: AbortSignal;
}

export async function createSimulationAgentChooserFromConfigUrl(
  options: CreateSimulationAgentChooserFromConfigUrlOptions
): Promise<{ agent: SimulationAgentChooser; config: CharacterConfig }> {
  const { character, config } = await createCharacterFromConfigUrl({
    configUrl: options.configUrl,
    engine: options.engine,
    fetch: options.fetch,
    signal: options.signal,
  });
  return {
    agent: new SimulationAgentChooser(options.agentId, character, config, options.chooserOptions),
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
