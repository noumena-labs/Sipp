import type {
  CharacterRuntime,
  CharacterConfig,
  CharacterChooseResult,
} from '@noumena-labs/cogentlm/character';
import { buildDecisionContext } from './decision-context.js';
import type { AgentGoal, AgentPerception, DecisionContext } from './types.js';

export interface SimulationAgentDecisionResult {
  readonly goal: AgentGoal | null;
  readonly status: CharacterChooseResult['status'];
  readonly errorMessage?: string;
  readonly rawText: string;
}

export class SimulationAgentChooser {
  private readonly character: CharacterRuntime;
  private readonly config: CharacterConfig;

  public readonly agentId: string;

  public constructor(
    agentId: string,
    character: CharacterRuntime,
    config: CharacterConfig
  ) {
    this.agentId = agentId;
    if (config.id !== agentId) {
      throw new Error(`character config id ${JSON.stringify(config.id)} must match simulation agent id ${JSON.stringify(agentId)}.`);
    }
    this.character = character;
    this.config = config;
  }

  public get characterConfig(): CharacterConfig {
    return this.config;
  }

  public async query(
    perception: AgentPerception,
    options: { signal?: AbortSignal; timeoutMs?: number } = {}
  ): Promise<SimulationAgentDecisionResult> {
    const decision = buildDecisionContext(perception);
    if (decision.options.length === 1) {
      return {
        goal: decision.options[0]!.goal,
        status: 'ok',
        rawText: '',
      };
    }

    const chooseResult = await this.character.choose(decision.prompt, {
      choices: decision.options.map((option, index) => ({
        id: String(index),
        label: option.label,
      })),
      signal: options.signal,
      timeoutMs: options.timeoutMs,
    });

    if (chooseResult.status !== 'ok') {
      return {
        goal: null,
        status: chooseResult.status,
        ...(chooseResult.errorMessage ? { errorMessage: chooseResult.errorMessage } : {}),
        rawText: chooseResult.rawText,
      };
    }

    const chosen = findOptionById(decision, chooseResult.selection);
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

function findOptionById(
  context: DecisionContext,
  id: string | null
): { label: string; goal: AgentGoal } | undefined {
  if (!id) return undefined;
  return Number(id) >= 0 ? context.options[Number(id)] : undefined;
}