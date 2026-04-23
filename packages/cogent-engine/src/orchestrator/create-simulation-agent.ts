//////////////////////////////////////////////////////////////////////////////
//
// create-simulation-agent.ts
//
// - Factory mirroring `createCharacterFromConfigUrl` — fetches a
//   character.json, validates that its action vocabulary matches the
//   fixed simulation set, and returns a ready-to-attach SimulationAgent.
//
//////////////////////////////////////////////////////////////////////////////

import type { CharacterAgentEngine } from '../character/character-agent.js';
import { parseCharacterConfig, type CharacterConfig } from '../character/character-config.js';
import { SimulationAgent, type SimulationAgentOptions } from './simulation-agent.js';

export interface CreateSimulationAgentOptions {
  readonly agentId: string;
  readonly configUrl: string;
  readonly engine: CharacterAgentEngine;
  readonly agentOptions?: SimulationAgentOptions;
  readonly fetch?: typeof globalThis.fetch;
  readonly signal?: AbortSignal;
}

export async function createSimulationAgentFromConfigUrl(
  options: CreateSimulationAgentOptions
): Promise<{ agent: SimulationAgent; config: CharacterConfig }> {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new Error(
      'createSimulationAgentFromConfigUrl requires a fetch implementation. Pass `fetch` explicitly in this runtime.'
    );
  }
  const response = await fetchImpl(options.configUrl, { signal: options.signal });
  if (!response.ok) {
    throw new Error(`character.json HTTP ${response.status}`);
  }
  const config = parseCharacterConfig(await response.json());
  const agent = new SimulationAgent(
    options.agentId,
    options.engine,
    config,
    options.agentOptions ?? {}
  );
  return { agent, config };
}
