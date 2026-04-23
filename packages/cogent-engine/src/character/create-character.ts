import { ActionBus } from './action-bus.js';
import { CharacterAgent, type CharacterAgentEngine, type CharacterAgentOptions } from './character-agent.js';
import { parseCharacterConfig, type CharacterConfig } from './character-config.js';

export interface CreateCharacterFromConfigUrlOptions {
  readonly configUrl: string;
  readonly engine: CharacterAgentEngine;
  readonly bus?: ActionBus;
  readonly agentOptions?: Omit<CharacterAgentOptions, 'bus'>;
  readonly fetch?: typeof globalThis.fetch;
  readonly signal?: AbortSignal;
}

export async function createCharacterFromConfigUrl(
  options: CreateCharacterFromConfigUrlOptions
): Promise<{ agent: CharacterAgent; config: CharacterConfig }> {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new Error(
      'createCharacterFromConfigUrl requires a fetch implementation. Pass `fetch` explicitly in this runtime.'
    );
  }

  const response = await fetchImpl(options.configUrl, { signal: options.signal });
  if (!response.ok) {
    throw new Error(`character.json HTTP ${response.status}`);
  }

  const config = parseCharacterConfig(await response.json());
  const agent = new CharacterAgent(options.engine, config, {
    ...options.agentOptions,
    ...(options.bus ? { bus: options.bus } : {}),
  });
  return { agent, config };
}
