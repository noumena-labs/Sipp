import { CharacterEventBus } from './action-bus.js';
import { CharacterRuntime, type CharacterRuntimeClient, type CharacterRuntimeOptions } from './character-agent.js';
import { parseCharacterConfig, type CharacterConfig } from './character-config.js';

export interface CreateCharacterFromConfigUrlOptions {
  readonly configUrl: string;
  /** Chat client used by the constructed CharacterRuntime. */
  readonly client: CharacterRuntimeClient;
  readonly bus?: CharacterEventBus;
  readonly runtimeOptions?: Omit<CharacterRuntimeOptions, 'bus'>;
  readonly fetch?: typeof globalThis.fetch;
  readonly signal?: AbortSignal;
}

export async function createCharacterFromConfigUrl(
  options: CreateCharacterFromConfigUrlOptions
): Promise<{ character: CharacterRuntime; config: CharacterConfig }> {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new Error('createCharacterFromConfigUrl requires a fetch implementation. Pass `fetch` explicitly in this runtime.');
  }
  const response = await fetchImpl(options.configUrl, { signal: options.signal });
  if (!response.ok) {
    throw new Error(`character.json HTTP ${response.status}`);
  }
  const config = parseCharacterConfig(await response.json());
  const character = new CharacterRuntime(options.client, config, {
    ...options.runtimeOptions,
    ...(options.bus ? { bus: options.bus } : {}),
  });
  return { character, config };
}
