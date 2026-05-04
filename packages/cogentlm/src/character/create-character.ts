import { CharacterEventBus } from './action-bus.js';
import { CharacterRuntime, type CharacterRuntimeEngine, type CharacterRuntimeOptions } from './character-agent.js';
import { parseCharacterConfig, type CharacterConfig } from './character-config.js';
import { loadJsonConfig } from '../utils/load-json-config.js';

export interface CreateCharacterFromConfigUrlOptions {
  readonly configUrl: string;
  readonly engine: CharacterRuntimeEngine;
  readonly bus?: CharacterEventBus;
  readonly runtimeOptions?: Omit<CharacterRuntimeOptions, 'bus'>;
  readonly fetch?: typeof globalThis.fetch;
  readonly signal?: AbortSignal;
}

export async function createCharacterFromConfigUrl(
  options: CreateCharacterFromConfigUrlOptions
): Promise<{ character: CharacterRuntime; config: CharacterConfig }> {
  const config = parseCharacterConfig(await loadJsonConfig(options.configUrl, {
    fetch: options.fetch,
    signal: options.signal,
    fetchLabel: 'createCharacterFromConfigUrl',
    httpLabel: 'character.json',
  }));
  const character = new CharacterRuntime(options.engine, config, {
    ...options.runtimeOptions,
    ...(options.bus ? { bus: options.bus } : {}),
  });
  return { character, config };
}

export interface CreateCharacterFromConfigOptions {
  readonly config: CharacterConfig;
  readonly engine: CharacterRuntimeEngine;
  readonly bus?: CharacterEventBus;
  readonly runtimeOptions?: Omit<CharacterRuntimeOptions, 'bus'>;
}

export function createCharacterFromConfig(
  options: CreateCharacterFromConfigOptions
): { character: CharacterRuntime; config: CharacterConfig } {
  const character = new CharacterRuntime(options.engine, options.config, {
    ...options.runtimeOptions,
    ...(options.bus ? { bus: options.bus } : {}),
  });
  return { character, config: options.config };
}
