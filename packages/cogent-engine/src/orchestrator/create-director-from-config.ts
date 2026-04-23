import { parseDirectorConfig } from './director-config.js';
import { DirectorRuntime } from './director-runtime.js';
import type { DirectorConfig } from './director-types.js';
import type { CharacterAgentEngine } from '../character/character-agent.js';
import type { DirectorRuntimeOptions } from './director-types.js';

export interface CreateDirectorFromConfigUrlOptions {
  readonly configUrl: string;
  readonly engine: CharacterAgentEngine;
  readonly runtimeOptions?: DirectorRuntimeOptions;
  readonly fetch?: typeof globalThis.fetch;
  readonly signal?: AbortSignal;
}

export async function createDirectorFromConfigUrl(
  options: CreateDirectorFromConfigUrlOptions
): Promise<{ director: DirectorRuntime; config: DirectorConfig }> {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new Error(
      'createDirectorFromConfigUrl requires a fetch implementation. Pass `fetch` explicitly in this runtime.'
    );
  }

  const response = await fetchImpl(options.configUrl, { signal: options.signal });
  if (!response.ok) {
    throw new Error(`director.json HTTP ${response.status}`);
  }

  const config = parseDirectorConfig(await response.json());
  const director = new DirectorRuntime(options.engine, config, options.runtimeOptions);
  return { director, config };
}
