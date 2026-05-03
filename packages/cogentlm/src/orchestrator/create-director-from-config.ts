import { parseDirectorConfig } from './director-config.js';
import { DirectorRuntime, type DirectorRuntimeEngine } from './director-runtime.js';
import type { DirectorConfig } from './director-types.js';
import type { DirectorRuntimeOptions } from './director-types.js';
import { loadJsonConfig } from '../utils/load-json-config.js';

export interface CreateDirectorFromConfigUrlOptions {
  readonly configUrl: string;
  readonly engine: DirectorRuntimeEngine;
  readonly runtimeOptions?: DirectorRuntimeOptions;
  readonly fetch?: typeof globalThis.fetch;
  readonly signal?: AbortSignal;
}

export async function createDirectorFromConfigUrl(
  options: CreateDirectorFromConfigUrlOptions
): Promise<{ director: DirectorRuntime; config: DirectorConfig }> {
  const config = parseDirectorConfig(await loadJsonConfig(options.configUrl, {
    fetch: options.fetch,
    signal: options.signal,
    fetchLabel: 'createDirectorFromConfigUrl',
    httpLabel: 'director.json',
  }));
  return createDirectorFromConfig({
    config,
    engine: options.engine,
    runtimeOptions: options.runtimeOptions,
  });
}

export interface CreateDirectorFromConfigOptions {
  readonly config: DirectorConfig;
  readonly engine: DirectorRuntimeEngine;
  readonly runtimeOptions?: DirectorRuntimeOptions;
}

export function createDirectorFromConfig(
  options: CreateDirectorFromConfigOptions
): { director: DirectorRuntime; config: DirectorConfig } {
  const director = new DirectorRuntime(options.engine, options.config, options.runtimeOptions);
  return { director, config: options.config };
}
