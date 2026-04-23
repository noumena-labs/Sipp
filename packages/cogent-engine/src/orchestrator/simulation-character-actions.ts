//////////////////////////////////////////////////////////////////////////////
//
// simulation-character-actions.ts
//
// - The fixed expressive-action vocabulary v1 SimulationAgents speak.
// - Validators that reject character.json files that do not expose this
//   exact set, so the LLM output grammar is always in sync with scene
//   rendering.
//
//////////////////////////////////////////////////////////////////////////////

import type { CharacterConfig } from '../character/character-config.js';
import type { SimulationActionName } from './simulation-types.js';

/**
 * Canonical ordering. Must stay in sync with
 * {@link SimulationActionName}. Exported as a tuple so grammar builders can
 * iterate it deterministically.
 */
export const SIMULATION_ACTION_NAMES = [
  'thinking',
  'curious',
  'happy',
  'confused',
  'alert',
  'frustrated',
  'sleepy',
  'celebrate',
] as const satisfies readonly SimulationActionName[];

/** `Set<string>` over {@link SIMULATION_ACTION_NAMES} for O(1) membership. */
export const SIMULATION_ACTION_NAME_SET: ReadonlySet<string> = new Set(
  SIMULATION_ACTION_NAMES
);

/** Fallback emotion used when output is invalid or no intent is active. */
export const DEFAULT_SIMULATION_EMOTION: SimulationActionName = 'confused';

export function isSimulationActionName(value: unknown): value is SimulationActionName {
  return typeof value === 'string' && SIMULATION_ACTION_NAME_SET.has(value);
}

/**
 * Throws if the given character.json does not expose exactly the fixed
 * simulation action vocabulary. We allow extra cue labels per action but
 * the action *names* themselves must match 1:1.
 */
export function assertCharacterActionsMatchSimulation(config: CharacterConfig): void {
  const declared = new Set<string>(config.actions.actions.map((a) => a.name));
  const missing: string[] = [];
  for (const required of SIMULATION_ACTION_NAMES) {
    if (!declared.has(required)) {
      missing.push(required);
    }
  }
  const extra: string[] = [];
  for (const name of declared) {
    if (!SIMULATION_ACTION_NAME_SET.has(name)) {
      extra.push(name);
    }
  }
  if (missing.length === 0 && extra.length === 0) {
    return;
  }
  const details: string[] = [];
  if (missing.length > 0) {
    details.push(`missing: [${missing.join(', ')}]`);
  }
  if (extra.length > 0) {
    details.push(`unexpected: [${extra.join(', ')}]`);
  }
  throw new Error(
    `character "${config.id}" does not expose the simulation action set (${details.join('; ')}).`
  );
}
