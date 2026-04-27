export const WORLD_EFFECT_ACTIONS = [
  'summon_familiar',
  'cast_starbolt',
  'raise_ward',
  'summon_rune_circle',
] as const;

export type WorldEffectActionName = (typeof WORLD_EFFECT_ACTIONS)[number];

export function isWorldEffectActionName(name: string): name is WorldEffectActionName {
  return (WORLD_EFFECT_ACTIONS as readonly string[]).includes(name);
}
