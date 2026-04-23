//////////////////////////////////////////////////////////////////////////////
//
// scenarios/courtyard-snack.ts
//
// - Hand-authored Banana Dash demo: 4 character agents race to carry the
//   banana to a shared goal while bumping and contesting each other.
//
//////////////////////////////////////////////////////////////////////////////

import type { ScenarioSeed } from '../runtime/types.js';

export interface ScenarioAgentAssignment {
  readonly agentId: string;
  readonly name: string;
  readonly archetype: string;
  readonly characterUrl: string;
  readonly color: string;
  readonly position: { x: number; z: number };
  readonly status: string;
}

export const COURTYARD_AGENTS: readonly ScenarioAgentAssignment[] = [
  {
    agentId: 'aria',
    name: 'Aria',
    archetype: 'aria',
    characterUrl: '/characters/aria/character.json',
    color: '#f4a261',
    position: { x: -5, z: -4 },
    status: 'eyeing the banana lane',
  },
  {
    agentId: 'beck',
    name: 'Beck',
    archetype: 'beck',
    characterUrl: '/characters/beck/character.json',
    color: '#2a9d8f',
    position: { x: 5, z: -4 },
    status: 'ready to sprint',
  },
  {
    agentId: 'mira',
    name: 'Mira',
    archetype: 'mira',
    characterUrl: '/characters/mira/character.json',
    color: '#e76f51',
    position: { x: -5, z: 4 },
    status: 'plotting a playful steal',
  },
  {
    agentId: 'sol',
    name: 'Sol',
    archetype: 'sol',
    characterUrl: '/characters/sol/character.json',
    color: '#8ab0ff',
    position: { x: 5, z: 4 },
    status: 'waiting for a clean opening',
  },
];

export const COURTYARD_SCENARIO: ScenarioSeed = {
  id: 'banana-dash',
  title: 'Banana Dash',
  bounds: { halfExtent: 8 },
  agents: COURTYARD_AGENTS.map((a) => ({
    id: a.agentId,
    name: a.name,
    archetype: a.archetype,
    position: a.position,
    status: a.status,
    speed: 1.35,
  })),
  objects: [
    {
      id: 'banana',
      kind: 'banana',
      label: 'banana',
      position: { x: 0, z: 0 },
      contested: true,
      collisionRadius: 0.2,
      tags: ['food', 'score'],
      affordances: [{ kind: 'pick_up', label: 'grab banana', status: 'grabbing the banana' }],
    },
    {
      id: 'home-base',
      kind: 'goal',
      label: 'home base',
      position: { x: 0, z: -6 },
      collisionRadius: 1.2,
      tags: ['goal'],
      affordances: [],
    },
    {
      id: 'crate-a',
      kind: 'crate',
      label: 'left crate',
      position: { x: -3, z: -1 },
      blocksMovement: true,
      collisionRadius: 0.65,
      tags: ['obstacle'],
      affordances: [],
    },
    {
      id: 'crate-b',
      kind: 'crate',
      label: 'right crate',
      position: { x: 3, z: 1 },
      blocksMovement: true,
      collisionRadius: 0.65,
      tags: ['obstacle'],
      affordances: [],
    },
    {
      id: 'rock-a',
      kind: 'rock',
      label: 'north rock',
      position: { x: -2, z: 3 },
      blocksMovement: true,
      collisionRadius: 0.55,
      tags: ['obstacle'],
      affordances: [],
    },
    {
      id: 'rock-b',
      kind: 'rock',
      label: 'south rock',
      position: { x: 2, z: -3 },
      blocksMovement: true,
      collisionRadius: 0.55,
      tags: ['obstacle'],
      affordances: [],
    },
  ],
  game: {
    title: 'Banana Dash',
    bananaObjectId: 'banana',
    goalObjectId: 'home-base',
    bananaSpawnPoints: [
      { x: 0, z: 0 },
      { x: -6, z: 1 },
      { x: 6, z: -1 },
      { x: -1, z: 5 },
      { x: 1, z: -5 },
      { x: 5, z: 5 },
      { x: -5, z: -5 },
    ],
  },
  directorNote: 'Banana Dash begins: score by carrying the banana to home base.',
  directorConfigUrl: '/directors/courtyard/director.json',
  directorCadenceTicks: 12,
  resolveRefereeQuery: 'resolve_referee_event',
  narrateQuery: 'narrate_scene',
};

export const AGENT_COLOR_BY_ID: ReadonlyMap<string, string> = new Map(
  COURTYARD_AGENTS.map((a) => [a.agentId, a.color])
);
