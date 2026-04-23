//////////////////////////////////////////////////////////////////////////////
//
// scenarios/courtyard-snack.ts
//
// - Hand-authored demo scenario: 4 agents meeting in a small courtyard
//   that has a single banana (contested), plus a bench, fountain, and
//   two potted plants as ambient objects.
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
    position: { x: -3, z: -2 },
    status: 'just arrived at the courtyard',
  },
  {
    agentId: 'beck',
    name: 'Beck',
    archetype: 'beck',
    characterUrl: '/characters/beck/character.json',
    color: '#2a9d8f',
    position: { x: 3, z: -2 },
    status: 'looking for something to eat',
  },
  {
    agentId: 'mira',
    name: 'Mira',
    archetype: 'mira',
    characterUrl: '/characters/mira/character.json',
    color: '#e76f51',
    position: { x: -3, z: 3 },
    status: 'watching the others curiously',
  },
  {
    agentId: 'sol',
    name: 'Sol',
    archetype: 'sol',
    characterUrl: '/characters/sol/character.json',
    color: '#8ab0ff',
    position: { x: 3, z: 3 },
    status: 'daydreaming by the fountain',
  },
];

export const COURTYARD_SCENARIO: ScenarioSeed = {
  id: 'courtyard-snack',
  title: 'Courtyard Snack',
  bounds: { halfExtent: 8 },
  agents: COURTYARD_AGENTS.map((a) => ({
    id: a.agentId,
    name: a.name,
    archetype: a.archetype,
    position: a.position,
    status: a.status,
    speed: 1.2,
  })),
  objects: [
    {
      id: 'banana',
      kind: 'banana',
      label: 'banana',
      position: { x: 0, z: 0 },
      contested: true,
      tags: ['food'],
      affordances: [{ kind: 'pick_up', label: 'pick up the banana', status: 'reaching for the banana' }],
    },
    {
      id: 'bench',
      kind: 'bench',
      label: 'bench',
      position: { x: 0, z: -5 },
      tags: ['seat'],
      affordances: [{ kind: 'use', label: 'sit on the bench', status: 'settling onto the bench' }],
    },
    {
      id: 'fountain',
      kind: 'fountain',
      label: 'fountain',
      position: { x: 0, z: 5 },
      tags: ['water'],
      affordances: [{ kind: 'use', label: 'linger by the fountain', status: 'lingering by the fountain' }],
    },
    {
      id: 'plant-a',
      kind: 'plant',
      label: 'left plant',
      position: { x: -5, z: 0 },
      tags: ['decor'],
      affordances: [{ kind: 'use', label: 'inspect the left plant', status: 'inspecting the left plant' }],
    },
    {
      id: 'plant-b',
      kind: 'plant',
      label: 'right plant',
      position: { x: 5, z: 0 },
      tags: ['decor'],
      affordances: [{ kind: 'use', label: 'inspect the right plant', status: 'inspecting the right plant' }],
    },
  ],
  directorNote: 'A single yellow banana sits between them on the courtyard tile.',
  directorConfigUrl: '/directors/courtyard/director.json',
  directorCadenceTicks: 10,
  resolveConflictQuery: 'resolve_pickup_conflict',
  narrateQuery: 'narrate_scene',
};

export const AGENT_COLOR_BY_ID: ReadonlyMap<string, string> = new Map(
  COURTYARD_AGENTS.map((a) => [a.agentId, a.color])
);
