//////////////////////////////////////////////////////////////////////////////
//
// scenarios/courtyard-snack.ts
//
// - Banana Dash demo: 4 fixed character agents race to carry the banana to a
//   shared fixed goal while the courtyard layout is generated per loaded run.
//
//////////////////////////////////////////////////////////////////////////////

import type { ScenarioObjectSeed, ScenarioSeed, Vec2 } from '../runtime/types.js';

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

const COURTYARD_BOUNDS = { halfExtent: 8 } as const;
const BANANA_POSITION = { x: 0, z: 0 } as const;
const HOME_BASE_POSITION = { x: 0, z: -6 } as const;
const MIN_FREE_COMPONENT_SIZE = 120;
const GENERATION_ATTEMPTS = 40;
const OBSTACLE_GROUP_COUNT_RANGE = { min: 3, max: 6 } as const;
const GROUP_LENGTH_RANGE = { min: 2, max: 5 } as const;
const DEFAULT_COUNT_OPTIONS = {
  obstacles: { enabled: true, target: 12 },
  bats: { enabled: true, target: 1 },
  iceCubes: { enabled: true, target: 1 },
} as const;

export interface CourtyardScenarioOptions {
  readonly seed?: number;
  readonly obstacles?: CountOption;
  readonly bats?: CountOption;
  readonly iceCubes?: CountOption;
}

export interface CountOption {
  readonly enabled: boolean;
  readonly target: number;
}

interface CandidateObject {
  readonly kind: 'crate' | 'rock';
  readonly position: Vec2;
}

interface LayoutDraft {
  readonly obstacles: readonly ScenarioObjectSeed[];
  readonly batPositions: readonly Vec2[];
  readonly icePositions: readonly Vec2[];
  readonly bananaRespawns: readonly Vec2[];
  readonly batRespawns: readonly (readonly Vec2[])[];
  readonly iceRespawns: readonly (readonly Vec2[])[];
}

interface ReservedZone {
  readonly center: Vec2;
  readonly radius: number;
}

export const COURTYARD_SCENARIO: ScenarioSeed = createCourtyardScenario({ seed: 1 });

export function createCourtyardScenario(options: CourtyardScenarioOptions = {}): ScenarioSeed {
  const seed = options.seed ?? createScenarioSeed();
  const counts = resolveScenarioCounts(seed, options);
  const layout = generateCourtyardLayout(seed, counts);
  return {
    id: 'banana-dash',
    title: 'Banana Dash',
    bounds: COURTYARD_BOUNDS,
    agents: COURTYARD_AGENTS.map((a) => ({
      id: a.agentId,
      name: a.name,
      archetype: a.archetype,
      position: a.position,
      status: a.status,
      speed: 1.35,
    })),
    objects: [
      createBananaObject(),
      createHomeBaseObject(),
      ...layout.batPositions.map((position, index) => createBatObject(index, position)),
      ...layout.icePositions.map((position, index) => createIceCubeObject(index, position)),
      ...layout.obstacles,
    ],
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home-base',
      respawnRules: [
        {
          objectId: 'banana',
          delayTicks: 1,
          spawnPoints: layout.bananaRespawns,
        },
        ...layout.batRespawns.map((spawnPoints, index) => ({
          objectId: powerUpId('bat-power-up', index),
          delayTicks: 12,
          spawnPoints,
        })),
        ...layout.iceRespawns.map((spawnPoints, index) => ({
          objectId: powerUpId('ice-power-up', index),
          delayTicks: 12,
          spawnPoints,
        })),
      ],
    },
    directorNote: 'Banana Dash begins: score by carrying the banana to home base, or peel off for slapstick power-ups on the wings.',
    directorConfigUrl: '/directors/courtyard/director.json',
    directorCadenceTicks: 12,
    resolveRefereeTask: 'resolve_referee_event',
    narrateTask: 'narrate_scene',
    refereeTimeoutMs: 30000,
    agentQueryTimeoutMs: 30000,
    narrationTimeoutMs: 15000,
    maxMoveTicksBeforeReevaluation: 15,
  };
}

export const AGENT_COLOR_BY_ID: ReadonlyMap<string, string> = new Map(
  COURTYARD_AGENTS.map((a) => [a.agentId, a.color])
);

function createBananaObject(): ScenarioObjectSeed {
  return {
    id: 'banana',
    kind: 'banana',
    label: 'banana',
    description: 'The bright yellow banana every agent is scrambling to carry home.',
    position: BANANA_POSITION,
    contested: true,
    collisionRadius: 0.2,
    tags: ['food', 'score'],
    affordances: [{ kind: 'pick_up', label: 'grab banana', status: 'grabbing the banana' }],
  };
}

function createHomeBaseObject(): ScenarioObjectSeed {
  return {
    id: 'home-base',
    kind: 'goal',
    label: 'home base',
    description: 'The shared scoring circle where a carrier cashes in the banana.',
    position: HOME_BASE_POSITION,
    collisionRadius: 1.2,
    tags: ['goal'],
    affordances: [],
  };
}

function createBatObject(index: number, position: Vec2): ScenarioObjectSeed {
  return {
    id: powerUpId('bat-power-up', index),
    kind: 'bat',
    label: 'baseball bat',
    description: 'A rubbery cartoon bat. One close smack guarantees a bonk and pops the banana loose.',
    position,
    contested: true,
    collisionRadius: 0.24,
    tags: ['power_up', 'weapon'],
    affordances: [{ kind: 'pick_up', label: 'grab baseball bat', status: 'reaching for the baseball bat' }],
  };
}

function createIceCubeObject(index: number, position: Vec2): ScenarioObjectSeed {
  return {
    id: powerUpId('ice-power-up', index),
    kind: 'ice_cube',
    label: 'ice cube',
    description: 'A chunky cartoon ice cube. One toss freezes a runner long enough to spill the banana.',
    position,
    contested: true,
    collisionRadius: 0.22,
    tags: ['power_up', 'weapon'],
    affordances: [{ kind: 'pick_up', label: 'grab ice cube', status: 'reaching for the ice cube' }],
  };
}

function powerUpId(baseId: string, index: number): string {
  return index === 0 ? baseId : `${baseId}-${index + 1}`;
}

function createScenarioSeed(): number {
  return Math.floor(Math.random() * 0xffffffff) >>> 0;
}

interface ResolvedScenarioCounts {
  readonly obstacles: number;
  readonly bats: number;
  readonly iceCubes: number;
}

function resolveScenarioCounts(seed: number, options: CourtyardScenarioOptions): ResolvedScenarioCounts {
  const rng = createRng(seed ^ 0xa53a9f31);
  const obstacles = options.obstacles ?? DEFAULT_COUNT_OPTIONS.obstacles;
  const bats = options.bats ?? DEFAULT_COUNT_OPTIONS.bats;
  const iceCubes = options.iceCubes ?? DEFAULT_COUNT_OPTIONS.iceCubes;
  return {
    obstacles: obstacles.enabled ? randomInt(rng, 1, obstacles.target) : 0,
    bats: bats.enabled ? randomInt(rng, 1, bats.target) : 0,
    iceCubes: iceCubes.enabled ? randomInt(rng, 1, iceCubes.target) : 0,
  };
}

function generateCourtyardLayout(seed: number, counts: ResolvedScenarioCounts): LayoutDraft {
  for (let attempt = 0; attempt < GENERATION_ATTEMPTS; attempt += 1) {
    const rng = createRng(seed + attempt * 0x9e3779b9);
    const obstacles = createObstacleSeeds(generateObstacleCandidates(rng, counts.obstacles));
    const reserved = createReservedZones();
    const freeCells = enumerateFreeCells(obstacles, reserved);
    const reachableCells = reachableFrom(BANANA_POSITION, freeCells);
    if (reachableCells.length < MIN_FREE_COMPONENT_SIZE) continue;
    if (!isReachable(HOME_BASE_POSITION, reachableCells)) continue;
    if (!COURTYARD_AGENTS.every((agent) => isReachable(agent.position, reachableCells))) continue;

    const powerUpAvoidance = [
      ...reserved,
      { center: BANANA_POSITION, radius: 3.2 },
      { center: HOME_BASE_POSITION, radius: 2.5 },
    ];
    const batPositions = pickPowerUpPositions(rng, reachableCells, counts.bats, powerUpAvoidance, []);
    const icePositions = pickPowerUpPositions(rng, reachableCells, counts.iceCubes, powerUpAvoidance, batPositions);
    const occupied = [
      ...reserved,
      ...batPositions.map((center) => ({ center, radius: 1.5 })),
      ...icePositions.map((center) => ({ center, radius: 1.5 })),
    ];

    return {
      obstacles,
      batPositions,
      icePositions,
      bananaRespawns: [BANANA_POSITION, ...pickRespawns(rng, reachableCells, 6, occupied)],
      batRespawns: batPositions.map((position) => [
        position,
        ...pickRespawns(rng, reachableCells, 3, occupied.filter((zone) => zone.center !== position)),
      ]),
      iceRespawns: icePositions.map((position) => [
        position,
        ...pickRespawns(rng, reachableCells, 3, occupied.filter((zone) => zone.center !== position)),
      ]),
    };
  }

  return createFallbackLayout(counts);
}

function generateObstacleCandidates(rng: Rng, count: number): CandidateObject[] {
  const occupied = new Set<string>();
  const obstacles: CandidateObject[] = [];
  const reserved = createReservedZones();
  const directions = [
    { x: 1, z: 0 },
    { x: 0, z: 1 },
    { x: 1, z: 1 },
    { x: 1, z: -1 },
  ] as const;

  for (let group = 0; obstacles.length < count && group < count * OBSTACLE_GROUP_COUNT_RANGE.max; group += 1) {
    const start = randomCell(rng, 6);
    const direction = directions[randomInt(rng, 0, directions.length - 1)]!;
    const length = randomInt(rng, GROUP_LENGTH_RANGE.min, GROUP_LENGTH_RANGE.max);
    const bendAt = randomInt(rng, 1, Math.max(1, length - 1));
    const bend = rng() < 0.45 ? { x: -direction.z, z: direction.x } : direction;

    for (let i = 0; i < length; i += 1) {
      const step = i < bendAt ? direction : bend;
      const jitter = rng() < 0.3 ? randomUnitOffset(rng) : { x: 0, z: 0 };
      const position = {
        x: clampCell(start.x + step.x * i + jitter.x),
        z: clampCell(start.z + step.z * i + jitter.z),
      };
      if (isReservedPosition(position, reserved)) continue;
      const key = cellKey(position);
      if (occupied.has(key)) continue;
      occupied.add(key);
      obstacles.push({
        kind: rng() < 0.55 ? 'crate' : 'rock',
        position,
      });
      if (obstacles.length >= count) break;
    }
  }

  if (obstacles.length < count) {
    const limit = COURTYARD_BOUNDS.halfExtent - 1;
    const fillCells: Vec2[] = [];
    for (let x = -limit; x <= limit; x += 1) {
      for (let z = -limit; z <= limit; z += 1) {
        const position = { x, z };
        if (isReservedPosition(position, reserved)) continue;
        if (occupied.has(cellKey(position))) continue;
        fillCells.push(position);
      }
    }
    for (const position of shuffle(fillCells, rng)) {
      occupied.add(cellKey(position));
      obstacles.push({
        kind: rng() < 0.55 ? 'crate' : 'rock',
        position,
      });
      if (obstacles.length >= count) break;
    }
  }

  return obstacles;
}

function createObstacleSeeds(candidates: readonly CandidateObject[]): ScenarioObjectSeed[] {
  return candidates.map((candidate, index) => ({
    id: `${candidate.kind}-${index + 1}`,
    kind: candidate.kind,
    label: candidate.kind === 'crate' ? `supply crate ${index + 1}` : `boulder ${index + 1}`,
    description: candidate.kind === 'crate'
      ? 'A heavy supply crate that blocks and redirects a courtyard lane.'
      : 'A low boulder that creates a rough edge in the running lane.',
    position: candidate.position,
    blocksMovement: true,
    collisionRadius: candidate.kind === 'crate' ? 0.65 : 0.55,
    tags: ['obstacle'],
    affordances: [],
  }));
}

function createFallbackLayout(counts: ResolvedScenarioCounts): LayoutDraft {
  const obstacles = createObstacleSeeds([
    { kind: 'crate', position: { x: -3, z: -1 } },
    { kind: 'crate', position: { x: 3, z: 1 } },
    { kind: 'rock', position: { x: -2, z: 3 } },
    { kind: 'rock', position: { x: 2, z: -3 } },
    { kind: 'crate', position: { x: -5, z: 0 } },
    { kind: 'rock', position: { x: 5, z: 0 } },
    { kind: 'crate', position: { x: -1, z: -3 } },
    { kind: 'rock', position: { x: 1, z: 3 } },
    { kind: 'crate', position: { x: -6, z: -2 } },
    { kind: 'rock', position: { x: 6, z: -2 } },
    { kind: 'crate', position: { x: -4, z: 3 } },
    { kind: 'rock', position: { x: 4, z: 3 } },
    { kind: 'crate', position: { x: -1, z: 6 } },
    { kind: 'rock', position: { x: 1, z: 6 } },
    { kind: 'crate', position: { x: -6, z: 5 } },
    { kind: 'rock', position: { x: 6, z: 5 } },
    { kind: 'crate', position: { x: -4, z: -6 } },
    { kind: 'rock', position: { x: 4, z: -6 } },
    { kind: 'crate', position: { x: 0, z: 2 } },
    { kind: 'rock', position: { x: 0, z: -2 } },
  ].slice(0, counts.obstacles));
  const batPositions = [
    { x: -6, z: 2 },
    { x: -5.5, z: 5 },
    { x: -6.2, z: -1.5 },
    { x: -3, z: 5.5 },
    { x: -5.5, z: -5 },
  ].slice(0, counts.bats);
  const icePositions = [
    { x: 6, z: 2 },
    { x: 5.4, z: 5 },
    { x: 6.1, z: -1.8 },
    { x: 3, z: 5.5 },
    { x: 5.5, z: -5 },
  ].slice(0, counts.iceCubes);
  return {
    obstacles,
    batPositions,
    icePositions,
    bananaRespawns: [
      BANANA_POSITION,
      { x: -6, z: 1 },
      { x: 6, z: -1 },
      { x: -1, z: 5 },
      { x: 1, z: -5 },
      { x: 5, z: 5 },
      { x: -5, z: -5 },
    ],
    batRespawns: batPositions.map((position) => [position, { x: -5.5, z: 5 }, { x: -6.2, z: -1.5 }]),
    iceRespawns: icePositions.map((position) => [position, { x: 5.4, z: 5 }, { x: 6.1, z: -1.8 }]),
  };
}

function createReservedZones(): ReservedZone[] {
  return [
    { center: BANANA_POSITION, radius: 2.4 },
    { center: HOME_BASE_POSITION, radius: 2.25 },
    ...COURTYARD_AGENTS.map((agent) => ({ center: agent.position, radius: 2.1 })),
  ];
}

function enumerateFreeCells(obstacles: readonly ScenarioObjectSeed[], reserved: readonly ReservedZone[]): Vec2[] {
  const cells: Vec2[] = [];
  const limit = COURTYARD_BOUNDS.halfExtent - 1;
  for (let x = -limit; x <= limit; x += 1) {
    for (let z = -limit; z <= limit; z += 1) {
      const cell = { x, z };
      if (isBlockedByObstacle(cell, obstacles)) continue;
      if (reserved.some((zone) => distance(cell, zone.center) < Math.min(1.15, zone.radius * 0.5))) continue;
      cells.push(cell);
    }
  }
  return cells;
}

function reachableFrom(start: Vec2, freeCells: readonly Vec2[]): Vec2[] {
  const free = new Set(freeCells.map(cellKey));
  const startCell = nearestFreeCell(start, freeCells);
  if (!startCell) return [];
  const visited = new Set<string>([cellKey(startCell)]);
  const queue = [startCell];
  for (let i = 0; i < queue.length; i += 1) {
    const current = queue[i]!;
    for (const next of neighbours(current)) {
      const key = cellKey(next);
      if (!free.has(key) || visited.has(key)) continue;
      visited.add(key);
      queue.push(next);
    }
  }
  return queue;
}

function isReachable(position: Vec2, reachableCells: readonly Vec2[]): boolean {
  const cell = nearestFreeCell(position, reachableCells);
  return cell != null && distance(position, cell) <= 1.5;
}

function pickFreeCell(rng: Rng, cells: readonly Vec2[], avoid: readonly ReservedZone[]): Vec2 {
  const candidates = cells.filter((cell) => !isReservedPosition(cell, avoid));
  return cloneVec(candidates[randomInt(rng, 0, candidates.length - 1)] ?? cells[randomInt(rng, 0, cells.length - 1)] ?? BANANA_POSITION);
}

function pickPowerUpPositions(
  rng: Rng,
  cells: readonly Vec2[],
  count: number,
  baseAvoidance: readonly ReservedZone[],
  otherPowerUps: readonly Vec2[]
): Vec2[] {
  const selected: Vec2[] = [];
  for (let i = 0; i < count; i += 1) {
    const position = pickFreeCell(rng, cells, [
      ...baseAvoidance,
      ...otherPowerUps.map((center) => ({ center, radius: 3 })),
      ...selected.map((center) => ({ center, radius: 3 })),
    ]);
    selected.push(position);
  }
  return selected;
}

function pickRespawns(rng: Rng, cells: readonly Vec2[], count: number, avoid: readonly ReservedZone[]): Vec2[] {
  const selected: Vec2[] = [];
  const candidates = shuffle(cells.filter((cell) => !isReservedPosition(cell, avoid)), rng);
  for (const candidate of candidates) {
    if (selected.some((point) => distance(point, candidate) < 2.25)) continue;
    selected.push(cloneVec(candidate));
    if (selected.length >= count) break;
  }
  return selected;
}

function isBlockedByObstacle(position: Vec2, obstacles: readonly ScenarioObjectSeed[]): boolean {
  return obstacles.some((obstacle) => distance(position, obstacle.position) < (obstacle.collisionRadius ?? 0.55) + 0.9);
}

function isReservedPosition(position: Vec2, reserved: readonly ReservedZone[]): boolean {
  return reserved.some((zone) => distance(position, zone.center) < zone.radius);
}

function neighbours(position: Vec2): Vec2[] {
  return [
    { x: position.x + 1, z: position.z },
    { x: position.x - 1, z: position.z },
    { x: position.x, z: position.z + 1 },
    { x: position.x, z: position.z - 1 },
  ];
}

function nearestFreeCell(position: Vec2, cells: readonly Vec2[]): Vec2 | null {
  let best: { cell: Vec2; distance: number } | null = null;
  for (const cell of cells) {
    const d = distance(position, cell);
    if (best == null || d < best.distance) {
      best = { cell, distance: d };
    }
  }
  return best?.cell ?? null;
}

function randomCell(rng: Rng, extent: number): Vec2 {
  return {
    x: randomInt(rng, -extent, extent),
    z: randomInt(rng, -extent, extent),
  };
}

function randomUnitOffset(rng: Rng): Vec2 {
  const offsets = [-1, 0, 1];
  return {
    x: offsets[randomInt(rng, 0, offsets.length - 1)]!,
    z: offsets[randomInt(rng, 0, offsets.length - 1)]!,
  };
}

function clampCell(value: number): number {
  const limit = COURTYARD_BOUNDS.halfExtent - 1;
  return Math.max(-limit, Math.min(limit, value));
}

function cellKey(position: Vec2): string {
  return `${Math.round(position.x)},${Math.round(position.z)}`;
}

function distance(a: Vec2, b: Vec2): number {
  return Math.hypot(a.x - b.x, a.z - b.z);
}

function cloneVec(position: Vec2): Vec2 {
  return { x: position.x, z: position.z };
}

function shuffle<T>(items: readonly T[], rng: Rng): T[] {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = randomInt(rng, 0, i);
    const tmp = result[i]!;
    result[i] = result[j]!;
    result[j] = tmp;
  }
  return result;
}

function randomInt(rng: Rng, min: number, max: number): number {
  return Math.floor(rng() * (max - min + 1)) + min;
}

type Rng = () => number;

function createRng(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
