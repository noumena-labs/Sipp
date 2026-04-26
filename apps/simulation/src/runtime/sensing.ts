import type {
  AgentPerception,
  PerceivedAgent,
  PerceivedObject,
  SimulationAgentState,
  SimulationGameState,
  SimulationObjectState,
  Vec2,
  WorldBounds,
} from './types.js';

export interface SensingOptions {
  readonly agentSightRadius?: number;
  readonly objectSightRadius?: number;
  readonly maxNeighbours?: number;
}

export function buildPerception(
  self: SimulationAgentState,
  agents: readonly SimulationAgentState[],
  objects: readonly SimulationObjectState[],
  tick: number,
  bounds: WorldBounds,
  directorNote: string | null,
  game: SimulationGameState,
  options: SensingOptions = {}
): AgentPerception {
  const agentRadius = options.agentSightRadius ?? 8;
  const objectRadius = options.objectSightRadius ?? 8;
  const maxNeighbours = options.maxNeighbours ?? 6;
  const mustSeeAgentIds = new Set<string>();
  const mustSeeObjectIds = new Set<string>([game.bananaObjectId, game.goalObjectId]);

  for (const other of agents) {
    if (other.holding === game.bananaObjectId) {
      mustSeeAgentIds.add(other.id);
    }
  }

  const nearbyAgents: PerceivedAgent[] = [];
  for (const other of agents) {
    if (other.id === self.id) continue;
    const distance = vec2Distance(self.position, other.position);
    if (distance > agentRadius && !mustSeeAgentIds.has(other.id)) continue;
    nearbyAgents.push({
      id: other.id,
      name: other.name,
      distance,
      direction: vec2Direction(self.position, other.position),
      emotion: other.emotion,
      status: other.status,
      holding: other.holding,
      powerUp: other.powerUp?.kind ?? null,
      frozenUntilTick: other.frozenUntilTick,
    });
  }
  nearbyAgents.sort((a, b) => compareByPriority(a.distance, b.distance, mustSeeAgentIds.has(a.id), mustSeeAgentIds.has(b.id)));
  trimToVisibleCount(nearbyAgents, maxNeighbours, (entry) => mustSeeAgentIds.has(entry.id));

  const nearbyObjects: PerceivedObject[] = [];
  for (const obj of objects) {
    if (!obj.active) continue;
    const distance = vec2Distance(self.position, obj.position);
    if (distance > objectRadius && !mustSeeObjectIds.has(obj.id)) continue;
    nearbyObjects.push({
      id: obj.id,
      kind: obj.kind,
      label: obj.label,
      description: obj.description,
      distance,
      direction: vec2Direction(self.position, obj.position),
      active: obj.active,
      heldBy: obj.heldBy,
      contested: obj.contested,
      affordances: obj.affordances,
      tags: obj.tags,
      blocksMovement: obj.blocksMovement,
      collisionRadius: obj.collisionRadius,
    });
  }
  nearbyObjects.sort((a, b) => compareByPriority(a.distance, b.distance, mustSeeObjectIds.has(a.id), mustSeeObjectIds.has(b.id)));
  trimToVisibleCount(nearbyObjects, maxNeighbours, (entry) => mustSeeObjectIds.has(entry.id));

  return { self, nearbyAgents, nearbyObjects, tick, bounds, directorNote, game };
}

function compareByPriority(aDistance: number, bDistance: number, aPinned: boolean, bPinned: boolean): number {
  if (aPinned !== bPinned) {
    return aPinned ? -1 : 1;
  }
  return aDistance - bDistance;
}

function trimToVisibleCount<T>(entries: T[], maxNeighbours: number, keep: (entry: T) => boolean): void {
  if (entries.length <= maxNeighbours) return;
  const pinned = entries.filter(keep);
  const rest = entries.filter((entry) => !keep(entry));
  entries.length = 0;
  for (const entry of pinned) {
    entries.push(entry);
  }
  for (const entry of rest) {
    if (entries.length >= maxNeighbours) break;
    entries.push(entry);
  }
}

export function vec2Distance(a: Vec2, b: Vec2): number {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dz * dz);
}

export function vec2Direction(from: Vec2, to: Vec2): Vec2 {
  const dx = to.x - from.x;
  const dz = to.z - from.z;
  const len = Math.sqrt(dx * dx + dz * dz);
  if (len < 1e-6) return { x: 0, z: 0 };
  return { x: dx / len, z: dz / len };
}

export function clampToBounds(position: Vec2, bounds: WorldBounds): Vec2 {
  const limit = bounds.halfExtent;
  return {
    x: Math.max(-limit, Math.min(limit, position.x)),
    z: Math.max(-limit, Math.min(limit, position.z)),
  };
}
