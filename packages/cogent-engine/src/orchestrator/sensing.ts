//////////////////////////////////////////////////////////////////////////////
//
// sensing.ts
//
// - Pure functions that derive an AgentPerception from the current world
//   state. Kept separate from the reducer so it is trivially testable.
//
//////////////////////////////////////////////////////////////////////////////

import type {
  AgentPerception,
  PerceivedAgent,
  PerceivedObject,
  SimulationAgentState,
  SimulationObjectState,
  Vec2,
  WorldBounds,
} from './simulation-types.js';

export interface SensingOptions {
  /** Max distance an agent perceives other agents. Default = 8. */
  readonly agentSightRadius?: number;
  /** Max distance an agent perceives objects. Default = 8. */
  readonly objectSightRadius?: number;
  /** Max number of neighbours returned per category. Default = 6. */
  readonly maxNeighbours?: number;
}

export function buildPerception(
  self: SimulationAgentState,
  agents: readonly SimulationAgentState[],
  objects: readonly SimulationObjectState[],
  tick: number,
  bounds: WorldBounds,
  directorNote: string | null,
  options: SensingOptions = {}
): AgentPerception {
  const agentRadius = options.agentSightRadius ?? 8;
  const objectRadius = options.objectSightRadius ?? 8;
  const maxNeighbours = options.maxNeighbours ?? 6;

  const nearbyAgents: PerceivedAgent[] = [];
  for (const other of agents) {
    if (other.id === self.id) continue;
    const distance = vec2Distance(self.position, other.position);
    if (distance > agentRadius) continue;
    nearbyAgents.push({
      id: other.id,
      name: other.name,
      distance,
      direction: vec2Direction(self.position, other.position),
      emotion: other.emotion,
      status: other.status,
      holding: other.holding,
    });
  }
  nearbyAgents.sort((a, b) => a.distance - b.distance);
  nearbyAgents.length = Math.min(nearbyAgents.length, maxNeighbours);

  const nearbyObjects: PerceivedObject[] = [];
  for (const obj of objects) {
    const distance = vec2Distance(self.position, obj.position);
    if (distance > objectRadius) continue;
    nearbyObjects.push({
      id: obj.id,
      kind: obj.kind,
      distance,
      direction: vec2Direction(self.position, obj.position),
      heldBy: obj.heldBy,
      contested: obj.contested,
    });
  }
  nearbyObjects.sort((a, b) => a.distance - b.distance);
  nearbyObjects.length = Math.min(nearbyObjects.length, maxNeighbours);

  return { self, nearbyAgents, nearbyObjects, tick, bounds, directorNote };
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
