//////////////////////////////////////////////////////////////////////////////
//
// world-reducer.ts
//
// - Pure, synchronous state transitions. The reducer is authoritative: the
//   LLM only *suggests* intents; this code decides what actually happens.
//
// Responsibilities each tick:
//   1. Advance time.
//   2. Apply each agent's active intent (movement, approach, drop).
//   3. Detect contested pick_up / use requests (multiple agents wanting the
//      same object on the same tick) and surface them as WorldConflict[].
//   4. Apply uncontested pick_up / use requests.
//   5. Apply director resolutions after the conflict query (second pass).
//
//////////////////////////////////////////////////////////////////////////////

import type {
  AgentIntent,
  DirectorDecision,
  SimulationAgentState,
  SimulationObjectState,
  Vec2,
  WorldBounds,
  WorldConflict,
} from './simulation-types.js';
import { clampToBounds, vec2Direction, vec2Distance } from './sensing.js';

/** Mutable, orchestrator-owned world state. Never leaked directly. */
export interface MutableWorldState {
  tick: number;
  timeSeconds: number;
  bounds: WorldBounds;
  agents: SimulationAgentState[];
  objects: SimulationObjectState[];
  directorNote: string | null;
}

/** Distance within which an agent can pick_up / use / drop / interact. */
export const INTERACTION_RADIUS = 0.75;

/**
 * Advance continuous movement for one time step. Pure: mutates nothing,
 * returns the new agent position and heading.
 */
export function stepMovement(
  agent: SimulationAgentState,
  objects: readonly SimulationObjectState[],
  agents: readonly SimulationAgentState[],
  dt: number,
  bounds: WorldBounds
): { position: Vec2; heading: number } {
  const intent = agent.intent;
  if (!intent) {
    return { position: agent.position, heading: agent.heading };
  }
  const target = resolveMovementTarget(intent, agent, objects, agents);
  if (!target) {
    return { position: agent.position, heading: agent.heading };
  }
  const dx = target.x - agent.position.x;
  const dz = target.z - agent.position.z;
  const dist = Math.sqrt(dx * dx + dz * dz);
  if (dist < 1e-4) {
    return { position: agent.position, heading: agent.heading };
  }
  const step = Math.min(dist, agent.speed * dt);
  const nx = agent.position.x + (dx / dist) * step;
  const nz = agent.position.z + (dz / dist) * step;
  const heading = Math.atan2(dx, dz);
  return { position: clampToBounds({ x: nx, z: nz }, bounds), heading };
}

function resolveMovementTarget(
  intent: AgentIntent,
  agent: SimulationAgentState,
  objects: readonly SimulationObjectState[],
  agents: readonly SimulationAgentState[]
): Vec2 | null {
  switch (intent.kind) {
    case 'move_to':
      return intent.target;
    case 'approach_agent': {
      const target = agents.find((a) => a.id === intent.agentId);
      return target ? target.position : null;
    }
    case 'pick_up':
    case 'use': {
      const target = objects.find((o) => o.id === intent.objectId);
      return target ? target.position : null;
    }
    case 'wander': {
      // Cheap deterministic wander: pick a point offset from current pos
      // that loops based on agent id hash + tick-independent heading.
      const offset = hashVec2(agent.id);
      return {
        x: agent.position.x + offset.x,
        z: agent.position.z + offset.z,
      };
    }
    case 'wait':
    case 'drop':
    default:
      return null;
  }
}

function hashVec2(id: string): Vec2 {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < id.length; i += 1) {
    h = Math.imul(h ^ id.charCodeAt(i), 16777619) >>> 0;
  }
  const angle = (h / 0xffffffff) * Math.PI * 2;
  return { x: Math.cos(angle) * 1.5, z: Math.sin(angle) * 1.5 };
}

export interface TickReducerResult {
  readonly conflicts: WorldConflict[];
}

/**
 * First pass: integrate movement, handle `drop`, and resolve uncontested
 * `pick_up` / `use`. Returns any conflicts the director must resolve next.
 */
export function applyTickFirstPass(state: MutableWorldState, dt: number): TickReducerResult {
  // Movement integration.
  for (const agent of state.agents) {
    const next = stepMovement(agent, state.objects, state.agents, dt, state.bounds);
    agent.position = next.position;
    agent.heading = next.heading;
    if (agent.intent?.emotion) {
      agent.emotion = agent.intent.emotion;
    }
  }

  // Move carried objects with their carrier.
  for (const obj of state.objects) {
    if (obj.heldBy) {
      const carrier = state.agents.find((a) => a.id === obj.heldBy);
      if (carrier) {
        obj.position = { x: carrier.position.x, z: carrier.position.z };
      }
    }
  }

  // Handle drops.
  for (const agent of state.agents) {
    if (agent.intent?.kind !== 'drop') continue;
    if (!agent.holding) {
      agent.intent = null;
      continue;
    }
    const held = state.objects.find((o) => o.id === agent.holding);
    if (held) {
      held.heldBy = null;
    }
    agent.holding = null;
    agent.intent = null;
  }

  // Collect pick_up / use requests that are within interaction radius.
  const requests = new Map<string, string[]>(); // objectId -> agentId[]
  for (const agent of state.agents) {
    const intent = agent.intent;
    if (!intent) continue;
    if (intent.kind !== 'pick_up' && intent.kind !== 'use') continue;
    const target = state.objects.find((o) => o.id === intent.objectId);
    if (!target) {
      agent.intent = null;
      continue;
    }
    if (vec2Distance(agent.position, target.position) > INTERACTION_RADIUS) continue;
    let bucket = requests.get(target.id);
    if (!bucket) {
      bucket = [];
      requests.set(target.id, bucket);
    }
    bucket.push(agent.id);
  }

  const conflicts: WorldConflict[] = [];
  for (const [objectId, contenders] of requests) {
    const obj = state.objects.find((o) => o.id === objectId);
    if (!obj) continue;
    if (obj.heldBy) {
      // Already held by someone; treat as a denied interaction.
      for (const id of contenders) {
        clearAgentIntent(state, id);
      }
      continue;
    }
    if (contenders.length === 1 && !obj.contested) {
      applyPickUp(state, contenders[0]!, objectId);
      continue;
    }
    conflicts.push({ kind: 'contested_object', objectId, contenderAgentIds: contenders });
  }

  // Clear wait intents so the agent is eligible for a fresh query next tick
  // that triggers it (orchestrator decides cadence).
  for (const agent of state.agents) {
    if (agent.intent?.kind === 'wait') {
      agent.intent = null;
    }
  }

  return { conflicts };
}

/** Second pass: apply director resolutions authoritatively. */
export function applyDirectorDecision(
  state: MutableWorldState,
  decision: DirectorDecision
): void {
  state.directorNote = decision.note.length > 0 ? decision.note : state.directorNote;
  for (const resolution of decision.resolutions) {
    const obj = state.objects.find((o) => o.id === resolution.objectId);
    if (!obj) continue;
    if (obj.heldBy) continue;
    if (resolution.winnerAgentId === null) {
      // Deny all contenders; clear their intents so they can choose afresh.
      for (const agent of state.agents) {
        if (agent.intent?.kind === 'pick_up' && agent.intent.objectId === obj.id) {
          agent.intent = null;
        }
        if (agent.intent?.kind === 'use' && agent.intent.objectId === obj.id) {
          agent.intent = null;
        }
      }
      continue;
    }
    applyPickUp(state, resolution.winnerAgentId, obj.id);
    // Everyone else loses their intent on this object.
    for (const agent of state.agents) {
      if (agent.id === resolution.winnerAgentId) continue;
      if (agent.intent?.kind === 'pick_up' && agent.intent.objectId === obj.id) {
        agent.intent = null;
      }
      if (agent.intent?.kind === 'use' && agent.intent.objectId === obj.id) {
        agent.intent = null;
      }
    }
  }
}

function applyPickUp(
  state: MutableWorldState,
  agentId: string,
  objectId: string
): void {
  const agent = state.agents.find((a) => a.id === agentId);
  const obj = state.objects.find((o) => o.id === objectId);
  if (!agent || !obj) return;
  if (agent.holding) {
    // Can't pick up while holding something — drop silently.
    const prev = state.objects.find((o) => o.id === agent.holding);
    if (prev) prev.heldBy = null;
    agent.holding = null;
  }
  obj.heldBy = agent.id;
  obj.position = { x: agent.position.x, z: agent.position.z };
  agent.holding = obj.id;
  agent.intent = null;
}

function clearAgentIntent(state: MutableWorldState, agentId: string): void {
  const agent = state.agents.find((a) => a.id === agentId);
  if (agent) agent.intent = null;
}
