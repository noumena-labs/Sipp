import { clampToBounds, vec2Distance } from './sensing.js';
import type {
  AgentIntent,
  DirectorDecision,
  RefereeState,
  SimulationAgentState,
  SimulationGameEvent,
  SimulationGameState,
  SimulationObjectState,
  SimulationScoreState,
  Vec2,
  WorldBounds,
  WorldConflict,
} from './types.js';

export interface MutableScoreState {
  deliveries: Record<string, number>;
  forcedDrops: Record<string, number>;
}

export interface MutableGameState extends Omit<SimulationGameState, 'score' | 'referee'> {
  score: MutableScoreState;
  referee: RefereeState;
  pendingRespawn: SimulationGameState['pendingRespawn'];
  nextSpawnIndex: number;
}

export interface MutableWorldState {
  tick: number;
  timeSeconds: number;
  bounds: WorldBounds;
  agents: SimulationAgentState[];
  objects: SimulationObjectState[];
  directorNote: string | null;
  game: MutableGameState;
}

export const INTERACTION_RADIUS = 0.75;
export const AGENT_RADIUS = 0.38;
export const GOAL_RADIUS = 1.2;
export const SABOTAGE_RADIUS = 0.95;
const DETOUR_PADDING = 0.35;
const DETOUR_REACHED_RADIUS = 0.35;
const BLOCKED_REPATH_TICKS = 2;
const SIDESTEP_ANGLES = [0, Math.PI / 6, -Math.PI / 6, Math.PI / 3, -Math.PI / 3, Math.PI / 2, -Math.PI / 2] as const;
const OBJECT_APPROACH_SLOT_RADIUS = 0.6;
const AGENT_APPROACH_SLOT_RADIUS = 0.85;

export interface TickReducerResult {
  readonly conflicts: WorldConflict[];
  readonly arrivedAgentIds: readonly string[];
  readonly events: readonly SimulationGameEvent[];
}

export function applyTickFirstPass(state: MutableWorldState, dt: number): TickReducerResult {
  const events: SimulationGameEvent[] = [];
  const arrivedAgentIds: string[] = [];

  processPendingRespawn(state, events);

  moveAgentsWithoutOverlap(state, dt, arrivedAgentIds);
  syncHeldObjects(state);

  processDrops(state, events);
  processDeliveries(state, events);

  const conflicts = [
    ...processPickupRequests(state, events),
    ...processSabotageRequests(state),
  ];

  for (const agent of state.agents) {
    if (agent.intent?.kind === 'wait') {
      agent.intent = null;
    }
  }

  return { conflicts, arrivedAgentIds, events };
}

export function applyDirectorDecision(
  state: MutableWorldState,
  decision: DirectorDecision
): SimulationGameEvent[] {
  state.directorNote = decision.note.length > 0 ? decision.note : state.directorNote;
  const events: SimulationGameEvent[] = [];

  for (const resolution of decision.resolutions) {
    const conflict = findConflictFromResolution(state, resolution.conflictId);
    if (conflict?.kind === 'contested_object') {
      const objectId = resolution.objectId ?? conflict.objectId;
      if (resolution.outcome === 'pickup' && resolution.winnerAgentId) {
        applyPickUp(state, resolution.winnerAgentId, objectId, events);
      } else {
        clearPickupIntents(state, objectId);
      }
      continue;
    }

    if (conflict?.kind === 'forced_drop') {
      applyForcedDropResolution(state, conflict, resolution.outcome, events);
    }
  }

  return events;
}

export function deterministicConflictResolution(
  state: MutableWorldState,
  conflicts: readonly WorldConflict[]
): DirectorDecision {
  return {
    note: 'The referee uses a quick house-rule ruling.',
    resolutions: conflicts.map((conflict) => {
      if (conflict.kind === 'contested_object') {
        return {
          conflictId: conflict.id,
          objectId: conflict.objectId,
          winnerAgentId: choosePickupWinner(state, conflict),
          outcome: 'pickup',
          note: 'closest grab wins',
        };
      }
      return {
        conflictId: conflict.id,
        objectId: conflict.objectId,
        winnerAgentId: null,
        outcome: 'drop',
        note: 'the bump shakes the banana loose',
      };
    }),
  };
}

function moveAgentsWithoutOverlap(
  state: MutableWorldState,
  dt: number,
  arrivedAgentIds: string[]
): void {
  const accepted = new Map<string, Vec2>();

  for (const agent of state.agents) {
    const next = stepMovement(agent, state, dt, accepted);
    const position = next.position;

    agent.position = position;
    agent.heading = next.heading;
    accepted.set(agent.id, position);

    if (agent.intent?.emotion) {
      agent.emotion = agent.intent.emotion;
    }
    if (agent.intent && hasReachedCurrentIntent(agent, state)) {
      arrivedAgentIds.push(agent.id);
    }
  }
}

function stepMovement(
  agent: SimulationAgentState,
  state: MutableWorldState,
  dt: number,
  accepted: ReadonlyMap<string, Vec2>
): { position: Vec2; heading: number } {
  const intent = agent.intent;
  if (!intent) {
    agent.navigation.detourTarget = null;
    agent.navigation.blockedTicks = 0;
    agent.navigation.obstacleId = null;
    return { position: agent.position, heading: agent.heading };
  }
  const finalTarget = resolveMovementTarget(agent, intent, state);
  if (!finalTarget) {
    agent.navigation.detourTarget = null;
    agent.navigation.blockedTicks = 0;
    agent.navigation.obstacleId = null;
    return { position: agent.position, heading: agent.heading };
  }
  if (agent.navigation.detourTarget && vec2Distance(agent.position, agent.navigation.detourTarget) <= DETOUR_REACHED_RADIUS) {
    agent.navigation.detourTarget = null;
    agent.navigation.obstacleId = null;
  }

  if (agent.navigation.detourTarget && hasClearLine(agent.position, finalTarget, state.objects)) {
    agent.navigation.detourTarget = null;
    agent.navigation.obstacleId = null;
  }

  const target = agent.navigation.detourTarget ?? finalTarget;
  const dx = target.x - agent.position.x;
  const dz = target.z - agent.position.z;
  const dist = Math.sqrt(dx * dx + dz * dz);
  if (dist < 1e-4) {
    agent.navigation.blockedTicks = 0;
    return { position: agent.position, heading: agent.heading };
  }
  const step = Math.min(dist, agent.speed * dt);
  const desiredHeading = Math.atan2(dx, dz);

  for (const angle of SIDESTEP_ANGLES) {
    const heading = desiredHeading + angle;
    const candidate = clampToBounds({
      x: agent.position.x + Math.sin(heading) * step,
      z: agent.position.z + Math.cos(heading) * step,
    }, state.bounds);
    if (!isCandidateBlocked(agent.id, candidate, state, accepted)) {
      if (angle === 0) {
        agent.navigation.blockedTicks = 0;
        if (agent.navigation.detourTarget && vec2Distance(candidate, finalTarget) < vec2Distance(agent.position, finalTarget)) {
          agent.navigation.detourTarget = null;
          agent.navigation.obstacleId = null;
        }
      } else {
        agent.navigation.blockedTicks = 0;
      }
      return { position: candidate, heading };
    }
  }

  agent.navigation.blockedTicks += 1;
  const blockingObstacle = findBlockingObstacleOnPath(agent.position, finalTarget, state.objects);
  if (blockingObstacle && (agent.navigation.detourTarget == null || agent.navigation.blockedTicks >= BLOCKED_REPATH_TICKS)) {
    const detour = chooseDetourTarget(agent, finalTarget, blockingObstacle, state, accepted);
    if (detour) {
      agent.navigation.detourTarget = detour.target;
      agent.navigation.obstacleId = blockingObstacle.id;
      agent.navigation.blockedTicks = 0;
      return { position: agent.position, heading: detour.heading };
    }
  }

  return { position: agent.position, heading: desiredHeading };
}

function resolveMovementTarget(
  agent: SimulationAgentState,
  intent: AgentIntent,
  state: MutableWorldState
): Vec2 | null {
  switch (intent.kind) {
    case 'move_to':
      return intent.target;
    case 'go_to_object': {
      const target = state.objects.find((o) => o.id === intent.objectId);
      if (!target) return null;
      if (usesObjectApproachSlots(state, target)) {
        return resolveApproachPointForObject(state, agent.id, intent.objectId, target.position);
      }
      return target.position;
    }
    case 'approach_agent':
    case 'sabotage': {
      const target = state.agents.find((a) => a.id === intent.agentId);
      if (!target) return null;
      if (usesAgentApproachSlots(state, intent.agentId)) {
        return resolveApproachPointForAgent(state, agent.id, intent.agentId, target.position);
      }
      return target.position;
    }
    case 'pick_up':
    case 'use':
    case 'deliver': {
      const target = state.objects.find((o) => o.id === intent.objectId);
      if (!target) return null;
      if (intent.kind === 'pick_up') {
        if (usesObjectApproachSlots(state, target)) {
          return resolveApproachPointForObject(state, agent.id, intent.objectId, target.position);
        }
      }
      return target.position;
    }
    case 'wait':
    case 'drop':
      return null;
  }
}

function collidesWithBlockingObject(position: Vec2, objects: readonly SimulationObjectState[]): boolean {
  return objects.some(
    (obj) => obj.blocksMovement && vec2Distance(position, obj.position) < AGENT_RADIUS + obj.collisionRadius
  );
}

function findBlockingObject(position: Vec2, objects: readonly SimulationObjectState[]): SimulationObjectState | null {
  for (const obj of objects) {
    if (!obj.blocksMovement) continue;
    if (vec2Distance(position, obj.position) < AGENT_RADIUS + obj.collisionRadius) {
      return obj;
    }
  }
  return null;
}

function collidesWithAgents(
  agentId: string,
  position: Vec2,
  agents: readonly SimulationAgentState[],
  accepted: ReadonlyMap<string, Vec2>
): boolean {
  for (const other of agents) {
    if (other.id === agentId) continue;
    const otherPosition = accepted.get(other.id) ?? other.position;
    if (vec2Distance(position, otherPosition) < AGENT_RADIUS * 2) {
      return true;
    }
  }
  return false;
}

function isCandidateBlocked(
  agentId: string,
  position: Vec2,
  state: MutableWorldState,
  accepted: ReadonlyMap<string, Vec2>
): boolean {
  return (
    collidesWithBlockingObject(position, state.objects) ||
    collidesWithAgents(agentId, position, state.agents, accepted)
  );
}

function usesObjectApproachSlots(
  state: MutableWorldState,
  object: SimulationObjectState
): boolean {
  return object.id === state.game.bananaObjectId || object.contested;
}

function usesAgentApproachSlots(state: MutableWorldState, targetAgentId: string): boolean {
  return getAgentApproachContenders(state, targetAgentId).length > 1;
}

function resolveApproachPointForObject(
  state: MutableWorldState,
  agentId: string,
  objectId: string,
  center: Vec2
): Vec2 {
  return resolveApproachPoint(
    agentId,
    getObjectApproachContenders(state, objectId),
    center,
    OBJECT_APPROACH_SLOT_RADIUS
  );
}

function resolveApproachPointForAgent(
  state: MutableWorldState,
  agentId: string,
  targetAgentId: string,
  center: Vec2
): Vec2 {
  return resolveApproachPoint(
    agentId,
    getAgentApproachContenders(state, targetAgentId),
    center,
    AGENT_APPROACH_SLOT_RADIUS
  );
}

function resolveApproachPoint(
  agentId: string,
  contenders: readonly SimulationAgentState[],
  center: Vec2,
  radius: number
): Vec2 {
  if (contenders.length <= 1) {
    return center;
  }
  const ordered = [...contenders].sort(compareApproachPriority(center));
  const index = ordered.findIndex((entry) => entry.id === agentId);
  if (index < 0) {
    return center;
  }
  return slotPointAround(center, index, ordered.length, approachAngleFrom(center, ordered[0]!.position), radius);
}

function getObjectApproachContenders(
  state: MutableWorldState,
  objectId: string
): SimulationAgentState[] {
  return state.agents.filter((agent) => {
    const intent = agent.intent;
    return (
      (intent?.kind === 'pick_up' && intent.objectId === objectId) ||
      (intent?.kind === 'go_to_object' && intent.objectId === objectId)
    );
  });
}

function getAgentApproachContenders(
  state: MutableWorldState,
  targetAgentId: string
): SimulationAgentState[] {
  return state.agents.filter((agent) => {
    const intent = agent.intent;
    return (
      (intent?.kind === 'sabotage' && intent.agentId === targetAgentId) ||
      (intent?.kind === 'approach_agent' && intent.agentId === targetAgentId)
    );
  });
}

function compareApproachPriority(center: Vec2) {
  return (a: SimulationAgentState, b: SimulationAgentState): number => {
    const angleDiff = approachAngleFrom(center, a.position) - approachAngleFrom(center, b.position);
    if (Math.abs(angleDiff) > 0.01) return angleDiff;
    const distanceDiff = vec2Distance(a.position, center) - vec2Distance(b.position, center);
    if (Math.abs(distanceDiff) > 0.01) return distanceDiff;
    const issuedDiff = a.intentIssuedAtTick - b.intentIssuedAtTick;
    if (issuedDiff !== 0) return issuedDiff;
    return a.id.localeCompare(b.id);
  };
}

function approachAngleFrom(center: Vec2, position: Vec2): number {
  const angle = Math.atan2(position.x - center.x, position.z - center.z);
  return angle >= 0 ? angle : angle + Math.PI * 2;
}

function slotPointAround(
  center: Vec2,
  index: number,
  total: number,
  startAngle: number,
  radius: number
): Vec2 {
  if (total <= 1) {
    return center;
  }
  const angle = startAngle + (Math.PI * 2 * index) / total;
  return {
    x: center.x + Math.sin(angle) * radius,
    z: center.z + Math.cos(angle) * radius,
  };
}

function hasClearLine(from: Vec2, to: Vec2, objects: readonly SimulationObjectState[]): boolean {
  return findBlockingObstacleOnPath(from, to, objects) == null;
}

function findBlockingObstacleOnPath(
  from: Vec2,
  to: Vec2,
  objects: readonly SimulationObjectState[]
): SimulationObjectState | null {
  let best: { object: SimulationObjectState; t: number } | null = null;
  for (const obj of objects) {
    if (!obj.blocksMovement) continue;
    const hit = distanceToSegment(obj.position, from, to);
    const radius = obj.collisionRadius + AGENT_RADIUS + DETOUR_PADDING;
    if (hit.distance > radius) continue;
    if (best == null || hit.t < best.t) {
      best = { object: obj, t: hit.t };
    }
  }
  return best?.object ?? null;
}

function chooseDetourTarget(
  agent: SimulationAgentState,
  finalTarget: Vec2,
  obstacle: SimulationObjectState,
  state: MutableWorldState,
  accepted: ReadonlyMap<string, Vec2>
): { target: Vec2; heading: number } | null {
  const padding = obstacle.collisionRadius + AGENT_RADIUS + DETOUR_PADDING;
  const awayX = agent.position.x - obstacle.position.x;
  const awayZ = agent.position.z - obstacle.position.z;
  const awayLen = Math.sqrt(awayX * awayX + awayZ * awayZ) || 1;
  const nx = awayX / awayLen;
  const nz = awayZ / awayLen;
  const tx = -nz;
  const tz = nx;

  const candidates: Vec2[] = [
    clampToBounds({ x: obstacle.position.x + tx * padding + nx * padding, z: obstacle.position.z + tz * padding + nz * padding }, state.bounds),
    clampToBounds({ x: obstacle.position.x - tx * padding + nx * padding, z: obstacle.position.z - tz * padding + nz * padding }, state.bounds),
  ];

  let best: { target: Vec2; score: number; heading: number } | null = null;
  for (const candidate of candidates) {
    if (findBlockingObject(candidate, state.objects)) continue;
    if (collidesWithAgents(agent.id, candidate, state.agents, accepted)) continue;
    const heading = Math.atan2(candidate.x - agent.position.x, candidate.z - agent.position.z);
    const score = vec2Distance(candidate, finalTarget);
    if (best == null || score < best.score) {
      best = { target: candidate, score, heading };
    }
  }
  return best ? { target: best.target, heading: best.heading } : null;
}

function distanceToSegment(point: Vec2, a: Vec2, b: Vec2): { distance: number; t: number } {
  const abx = b.x - a.x;
  const abz = b.z - a.z;
  const apx = point.x - a.x;
  const apz = point.z - a.z;
  const abLenSq = abx * abx + abz * abz;
  if (abLenSq < 1e-6) {
    return { distance: vec2Distance(point, a), t: 0 };
  }
  const unclampedT = (apx * abx + apz * abz) / abLenSq;
  const t = Math.max(0, Math.min(1, unclampedT));
  const closest = { x: a.x + abx * t, z: a.z + abz * t };
  return { distance: vec2Distance(point, closest), t };
}

function syncHeldObjects(state: MutableWorldState): void {
  for (const obj of state.objects) {
    if (!obj.heldBy) continue;
    const carrier = state.agents.find((a) => a.id === obj.heldBy);
    if (carrier) {
      obj.position = { x: carrier.position.x, z: carrier.position.z };
    }
  }
}

function processDrops(state: MutableWorldState, events: SimulationGameEvent[]): void {
  for (const agent of state.agents) {
    if (agent.intent?.kind !== 'drop') continue;
    if (!agent.holding) {
      agent.intent = null;
      continue;
    }
    dropHeldObject(state, agent, events, `${agent.name} drops the ${agent.holding}.`);
  }
}

function processDeliveries(state: MutableWorldState, events: SimulationGameEvent[]): void {
  const banana = getObject(state, state.game.bananaObjectId);
  const goal = getObject(state, state.game.goalObjectId);
  if (!banana || !goal) return;

  for (const agent of state.agents) {
    if (agent.holding !== banana.id) continue;
    if (vec2Distance(agent.position, goal.position) > GOAL_RADIUS) continue;
    const previousCarrierId = agent.id;
    incrementScore(state.game.score.deliveries, agent.id);
    agent.holding = null;
    agent.intent = null;
    agent.goal = null;
    agent.status = 'scored a banana delivery';
    agent.emotion = 'happy';
    banana.heldBy = null;
    events.push({
      kind: 'delivery',
      agentId: agent.id,
      objectId: banana.id,
      position: { x: goal.position.x, z: goal.position.z },
      points: 1,
    });
    scheduleRespawn(state, banana.id, 1);
    clearCarrierPursuits(state, previousCarrierId);
  }
}

function processPickupRequests(
  state: MutableWorldState,
  events: SimulationGameEvent[]
): WorldConflict[] {
  const requests = new Map<string, string[]>();
  for (const agent of state.agents) {
    const intent = agent.intent;
    if (intent?.kind !== 'pick_up') continue;
    const target = getObject(state, intent.objectId);
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
    const obj = getObject(state, objectId);
    if (!obj) continue;
    if (obj.heldBy) {
      for (const id of contenders) clearAgentIntent(state, id);
      continue;
    }
    if (contenders.length === 1) {
      applyPickUp(state, contenders[0]!, objectId, events);
      continue;
    }
    conflicts.push({
      id: `pickup:${objectId}:${state.tick}`,
      kind: 'contested_object',
      objectId,
      contenderAgentIds: contenders,
    });
  }
  return conflicts;
}

function processSabotageRequests(state: MutableWorldState): WorldConflict[] {
  const banana = getObject(state, state.game.bananaObjectId);
  if (!banana) return [];

  const conflicts: WorldConflict[] = [];
  for (const agent of state.agents) {
    const intent = agent.intent;
    if (intent?.kind !== 'sabotage') continue;
    const target = state.agents.find((entry) => entry.id === intent.agentId);
    if (!target || target.holding !== banana.id) {
      agent.intent = null;
      continue;
    }
    if (vec2Distance(agent.position, target.position) > SABOTAGE_RADIUS) continue;
    conflicts.push({
      id: `drop:${agent.id}:${target.id}:${state.tick}`,
      kind: 'forced_drop',
      attackerAgentId: agent.id,
      targetAgentId: target.id,
      objectId: banana.id,
    });
  }
  return conflicts;
}

function processPendingRespawn(state: MutableWorldState, events: SimulationGameEvent[]): void {
  const pending = state.game.pendingRespawn;
  if (!pending || state.tick < pending.activateAtTick) return;
  const banana = getObject(state, pending.objectId);
  state.game.pendingRespawn = null;
  if (!banana) return;
  banana.position = { x: pending.spawnPosition.x, z: pending.spawnPosition.z };
  banana.heldBy = null;
  events.push({
    kind: 'respawn',
    objectId: banana.id,
    position: { x: banana.position.x, z: banana.position.z },
  });
}

function scheduleRespawn(state: MutableWorldState, objectId: string, delayTicks: number): void {
  const spawn = nextValidBananaSpawn(state);
  state.game.pendingRespawn = {
    objectId,
    spawnPosition: spawn,
    activateAtTick: state.tick + delayTicks,
  };
}

function nextValidBananaSpawn(state: MutableWorldState): Vec2 {
  const spawns = state.game.bananaSpawnPoints;
  for (let i = 0; i < spawns.length; i += 1) {
    const index = (state.game.nextSpawnIndex + i) % spawns.length;
    const candidate = spawns[index]!;
    if (isValidSpawn(state, candidate)) {
      state.game.nextSpawnIndex = (index + 1) % spawns.length;
      return candidate;
    }
  }
  state.game.nextSpawnIndex = (state.game.nextSpawnIndex + 1) % spawns.length;
  return spawns[state.game.nextSpawnIndex] ?? { x: 0, z: 0 };
}

function isValidSpawn(state: MutableWorldState, position: Vec2): boolean {
  if (collidesWithBlockingObject(position, state.objects)) return false;
  for (const agent of state.agents) {
    if (vec2Distance(agent.position, position) < AGENT_RADIUS * 2.5) return false;
  }
  const goal = getObject(state, state.game.goalObjectId);
  if (goal && vec2Distance(goal.position, position) < GOAL_RADIUS + 0.5) return false;
  return true;
}

function applyPickUp(
  state: MutableWorldState,
  agentId: string,
  objectId: string,
  events: SimulationGameEvent[]
): void {
  const agent = state.agents.find((a) => a.id === agentId);
  const obj = getObject(state, objectId);
  if (!agent || !obj || obj.heldBy) return;
  if (agent.holding) {
    const prev = getObject(state, agent.holding);
    if (prev) prev.heldBy = null;
    agent.holding = null;
  }
  obj.heldBy = agent.id;
  obj.position = { x: agent.position.x, z: agent.position.z };
  agent.holding = obj.id;
  const goal = getObject(state, state.game.goalObjectId);
  agent.intent = goal ? { kind: 'go_to_object', objectId: goal.id, emotion: 'deliver' } : null;
  agent.goal = goal ? { kind: 'deliver', objectId: goal.id, label: 'run to home base' } : null;
  agent.status = goal ? `carrying the ${obj.label} to home base` : `carrying the ${obj.label}`;
  agent.emotion = 'happy';
  events.push({
    kind: 'pickup',
    agentId: agent.id,
    objectId: obj.id,
    position: { x: agent.position.x, z: agent.position.z },
  });
}

function applyForcedDropResolution(
  state: MutableWorldState,
  conflict: Extract<WorldConflict, { kind: 'forced_drop' }>,
  outcome: string,
  events: SimulationGameEvent[]
): void {
  const attacker = state.agents.find((agent) => agent.id === conflict.attackerAgentId);
  const target = state.agents.find((agent) => agent.id === conflict.targetAgentId);
  const obj = getObject(state, conflict.objectId);
  if (!attacker || !target || !obj) return;

  attacker.intent = null;
  attacker.goal = null;

  if (outcome === 'drop' && target.holding === obj.id) {
    dropHeldObject(state, target, events, `${attacker.name} bumps ${target.name}, and the banana drops!`, 'forced');
    incrementScore(state.game.score.forcedDrops, attacker.id);
    events.push({
      kind: 'forced_drop',
      attackerAgentId: attacker.id,
      targetAgentId: target.id,
      objectId: obj.id,
      position: { x: target.position.x, z: target.position.z },
      outcome: 'drop',
    });
    return;
  }

  if (outcome === 'attacker_fumbles') {
    attacker.status = 'fumbled a sabotage attempt';
    attacker.emotion = 'surprised';
    events.push({
      kind: 'forced_drop',
      attackerAgentId: attacker.id,
      targetAgentId: target.id,
      objectId: obj.id,
      position: { x: attacker.position.x, z: attacker.position.z },
      outcome: 'attacker_fumbles',
    });
    return;
  }

  target.status = 'kept hold of the banana';
  target.emotion = 'alert';
  events.push({
    kind: 'forced_drop',
    attackerAgentId: attacker.id,
    targetAgentId: target.id,
    objectId: obj.id,
    position: { x: target.position.x, z: target.position.z },
    outcome: 'hold',
  });
}

function dropHeldObject(
  state: MutableWorldState,
  agent: SimulationAgentState,
  events: SimulationGameEvent[],
  _message: string,
  cause: 'voluntary' | 'forced' = 'voluntary'
): void {
  if (!agent.holding) return;
  const previousCarrierId = agent.id;
  const from = { x: agent.position.x, z: agent.position.z };
  const to = cause === 'forced'
    ? chooseForcedDropLandingPoint(state, agent)
    : chooseVoluntaryDropLandingPoint(state, from);
  const held = getObject(state, agent.holding);
  if (held) {
    held.heldBy = null;
    held.position = to;
  }
  const objectId = agent.holding;
  agent.holding = null;
  agent.intent = null;
  agent.goal = null;
  agent.status = 'dropped the banana';
  agent.emotion = 'surprised';
  clearCarrierPursuits(state, previousCarrierId);
  events.push({ kind: 'drop', agentId: agent.id, objectId, from, to, cause });
}

function chooseVoluntaryDropLandingPoint(state: MutableWorldState, origin: Vec2): Vec2 {
  const radii = [0.9, 0.75, 1.1, 0.55];
  for (const radius of radii) {
    for (let i = 0; i < 10; i += 1) {
      const angle = (Math.PI * 2 * i) / 10;
      const candidate = clampToBounds(
        { x: origin.x + Math.sin(angle) * radius, z: origin.z + Math.cos(angle) * radius },
        state.bounds
      );
      if (findBlockingObject(candidate, state.objects)) continue;
      if (state.agents.some((other) => vec2Distance(other.position, candidate) < AGENT_RADIUS * 1.5)) continue;
      return candidate;
    }
  }
  return { x: origin.x, z: origin.z };
}

function chooseForcedDropLandingPoint(
  state: MutableWorldState,
  carrier: SimulationAgentState
): Vec2 {
  const origin = carrier.position;
  const nearbyAgents = state.agents.filter((agent) => vec2Distance(agent.position, origin) <= 3.5);
  const crowdCenter = nearbyAgents.length > 0
    ? averagePosition(nearbyAgents.map((agent) => agent.position))
    : origin;
  const awayFromCrowd = normalizeVec({
    x: origin.x - crowdCenter.x,
    z: origin.z - crowdCenter.z,
  });
  const fallbackDirection = {
    x: Math.sin(carrier.heading || 0),
    z: Math.cos(carrier.heading || 0),
  };
  const baseDirection = awayFromCrowd ?? normalizeVec(fallbackDirection) ?? { x: 0, z: 1 };
  const radii = [2.6, 3.1, 3.6, 4.2, 4.8];
  const candidateAngles = [0, Math.PI / 8, -Math.PI / 8, Math.PI / 4, -Math.PI / 4, Math.PI / 2, -Math.PI / 2, Math.PI, Math.PI * 0.75, -Math.PI * 0.75];

  let best: { point: Vec2; score: number } | null = null;
  for (const radius of radii) {
    for (const angleOffset of candidateAngles) {
      const direction = rotateVec(baseDirection, angleOffset);
      const candidate = clampToBounds(
        {
          x: origin.x + direction.x * radius,
          z: origin.z + direction.z * radius,
        },
        state.bounds
      );
      const score = scoreForcedDropCandidate(state, candidate, carrier.id);
      if (score == null) continue;
      if (best == null || score > best.score) {
        best = { point: candidate, score };
      }
    }
  }

  if (best) {
    return best.point;
  }

  const fallback = chooseVoluntaryDropLandingPoint(state, origin);
  if (fallback.x !== origin.x || fallback.z !== origin.z) {
    return fallback;
  }

  return clampToBounds(
    {
      x: origin.x + baseDirection.x * 2.4,
      z: origin.z + baseDirection.z * 2.4,
    },
    state.bounds
  );
}

function scoreForcedDropCandidate(
  state: MutableWorldState,
  candidate: Vec2,
  carrierId: string
): number | null {
  if (findBlockingObject(candidate, state.objects)) return null;

  let nearestDistance = Number.POSITIVE_INFINITY;
  let nearbyPenalty = 0;
  for (const agent of state.agents) {
    const distance = vec2Distance(agent.position, candidate);
    if (agent.id === carrierId && distance < AGENT_RADIUS * 3.6) return null;
    if (distance < AGENT_RADIUS * 2.4) return null;
    nearestDistance = Math.min(nearestDistance, distance);
    nearbyPenalty += Math.max(0, 2.6 - distance);
  }

  const goal = getObject(state, state.game.goalObjectId);
  if (goal && vec2Distance(goal.position, candidate) < GOAL_RADIUS + 0.6) {
    return null;
  }

  const banana = getObject(state, state.game.bananaObjectId);
  let obstacleClearance = 0;
  for (const object of state.objects) {
    if (object.id === banana?.id) continue;
    const distance = vec2Distance(object.position, candidate) - object.collisionRadius;
    obstacleClearance = Math.max(obstacleClearance, Math.max(0, 1.4 - distance));
  }

  return nearestDistance * 2.4 - nearbyPenalty * 1.7 - obstacleClearance;
}

function averagePosition(points: readonly Vec2[]): Vec2 {
  if (points.length === 0) {
    return { x: 0, z: 0 };
  }

  let totalX = 0;
  let totalZ = 0;
  for (const point of points) {
    totalX += point.x;
    totalZ += point.z;
  }
  return {
    x: totalX / points.length,
    z: totalZ / points.length,
  };
}

function normalizeVec(vector: Vec2): Vec2 | null {
  const length = Math.sqrt(vector.x * vector.x + vector.z * vector.z);
  if (length < 1e-6) {
    return null;
  }
  return {
    x: vector.x / length,
    z: vector.z / length,
  };
}

function rotateVec(vector: Vec2, angle: number): Vec2 {
  const sin = Math.sin(angle);
  const cos = Math.cos(angle);
  return {
    x: vector.x * cos - vector.z * sin,
    z: vector.x * sin + vector.z * cos,
  };
}

function clearCarrierPursuits(state: MutableWorldState, carrierId: string): void {
  for (const agent of state.agents) {
    if (agent.id === carrierId) continue;
    if (agent.goal?.kind === 'sabotage_agent' && agent.goal.agentId === carrierId) {
      agent.goal = null;
      agent.intent = null;
      continue;
    }
    if (agent.goal?.kind === 'go_to_agent' && agent.goal.agentId === carrierId) {
      agent.goal = null;
      agent.intent = null;
      continue;
    }
    if (agent.intent?.kind === 'sabotage' && agent.intent.agentId === carrierId) {
      agent.intent = null;
      agent.goal = null;
      continue;
    }
    if (agent.intent?.kind === 'approach_agent' && agent.intent.agentId === carrierId) {
      agent.intent = null;
      agent.goal = null;
    }
  }
}

function choosePickupWinner(
  state: MutableWorldState,
  conflict: Extract<WorldConflict, { kind: 'contested_object' }>
): string | null {
  const obj = getObject(state, conflict.objectId);
  if (!obj) return conflict.contenderAgentIds[0] ?? null;
  const contenders = conflict.contenderAgentIds
    .map((id) => state.agents.find((agent) => agent.id === id))
    .filter((agent): agent is SimulationAgentState => agent != null)
    .sort((a, b) => {
      const distanceDiff = vec2Distance(a.position, obj.position) - vec2Distance(b.position, obj.position);
      if (Math.abs(distanceDiff) > 0.01) return distanceDiff;
      const issuedDiff = a.intentIssuedAtTick - b.intentIssuedAtTick;
      if (issuedDiff !== 0) return issuedDiff;
      return a.id.localeCompare(b.id);
    });
  return contenders[0]?.id ?? null;
}

function clearPickupIntents(state: MutableWorldState, objectId: string): void {
  for (const agent of state.agents) {
    if (agent.intent?.kind === 'pick_up' && agent.intent.objectId === objectId) {
      agent.intent = null;
    }
  }
}

function clearAgentIntent(state: MutableWorldState, agentId: string): void {
  const agent = state.agents.find((a) => a.id === agentId);
  if (agent) agent.intent = null;
}

function findConflictFromResolution(
  state: MutableWorldState,
  conflictId: string
): WorldConflict | null {
  const referee = state.game.referee;
  if (referee.status === 'ruling' && referee.conflict.id === conflictId) {
    return referee.conflict;
  }
  return null;
}

export function hasReachedCurrentIntent(agent: SimulationAgentState, state: MutableWorldState): boolean {
  const intent = agent.intent;
  if (!intent) return false;
  switch (intent.kind) {
    case 'move_to':
      return vec2Distance(agent.position, intent.target) <= 0.35;
    case 'go_to_object': {
      const target = getObject(state, intent.objectId);
      if (!target) return true;
      const reachRadius = usesObjectApproachSlots(state, target) ? INTERACTION_RADIUS : 0.35;
      return vec2Distance(agent.position, target.position) <= reachRadius;
    }
    case 'wait':
    case 'drop':
      return true;
    case 'approach_agent': {
      const target = state.agents.find((entry) => entry.id === intent.agentId);
      return target ? vec2Distance(agent.position, target.position) <= 1.25 : true;
    }
    case 'sabotage': {
      const target = state.agents.find((entry) => entry.id === intent.agentId);
      return target ? vec2Distance(agent.position, target.position) <= SABOTAGE_RADIUS : true;
    }
    case 'pick_up':
    case 'use': {
      const target = getObject(state, intent.objectId);
      return target ? vec2Distance(agent.position, target.position) <= INTERACTION_RADIUS : true;
    }
    case 'deliver': {
      const target = getObject(state, intent.objectId);
      return target ? vec2Distance(agent.position, target.position) <= GOAL_RADIUS : true;
    }
  }
}

function getObject(state: MutableWorldState, objectId: string): SimulationObjectState | undefined {
  return state.objects.find((object) => object.id === objectId);
}

function incrementScore(score: Record<string, number>, agentId: string): void {
  score[agentId] = (score[agentId] ?? 0) + 1;
}

export function cloneScore(score: MutableScoreState): SimulationScoreState {
  return {
    deliveries: { ...score.deliveries },
    forcedDrops: { ...score.forcedDrops },
  };
}
