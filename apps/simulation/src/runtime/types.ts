export interface Vec2 {
  readonly x: number;
  readonly z: number;
}

export interface WorldBounds {
  readonly halfExtent: number;
}

export interface ObjectAffordance {
  readonly kind: 'pick_up' | 'use';
  readonly label: string;
  readonly status?: string;
}

export type AgentIntent =
  | { kind: 'wait'; emotion: string; reason?: string }
  | { kind: 'move_to'; target: Vec2; emotion: string }
  | { kind: 'go_to_object'; objectId: string; emotion: string }
  | { kind: 'approach_agent'; agentId: string; emotion: string }
  | { kind: 'pick_up'; objectId: string; emotion: string }
  | { kind: 'drop'; emotion: string }
  | { kind: 'deliver'; objectId: string; emotion: string }
  | { kind: 'sabotage'; agentId: string; emotion: string }
  | { kind: 'use'; objectId: string; emotion: string };

export interface SimulationAgentState {
  readonly id: string;
  readonly name: string;
  readonly archetype?: string;
  position: Vec2;
  heading: number;
  speed: number;
  emotion: string | null;
  status: string;
  intent: AgentIntent | null;
  goal: AgentGoal | null;
  holding: string | null;
  intentIssuedAtTick: number;
  thinking: boolean;
  navigation: AgentNavigationState;
}

export interface AgentNavigationState {
  detourTarget: Vec2 | null;
  blockedTicks: number;
  obstacleId: string | null;
}

export interface SimulationObjectState {
  readonly id: string;
  readonly kind: string;
  readonly label: string;
  position: Vec2;
  readonly contested: boolean;
  heldBy: string | null;
  readonly tags: readonly string[];
  readonly affordances: readonly ObjectAffordance[];
  readonly blocksMovement: boolean;
  readonly collisionRadius: number;
}

export interface SimulationScoreState {
  readonly deliveries: Readonly<Record<string, number>>;
  readonly forcedDrops: Readonly<Record<string, number>>;
}

export interface SimulationGameState {
  readonly title: string;
  readonly bananaObjectId: string;
  readonly goalObjectId: string;
  readonly bananaSpawnPoints: readonly Vec2[];
  readonly score: SimulationScoreState;
  readonly referee: RefereeState;
}

export type RefereeState =
  | { readonly status: 'idle' }
  | { readonly status: 'ruling'; readonly conflict: WorldConflict; readonly startedAtTick: number };

export interface WorldSnapshot {
  readonly tick: number;
  readonly timeSeconds: number;
  readonly bounds: WorldBounds;
  readonly agents: readonly SimulationAgentState[];
  readonly objects: readonly SimulationObjectState[];
  readonly directorNote: string | null;
  readonly game: SimulationGameState;
}

export interface PerceivedAgent {
  readonly id: string;
  readonly name: string;
  readonly distance: number;
  readonly direction: Vec2;
  readonly emotion: string | null;
  readonly status: string;
  readonly holding: string | null;
}

export interface PerceivedObject {
  readonly id: string;
  readonly kind: string;
  readonly label: string;
  readonly distance: number;
  readonly direction: Vec2;
  readonly heldBy: string | null;
  readonly contested: boolean;
  readonly affordances: readonly ObjectAffordance[];
  readonly tags: readonly string[];
  readonly blocksMovement: boolean;
  readonly collisionRadius: number;
}

export interface AgentPerception {
  readonly self: SimulationAgentState;
  readonly nearbyAgents: readonly PerceivedAgent[];
  readonly nearbyObjects: readonly PerceivedObject[];
  readonly tick: number;
  readonly bounds: WorldBounds;
  readonly directorNote: string | null;
  readonly game: SimulationGameState;
}

export type AgentGoal =
  | { kind: 'wait'; label: string }
  | { kind: 'go_to_object'; objectId: string; label: string }
  | { kind: 'go_to_agent'; agentId: string; label: string }
  | { kind: 'object_action'; objectId: string; affordance: ObjectAffordance; label: string }
  | { kind: 'deliver'; objectId: string; label: string }
  | { kind: 'sabotage_agent'; agentId: string; label: string }
  | { kind: 'drop'; label: string };

export interface DecisionOption {
  readonly label: string;
  readonly goal: AgentGoal;
}

export interface DecisionContext {
  readonly prompt: string;
  readonly options: readonly DecisionOption[];
}

export interface ContestedObjectConflict {
  readonly id: string;
  readonly kind: 'contested_object';
  readonly objectId: string;
  readonly contenderAgentIds: readonly string[];
}

export interface ForcedDropConflict {
  readonly id: string;
  readonly kind: 'forced_drop';
  readonly attackerAgentId: string;
  readonly targetAgentId: string;
  readonly objectId: string;
}

export type WorldConflict = ContestedObjectConflict | ForcedDropConflict;

export type DirectorResolutionOutcome =
  | 'pickup'
  | 'deny'
  | 'drop'
  | 'hold'
  | 'attacker_fumbles';

export interface DirectorResolution {
  readonly conflictId: string;
  readonly objectId?: string;
  readonly winnerAgentId: string | null;
  readonly outcome: DirectorResolutionOutcome;
  readonly note?: string;
}

export interface DirectorDecision {
  readonly resolutions: readonly DirectorResolution[];
  readonly note: string;
}

export interface SimulationGameEvent {
  readonly kind:
    | 'delivery'
    | 'respawn'
    | 'pickup'
    | 'drop'
    | 'forced_drop'
    | 'fallback';
  readonly message: string;
  readonly agentId?: string;
  readonly objectId?: string;
  readonly points?: number;
}

export interface ScenarioAgentSeed {
  readonly id: string;
  readonly name: string;
  readonly archetype?: string;
  readonly position: Vec2;
  readonly heading?: number;
  readonly speed?: number;
  readonly status?: string;
}

export interface ScenarioObjectSeed {
  readonly id: string;
  readonly kind: string;
  readonly label?: string;
  readonly position: Vec2;
  readonly contested?: boolean;
  readonly tags?: readonly string[];
  readonly affordances?: readonly ObjectAffordance[];
  readonly blocksMovement?: boolean;
  readonly collisionRadius?: number;
}

export interface ScenarioGameSeed {
  readonly title: string;
  readonly bananaObjectId: string;
  readonly goalObjectId: string;
  readonly bananaSpawnPoints: readonly Vec2[];
}

export interface ScenarioSeed {
  readonly id: string;
  readonly title: string;
  readonly bounds?: WorldBounds;
  readonly agents: readonly ScenarioAgentSeed[];
  readonly objects: readonly ScenarioObjectSeed[];
  readonly game: ScenarioGameSeed;
  readonly directorNote?: string;
  readonly directorConfigUrl: string;
  readonly directorCadenceTicks?: number;
  readonly resolveRefereeQuery?: string;
  readonly narrateQuery?: string;
}
