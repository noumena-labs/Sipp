export interface Vec2 {
  readonly x: number;
  readonly z: number;
}

export interface WorldBounds {
  readonly halfExtent: number;
}

export type AgentIntent =
  | { kind: 'wait'; emotion: string; reason?: string }
  | { kind: 'wander'; emotion: string }
  | { kind: 'move_to'; target: Vec2; emotion: string }
  | { kind: 'approach_agent'; agentId: string; emotion: string }
  | { kind: 'pick_up'; objectId: string; emotion: string }
  | { kind: 'drop'; emotion: string }
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
  holding: string | null;
  intentIssuedAtTick: number;
}

export interface SimulationObjectState {
  readonly id: string;
  readonly kind: string;
  position: Vec2;
  readonly contested: boolean;
  heldBy: string | null;
  readonly tags: readonly string[];
}

export interface WorldSnapshot {
  readonly tick: number;
  readonly timeSeconds: number;
  readonly bounds: WorldBounds;
  readonly agents: readonly SimulationAgentState[];
  readonly objects: readonly SimulationObjectState[];
  readonly directorNote: string | null;
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
  readonly distance: number;
  readonly direction: Vec2;
  readonly heldBy: string | null;
  readonly contested: boolean;
}

export interface AgentPerception {
  readonly self: SimulationAgentState;
  readonly nearbyAgents: readonly PerceivedAgent[];
  readonly nearbyObjects: readonly PerceivedObject[];
  readonly tick: number;
  readonly bounds: WorldBounds;
  readonly directorNote: string | null;
}

export interface ContestedObjectConflict {
  readonly kind: 'contested_object';
  readonly objectId: string;
  readonly contenderAgentIds: readonly string[];
}

export type WorldConflict = ContestedObjectConflict;

export interface DirectorResolution {
  readonly objectId: string;
  readonly winnerAgentId: string | null;
  readonly note?: string;
}

export interface DirectorDecision {
  readonly resolutions: readonly DirectorResolution[];
  readonly note: string;
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
  readonly position: Vec2;
  readonly contested?: boolean;
  readonly tags?: readonly string[];
}

export interface ScenarioSeed {
  readonly id: string;
  readonly title: string;
  readonly bounds?: WorldBounds;
  readonly agents: readonly ScenarioAgentSeed[];
  readonly objects: readonly ScenarioObjectSeed[];
  readonly directorNote?: string;
  readonly directorConfigUrl: string;
  readonly directorCadenceTicks?: number;
  readonly resolveConflictQuery?: string;
  readonly narrateQuery?: string;
}
