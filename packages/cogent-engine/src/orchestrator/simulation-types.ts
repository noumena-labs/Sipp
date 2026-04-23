//////////////////////////////////////////////////////////////////////////////
//
// simulation-types.ts
//
// - Core type definitions for the rendering-agnostic orchestrator layer.
// - Describes world state, agents, objects, intents, perceptions, and
//   director decisions. No three.js / DOM references.
//
//////////////////////////////////////////////////////////////////////////////

/**
 * Canonical emoji-action name set understood by v1 SimulationAgents. The app
 * layer decides how to render each one (glyph overlays, VRM expression, etc).
 * Keep this in lockstep with `SIMULATION_ACTION_NAMES` in
 * `simulation-character-actions.ts`.
 */
export type SimulationActionName =
  | 'thinking'
  | 'curious'
  | 'happy'
  | 'confused'
  | 'alert'
  | 'frustrated'
  | 'sleepy'
  | 'celebrate';

/** 2D ground-plane vector. Y is implicitly 0 — the world is flat in v1. */
export interface Vec2 {
  readonly x: number;
  readonly z: number;
}

/** Square, origin-centred world bounds on the XZ plane. */
export interface WorldBounds {
  /** Half-extent on each axis. A value of 8 yields a 16×16 world. */
  readonly halfExtent: number;
}

/** A stateful simulation agent (a character in the world). */
export interface SimulationAgentState {
  /** Stable runtime id. Independent of the underlying character.json id
   *  so one archetype can spawn multiple agents. */
  readonly id: string;
  /** Display name surfaced in prompts and UI. */
  readonly name: string;
  /** Archetype id (usually the `CharacterConfig.id`). Optional metadata. */
  readonly archetype?: string;
  /** Current position on the ground plane. */
  position: Vec2;
  /** Current heading in radians (0 faces +X). */
  heading: number;
  /** Move speed in world units / second. */
  speed: number;
  /** Current expressive action. `null` until the agent first emotes. */
  emotion: SimulationActionName | null;
  /** Short free-form status line the director or agent set. */
  status: string;
  /** Latest intent authored by the agent, or `null` if none is active. */
  intent: AgentIntent | null;
  /** The object id the agent is currently carrying, or `null` if empty-handed. */
  holding: string | null;
  /** Monotonically increasing tick at which `intent` was last issued. */
  intentIssuedAtTick: number;
}

/** A world object the agents can perceive and interact with. */
export interface SimulationObjectState {
  readonly id: string;
  /** Short descriptive kind (`"banana"`, `"bench"`, `"fountain"`, …). */
  readonly kind: string;
  position: Vec2;
  /** When true, only one agent may pick up / use this object per tick. */
  readonly contested: boolean;
  /** `null` if free, otherwise the id of the owning agent. */
  heldBy: string | null;
  /** Arbitrary tags the app layer may use for rendering hints. */
  readonly tags: readonly string[];
}

/** Immutable snapshot of the entire world, produced by the orchestrator. */
export interface WorldSnapshot {
  readonly tick: number;
  readonly timeSeconds: number;
  readonly bounds: WorldBounds;
  readonly agents: readonly SimulationAgentState[];
  readonly objects: readonly SimulationObjectState[];
  /** Director-authored narration for the current tick, if any. */
  readonly directorNote: string | null;
}

//////////////////////////////////////////////////////////////////////////////
// Intents
//////////////////////////////////////////////////////////////////////////////

export type AgentIntentKind =
  | 'wait'
  | 'wander'
  | 'move_to'
  | 'approach_agent'
  | 'pick_up'
  | 'drop'
  | 'use';

/** What a SimulationAgent wants to do next. Reducers decide if it happens. */
export type AgentIntent =
  | { kind: 'wait'; emotion: SimulationActionName; reason?: string }
  | { kind: 'wander'; emotion: SimulationActionName }
  | { kind: 'move_to'; target: Vec2; emotion: SimulationActionName }
  | { kind: 'approach_agent'; agentId: string; emotion: SimulationActionName }
  | { kind: 'pick_up'; objectId: string; emotion: SimulationActionName }
  | { kind: 'drop'; emotion: SimulationActionName }
  | { kind: 'use'; objectId: string; emotion: SimulationActionName };

//////////////////////////////////////////////////////////////////////////////
// Perception
//////////////////////////////////////////////////////////////////////////////

export interface PerceivedAgent {
  readonly id: string;
  readonly name: string;
  readonly distance: number;
  readonly direction: Vec2;
  readonly emotion: SimulationActionName | null;
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

/** What a single agent sees during a tick. Drives the prompt to the LLM. */
export interface AgentPerception {
  readonly self: SimulationAgentState;
  readonly nearbyAgents: readonly PerceivedAgent[];
  readonly nearbyObjects: readonly PerceivedObject[];
  readonly tick: number;
  readonly bounds: WorldBounds;
  readonly directorNote: string | null;
}

//////////////////////////////////////////////////////////////////////////////
// Director decisions
//////////////////////////////////////////////////////////////////////////////

/**
 * A conflict surfaced by the reducer to the director. In v1 the only
 * conflict kind is `contested_object` — multiple agents tried to pick up or
 * use the same object on the same tick.
 */
export interface ContestedObjectConflict {
  readonly kind: 'contested_object';
  readonly objectId: string;
  readonly contenderAgentIds: readonly string[];
}

export type WorldConflict = ContestedObjectConflict;

export interface DirectorResolution {
  /** Object id being resolved. */
  readonly objectId: string;
  /** Agent id that wins the conflict, or `null` to deny everyone. */
  readonly winnerAgentId: string | null;
  /** Optional short narration the app can render. */
  readonly note?: string;
}

export interface DirectorDecision {
  readonly resolutions: readonly DirectorResolution[];
  readonly note: string;
}

//////////////////////////////////////////////////////////////////////////////
// Scenario definition (for app-level use; kept here because it is pure data)
//////////////////////////////////////////////////////////////////////////////

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
  readonly bounds?: WorldBounds;
  readonly agents: readonly ScenarioAgentSeed[];
  readonly objects: readonly ScenarioObjectSeed[];
  readonly directorNote?: string;
}
