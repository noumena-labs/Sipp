import type { DirectorRuntime, JsonValue } from 'cogent-engine/orchestrator';
import { SimulationBus, type SimulationEvent } from './bus.js';
import { buildPerception } from './sensing.js';
import { SimulationAgentChooser } from './agent-chooser.js';
import {
  applyDirectorDecision,
  applyTickFirstPass,
  type MutableWorldState,
} from './reducer.js';
import type {
  AgentGoal,
  AgentIntent,
  DirectorDecision,
  DirectorResolution,
  ScenarioAgentSeed,
  ScenarioObjectSeed,
  SimulationAgentState,
  SimulationObjectState,
  WorldBounds,
  WorldConflict,
  WorldSnapshot,
} from './types.js';

export interface SimulationRuntimeOptions {
  readonly id?: string;
  readonly bounds?: WorldBounds;
  readonly initialDirectorNote?: string | null;
  readonly directorCadenceTicks?: number;
  readonly resolveConflictQuery?: string;
  readonly narrateQuery?: string;
  readonly bus?: SimulationBus;
}

export class SimulationRuntime {
  public readonly bus: SimulationBus;
  public readonly id: string;

  private readonly state: MutableWorldState;
  private readonly simulationAgents: Map<string, SimulationAgentChooser> = new Map();
  private readonly director: DirectorRuntime | null;
  private readonly directorCadenceTicks: number;
  private readonly resolveConflictQuery: string;
  private readonly narrateQuery: string;

  private disposed = false;
  private inFlightTick: Promise<void> | null = null;
  private queryCursor = 0;
  private activeController: AbortController | null = null;

  public constructor(director: DirectorRuntime | null, options: SimulationRuntimeOptions = {}) {
    this.id = options.id ?? 'simulation';
    this.bus = options.bus ?? new SimulationBus();
    this.director = director;
    this.directorCadenceTicks = Math.max(1, Math.floor(options.directorCadenceTicks ?? 10));
    this.resolveConflictQuery = options.resolveConflictQuery ?? 'resolve_pickup_conflict';
    this.narrateQuery = options.narrateQuery ?? 'narrate_scene';
    this.state = {
      tick: 0,
      timeSeconds: 0,
      bounds: options.bounds ?? { halfExtent: 8 },
      agents: [],
      objects: [],
      directorNote: options.initialDirectorNote ?? null,
    };
  }

  public getSnapshot(): WorldSnapshot {
    return {
      tick: this.state.tick,
      timeSeconds: this.state.timeSeconds,
      bounds: this.state.bounds,
      agents: this.state.agents.map(cloneAgent),
      objects: this.state.objects.map(cloneObject),
      directorNote: this.state.directorNote,
    };
  }

  public async step(dtSeconds: number): Promise<void> {
    if (this.disposed) return;
    if (this.inFlightTick) {
      await this.inFlightTick;
      return;
    }
    this.inFlightTick = this.runTick(dtSeconds).finally(() => {
      this.inFlightTick = null;
    });
    await this.inFlightTick;
  }

  public async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    if (this.activeController) {
      this.activeController.abort();
    }
    if (this.inFlightTick) {
      try {
        await this.inFlightTick;
      } catch {
        // Ignore.
      }
    }
  }

  public addAgent(agent: SimulationAgentChooser, seed: ScenarioAgentSeed): void {
    if (this.simulationAgents.has(seed.id)) {
      throw new Error(`agent ${JSON.stringify(seed.id)} already exists`);
    }
    this.simulationAgents.set(seed.id, agent);
    this.state.agents.push({
      id: seed.id,
      name: seed.name,
      ...(seed.archetype ? { archetype: seed.archetype } : {}),
      position: { x: seed.position.x, z: seed.position.z },
      heading: seed.heading ?? 0,
      speed: seed.speed ?? 1.2,
      emotion: null,
      status: seed.status ?? '',
      intent: null,
      goal: null,
      holding: null,
      intentIssuedAtTick: -1,
    });
  }

  public removeAgent(agentId: string): void {
    this.simulationAgents.delete(agentId);
    this.state.agents = this.state.agents.filter((a) => a.id !== agentId);
    for (const obj of this.state.objects) {
      if (obj.heldBy === agentId) obj.heldBy = null;
    }
  }

  public upsertObject(seed: ScenarioObjectSeed): void {
    const existing = this.state.objects.find((o) => o.id === seed.id);
    if (existing) {
      existing.position = { x: seed.position.x, z: seed.position.z };
      return;
    }
    this.state.objects.push({
      id: seed.id,
      kind: seed.kind,
      label: seed.label ?? seed.kind,
      position: { x: seed.position.x, z: seed.position.z },
      contested: seed.contested ?? false,
      heldBy: null,
      tags: seed.tags ?? [],
      affordances: seed.affordances ?? [],
    });
  }

  public removeObject(objectId: string): void {
    this.state.objects = this.state.objects.filter((o) => o.id !== objectId);
    for (const agent of this.state.agents) {
      if (agent.holding === objectId) agent.holding = null;
    }
  }

  private async runTick(dtSeconds: number): Promise<void> {
    if (this.disposed) return;
    this.state.tick += 1;
    this.state.timeSeconds += dtSeconds;

    const controller = new AbortController();
    this.activeController = controller;
    const signal = controller.signal;

    try {
      this.emit({ kind: 'tick-start', tick: this.state.tick, timeSeconds: this.state.timeSeconds });

      await this.maybeQueryOneAgent(signal);
      if (signal.aborted) return;

      const { conflicts, arrivedAgentIds } = applyTickFirstPass(this.state, dtSeconds);
      this.clearSatisfiedGoals(arrivedAgentIds);

      if (conflicts.length > 0 && this.director) {
        this.emit({ kind: 'director-conflict', tick: this.state.tick, conflicts });
        const snapshotBefore = this.getSnapshot();
        const result = await this.director.query(this.resolveConflictQuery, {
          state: snapshotBefore as unknown as JsonValue,
          conflicts: conflicts as unknown as JsonValue,
        }, { signal });
        if (signal.aborted) return;
        const decision = coerceConflictDecision(result.data, conflicts);
        applyDirectorDecision(this.state, decision);
        this.clearSatisfiedGoals([]);
        this.emit({ kind: 'director-decision', tick: this.state.tick, decision });
        if (decision.note) {
          this.emit({ kind: 'world-note', tick: this.state.tick, note: decision.note });
        }
      } else if (
        this.director &&
        this.simulationAgents.size > 0 &&
        this.state.tick % this.directorCadenceTicks === 0
      ) {
        const snapshotBefore = this.getSnapshot();
        const result = await this.director.query(this.narrateQuery, {
          state: snapshotBefore as unknown as JsonValue,
        }, { signal });
        if (signal.aborted) return;
        const decision = coerceNarrationDecision(result.data, snapshotBefore);
        this.state.directorNote = decision.note || this.state.directorNote;
        this.emit({ kind: 'director-decision', tick: this.state.tick, decision });
        if (decision.note) {
          this.emit({ kind: 'world-note', tick: this.state.tick, note: decision.note });
        }
      }

      for (const agent of this.state.agents) {
        this.emit({ kind: 'agent-state', tick: this.state.tick, agent: cloneAgent(agent) });
        if (agent.emotion) {
          this.emit({ kind: 'agent-action', tick: this.state.tick, agentId: agent.id, emotion: agent.emotion });
        }
      }
      this.emit({ kind: 'tick-end', tick: this.state.tick, snapshot: this.getSnapshot() });
    } finally {
      if (this.activeController === controller) {
        this.activeController = null;
      }
    }
  }

  private async maybeQueryOneAgent(signal: AbortSignal): Promise<void> {
    if (this.simulationAgents.size === 0) return;
    const ids = Array.from(this.simulationAgents.keys());
    let chosenId: string | null = null;
    for (let i = 0; i < ids.length; i += 1) {
      const candidate = ids[(this.queryCursor + i) % ids.length]!;
      const agentState = this.state.agents.find((a) => a.id === candidate);
      if (!agentState) continue;
      if (!this.needsDecision(agentState)) continue;
      chosenId = candidate;
      this.queryCursor = (this.queryCursor + i + 1) % ids.length;
      break;
    }
    if (!chosenId) return;

    const agentState = this.state.agents.find((a) => a.id === chosenId);
    const agent = this.simulationAgents.get(chosenId);
    if (!agentState || !agent) return;

    const perception = buildPerception(
      agentState,
      this.state.agents,
      this.state.objects,
      this.state.tick,
      this.state.bounds,
      this.state.directorNote
    );

    this.emit({ kind: 'agent-query-start', tick: this.state.tick, agentId: chosenId });
    const result = await agent.query(perception, { signal });
    if (signal.aborted) return;

    agentState.goal = result.goal;
    const intent = this.mapGoalToIntent(result.goal);
    agentState.intent = intent;
    agentState.intentIssuedAtTick = this.state.tick;
    agentState.status = this.describeActiveGoal(result.goal);
    if (intent.emotion) {
      agentState.emotion = intent.emotion;
    }

    this.emit({
      kind: 'agent-query-end',
      tick: this.state.tick,
      agentId: chosenId,
      goal: result.goal,
      intent,
      status: agentState.status,
      emotion: intent.emotion ?? null,
      cancelled: result.cancelled,
      ...(result.errorMessage ? { errorMessage: result.errorMessage } : {}),
    });
    this.emit({
      kind: 'agent-intent',
      tick: this.state.tick,
      agentId: chosenId,
      goal: result.goal,
      intent,
      status: agentState.status,
    });
  }

  private mapGoalToIntent(goal: AgentGoal): AgentIntent {
    switch (goal.kind) {
      case 'wait':
        return { kind: 'wait', emotion: inferEmotionFromGoal(goal), reason: goal.label };
      case 'go_to_object': {
        const object = this.state.objects.find((entry) => entry.id === goal.objectId);
        if (!object) {
          return { kind: 'wait', emotion: inferEmotionFromGoal(goal), reason: 'missing-object' };
        }
        return { kind: 'move_to', target: object.position, emotion: inferEmotionFromGoal(goal) };
      }
      case 'go_to_agent':
        return { kind: 'approach_agent', agentId: goal.agentId, emotion: inferEmotionFromGoal(goal) };
      case 'object_action': {
        const object = this.state.objects.find((entry) => entry.id === goal.objectId);
        if (!object) {
          return { kind: 'wait', emotion: inferEmotionFromGoal(goal), reason: 'missing-object' };
        }
        if (goal.affordance.kind === 'pick_up') {
          return { kind: 'pick_up', objectId: goal.objectId, emotion: inferEmotionFromGoal(goal) };
        }
        return { kind: 'use', objectId: goal.objectId, emotion: inferEmotionFromGoal(goal) };
      }
      case 'drop':
        return { kind: 'drop', emotion: inferEmotionFromGoal(goal) };
    }
  }

  private needsDecision(agent: SimulationAgentState): boolean {
    if (!agent.goal) return true;
    if (!agent.intent) return true;
    return false;
  }

  private clearSatisfiedGoals(arrivedAgentIds: readonly string[]): void {
    const arrived = new Set(arrivedAgentIds);
    for (const agent of this.state.agents) {
      if (!agent.goal) continue;
      if (!agent.intent) {
        if (
          agent.goal.kind === 'object_action' ||
          agent.goal.kind === 'drop' ||
          agent.goal.kind === 'wait'
        ) {
          agent.status = this.describeCompletedGoal(agent);
          agent.goal = null;
        }
        continue;
      }
      if (arrived.has(agent.id)) {
        if (agent.goal.kind === 'go_to_object') {
          const chainedGoal = this.buildImmediateObjectActionGoal(agent.goal.objectId);
          if (chainedGoal) {
            const intent = this.mapGoalToIntent(chainedGoal);
            agent.goal = chainedGoal;
            agent.intent = intent;
            agent.intentIssuedAtTick = this.state.tick;
            agent.status = this.describeActiveGoal(chainedGoal);
            if (intent.emotion) {
              agent.emotion = intent.emotion;
            }
            this.emit({
              kind: 'agent-intent',
              tick: this.state.tick,
              agentId: agent.id,
              goal: chainedGoal,
              intent,
              status: agent.status,
            });
            continue;
          }
          agent.status = this.describeCompletedGoal(agent);
          agent.goal = null;
          agent.intent = null;
          continue;
        }
        if (agent.goal.kind === 'go_to_agent') {
          agent.status = this.describeCompletedGoal(agent);
          agent.goal = null;
          agent.intent = null;
        }
        continue;
      }
      if (this.isGoalInvalid(agent)) {
        agent.status = this.describeBlockedGoal(agent.goal);
        agent.goal = null;
        agent.intent = null;
      }
    }
  }

  private isGoalInvalid(agent: SimulationAgentState): boolean {
    const goal = agent.goal;
    if (!goal || !agent.intent) return false;
    switch (goal.kind) {
      case 'go_to_object':
      case 'object_action': {
        const object = this.state.objects.find((entry) => entry.id === goal.objectId);
        if (!object) return true;
        if (goal.kind === 'go_to_object' && object.heldBy && object.affordances.some((affordance) => affordance.kind === 'pick_up')) {
          return true;
        }
        if (goal.kind === 'object_action' && goal.affordance.kind === 'pick_up' && object.heldBy && object.heldBy !== agent.id) {
          return true;
        }
        return false;
      }
      case 'go_to_agent': {
        const target = this.state.agents.find((entry) => entry.id === goal.agentId);
        return !target;
      }
      default:
        return false;
    }
  }

  private buildImmediateObjectActionGoal(objectId: string): AgentGoal | null {
    const object = this.state.objects.find((entry) => entry.id === objectId);
    if (!object) return null;
    const affordance = object.affordances.find((entry) => {
      if (entry.kind === 'pick_up') {
        return object.heldBy == null;
      }
      return true;
    });
    if (!affordance) return null;
    return { kind: 'object_action', objectId, affordance, label: affordance.label };
  }

  private describeActiveGoal(goal: AgentGoal): string {
    switch (goal.kind) {
      case 'wait':
        return 'pausing to watch the courtyard';
      case 'go_to_object': {
        const object = this.state.objects.find((entry) => entry.id === goal.objectId);
        return object ? `heading to the ${object.label}` : goal.label;
      }
      case 'go_to_agent': {
        const target = this.state.agents.find((entry) => entry.id === goal.agentId);
        return target ? `approaching ${target.name}` : goal.label;
      }
      case 'object_action':
        return goal.affordance.status ?? goal.label;
      case 'drop':
        return goal.label.replace(/^drop /, 'putting down ');
    }
  }

  private describeCompletedGoal(agent: SimulationAgentState): string {
    const goal = agent.goal;
    if (!goal) return agent.status;
    switch (goal.kind) {
      case 'wait':
        return 'watching quietly';
      case 'go_to_object': {
        const object = this.state.objects.find((entry) => entry.id === goal.objectId);
        return object ? `arrived at the ${object.label}` : 'arrived';
      }
      case 'go_to_agent': {
        const target = this.state.agents.find((entry) => entry.id === goal.agentId);
        return target ? `reached ${target.name}` : 'arrived';
      }
      case 'object_action':
        if (goal.affordance.kind === 'pick_up') {
          const object = this.state.objects.find((entry) => entry.id === goal.objectId);
          if (object?.heldBy === agent.id) {
            return `picked up the ${object.label}`;
          }
          return object ? `missed the ${object.label}` : 'missed the target';
        }
        return goal.affordance.status ?? goal.label;
      case 'drop':
        return 'set something down';
    }
  }

  private describeBlockedGoal(goal: AgentGoal): string {
    switch (goal.kind) {
      case 'go_to_object':
      case 'object_action': {
        const object = this.state.objects.find((entry) => entry.id === goal.objectId);
        return object ? `lost access to the ${object.label}` : 'lost the target';
      }
      case 'go_to_agent':
        return 'lost sight of the target';
      case 'wait':
      case 'drop':
        return 'reconsidering';
    }
  }

  private emit(event: SimulationEvent): void {
    this.bus.emit(event);
  }
}

function coerceConflictDecision(
  value: JsonValue | null,
  conflicts: readonly WorldConflict[]
): DirectorDecision {
  if (isRecord(value) && typeof value.note === 'string' && Array.isArray(value.resolutions)) {
    const resolutions: DirectorResolution[] = [];
    for (const entry of value.resolutions) {
      if (!isRecord(entry)) continue;
      if (typeof entry.objectId !== 'string') continue;
      const winnerAgentId =
        entry.winnerAgentId === null || typeof entry.winnerAgentId === 'string'
          ? entry.winnerAgentId
          : null;
      const note = typeof entry.note === 'string' && entry.note.length > 0 ? entry.note : undefined;
      resolutions.push({ objectId: entry.objectId, winnerAgentId, ...(note ? { note } : {}) });
    }
    if (resolutions.length === conflicts.length) {
      return { note: value.note, resolutions };
    }
  }
  return deterministicConflictResolution(conflicts);
}

function coerceNarrationDecision(value: JsonValue | null, snapshot: WorldSnapshot): DirectorDecision {
  if (isRecord(value) && typeof value.note === 'string') {
    return { note: value.note, resolutions: [] };
  }
  return { note: `Tick ${snapshot.tick}: the courtyard carries on.`, resolutions: [] };
}

function deterministicConflictResolution(conflicts: readonly WorldConflict[]): DirectorDecision {
  return {
    note: 'Director fell back to deterministic tie-break.',
    resolutions: conflicts.map((conflict) => ({
      objectId: conflict.objectId,
      winnerAgentId: conflict.contenderAgentIds[0] ?? null,
      note: 'first-come tie break',
    })),
  };
}

function isRecord(value: JsonValue | null): value is Record<string, JsonValue> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function cloneAgent(agent: SimulationAgentState): SimulationAgentState {
  return {
    id: agent.id,
    name: agent.name,
    ...(agent.archetype ? { archetype: agent.archetype } : {}),
    position: { x: agent.position.x, z: agent.position.z },
    heading: agent.heading,
    speed: agent.speed,
    emotion: agent.emotion,
    status: agent.status,
    intent: agent.intent ? { ...agent.intent } : null,
    goal: agent.goal ? { ...agent.goal } : null,
    holding: agent.holding,
    intentIssuedAtTick: agent.intentIssuedAtTick,
  };
}

function cloneObject(obj: SimulationObjectState): SimulationObjectState {
  return {
    id: obj.id,
    kind: obj.kind,
    label: obj.label,
    position: { x: obj.position.x, z: obj.position.z },
    contested: obj.contested,
    heldBy: obj.heldBy,
    tags: obj.tags,
    affordances: obj.affordances,
  };
}

function inferEmotionFromGoal(goal: AgentGoal): string {
  switch (goal.kind) {
    case 'wait':
      return 'thinking';
    case 'drop':
      return 'alert';
    case 'go_to_agent':
      return 'curious';
    case 'go_to_object':
      return 'alert';
    case 'object_action':
      return goal.affordance.kind === 'pick_up' ? 'happy' : 'curious';
  }
}
