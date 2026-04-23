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
    this.bus.clear();
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
      position: { x: seed.position.x, z: seed.position.z },
      contested: seed.contested ?? false,
      heldBy: null,
      tags: seed.tags ?? [],
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

      const { conflicts } = applyTickFirstPass(this.state, dtSeconds);

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
      if (agentState.intent) continue;
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

    agentState.intent = result.intent;
    agentState.intentIssuedAtTick = this.state.tick;
    if (result.intent.kind === 'wait' && result.intent.reason) {
      agentState.status = result.intent.reason;
    }
    if (result.intent.emotion) {
      agentState.emotion = result.intent.emotion;
    }

    this.emit({
      kind: 'agent-query-end',
      tick: this.state.tick,
      agentId: chosenId,
      intent: result.intent,
      emotion: result.intent.emotion ?? null,
      cancelled: result.cancelled,
      ...(result.errorMessage ? { errorMessage: result.errorMessage } : {}),
    });
    this.emit({
      kind: 'agent-intent',
      tick: this.state.tick,
      agentId: chosenId,
      intent: result.intent,
    });
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
    holding: agent.holding,
    intentIssuedAtTick: agent.intentIssuedAtTick,
  };
}

function cloneObject(obj: SimulationObjectState): SimulationObjectState {
  return {
    id: obj.id,
    kind: obj.kind,
    position: { x: obj.position.x, z: obj.position.z },
    contested: obj.contested,
    heldBy: obj.heldBy,
    tags: obj.tags,
  };
}
