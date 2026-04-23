//////////////////////////////////////////////////////////////////////////////
//
// world-orchestrator.ts
//
// - The "game brain". Owns the world state and drives a fixed-rate tick
//   loop that:
//     1. Emits `tick-start`.
//     2. Optionally queries at most one agent (scheduled round-robin).
//     3. Runs the reducer (movement, interactions, conflict detection).
//     4. Queries the director when conflicts exist or on cadence ticks.
//     5. Applies the director decision to state.
//     6. Emits `tick-end` with an immutable snapshot.
//
//   Ticks never overlap: if LLM inference runs past the nominal tick
//   interval, the next tick simply waits.
//
//////////////////////////////////////////////////////////////////////////////

import { SimulationBus, type SimulationEvent } from './simulation-bus.js';
import { buildPerception } from './sensing.js';
import { SimulationAgent } from './simulation-agent.js';
import { WorldDirector } from './world-director.js';
import {
  applyDirectorDecision,
  applyTickFirstPass,
  type MutableWorldState,
} from './world-reducer.js';
import type {
  ScenarioAgentSeed,
  ScenarioObjectSeed,
  SimulationAgentState,
  SimulationObjectState,
  WorldBounds,
  WorldSnapshot,
} from './simulation-types.js';

export interface WorldOrchestratorOptions {
  /** Stable identifier used to scope engine context keys for the director. */
  readonly id?: string;
  /** Initial world bounds. Default = half-extent 8 (16×16 world). */
  readonly bounds?: WorldBounds;
  /** Ticks per second. Default = 1.5. */
  readonly tickHz?: number;
  /** Ticks between director narration queries. Default = 10. */
  readonly directorCadenceTicks?: number;
  /** Initial director note shown before the first tick completes. */
  readonly initialDirectorNote?: string | null;
  /** Pre-built event bus to share with external consumers. */
  readonly bus?: SimulationBus;
}

export interface AttachedSimulationAgent {
  readonly agent: SimulationAgent;
  readonly seed: ScenarioAgentSeed;
}

export class WorldOrchestrator {
  public readonly bus: SimulationBus;
  public readonly id: string;

  private readonly state: MutableWorldState;
  private readonly simulationAgents: Map<string, SimulationAgent> = new Map();
  private readonly director: WorldDirector | null;
  private readonly directorCadenceTicks: number;

  private tickHz: number;
  private running = false;
  private disposed = false;
  private tickTimer: ReturnType<typeof setTimeout> | null = null;
  private inFlightTick: Promise<void> | null = null;
  private queryCursor = 0;
  private activeController: AbortController | null = null;

  public constructor(director: WorldDirector | null, options: WorldOrchestratorOptions = {}) {
    this.id = options.id ?? 'world';
    this.bus = options.bus ?? new SimulationBus();
    this.tickHz = clampHz(options.tickHz ?? 1.5);
    this.directorCadenceTicks = Math.max(1, Math.floor(options.directorCadenceTicks ?? 10));
    this.director = director;
    this.state = {
      tick: 0,
      timeSeconds: 0,
      bounds: options.bounds ?? { halfExtent: 8 },
      agents: [],
      objects: [],
      directorNote: options.initialDirectorNote ?? null,
    };
  }

  // ---------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------

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

  public getTickHz(): number {
    return this.tickHz;
  }

  public setTickHz(hz: number): void {
    this.tickHz = clampHz(hz);
  }

  public isRunning(): boolean {
    return this.running;
  }

  public start(): void {
    if (this.disposed || this.running) return;
    this.running = true;
    this.scheduleNextTick(0);
  }

  public pause(): void {
    this.running = false;
    if (this.tickTimer !== null) {
      clearTimeout(this.tickTimer);
      this.tickTimer = null;
    }
  }

  /** Runs exactly one tick. Resolves after the tick fully completes. */
  public async step(): Promise<void> {
    if (this.disposed) return;
    if (this.inFlightTick) {
      await this.inFlightTick;
      return;
    }
    this.inFlightTick = this.runTick().finally(() => {
      this.inFlightTick = null;
    });
    await this.inFlightTick;
  }

  public async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.running = false;
    if (this.tickTimer !== null) {
      clearTimeout(this.tickTimer);
      this.tickTimer = null;
    }
    if (this.activeController) {
      this.activeController.abort();
    }
    if (this.inFlightTick) {
      try {
        await this.inFlightTick;
      } catch {
        // ignore
      }
    }
    this.bus.clear();
  }

  public addAgent(agent: SimulationAgent, seed: ScenarioAgentSeed): void {
    if (this.simulationAgents.has(seed.id)) {
      throw new Error(`agent "${seed.id}" already exists`);
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
    // Release any objects held by this agent.
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

  // ---------------------------------------------------------------------
  // Tick loop
  // ---------------------------------------------------------------------

  private scheduleNextTick(delayMs: number): void {
    if (!this.running || this.disposed) return;
    this.tickTimer = setTimeout(() => {
      this.tickTimer = null;
      if (!this.running || this.disposed) return;
      if (this.inFlightTick) {
        // Previous tick is still running; try again soon.
        this.scheduleNextTick(1);
        return;
      }
      const tickStart = performance.now();
      this.inFlightTick = this.runTick()
        .catch(() => {
          // Errors are surfaced via bus events; the loop keeps going.
        })
        .finally(() => {
          this.inFlightTick = null;
          const elapsedMs = performance.now() - tickStart;
          const intervalMs = 1000 / this.tickHz;
          const next = Math.max(0, intervalMs - elapsedMs);
          this.scheduleNextTick(next);
        });
    }, delayMs);
  }

  private async runTick(): Promise<void> {
    if (this.disposed) return;
    const dt = 1 / this.tickHz;
    this.state.tick += 1;
    this.state.timeSeconds += dt;

    const controller = new AbortController();
    this.activeController = controller;
    const signal = controller.signal;

    try {
      this.emit({ kind: 'tick-start', tick: this.state.tick, timeSeconds: this.state.timeSeconds });

      // 1. Optionally query one agent (round-robin, only agents that lack an intent).
      await this.maybeQueryOneAgent(signal);

      if (signal.aborted) return;

      // 2. Reduce physics + interactions.
      const { conflicts } = applyTickFirstPass(this.state, dt);

      // 3. Resolve conflicts if any.
      if (conflicts.length > 0 && this.director) {
        this.emit({ kind: 'director-conflict', tick: this.state.tick, conflicts });
        const snapshotBefore = this.getSnapshot();
        const result = await this.director.resolveConflicts(snapshotBefore, conflicts, {
          signal,
        });
        if (signal.aborted) return;
        applyDirectorDecision(this.state, result.decision);
        this.emit({
          kind: 'director-decision',
          tick: this.state.tick,
          decision: result.decision,
        });
        if (result.decision.note) {
          this.emit({ kind: 'world-note', tick: this.state.tick, note: result.decision.note });
        }
      } else if (
        this.director &&
        this.state.tick % this.directorCadenceTicks === 0 &&
        this.simulationAgents.size > 0
      ) {
        // 4. Cadence narration.
        const snapshotBefore = this.getSnapshot();
        const result = await this.director.narrate(snapshotBefore, { signal });
        if (signal.aborted) return;
        this.state.directorNote = result.decision.note || this.state.directorNote;
        this.emit({
          kind: 'director-decision',
          tick: this.state.tick,
          decision: result.decision,
        });
        if (result.decision.note) {
          this.emit({ kind: 'world-note', tick: this.state.tick, note: result.decision.note });
        }
      }

      // 5. Emit per-agent state events + tick end.
      for (const agent of this.state.agents) {
        this.emit({ kind: 'agent-state', tick: this.state.tick, agent: cloneAgent(agent) });
        if (agent.emotion) {
          this.emit({
            kind: 'agent-action',
            tick: this.state.tick,
            agentId: agent.id,
            emotion: agent.emotion,
          });
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
    // Round-robin starting at queryCursor; pick the first agent without an active intent.
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

    agentState.intent = result.output.intent;
    agentState.intentIssuedAtTick = this.state.tick;
    if (result.output.status.length > 0) {
      agentState.status = result.output.status;
    }
    if (result.output.intent.emotion) {
      agentState.emotion = result.output.intent.emotion;
    }

    this.emit({
      kind: 'agent-query-end',
      tick: this.state.tick,
      agentId: chosenId,
      intent: result.output.intent,
      emotion: result.output.intent.emotion ?? null,
      cancelled: result.cancelled,
      ...(result.errorMessage ? { errorMessage: result.errorMessage } : {}),
    });
    this.emit({
      kind: 'agent-intent',
      tick: this.state.tick,
      agentId: chosenId,
      intent: result.output.intent,
    });
  }

  private emit(event: SimulationEvent): void {
    this.bus.emit(event);
  }
}

function clampHz(hz: number): number {
  if (!Number.isFinite(hz) || hz <= 0) return 1.5;
  return Math.max(0.25, Math.min(20, hz));
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
