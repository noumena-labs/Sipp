import type { DirectorChoice, DirectorRuntime, JsonValue } from 'cogent-engine/orchestrator';
import { SimulationBus, type SimulationEvent } from './bus.js';
import { buildPerception, vec2Distance } from './sensing.js';
import { SimulationAgentChooser } from './agent-chooser.js';
import {
  applyDirectorDecision,
  applyTickFirstPass,
  cloneScore,
  deterministicConflictResolution,
  FORCED_DROP_HISTORY_LIMIT,
  SABOTAGE_COOLDOWN_TICKS,
  SABOTAGE_RADIUS,
  type MutableGameState,
  type MutableWorldState,
} from './reducer.js';
import type {
  AgentGoal,
  DirectorDecision,
  DirectorResolution,
  ForcedDropOutcome,
  ForcedDropRulingRecord,
  ScenarioAgentSeed,
  ScenarioGameSeed,
  ScenarioObjectSeed,
  SimulationAgentState,
  SimulationGameEvent,
  SimulationGameState,
  SimulationObjectState,
  WorldBounds,
  WorldConflict,
  WorldSnapshot,
} from './types.js';

const FORCED_DROP_OUTCOMES: readonly ForcedDropOutcome[] = ['drop', 'hold', 'attacker_fumbles'];
const FORCED_DROP_REPEAT_SUPPRESSION_COUNT = 3;

interface ForcedDropPolicy {
  readonly availableOutcomes: readonly ForcedDropOutcome[];
  readonly suppressedOutcomes: readonly ForcedDropOutcome[];
  readonly fallbackOutcome: ForcedDropOutcome;
  readonly samePairHistory: readonly ForcedDropRulingRecord[];
  readonly recentHistory: readonly ForcedDropRulingRecord[];
  readonly suppressionNotes: readonly string[];
  readonly varietyNote: string;
}

export interface SimulationRuntimeOptions {
  readonly id?: string;
  readonly bounds?: WorldBounds;
  readonly game: ScenarioGameSeed;
  readonly initialDirectorNote?: string | null;
  readonly directorCadenceTicks?: number;
  readonly resolveRefereeTask?: string;
  readonly narrateTask?: string;
  readonly refereeTimeoutMs?: number;
  readonly narrationTimeoutMs?: number;
  readonly agentQueryTimeoutMs?: number;
  readonly bus?: SimulationBus;
}

interface AgentQueryInFlight {
  readonly agentId: string;
  readonly controller: AbortController;
  readonly done: Promise<void>;
}

export class SimulationRuntime {
  public readonly bus: SimulationBus;
  public readonly id: string;

  private readonly state: MutableWorldState;
  private readonly simulationAgents: Map<string, SimulationAgentChooser> = new Map();
  private readonly director: DirectorRuntime;
  private readonly directorCadenceTicks: number;
  private readonly resolveRefereeTask: string;
  private readonly narrateTask: string;
  private readonly refereeTimeoutMs: number;
  private readonly narrationTimeoutMs: number;
  private readonly agentQueryTimeoutMs: number;
  private readonly activeControllers: Set<AbortController> = new Set();
  private readonly movementSubstepSeconds = 0.15;

  private disposed = false;
  private inFlightTick: Promise<void> | null = null;
  private agentQueryInFlight: AgentQueryInFlight | null = null;
  private narrationInFlight: Promise<void> | null = null;
  private narrationController: AbortController | null = null;
  private queryCursor = 0;

  public constructor(director: DirectorRuntime, options: SimulationRuntimeOptions) {
    this.id = options.id ?? 'simulation';
    this.bus = options.bus ?? new SimulationBus();
    this.director = director;
    this.directorCadenceTicks = Math.max(1, Math.floor(options.directorCadenceTicks ?? 12));
    this.resolveRefereeTask = options.resolveRefereeTask ?? 'resolve_referee_event';
    this.narrateTask = options.narrateTask ?? 'narrate_scene';
    this.refereeTimeoutMs = Math.max(1000, Math.floor(options.refereeTimeoutMs ?? 30000));
    this.narrationTimeoutMs = Math.max(1000, Math.floor(options.narrationTimeoutMs ?? 15000));
    this.agentQueryTimeoutMs = Math.max(1000, Math.floor(options.agentQueryTimeoutMs ?? 30000));
    this.state = {
      tick: 0,
      timeSeconds: 0,
      bounds: options.bounds ?? { halfExtent: 8 },
      agents: [],
      objects: [],
      directorNote: options.initialDirectorNote ?? null,
      game: createGameState(options.game),
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
      game: cloneGame(this.state.game),
    };
  }

  public async step(dtSeconds: number): Promise<void> {
    if (this.disposed) return;
    if (this.inFlightTick) {
      await this.inFlightTick;
      return;
    }
    this.inFlightTick = this.runStep(dtSeconds).finally(() => {
      this.inFlightTick = null;
    });
    await this.inFlightTick;
  }

  public isBusy(): boolean {
    return this.agentQueryInFlight != null || this.state.game.referee.status === 'ruling';
  }

  public async waitForIdle(): Promise<void> {
    while (!this.disposed && this.isBusy()) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  }

  public async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    for (const controller of this.activeControllers) {
      controller.abort();
    }
    if (this.agentQueryInFlight) {
      this.agentQueryInFlight.controller.abort();
      await this.agentQueryInFlight.done.catch(() => undefined);
    }
    if (this.inFlightTick) {
      await this.inFlightTick.catch(() => undefined);
    }
    if (this.narrationInFlight) {
      await this.narrationInFlight.catch(() => undefined);
    }
    this.bus.clear();
  }

  public addAgent(agent: SimulationAgentChooser, seed: ScenarioAgentSeed): void {
    if (this.simulationAgents.has(seed.id)) {
      throw new Error(`agent ${JSON.stringify(seed.id)} already exists`);
    }
    this.simulationAgents.set(seed.id, agent);
    this.state.game.score.deliveries[seed.id] = 0;
    this.state.game.score.forcedDrops[seed.id] = 0;
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
      powerUp: null,
      frozenUntilTick: 0,
      intentIssuedAtTick: -1,
      thinking: false,
      cooldowns: {
        sabotageUntilTick: 0,
      },
      navigation: {
        detourTarget: null,
        blockedTicks: 0,
        obstacleId: null,
      },
    });
  }

  public removeAgent(agentId: string): void {
    this.simulationAgents.delete(agentId);
    this.state.agents = this.state.agents.filter((a) => a.id !== agentId);
    delete this.state.game.score.deliveries[agentId];
    delete this.state.game.score.forcedDrops[agentId];
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
      description: seed.description ?? seed.label ?? seed.kind,
      position: { x: seed.position.x, z: seed.position.z },
      active: seed.active ?? true,
      contested: seed.contested ?? false,
      heldBy: null,
      tags: seed.tags ?? [],
      affordances: seed.affordances ?? [],
      blocksMovement: seed.blocksMovement ?? false,
      collisionRadius: seed.collisionRadius ?? 0.45,
    });
  }

  public removeObject(objectId: string): void {
    this.state.objects = this.state.objects.filter((o) => o.id !== objectId);
    for (const agent of this.state.agents) {
      if (agent.holding === objectId) agent.holding = null;
    }
  }

  private async runStep(dtSeconds: number): Promise<void> {
    let remaining = dtSeconds;
    while (remaining > 1e-6) {
      const substep = Math.min(remaining, this.movementSubstepSeconds);
      await this.runTick(substep);
      remaining -= substep;
      if (this.disposed || this.isBusy()) {
        return;
      }
    }
  }

  private async runTick(dtSeconds: number): Promise<void> {
    if (this.disposed) return;
    this.state.tick += 1;
    this.state.timeSeconds += dtSeconds;

    this.emit({ kind: 'tick-start', tick: this.state.tick, timeSeconds: this.state.timeSeconds });

    const { conflicts, arrivedAgentIds, events } = applyTickFirstPass(this.state, dtSeconds);
    this.emitGameEvents(events);
    this.clearSatisfiedGoals(arrivedAgentIds);

    if (conflicts.length > 0) {
      await this.handleRefereeConflict(conflicts[0]!);
    } else {
      this.maybeStartOneAgentQuery();
      this.maybeStartNarration();
    }

    for (const agent of this.state.agents) {
      this.emit({ kind: 'agent-state', tick: this.state.tick, agent: cloneAgent(agent) });
      if (agent.emotion) {
        this.emit({ kind: 'agent-action', tick: this.state.tick, agentId: agent.id, emotion: agent.emotion });
      }
    }
    this.emit({ kind: 'tick-end', tick: this.state.tick, snapshot: this.getSnapshot() });
  }

  private async handleRefereeConflict(conflict: WorldConflict): Promise<void> {
    if (this.agentQueryInFlight) {
      this.agentQueryInFlight.controller.abort();
      await this.agentQueryInFlight.done.catch(() => undefined);
    }

    this.cancelNarration();

    this.state.game.referee = { status: 'ruling', conflict, startedAtTick: this.state.tick };
    this.emit({ kind: 'director-conflict', tick: this.state.tick, conflicts: [conflict] });
    this.emit({
      kind: 'game-event',
      tick: this.state.tick,
      event: { kind: 'fallback', message: `Director is ruling on ${describeConflict(conflict)}.` },
    });

    let appliedDecision = false;
    try {
      const decision = await this.queryRefereeWithTimeout(conflict);
      if (decision) {
        const events = applyDirectorDecision(this.state, decision);
        this.emit({ kind: 'director-decision', tick: this.state.tick, decision });
        if (decision.note) {
          this.emit({ kind: 'world-note', tick: this.state.tick, note: decision.note });
        }
        this.emitGameEvents(events);
        appliedDecision = true;
      }
    } finally {
      this.state.game.referee = { status: 'idle' };
      if (appliedDecision) {
        this.maybeStartOneAgentQuery();
      }
    }
  }

  private async queryRefereeWithTimeout(conflict: WorldConflict): Promise<DirectorDecision | null> {
    const controller = new AbortController();
    this.activeControllers.add(controller);
    try {
      try {
        const choices = buildRefereeChoices(this.state, conflict);
        const result = await this.director.run<DirectorResolution>(this.resolveRefereeTask, {
          inputs: buildRefereePayload(this.state, conflict) as Record<string, JsonValue>,
          choices,
          signal: controller.signal,
          timeoutMs: this.refereeTimeoutMs,
        });
        if (result.status === 'aborted') {
          return null;
        }
        if (result.status !== 'ok') {
          return this.fallbackRefereeDecision(
            conflict,
            `Director referee task failed: ${result.errorMessage ?? result.status}`
          );
        }
        const resolution = result.selections[0]?.payload;
        if (!resolution) {
          return this.fallbackRefereeDecision(
            conflict,
            'Director referee task did not select a valid ruling.'
          );
        }
        return {
          note: describeRefereeResolution(this.state, conflict, resolution),
          resolutions: [{ ...resolution, note: resolution.note ?? describeResolutionNote(resolution) }],
        };
      } catch (error) {
        if (controller.signal.aborted || this.disposed) {
          return null;
        }
        return this.fallbackRefereeDecision(
          conflict,
          `Director referee task failed: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } finally {
      this.activeControllers.delete(controller);
    }
  }

  private fallbackRefereeDecision(conflict: WorldConflict, message: string): DirectorDecision {
    this.emitRuntimeIssue({
      severity: 'warning',
      source: 'referee',
      message: `${message}; applying deterministic house-rule fallback.`,
      conflictId: conflict.id,
      taskName: this.resolveRefereeTask,
    });
    if (conflict.kind === 'forced_drop') {
      const policy = buildForcedDropPolicy(this.state, conflict);
      return {
        note: 'The referee uses a quick house-rule ruling.',
        resolutions: [
          {
            conflictId: conflict.id,
            objectId: conflict.objectId,
            winnerAgentId: null,
            outcome: policy.fallbackOutcome,
            note: `fallback ${policy.fallbackOutcome}`,
          },
        ],
      };
    }
    return deterministicConflictResolution(this.state, [conflict]);
  }

  private maybeStartOneAgentQuery(): void {
    if (this.disposed || this.agentQueryInFlight) return;
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
      this.state.directorNote,
      cloneGame(this.state.game)
    );

    const controller = new AbortController();
    agentState.thinking = true;
    this.emit({ kind: 'agent-query-start', tick: this.state.tick, agentId: chosenId });
    const done = agent.query(perception, {
      signal: controller.signal,
      timeoutMs: this.agentQueryTimeoutMs,
    }).then((result) => {
      if (this.disposed || controller.signal.aborted) return;
      const current = this.state.agents.find((a) => a.id === chosenId);
      if (!current) return;
      if (result.status !== 'ok' || !result.goal) {
        current.thinking = false;
        current.status = 'decision query failed';
        this.emit({
          kind: 'agent-query-end',
          tick: this.state.tick,
          agentId: chosenId,
          goal: null,
          intent: null,
          status: current.status,
          emotion: null,
          queryStatus: result.status,
          ...(result.errorMessage ? { errorMessage: result.errorMessage } : {}),
        });
        this.emitRuntimeIssue({
          severity: 'critical',
          source: 'agent',
          message: `${current.name} decision query failed: ${result.errorMessage ?? result.status}`,
          agentId: chosenId,
        });
        return;
      }
      current.goal = result.goal;
      const intent = this.mapGoalToIntent(result.goal);
      current.intent = intent;
      current.intentIssuedAtTick = this.state.tick;
      current.status = result.goal.label;
      current.thinking = false;
      current.emotion = intent.emotion;
      this.emit({
        kind: 'agent-query-end',
        tick: this.state.tick,
        agentId: chosenId,
        goal: result.goal,
        intent,
        status: result.goal.label,
        emotion: intent.emotion,
        queryStatus: result.status,
        ...(result.errorMessage ? { errorMessage: result.errorMessage } : {}),
      });
      this.emit({
        kind: 'agent-intent',
        tick: this.state.tick,
        agentId: chosenId,
        goal: result.goal,
        intent,
        status: result.goal.label,
      });
    }).catch((error) => {
      if (!controller.signal.aborted) {
        this.emit({
          kind: 'agent-query-end',
          tick: this.state.tick,
          agentId: chosenId,
          goal: null,
          intent: null,
          status: 'query failed',
          emotion: null,
          queryStatus: 'failed',
          errorMessage: error instanceof Error ? error.message : String(error),
        });
        this.emitRuntimeIssue({
          severity: 'critical',
          source: 'agent',
          message: `${chosenId} decision query failed: ${error instanceof Error ? error.message : String(error)}`,
          agentId: chosenId,
        });
      }
    }).finally(() => {
      const current = this.state.agents.find((a) => a.id === chosenId);
      if (current) current.thinking = false;
      if (this.agentQueryInFlight?.agentId === chosenId) {
        this.agentQueryInFlight = null;
      }
    });

    this.agentQueryInFlight = { agentId: chosenId, controller, done };
  }

  private maybeStartNarration(): void {
    if (this.agentQueryInFlight || this.narrationInFlight) return;
    if (this.simulationAgents.size === 0 || this.state.tick % this.directorCadenceTicks !== 0) return;

    const controller = new AbortController();
    this.narrationController = controller;
    this.activeControllers.add(controller);
    this.narrationInFlight = this.director.run(this.narrateTask, {
      inputs: buildNarrationPayload(this.state) as Record<string, JsonValue>,
      signal: controller.signal,
      timeoutMs: this.narrationTimeoutMs,
    }).then((result) => {
      if (this.disposed || controller.signal.aborted) return;
      if (result.status !== 'ok') {
        this.emitRuntimeIssue({
          severity: 'warning',
          source: 'narration',
          message: `Director narration task failed: ${result.errorMessage ?? result.status}`,
          taskName: this.narrateTask,
        });
        return;
      }
      const decision = coerceNarrationDecision(result.text, this.getSnapshot());
      this.state.directorNote = decision.note || this.state.directorNote;
      this.emit({ kind: 'director-decision', tick: this.state.tick, decision });
      if (decision.note) {
        this.emit({ kind: 'world-note', tick: this.state.tick, note: decision.note });
      }
    }).finally(() => {
      this.activeControllers.delete(controller);
      if (this.narrationController === controller) {
        this.narrationController = null;
      }
      this.narrationInFlight = null;
    });
  }

  private cancelNarration(): void {
    this.narrationController?.abort();
  }

  private mapGoalToIntent(goal: AgentGoal): import('./types.js').AgentIntent {
    switch (goal.kind) {
      case 'wait':
        return { kind: 'wait', emotion: inferEmotionFromGoal(goal), reason: goal.label };
      case 'go_to_object': {
        const object = this.state.objects.find((entry) => entry.id === goal.objectId);
        if (!object) return { kind: 'wait', emotion: inferEmotionFromGoal(goal), reason: 'missing-object' };
        return { kind: 'go_to_object', objectId: object.id, emotion: inferEmotionFromGoal(goal) };
      }
      case 'go_to_agent':
        return { kind: 'approach_agent', agentId: goal.agentId, emotion: inferEmotionFromGoal(goal) };
      case 'object_action': {
        const object = this.state.objects.find((entry) => entry.id === goal.objectId);
        if (!object) return { kind: 'wait', emotion: inferEmotionFromGoal(goal), reason: 'missing-object' };
        if (goal.affordance.kind === 'pick_up') {
          return { kind: 'pick_up', objectId: goal.objectId, emotion: inferEmotionFromGoal(goal) };
        }
        return { kind: 'use', objectId: goal.objectId, emotion: inferEmotionFromGoal(goal) };
      }
      case 'deliver':
        return { kind: 'deliver', objectId: goal.objectId, emotion: inferEmotionFromGoal(goal) };
      case 'sabotage_agent':
        return { kind: 'sabotage', agentId: goal.agentId, method: goal.method, emotion: inferEmotionFromGoal(goal) };
      case 'drop':
        return { kind: 'drop', emotion: inferEmotionFromGoal(goal) };
    }
  }

  private needsDecision(agent: SimulationAgentState): boolean {
    if (agent.thinking) return false;
    if (agent.frozenUntilTick > this.state.tick) return false;
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
          agent.goal.kind === 'wait' ||
          agent.goal.kind === 'deliver' ||
          agent.goal.kind === 'sabotage_agent'
        ) {
          agent.goal = null;
        }
        continue;
      }
      if (arrived.has(agent.id)) {
        if (agent.goal.kind === 'go_to_object' || agent.goal.kind === 'go_to_agent') {
          agent.goal = null;
          agent.intent = null;
        }
        continue;
      }
      if (this.isGoalInvalid(agent)) {
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
        if (!object.active) return true;
        if (goal.kind === 'go_to_object' && object.heldBy && object.id === this.state.game.bananaObjectId) {
          return object.heldBy !== agent.id;
        }
        if (goal.kind === 'object_action' && goal.affordance.kind === 'pick_up' && object.heldBy && object.heldBy !== agent.id) {
          return true;
        }
        return false;
      }
      case 'deliver':
        return agent.holding !== this.state.game.bananaObjectId;
      case 'sabotage_agent': {
        const target = this.state.agents.find((entry) => entry.id === goal.agentId);
        if (!target || target.holding !== this.state.game.bananaObjectId) return true;
        if (goal.method !== 'bump' && agent.powerUp?.kind !== goal.method) return true;
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

  private emitGameEvents(events: readonly SimulationGameEvent[]): void {
    for (const event of events) {
      this.emit({ kind: 'game-event', tick: this.state.tick, event });
    }
  }

  private emit(event: SimulationEvent): void {
    this.bus.emit(event);
  }

  private emitRuntimeIssue(args: {
    readonly severity: 'critical' | 'warning';
    readonly source: 'agent' | 'referee' | 'narration';
    readonly message: string;
    readonly agentId?: string;
    readonly conflictId?: string;
    readonly taskName?: string;
  }): void {
    const prefix = `[SimulationRuntime] ${args.source} ${args.severity}`;
    if (args.severity === 'critical') {
      console.error(`${prefix}: ${args.message}`);
    } else {
      console.warn(`${prefix}: ${args.message}`);
    }
    this.emit({ kind: 'runtime-error', tick: this.state.tick, ...args });
  }
}

function createGameState(seed: ScenarioGameSeed): MutableGameState {
  return {
    title: seed.title,
    bananaObjectId: seed.bananaObjectId,
    goalObjectId: seed.goalObjectId,
    respawnRules: seed.respawnRules.map((rule) => ({
      objectId: rule.objectId,
      delayTicks: rule.delayTicks,
      spawnPoints: rule.spawnPoints.map((point) => ({ x: point.x, z: point.z })),
    })),
    score: { deliveries: {}, forcedDrops: {} },
    referee: { status: 'idle' },
    refereeMemory: { forcedDrops: [] },
    pendingRespawns: [],
    nextSpawnIndexByObjectId: {},
  };
}

function cloneGame(game: MutableGameState): SimulationGameState {
  return {
    title: game.title,
    bananaObjectId: game.bananaObjectId,
    goalObjectId: game.goalObjectId,
    respawnRules: game.respawnRules.map((rule) => ({
      objectId: rule.objectId,
      delayTicks: rule.delayTicks,
      spawnPoints: rule.spawnPoints.map((point) => ({ x: point.x, z: point.z })),
    })),
    score: cloneScore(game.score),
    referee: game.referee,
    refereeMemory: {
      forcedDrops: game.refereeMemory.forcedDrops.map(cloneForcedDropRulingRecord),
    },
    pendingRespawns: game.pendingRespawns.map((pending) => ({
      objectId: pending.objectId,
      spawnPosition: { x: pending.spawnPosition.x, z: pending.spawnPosition.z },
      activateAtTick: pending.activateAtTick,
    })),
  };
}

export function buildRefereePayload(state: MutableWorldState, conflict: WorldConflict): JsonValue {
  return {
    referee_event: summarizeConflict(state, conflict),
    scoreboard: state.game.score.deliveries,
    scene_summary: buildSceneSummary(state),
  };
}

function buildNarrationPayload(state: MutableWorldState): JsonValue {
  return {
    scoreboard: state.game.score.deliveries,
    scene_summary: buildSceneSummary(state),
  };
}

function summarizeConflict(state: MutableWorldState, conflict: WorldConflict): JsonValue {
  if (conflict.kind === 'contested_object') {
    const object = state.objects.find((entry) => entry.id === conflict.objectId);
    return {
      conflictId: conflict.id,
      kind: conflict.kind,
      objectId: conflict.objectId,
      objectLabel: object?.label ?? conflict.objectId,
      contenders: conflict.contenderAgentIds.map((id) => summarizeAgent(state, id)),
    };
  }
  const policy = buildForcedDropPolicy(state, conflict);
  return {
    conflictId: conflict.id,
    kind: conflict.kind,
    objectId: conflict.objectId,
    attacker: summarizeAgent(state, conflict.attackerAgentId),
    target: summarizeAgent(state, conflict.targetAgentId),
    attempt: summarizeForcedDropAttempt(state, conflict),
    recent_history: {
      same_pair: policy.samePairHistory.map(summarizeForcedDropRuling),
      recent_forced_drops: policy.recentHistory.map(summarizeForcedDropRuling),
    },
    ruling_policy: {
      availableOutcomes: policy.availableOutcomes,
      suppressedOutcomes: policy.suppressedOutcomes,
      suppressionNotes: policy.suppressionNotes,
      fallbackOutcome: policy.fallbackOutcome,
      varietyNote: policy.varietyNote,
      sabotageCooldownTicksAfterRuling: SABOTAGE_COOLDOWN_TICKS,
    },
  };
}

function buildSceneSummary(state: MutableWorldState): JsonValue {
  const banana = state.objects.find((entry) => entry.id === state.game.bananaObjectId);
  return {
    tick: state.tick,
    banana: banana
      ? { position: jsonVec(banana.position), heldBy: banana.heldBy }
      : null,
    agents: state.agents.map((agent) => ({
      id: agent.id,
      name: agent.name,
      position: jsonVec(agent.position),
      holding: agent.holding,
      powerUp: agent.powerUp?.kind ?? null,
      frozenRemainingTicks: Math.max(0, agent.frozenUntilTick - state.tick),
      status: agent.status,
      sabotageCooldownRemainingTicks: Math.max(0, agent.cooldowns.sabotageUntilTick - state.tick),
    })),
    active_objects: state.objects
      .filter((object) => object.active)
      .map((object) => ({
        id: object.id,
        kind: object.kind,
        label: object.label,
        position: jsonVec(object.position),
        heldBy: object.heldBy,
      })),
  };
}

function summarizeAgent(state: MutableWorldState, agentId: string): JsonValue {
  const agent = state.agents.find((entry) => entry.id === agentId);
  if (!agent) return { id: agentId, missing: true };
  return {
    id: agent.id,
    name: agent.name,
    position: jsonVec(agent.position),
    holding: agent.holding,
    powerUp: agent.powerUp?.kind ?? null,
    status: agent.status,
    intentIssuedAtTick: agent.intentIssuedAtTick,
    frozenRemainingTicks: Math.max(0, agent.frozenUntilTick - state.tick),
    sabotageCooldownRemainingTicks: Math.max(0, agent.cooldowns.sabotageUntilTick - state.tick),
    score: state.game.score.deliveries[agent.id] ?? 0,
  };
}

function jsonVec(position: { readonly x: number; readonly z: number }): JsonValue {
  return { x: position.x, z: position.z };
}

function summarizeForcedDropAttempt(
  state: MutableWorldState,
  conflict: Extract<WorldConflict, { kind: 'forced_drop' }>
): JsonValue {
  const attacker = state.agents.find((entry) => entry.id === conflict.attackerAgentId);
  const target = state.agents.find((entry) => entry.id === conflict.targetAgentId);
  const object = state.objects.find((entry) => entry.id === conflict.objectId);
  const goal = state.objects.find((entry) => entry.id === state.game.goalObjectId);
  return {
    attackerAgentId: conflict.attackerAgentId,
    attackerName: attacker?.name ?? conflict.attackerAgentId,
    targetAgentId: conflict.targetAgentId,
    targetName: target?.name ?? conflict.targetAgentId,
    objectId: conflict.objectId,
    currentHolder: object?.heldBy ?? null,
    distance: attacker && target ? roundForPrompt(vec2Distance(attacker.position, target.position)) : null,
    sabotageRadius: SABOTAGE_RADIUS,
    attackerIntentAgeTicks: attacker ? Math.max(0, state.tick - attacker.intentIssuedAtTick) : null,
    targetDistanceToGoal: target && goal ? roundForPrompt(vec2Distance(target.position, goal.position)) : null,
    score: {
      deliveries: state.game.score.deliveries,
      forcedDrops: state.game.score.forcedDrops,
    },
  };
}

function summarizeForcedDropRuling(record: ForcedDropRulingRecord): JsonValue {
  return {
    tick: record.tick,
    attackerAgentId: record.attackerAgentId,
    targetAgentId: record.targetAgentId,
    objectId: record.objectId,
    outcome: record.outcome,
  };
}

function buildForcedDropPolicy(
  state: MutableWorldState,
  conflict: Extract<WorldConflict, { kind: 'forced_drop' }>
): ForcedDropPolicy {
  const recentHistory = state.game.refereeMemory.forcedDrops.slice(-FORCED_DROP_HISTORY_LIMIT);
  const samePairHistory = recentHistory.filter(
    (record) =>
      record.attackerAgentId === conflict.attackerAgentId &&
      record.targetAgentId === conflict.targetAgentId
  );
  const suppressed = new Set<ForcedDropOutcome>();
  const suppressionNotes: string[] = [];

  const lastSamePair = samePairHistory[samePairHistory.length - 1];
  if (lastSamePair?.outcome === 'attacker_fumbles') {
    suppressed.add('attacker_fumbles');
    suppressionNotes.push('Same attacker-target pair just fumbled; omit attacker_fumbles to keep the next ruling varied.');
  }

  const lastGlobal = recentHistory.slice(-FORCED_DROP_REPEAT_SUPPRESSION_COUNT);
  if (lastGlobal.length === FORCED_DROP_REPEAT_SUPPRESSION_COUNT) {
    const repeatedOutcome = lastGlobal[0]!.outcome;
    if (!suppressed.has(repeatedOutcome) && lastGlobal.every((record) => record.outcome === repeatedOutcome)) {
      const alternativesAfterSuppression = FORCED_DROP_OUTCOMES.filter(
        (outcome) => outcome !== repeatedOutcome && !suppressed.has(outcome)
      );
      if (alternativesAfterSuppression.length >= 2) {
        suppressed.add(repeatedOutcome);
        suppressionNotes.push(`The last ${FORCED_DROP_REPEAT_SUPPRESSION_COUNT} forced-drop rulings were ${repeatedOutcome}; omit it for variety.`);
      }
    }
  }

  const availableOutcomes = FORCED_DROP_OUTCOMES.filter((outcome) => !suppressed.has(outcome));
  return {
    availableOutcomes,
    suppressedOutcomes: FORCED_DROP_OUTCOMES.filter((outcome) => suppressed.has(outcome)),
    fallbackOutcome: chooseForcedDropFallback(availableOutcomes),
    samePairHistory,
    recentHistory,
    suppressionNotes,
    varietyNote: suppressionNotes.length > 0
      ? 'Choose from the available outcomes only and prefer a fair variation from recent rulings.'
      : 'All outcomes are legal; attacker_fumbles should be occasional, not the default.',
  };
}

function chooseForcedDropFallback(availableOutcomes: readonly ForcedDropOutcome[]): ForcedDropOutcome {
  if (availableOutcomes.includes('drop')) return 'drop';
  if (availableOutcomes.includes('hold')) return 'hold';
  return availableOutcomes[0] ?? 'hold';
}

function roundForPrompt(value: number): number {
  return Math.round(value * 100) / 100;
}

export function buildRefereeChoices(
  state: MutableWorldState,
  conflict: WorldConflict
): readonly DirectorChoice<DirectorResolution>[] {
  if (conflict.kind === 'contested_object') {
    const object = state.objects.find((entry) => entry.id === conflict.objectId);
    const choices: DirectorChoice<DirectorResolution>[] = conflict.contenderAgentIds.map((agentId) => {
      const agent = state.agents.find((entry) => entry.id === agentId);
      return {
        id: `pickup:${agentId}`,
        label: `award pickup to ${agent?.name ?? agentId}`,
        description: `Let ${agent?.name ?? agentId} win the ${object?.label ?? conflict.objectId} scramble.`,
        payload: {
          conflictId: conflict.id,
          objectId: conflict.objectId,
          winnerAgentId: agentId,
          outcome: 'pickup',
        },
      };
    });
    choices.push({
      id: 'deny',
      label: 'deny pickup',
      description: 'No one gets the object from this scramble.',
      payload: {
        conflictId: conflict.id,
        objectId: conflict.objectId,
        winnerAgentId: null,
        outcome: 'deny',
      },
    });
    return choices;
  }

  const attacker = state.agents.find((entry) => entry.id === conflict.attackerAgentId);
  const target = state.agents.find((entry) => entry.id === conflict.targetAgentId);
  const policy = buildForcedDropPolicy(state, conflict);
  return policy.availableOutcomes.map((outcome) => ({
    id: outcome,
    label: forcedDropChoiceLabel(outcome),
    description: describeForcedDropChoice(
      outcome,
      attacker?.name ?? conflict.attackerAgentId,
      target?.name ?? conflict.targetAgentId
    ),
    payload: {
      conflictId: conflict.id,
      objectId: conflict.objectId,
      winnerAgentId: null,
      outcome,
    },
  }));
}

function forcedDropChoiceLabel(outcome: ForcedDropOutcome): string {
  switch (outcome) {
    case 'drop':
      return 'carrier drops';
    case 'hold':
      return 'carrier holds';
    case 'attacker_fumbles':
      return 'attacker fumbles';
  }
}

function describeForcedDropChoice(
  outcome: ForcedDropOutcome,
  attackerName: string,
  targetName: string
): string {
  switch (outcome) {
    case 'drop':
      return `Clean contact: ${attackerName} lands the bump and ${targetName} drops the banana.`;
    case 'hold':
      return `Balanced or braced contact: ${targetName} absorbs the bump and keeps the banana.`;
    case 'attacker_fumbles':
      return `Reckless or overextended contact: ${attackerName} mistimes the bump. Use occasionally, not as the default.`;
  }
}

function describeRefereeResolution(
  state: MutableWorldState,
  conflict: WorldConflict,
  resolution: DirectorResolution
): string {
  if (conflict.kind === 'contested_object') {
    const object = state.objects.find((entry) => entry.id === conflict.objectId);
    if (resolution.outcome === 'pickup' && resolution.winnerAgentId) {
      const winner = state.agents.find((entry) => entry.id === resolution.winnerAgentId);
      return `${winner?.name ?? resolution.winnerAgentId} gets the ${object?.label ?? conflict.objectId} after the scramble.`;
    }
    return `The ${object?.label ?? conflict.objectId} scramble is waved off.`;
  }

  const attacker = state.agents.find((entry) => entry.id === conflict.attackerAgentId);
  const target = state.agents.find((entry) => entry.id === conflict.targetAgentId);
  switch (resolution.outcome) {
    case 'drop':
      return `${attacker?.name ?? conflict.attackerAgentId} bumps ${target?.name ?? conflict.targetAgentId}, and the banana pops loose.`;
    case 'attacker_fumbles':
      return `${attacker?.name ?? conflict.attackerAgentId} overcooks the bump and fumbles the play.`;
    default:
      return `${target?.name ?? conflict.targetAgentId} braces through the bump and keeps the banana.`;
  }
}

function describeResolutionNote(resolution: DirectorResolution): string {
  switch (resolution.outcome) {
    case 'pickup':
      return 'pickup awarded';
    case 'deny':
      return 'pickup denied';
    case 'drop':
      return 'forced drop';
    case 'hold':
      return 'carrier holds';
    case 'attacker_fumbles':
      return 'attacker fumbles';
  }
}

function coerceNarrationDecision(value: string, snapshot: WorldSnapshot): DirectorDecision {
  if (value.length > 0) {
    return { note: value, resolutions: [] };
  }
  return { note: `Tick ${snapshot.tick}: Banana Dash keeps moving.`, resolutions: [] };
}

function describeConflict(conflict: WorldConflict): string {
  if (conflict.kind === 'contested_object') {
    return `${conflict.objectId} contested by ${conflict.contenderAgentIds.join(', ')}`;
  }
  return `${conflict.attackerAgentId} bumping ${conflict.targetAgentId}`;
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
    powerUp: agent.powerUp ? { ...agent.powerUp } : null,
    frozenUntilTick: agent.frozenUntilTick,
    intentIssuedAtTick: agent.intentIssuedAtTick,
    thinking: agent.thinking,
    cooldowns: {
      sabotageUntilTick: agent.cooldowns.sabotageUntilTick,
    },
    navigation: {
      detourTarget: agent.navigation.detourTarget
        ? { x: agent.navigation.detourTarget.x, z: agent.navigation.detourTarget.z }
        : null,
      blockedTicks: agent.navigation.blockedTicks,
      obstacleId: agent.navigation.obstacleId,
    },
  };
}

function cloneForcedDropRulingRecord(record: ForcedDropRulingRecord): ForcedDropRulingRecord {
  return {
    tick: record.tick,
    attackerAgentId: record.attackerAgentId,
    targetAgentId: record.targetAgentId,
    objectId: record.objectId,
    outcome: record.outcome,
  };
}

function cloneObject(obj: SimulationObjectState): SimulationObjectState {
  return {
    id: obj.id,
    kind: obj.kind,
    label: obj.label,
    description: obj.description,
    position: { x: obj.position.x, z: obj.position.z },
    active: obj.active,
    contested: obj.contested,
    heldBy: obj.heldBy,
    tags: obj.tags,
    affordances: obj.affordances,
    blocksMovement: obj.blocksMovement,
    collisionRadius: obj.collisionRadius,
  };
}

function inferEmotionFromGoal(goal: AgentGoal): string {
  switch (goal.kind) {
    case 'wait':
      return 'thinking';
    case 'drop':
      return 'alert';
    case 'go_to_agent':
    case 'sabotage_agent':
      return 'curious';
    case 'go_to_object':
    case 'deliver':
      return 'alert';
    case 'object_action':
      return goal.affordance.kind === 'pick_up' ? 'happy' : 'curious';
  }
}
