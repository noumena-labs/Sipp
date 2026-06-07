import type { DirectorChoice, DirectorRuntime, JsonValue } from '@noumena-labs/cogentlm/director';
import { SimulationBus, type SimulationEvent } from './bus.js';
import { buildPerception, vec2Distance } from './sensing.js';
import { SimulationAgentChooser } from './agent-chooser.js';
import {
  applyDirectorDecision,
  applyTickFirstPass,
  BAT_SWING_RADIUS,
  CHASE_MIN_DISTANCE,
  cloneScore,
  deterministicConflictResolution,
  FORCED_DROP_HISTORY_LIMIT,
  GOAL_RADIUS,
  ICE_THROW_RADIUS,
  SABOTAGE_COOLDOWN_TICKS,
  SABOTAGE_RADIUS,
  type MutableGameState,
  type MutableWorldState,
} from './reducer.js';
import type {
  AgentGoal,
  AgentIntent,
  DirectorDecision,
  DirectorResolution,
  ForcedDropOutcome,
  ForcedDropRulingRecord,
  ObjectAffordance,
  RefereeState,
  ScenarioAgentSeed,
  ScenarioGameSeed,
  ScenarioObjectSeed,
  SimulationAgentState,
  SimulationGameEvent,
  SimulationGameState,
  SimulationObjectState,
  PowerUpKind,
  WorldBounds,
  WorldConflict,
  WorldSnapshot,
} from './types.js';

const FORCED_DROP_OUTCOMES: readonly ForcedDropOutcome[] = ['drop', 'hold', 'attacker_fumbles'];
const FORCED_DROP_REPEAT_SUPPRESSION_COUNT = 3;
const DEFAULT_MAX_MOVE_TICKS_BEFORE_REEVALUATION = 15;
const MAX_RECENT_NARRATION_EVENTS = 8;
interface ForcedDropPolicy {
  readonly availableOutcomes: readonly ForcedDropOutcome[];
  readonly suppressedOutcomes: readonly ForcedDropOutcome[];
  readonly fallbackOutcome: ForcedDropOutcome;
  readonly samePairHistory: readonly ForcedDropRulingRecord[];
  readonly recentHistory: readonly ForcedDropRulingRecord[];
  readonly suppressionNotes: readonly string[];
  readonly varietyNote: string;
}

interface NarrationEventSummary {
  readonly tick: number;
  readonly text: string;
}

type NarrationWorldState = Pick<WorldSnapshot, 'agents' | 'objects' | 'game' | 'tick' | 'bounds'>;

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
  readonly maxMoveTicksBeforeReevaluation?: number;
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
  private readonly maxMoveTicksBeforeReevaluation: number;
  private readonly activeControllers: Set<AbortController> = new Set();
  private readonly recentNarrationEvents: NarrationEventSummary[] = [];
  private readonly lastDecisionByAgentId: Map<string, string> = new Map();
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
    this.maxMoveTicksBeforeReevaluation = Math.max(
      1,
      Math.floor(options.maxMoveTicksBeforeReevaluation ?? DEFAULT_MAX_MOVE_TICKS_BEFORE_REEVALUATION)
    );
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
    return this.agentQueryInFlight != null || this.narrationInFlight != null || this.state.game.referee.status === 'ruling';
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
    this.lastDecisionByAgentId.delete(agentId);
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
    this.expireStaleMovementGoals();

    if (conflicts.length > 0) {
      await this.handleRefereeConflict(conflicts[0]!);
    } else {
      this.maybeStartOneAgentQuery();
      this.maybeStartNarration();
    }

    for (const agent of this.state.agents) {
      this.emit({ kind: 'agent-state', tick: this.state.tick, agent: cloneAgent(agent) });
      if (agent.emotion) {
        this.emit({ kind: 'agent-expression', tick: this.state.tick, agentId: agent.id, emotion: agent.emotion });
      }
    }
    this.emit({ kind: 'tick-end', tick: this.state.tick, snapshot: this.getSnapshot() });
  }

  private async handleRefereeConflict(conflict: WorldConflict): Promise<void> {
    if (this.agentQueryInFlight) {
      this.agentQueryInFlight.controller.abort();
      await this.agentQueryInFlight.done.catch(() => undefined);
    }

    await this.cancelNarration();

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
      cloneGame(this.state.game),
      {},
      this.lastDecisionByAgentId.get(agentState.id) ?? null
    );

    const controller = new AbortController();
    agentState.thinking = true;
    this.emit({ kind: 'agent-query-start', tick: this.state.tick, agentId: chosenId });
    const done = agent.query(perception, {
      signal: controller.signal,
      timeoutMs: this.agentQueryTimeoutMs,
    }).then((result) => {
      if (this.disposed || controller.signal.aborted || result.status === 'aborted') return;
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
      const goal = this.coerceCloseRangeGoal(current, result.goal);
      current.goal = goal;
      this.lastDecisionByAgentId.set(current.id, goal.label);
      const intent = this.mapGoalToIntent(goal);
      current.intent = intent;
      current.intentIssuedAtTick = this.state.tick;
      current.status = goal.label;
      current.thinking = false;
      current.emotion = intent.emotion;
      this.emit({
        kind: 'agent-query-end',
        tick: this.state.tick,
        agentId: chosenId,
        goal,
        intent,
        status: goal.label,
        emotion: intent.emotion,
        queryStatus: result.status,
        ...(result.errorMessage ? { errorMessage: result.errorMessage } : {}),
      });
      this.emit({
        kind: 'agent-intent',
        tick: this.state.tick,
        agentId: chosenId,
        goal,
        intent,
        status: goal.label,
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
    const narrationEvents = [...this.recentNarrationEvents];
    const requestTick = this.state.tick;
    const requestSnapshot = this.getSnapshot();
    this.narrationInFlight = this.director.run(this.narrateTask, {
      inputs: buildNarrationPayload(this.state, narrationEvents) as Record<string, JsonValue>,
      signal: controller.signal,
      timeoutMs: this.narrationTimeoutMs,
      maxOutputTokens: 128,
    }).then((result) => {
      if (this.disposed || controller.signal.aborted) return;
      if (result.status !== 'ok') {
        this.emitRuntimeIssue({
          severity: 'warning',
          source: 'narration',
          message: `Director narration task failed: ${result.errorMessage ?? result.status}`,
          taskName: this.narrateTask,
        });
        this.recentNarrationEvents.splice(0, narrationEvents.length);
        return;
      }
      const decision = coerceNarrationDecision(result.text, requestSnapshot, narrationEvents);
      this.emit({
        kind: 'director-narration-trace',
        tick: requestTick,
        rawText: result.rawText,
        parsedText: result.text,
        accepted: decision.note.length > 0,
        ...(decision.note.length === 0 ? { reason: describeRejectedNarration(result.text, requestSnapshot) } : {}),
      });
      this.state.directorNote = decision.note || this.state.directorNote;
      this.emit({ kind: 'director-decision', tick: requestTick, decision });
      if (decision.note) {
        this.emit({ kind: 'world-note', tick: requestTick, note: decision.note });
      }
      this.recentNarrationEvents.splice(0, narrationEvents.length);
    }).finally(() => {
      this.activeControllers.delete(controller);
      if (this.narrationController === controller) {
        this.narrationController = null;
      }
      this.narrationInFlight = null;
    });
  }

  private async cancelNarration(): Promise<void> {
    this.narrationController?.abort();
    if (this.narrationInFlight) {
      await this.narrationInFlight.catch(() => undefined);
    }
  }

  private mapGoalToIntent(goal: AgentGoal): AgentIntent {
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
      case 'push_agent':
        return { kind: 'push', agentId: goal.agentId, emotion: inferEmotionFromGoal(goal) };
      case 'sabotage_agent':
        return { kind: 'sabotage', agentId: goal.agentId, method: goal.method, emotion: inferEmotionFromGoal(goal) };
      case 'drop':
        return { kind: 'drop', emotion: inferEmotionFromGoal(goal) };
    }
  }

  private coerceCloseRangeGoal(agent: SimulationAgentState, goal: AgentGoal): AgentGoal {
    if (agent.holding === this.state.game.bananaObjectId) {
      const home = this.state.objects.find((entry) => entry.id === this.state.game.goalObjectId);
      if (!home) return goal;
      if (goal.kind === 'deliver' && goal.objectId === home.id) return goal;
      if (goal.kind === 'go_to_object' && goal.objectId === home.id) return goal;
      const label = vec2Distance(agent.position, home.position) <= GOAL_RADIUS
        ? 'score at home base'
        : 'keep running to base';
      return { kind: 'deliver', objectId: home.id, label };
    }

    if (goal.kind === 'sabotage_agent' && goal.method === 'bump' && agent.powerUp?.kind === 'bat') {
      const target = this.state.agents.find((entry) => entry.id === goal.agentId);
      if (target) {
        return { kind: 'sabotage_agent', agentId: target.id, method: 'bat', label: `smack ${target.name} with the bat` };
      }
    }

    if (goal.kind === 'push_agent') {
      const target = this.state.agents.find((entry) => entry.id === goal.agentId);
      if (target?.holding === this.state.game.bananaObjectId && agent.cooldowns.sabotageUntilTick <= this.state.tick) {
        return this.closeRangeGoalForTarget(agent, target);
      }
      return goal;
    }

    if (goal.kind === 'go_to_agent') {
      const target = this.state.agents.find((entry) => entry.id === goal.agentId);
      if (target && vec2Distance(agent.position, target.position) <= CHASE_MIN_DISTANCE) {
        return this.closeRangeGoalForTarget(agent, target);
      }
      return goal;
    }

    if (goal.kind === 'wait') {
      const target = nearestAgentWithin(agent, this.state.agents, CHASE_MIN_DISTANCE);
      if (target) {
        return this.closeRangeGoalForTarget(agent, target);
      }
    }

    return goal;
  }

  private closeRangeGoalForTarget(agent: SimulationAgentState, target: SimulationAgentState): AgentGoal {
    if (target.holding === this.state.game.bananaObjectId && agent.cooldowns.sabotageUntilTick <= this.state.tick) {
      if (agent.powerUp) {
        const distance = vec2Distance(agent.position, target.position);
        if (agent.powerUp.kind === 'ice_cube' && distance <= ICE_THROW_RADIUS) {
          return { kind: 'sabotage_agent', agentId: target.id, method: 'ice_cube', label: `freeze ${target.name} with the ice cube` };
        }
        const label = agent.powerUp.kind === 'bat'
          ? `smack ${target.name} with the bat`
          : `freeze ${target.name} with the ice cube`;
        return { kind: 'sabotage_agent', agentId: target.id, method: agent.powerUp.kind, label };
      }
      return { kind: 'sabotage_agent', agentId: target.id, method: 'bump', label: `bump ${target.name}` };
    }

    return { kind: 'push_agent', agentId: target.id, label: `push ${target.name}` };
  }

  private needsDecision(agent: SimulationAgentState): boolean {
    if (agent.thinking) return false;
    if (agent.frozenUntilTick > this.state.tick) return false;
    if (!agent.goal) return true;
    if (!agent.intent) return true;
    return false;
  }

  private expireStaleMovementGoals(): void {
    for (const agent of this.state.agents) {
      if (!isReevaluableMovementGoal(agent.goal)) continue;
      if (this.state.tick - agent.intentIssuedAtTick < this.maxMoveTicksBeforeReevaluation) continue;
      this.clearAgentForReevaluation(agent, 'reconsidering the route');
    }
  }

  private invalidateMovementGoalsForBananaDrop(): void {
    for (const agent of this.state.agents) {
      if (!isLooseBananaReevaluationGoal(agent.goal)) continue;
      this.clearAgentForReevaluation(agent, 'banana is loose; changing plans');
    }
  }

  private clearAgentForReevaluation(agent: SimulationAgentState, status: string): void {
    if (agent.goal) {
      this.lastDecisionByAgentId.set(agent.id, agent.goal.label);
    }
    agent.goal = null;
    agent.intent = null;
    agent.status = status;
    agent.navigation.detourTarget = null;
    agent.navigation.blockedTicks = 0;
    agent.navigation.obstacleId = null;
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
          agent.goal.kind === 'push_agent' ||
          agent.goal.kind === 'sabotage_agent'
        ) {
          agent.goal = null;
        }
        continue;
      }
      if (arrived.has(agent.id)) {
        if (this.completeArrivedPickupGoal(agent)) {
          continue;
        }
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

  private completeArrivedPickupGoal(agent: SimulationAgentState): boolean {
    const goal = agent.goal;
    if (!goal || goal.kind !== 'go_to_object') return false;
    const object = this.state.objects.find((entry) => entry.id === goal.objectId);
    const affordance = object?.affordances.find((entry) => entry.kind === 'pick_up');
    if (!object || !object.active || object.heldBy || !affordance) return false;
    const label = affordance.label;
    agent.goal = {
      kind: 'object_action',
      objectId: object.id,
      affordance: { ...affordance, label },
      label,
    };
    agent.intent = { kind: 'pick_up', objectId: object.id, emotion: inferEmotionFromGoal(agent.goal) };
    agent.intentIssuedAtTick = this.state.tick;
    agent.status = label;
    this.lastDecisionByAgentId.set(agent.id, label);
    return true;
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
      case 'push_agent': {
        const target = this.state.agents.find((entry) => entry.id === goal.agentId);
        return !target;
      }
      case 'sabotage_agent': {
        const target = this.state.agents.find((entry) => entry.id === goal.agentId);
        if (!target) return true;
        if (goal.method === 'bump' && target.holding !== this.state.game.bananaObjectId) return true;
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
      this.recordNarrationEvent(event);
      this.emit({ kind: 'game-event', tick: this.state.tick, event });
      if (event.kind === 'drop' && event.objectId === this.state.game.bananaObjectId) {
        this.invalidateMovementGoalsForBananaDrop();
      }
      if (shouldEmitImmediateWorldSync(event)) {
        this.emit({ kind: 'world-sync', tick: this.state.tick, snapshot: this.getSnapshot() });
      }
    }
  }

  private recordNarrationEvent(event: SimulationGameEvent): void {
    const text = describeNarrationGameEvent(this.state, event);
    if (!text) return;
    this.recentNarrationEvents.push({ tick: this.state.tick, text });
    if (this.recentNarrationEvents.length > MAX_RECENT_NARRATION_EVENTS) {
      this.recentNarrationEvents.splice(0, this.recentNarrationEvents.length - MAX_RECENT_NARRATION_EVENTS);
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
    pendingIceImpacts: [],
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
    referee: cloneRefereeState(game.referee),
    refereeMemory: {
      forcedDrops: game.refereeMemory.forcedDrops.map(cloneForcedDropRulingRecord),
    },
    pendingRespawns: game.pendingRespawns.map((pending) => ({
      objectId: pending.objectId,
      spawnPosition: { x: pending.spawnPosition.x, z: pending.spawnPosition.z },
      activateAtTick: pending.activateAtTick,
    })),
    pendingIceImpacts: game.pendingIceImpacts.map((pending) => ({
      objectId: pending.objectId,
      attackerAgentId: pending.attackerAgentId,
      targetAgentId: pending.targetAgentId,
      launchedFrom: { x: pending.launchedFrom.x, z: pending.launchedFrom.z },
      activateAtTick: pending.activateAtTick,
      launchedAtTick: pending.launchedAtTick,
    })),
  };
}

export function buildRefereePayload(state: MutableWorldState, conflict: WorldConflict): JsonValue {
  return {
    referee_event: summarizeConflict(state, conflict),
    scoreboard: state.game.score.deliveries,
    scene_summary: buildRefereeSceneSummary(state),
  };
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// NARRATION
//////////////////////////////////////////////////////////////////////////////////////////////////////////

function buildNarrationPayload(
  state: MutableWorldState,
  recentEvents: readonly NarrationEventSummary[]
): JsonValue {
  return {
    narration_brief: buildNarrationBrief(state, recentEvents),
  };
}

function buildNarrationBrief(
  state: MutableWorldState,
  recentEvents: readonly NarrationEventSummary[]
): string {
  const observations = collectNarrationObservations(state, recentEvents);
  return [
    'Write exactly one complete sentence as an old-timey sports caller at an active game.',
    'The sentence must use ALL the observations below, include at least one player name, describe live action, and mention the stakes.',
    'Do not answer with only a player name, label, list, fragment, or JSON.',
    'Use 8 to 24 words.',
    '',
    'Observations:',
    ...observations.map((observation) => `- ${observation}`),
  ].join('\n');
}

function collectNarrationObservations(
  state: MutableWorldState,
  recentEvents: readonly NarrationEventSummary[]
): string[] {
  const observations: string[] = [];
  const previousNote = normalizePreviousNarrationNote(state.directorNote);

  if (previousNote) {
    observations.push(`Previous call to avoid: ${asSentence(previousNote)}`);
  }
  observations.push(`Score: ${asSentence(formatNarrationScore(state))}`);
  observations.push(...describeBananaNarrationObservations(state));
  observations.push(...describeAgentIntentObservations(state));
  observations.push(...describePowerUpNarrationObservations(state));
  observations.push(...describeFrozenNarrationObservations(state));
  observations.push(...formatRecentNarrationEventObservations(recentEvents));

  return dedupeNarrationObservations(observations).slice(0, 12);
}

function normalizePreviousNarrationNote(note: string | null): string | null {
  const trimmed = note?.trim();
  if (!trimmed) return null;
  return trimmed.replace(/[.!?]+$/g, '').trim() || null;
}

function formatRecentNarrationEventObservations(events: readonly NarrationEventSummary[]): string[] {
  return events
    .slice(-2)
    .map((entry) => `Recent event: ${asSentence(entry.text)}`);
}

function formatNarrationScore(state: MutableWorldState): string {
  const entries = Object.entries(state.game.score.deliveries)
    .map(([agentId, score]) => ({ agent: state.agents.find((entry) => entry.id === agentId), score }))
    .filter((entry): entry is { agent: SimulationAgentState; score: number } => entry.agent != null)
    .sort((a, b) => b.score - a.score || a.agent.name.localeCompare(b.agent.name));
  if (entries.length === 0) return 'No score yet';

  const leaderScore = entries[0]!.score;
  const leaders = entries.filter((entry) => entry.score === leaderScore);
  if (leaderScore === 0) {
    return 'Scoreless tie';
  }
  if (leaders.length === entries.length) {
    return `All tied at ${leaderScore}`;
  }
  return `${formatNameList(leaders.map((entry) => entry.agent.name))} lead${leaders.length === 1 ? 's' : ''} with ${leaderScore}`;
}

function describeBananaNarrationObservations(state: MutableWorldState): string[] {
  const banana = state.objects.find((entry) => entry.id === state.game.bananaObjectId);
  if (!banana) return ['The banana is not visible.'];

  if (banana.heldBy) {
    const carrier = state.agents.find((entry) => entry.id === banana.heldBy);
    if (!carrier) return [`The banana is marked as held by ${banana.heldBy}.`];
    const goal = state.objects.find((entry) => entry.id === state.game.goalObjectId);
    const goalDistance = goal ? roundForPrompt(vec2Distance(carrier.position, goal.position)) : null;
    const observations = [`${carrier.name} has the banana.`];
    if (goalDistance != null) {
      observations.push(`${carrier.name} is ${describeGoalProgress(goalDistance)}.`);
    }
    const chasers = nearestAgentsTo(carrier, state.agents, 2)
      .filter((entry) => entry.agent.frozenUntilTick <= state.tick)
      .map((entry) => entry.agent.name);
    if (chasers.length > 0) {
      observations.push(`${formatNameList(chasers)} ${chasers.length === 1 ? 'is' : 'are'} chasing ${carrier.name}.`);
    }
    return observations;
  }

  const rushers = state.agents
    .map((agent) => ({ agent, distance: vec2Distance(agent.position, banana.position) }))
    .filter((entry) => entry.agent.frozenUntilTick <= state.tick)
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 2);
  const activeRushers = rushers
    .filter((entry) => isAgentDrivingAtBanana(entry.agent))
    .map((entry) => entry.agent.name);
  const names = activeRushers.length > 0 ? activeRushers : rushers.map((entry) => entry.agent.name);
  const observations = [`The banana is loose on the ${describeFieldZone(state, banana.position)}.`];
  if (names.length > 0) {
    observations.push(`${formatNameList(names)} ${names.length === 1 ? 'is' : 'are'} charging toward the banana.`);
  }
  return observations;
}

function describeAgentIntentObservations(state: MutableWorldState): string[] {
  return state.agents
    .filter((agent) => agent.frozenUntilTick <= state.tick)
    .map((agent) => describeAgentIntentObservation(state, agent))
    .filter((entry): entry is string => entry != null)
    .slice(0, 5);
}

function describeAgentIntentObservation(
  state: MutableWorldState,
  agent: SimulationAgentState
): string | null {
  const intent = agent.intent;
  if (!intent) return null;
  switch (intent.kind) {
    case 'go_to_object':
      return `${agent.name} is moving toward ${describeNarrationObject(state, intent.objectId)}.`;
    case 'pick_up':
      return `${agent.name} is trying to grab ${describeNarrationObject(state, intent.objectId)}.`;
    case 'deliver':
      return `${agent.name} is heading to home base.`;
    case 'approach_agent':
      return `${agent.name} is chasing ${agentName(state, intent.agentId)}.`;
    case 'push':
      return `${agent.name} is trying to shove ${agentName(state, intent.agentId)}.`;
    case 'sabotage':
      return `${agent.name} is trying to ${describeSabotageObservation(state, intent.agentId, intent.method)}.`;
    case 'use':
      return `${agent.name} is using ${describeNarrationObject(state, intent.objectId)}.`;
    case 'drop':
      return `${agent.name} is dropping what they carry.`;
    case 'move_to':
    case 'wait':
      return null;
  }
}

function describePowerUpNarrationObservations(state: MutableWorldState): string[] {
  return state.agents
    .filter((agent) => agent.powerUp != null)
    .map((agent) => `${agent.name} has ${labelForNarrationPowerUp(agent.powerUp!.kind)}.`)
    .slice(0, 4);
}

function describeFrozenNarrationObservations(state: MutableWorldState): string[] {
  return state.agents
    .filter((agent) => agent.frozenUntilTick > state.tick)
    .map((agent) => `${agent.name} is frozen for ${agent.frozenUntilTick - state.tick} more ticks.`)
    .slice(0, 3);
}

function describeGoalProgress(distance: number): string {
  if (distance <= GOAL_RADIUS + 1.2) return 'on the doorstep of home base';
  if (distance <= 4) return 'closing on home base';
  if (distance <= 7) return 'midfield from home base';
  return 'far from home base';
}

function describeSabotageObservation(
  state: MutableWorldState,
  targetAgentId: string,
  method: 'bump' | PowerUpKind
): string {
  const target = state.agents.find((entry) => entry.id === targetAgentId);
  const targetName = target?.name ?? targetAgentId;
  const targetHasBanana = target?.holding === state.game.bananaObjectId;
  switch (method) {
    case 'bat':
      return `smack ${targetName} with the bat${targetHasBanana ? ' and knock the banana loose' : ''}`;
    case 'ice_cube':
      return `freeze ${targetName} with ice${targetHasBanana ? ' and stop the banana run' : ''}`;
    case 'bump':
      return `bump ${targetName}${targetHasBanana ? ' and knock the banana loose' : ''}`;
  }
}

function dedupeNarrationObservations(observations: readonly string[]): string[] {
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const observation of observations) {
    const normalized = normalizeNarrationText(observation);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    unique.push(observation);
  }
  return unique;
}

function asSentence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return '';
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

function nearestAgentsTo(
  self: SimulationAgentState,
  agents: readonly SimulationAgentState[],
  limit: number
): Array<{ agent: SimulationAgentState; distance: number }> {
  return agents
    .filter((agent) => agent.id !== self.id)
    .map((agent) => ({ agent, distance: vec2Distance(self.position, agent.position) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, limit);
}

function isAgentDrivingAtBanana(agent: SimulationAgentState): boolean {
  return /banana|rush|grab|score/i.test(agent.status);
}

function labelForNarrationPowerUp(powerUp: PowerUpKind): string {
  return powerUp === 'bat' ? 'the bat' : 'an ice cube';
}

function describeNarrationObject(state: NarrationWorldState, objectId: string): string {
  if (objectId === state.game.goalObjectId) return 'home base';
  const object = state.objects.find((entry) => entry.id === objectId);
  const label = object?.label ?? objectId;
  return label === 'banana' ? 'the banana' : label;
}

function agentName(state: NarrationWorldState, agentId: string): string {
  return state.agents.find((entry) => entry.id === agentId)?.name ?? agentId;
}

function describeNarrationGameEvent(state: MutableWorldState, event: SimulationGameEvent): string | null {
  switch (event.kind) {
    case 'delivery':
      return `${agentName(state, event.agentId)} scored with ${describeNarrationObject(state, event.objectId)}`;
    case 'pickup': {
      const object = state.objects.find((entry) => entry.id === event.objectId);
      if (event.objectId === state.game.bananaObjectId) {
        return `${agentName(state, event.agentId)} grabbed the banana and turned for home`;
      }
      return `${agentName(state, event.agentId)} snagged ${object?.label ?? event.objectId} for trouble`;
    }
    case 'drop':
      return `${agentName(state, event.agentId)} lost ${describeNarrationObject(state, event.objectId)} after ${labelForNarrationDropCause(event.cause)}`;
    case 'forced_drop':
      return describeForcedDropNarrationEvent(state, event);
    case 'bump_whiff':
      return `${agentName(state, event.attackerAgentId)} whiffed the bump on ${agentName(state, event.targetAgentId)}`;
    case 'push':
      return `${agentName(state, event.agentId)} shoved ${agentName(state, event.targetAgentId)} off the line`;
    case 'power_up_throw':
      return `${agentName(state, event.agentId)} threw ice at ${agentName(state, event.targetAgentId)}`;
    case 'bat_swing': {
      const hits = event.hits.map((hit) => agentName(state, hit.agentId));
      return hits.length > 0
        ? `${agentName(state, event.agentId)} swung the bat and rattled ${formatNameList(hits)}`
        : `${agentName(state, event.agentId)} swung the bat and found nothing but breeze`;
    }
    case 'power_up_use':
      return `${agentName(state, event.agentId)} froze ${agentName(state, event.targetAgentId)} with ice`;
    case 'respawn':
      return event.objectId === state.game.bananaObjectId
        ? `the banana respawned on the ${describeFieldZone(state, event.position)}`
        : null;
    case 'fallback':
      return null;
  }
}

function describeForcedDropNarrationEvent(
  state: NarrationWorldState,
  event: Extract<SimulationGameEvent, { kind: 'forced_drop' }>
): string {
  const attacker = agentName(state, event.attackerAgentId);
  const target = agentName(state, event.targetAgentId);
  switch (event.outcome) {
    case 'drop':
      return `${attacker} knocked the banana loose from ${target}`;
    case 'hold':
      return `${target} kept the banana through ${attacker}'s challenge`;
    case 'attacker_fumbles':
      return `${attacker} fumbled the challenge on ${target}`;
  }
}

function labelForNarrationDropCause(cause: Extract<SimulationGameEvent, { kind: 'drop' }>['cause']): string {
  switch (cause) {
    case 'bat':
      return 'a bat bonk';
    case 'bump':
      return 'a bump';
    case 'ice':
      return 'an ice mishap';
    case 'voluntary':
      return 'a voluntary drop';
  }
}

function describeFieldZone(state: MutableWorldState, position: { readonly x: number; readonly z: number }): string {
  const edge = state.bounds.halfExtent * 0.45;
  const horizontal = position.x < -edge ? 'left-field' : position.x > edge ? 'right-field' : 'center';
  if (position.z < -edge) {
    return horizontal === 'center' ? 'home stretch' : `${horizontal} corner near home`;
  }
  if (position.z > edge) {
    return horizontal === 'center' ? 'deep outfield' : `deep ${horizontal}`;
  }
  return horizontal === 'center' ? 'middle of the diamond' : `${horizontal} line`;
}

function formatNameList(names: readonly string[]): string {
  if (names.length === 0) return 'nobody';
  if (names.length === 1) return names[0]!;
  if (names.length === 2) return `${names[0]} and ${names[1]}`;
  return `${names.slice(0, -1).join(', ')}, and ${names[names.length - 1]}`;
}

function summarizeConflict(state: MutableWorldState, conflict: WorldConflict): JsonValue {
  if (conflict.kind === 'contested_object') {
    const object = state.objects.find((entry) => entry.id === conflict.objectId);
    return {
      conflictId: conflict.id,
      kind: conflict.kind,
      objectId: conflict.objectId,
      objectLabel: object?.label ?? conflict.objectId,
      contenders: conflict.contenderAgentIds.map((id) => summarizeConflictAgent(state, id, object?.position)),
    };
  }
  const policy = buildForcedDropPolicy(state, conflict);
  return {
    conflictId: conflict.id,
    kind: conflict.kind,
    objectId: conflict.objectId,
    attacker: summarizeConflictAgent(state, conflict.attackerAgentId),
    target: summarizeConflictAgent(state, conflict.targetAgentId),
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

function buildRefereeSceneSummary(state: MutableWorldState): JsonValue {
  const banana = state.objects.find((entry) => entry.id === state.game.bananaObjectId);
  const goal = state.objects.find((entry) => entry.id === state.game.goalObjectId);
  return {
    tick: state.tick,
    banana: banana ? { heldBy: banana.heldBy, position: jsonVec(banana.position) } : null,
    goal: goal ? { position: jsonVec(goal.position) } : null,
  };
}

function summarizeConflictAgent(
  state: MutableWorldState,
  agentId: string,
  referencePosition?: { readonly x: number; readonly z: number }
): JsonValue {
  const agent = state.agents.find((entry) => entry.id === agentId);
  if (!agent) return { id: agentId, missing: true };
  const summary: Record<string, JsonValue> = {
    id: agent.id,
    name: agent.name,
    holding: agent.holding,
    powerUp: agent.powerUp?.kind ?? null,
    status: agent.status,
    frozenRemainingTicks: Math.max(0, agent.frozenUntilTick - state.tick),
    sabotageCooldownRemainingTicks: Math.max(0, agent.cooldowns.sabotageUntilTick - state.tick),
    score: state.game.score.deliveries[agent.id] ?? 0,
  };
  if (referencePosition) {
    summary.distance = roundForPrompt(vec2Distance(agent.position, referencePosition));
  }
  return summary;
}

function nearestAgentWithin(
  self: SimulationAgentState,
  agents: readonly SimulationAgentState[],
  maxDistance: number
): SimulationAgentState | null {
  let best: { agent: SimulationAgentState; distance: number } | null = null;
  for (const agent of agents) {
    if (agent.id === self.id) continue;
    const distance = vec2Distance(self.position, agent.position);
    if (distance > maxDistance) continue;
    if (best == null || distance < best.distance) {
      best = { agent, distance };
    }
  }
  return best?.agent ?? null;
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
    batSwingRadius: BAT_SWING_RADIUS,
    attackerIntentAgeTicks: attacker ? Math.max(0, state.tick - attacker.intentIssuedAtTick) : null,
    targetDistanceToGoal: target && goal ? roundForPrompt(vec2Distance(target.position, goal.position)) : null,
    forcedDrops: state.game.score.forcedDrops,
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
    suppressionNotes.push('Same pair just fumbled; omit attacker_fumbles.');
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
        suppressionNotes.push(`Last ${FORCED_DROP_REPEAT_SUPPRESSION_COUNT} forced-drop rulings were ${repeatedOutcome}; omit it.`);
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
      ? 'Choose an available fair variation.'
      : 'All outcomes legal; fumbles occasional.',
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

function coerceNarrationDecision(
  value: string,
  snapshot: WorldSnapshot,
  _recentEvents: readonly NarrationEventSummary[]
): DirectorDecision {
  const note = value.trim();
  if (
    note.length > 0 &&
    !isTooShortNarration(note) &&
    !isGenericNarration(note) &&
    !isRepeatedNarration(note, snapshot.directorNote)
  ) {
    return { note, resolutions: [] };
  }
  return { note: '', resolutions: [] };
}

function describeRejectedNarration(note: string, snapshot: WorldSnapshot): string {
  const trimmed = note.trim();
  if (trimmed.length === 0) return 'empty response';
  if (isTooShortNarration(trimmed)) return 'too short';
  if (isGenericNarration(trimmed)) return 'generic instruction text';
  if (isRepeatedNarration(trimmed, snapshot.directorNote)) return 'repeated previous call';
  return 'did not pass narration validation';
}

function isTooShortNarration(note: string): boolean {
  const words = note.match(/[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*/g) ?? [];
  return note.trim().length < 16 || words.length < 4;
}

function isGenericNarration(note: string): boolean {
  const trimmed = note.trim();
  return /^(power[- ]?ups? available|power[- ]?ups? are available|available power[- ]?ups?)[.!]?$/i.test(trimmed)
    || /^return plain text(?: under \d+(?: characters?)?)?[.!]?$/i.test(trimmed)
    || /^write (?:only the final answer|the final answer only)(?:,? under \d+(?: characters?)?)?[.!]?$/i.test(
      trimmed
    );
}

function isRepeatedNarration(note: string, previousNote: string | null): boolean {
  if (!previousNote) return false;
  return normalizeNarrationText(note) === normalizeNarrationText(previousNote);
}

function normalizeNarrationText(note: string): string {
  return note.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
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
    intent: agent.intent ? cloneAgentIntent(agent.intent) : null,
    goal: agent.goal ? cloneAgentGoal(agent.goal) : null,
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

function cloneRefereeState(referee: RefereeState): RefereeState {
  if (referee.status === 'idle') {
    return { status: 'idle' };
  }
  return {
    status: 'ruling',
    conflict: cloneWorldConflict(referee.conflict),
    startedAtTick: referee.startedAtTick,
  };
}

function cloneWorldConflict(conflict: WorldConflict): WorldConflict {
  if (conflict.kind === 'contested_object') {
    return {
      id: conflict.id,
      kind: 'contested_object',
      objectId: conflict.objectId,
      contenderAgentIds: [...conflict.contenderAgentIds],
    };
  }
  return {
    id: conflict.id,
    kind: 'forced_drop',
    attackerAgentId: conflict.attackerAgentId,
    targetAgentId: conflict.targetAgentId,
    objectId: conflict.objectId,
  };
}

function cloneAgentIntent(intent: AgentIntent): AgentIntent {
  switch (intent.kind) {
    case 'move_to':
      return { ...intent, target: { x: intent.target.x, z: intent.target.z } };
    default:
      return { ...intent };
  }
}

function cloneAgentGoal(goal: AgentGoal): AgentGoal {
  switch (goal.kind) {
    case 'object_action':
      return { ...goal, affordance: cloneObjectAffordance(goal.affordance) };
    default:
      return { ...goal };
  }
}

function cloneObjectAffordance(affordance: ObjectAffordance): ObjectAffordance {
  return {
    kind: affordance.kind,
    label: affordance.label,
    ...(affordance.status ? { status: affordance.status } : {}),
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
    tags: [...obj.tags],
    affordances: obj.affordances.map(cloneObjectAffordance),
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
    case 'push_agent':
    case 'sabotage_agent':
      return 'curious';
    case 'go_to_object':
    case 'deliver':
      return 'alert';
    case 'object_action':
      return goal.affordance.kind === 'pick_up' ? 'happy' : 'curious';
  }
}

function isReevaluableMovementGoal(
  goal: AgentGoal | null
): goal is Extract<AgentGoal, { kind: 'go_to_object' | 'go_to_agent' }> {
  return goal?.kind === 'go_to_object' || goal?.kind === 'go_to_agent';
}

function isLooseBananaReevaluationGoal(
  goal: AgentGoal | null
): goal is Extract<AgentGoal, { kind: 'go_to_object' | 'go_to_agent' | 'push_agent' | 'sabotage_agent' }> {
  return goal?.kind === 'go_to_object'
    || goal?.kind === 'go_to_agent'
    || goal?.kind === 'push_agent'
    || goal?.kind === 'sabotage_agent';
}

function shouldEmitImmediateWorldSync(event: SimulationGameEvent): boolean {
  switch (event.kind) {
    case 'bat_swing':
    case 'power_up_use':
    case 'drop':
    case 'push':
    case 'delivery':
    case 'respawn':
      return true;
    case 'pickup':
    case 'forced_drop':
    case 'bump_whiff':
    case 'power_up_throw':
    case 'fallback':
      return false;
  }
}
