import assert from 'node:assert/strict';
import test from 'node:test';

import type { CharacterAgentEngine } from '@noumena-labs/cogent-engine/character';
import { DirectorRuntime, parseDirectorConfig, type JsonValue } from '@noumena-labs/cogent-engine/orchestrator';

import { SimulationBus, type SimulationEvent } from './src/runtime/bus.ts';
import { applyDirectorDecision, applyTickFirstPass, type MutableWorldState } from './src/runtime/reducer.ts';
import { buildRefereeChoices, buildRefereePayload, SimulationRuntime } from './src/runtime/simulation-runtime.ts';
import { buildDecisionContext } from './src/runtime/decision-context.ts';
import type {
  AgentPerception,
  DirectorResolution,
  SimulationAgentState,
  SimulationObjectState,
  Vec2,
  WorldConflict,
} from './src/runtime/types.ts';

const DIRECTOR_CONFIG = parseDirectorConfig({
  id: 'banana-dash-referee',
  director: {
    role: 'Banana Dash referee',
    instructions: ['Resolve conflicts quickly using only the supplied scene state.'],
  },
  inputs: {
    referee_event: { kind: 'data', description: 'Referee event to resolve.' },
    scoreboard: { kind: 'data', description: 'Current score.' },
    scene_summary: { kind: 'data', description: 'Current scene summary.' },
  },
  tasks: {
    resolve_referee_event: {
      inputs: ['referee_event', 'scoreboard', 'scene_summary'],
      output: { shape: 'select_one', choices: 'runtime' },
    },
    narrate_scene: {
      inputs: ['scoreboard', 'scene_summary'],
      output: { shape: 'text', maxLength: 180 },
    },
  },
});

interface MutableRuntimeInternals {
  state: MutableWorldState;
  refereeTimeoutMs: number;
}

function createTimeoutEngine(): CharacterAgentEngine & { cancelCalls: number[] } {
  const cancelCalls: number[] = [];

  return {
    cancelCalls,
    async applyChatTemplate(messages, _addAssistant) {
      return messages.map((message) => `${message.role}: ${message.content}`).join('\n');
    },
    async queuePrompt(_contextKey, _promptText, _options) {
      return 1;
    },
    async runQueuedRequest(requestId, options = {}) {
      const signal = options.signal;
      if (!signal) {
        throw new Error('Expected an abortable director task.');
      }
      if (!signal.aborted) {
        await new Promise<void>((resolve) => {
          signal.addEventListener('abort', () => resolve(), { once: true });
        });
      }
      return {
        requestId,
        completed: false,
        failed: false,
        cancelled: true,
        outputText: '',
      };
    },
    async cancelQueuedRequest(requestId) {
      cancelCalls.push(requestId);
      return true;
    },
  };
}

function createOutputEngine(outputText: string): CharacterAgentEngine & { grammar?: string; promptText?: string } {
  let grammar: string | undefined;
  let promptText: string | undefined;
  return {
    get grammar() {
      return grammar;
    },
    get promptText() {
      return promptText;
    },
    async applyChatTemplate(messages, _addAssistant) {
      return messages.map((message) => `${message.role}: ${message.content}`).join('\n');
    },
    async queuePrompt(_contextKey, queuedPromptText, options) {
      promptText = queuedPromptText;
      grammar = typeof options === 'object' ? options.grammar : undefined;
      return 1;
    },
    async runQueuedRequest(requestId) {
      return {
        requestId,
        completed: true,
        failed: false,
        cancelled: false,
        outputText,
      };
    },
  };
}

function createAgent(
  id: string,
  name: string,
  position: Vec2,
  overrides: Partial<SimulationAgentState> = {}
): SimulationAgentState {
  return {
    id,
    name,
    position,
    heading: 0,
    speed: 0,
    emotion: null,
    status: '',
    intent: null,
    goal: null,
    holding: null,
    powerUp: null,
    frozenUntilTick: 0,
    intentIssuedAtTick: 0,
    thinking: false,
    cooldowns: {
      sabotageUntilTick: 0,
    },
    navigation: {
      detourTarget: null,
      blockedTicks: 0,
      obstacleId: null,
    },
    ...overrides,
  };
}

function createObject(
  id: string,
  kind: string,
  position: Vec2,
  overrides: Partial<SimulationObjectState> = {}
): SimulationObjectState {
  return {
    id,
    kind,
    label: kind,
    description: kind,
    position,
    active: true,
    contested: false,
    heldBy: null,
    tags: [],
    affordances: [],
    blocksMovement: false,
    collisionRadius: 0.45,
    ...overrides,
  };
}

function createWorldState(): MutableWorldState {
  return {
    tick: 0,
    timeSeconds: 0,
    bounds: { halfExtent: 8 },
    agents: [],
    objects: [],
    directorNote: null,
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home',
      respawnRules: [{ objectId: 'banana', delayTicks: 1, spawnPoints: [{ x: -4, z: -4 }] }],
      score: {
        deliveries: {},
        forcedDrops: {},
      },
      referee: { status: 'idle' },
      refereeMemory: { forcedDrops: [] },
      pendingRespawns: [],
      pendingIceImpacts: [],
      nextSpawnIndexByObjectId: {},
    },
  };
}

function expectJsonObject(value: JsonValue | undefined): Record<string, JsonValue> {
  assert.ok(value != null && typeof value === 'object' && !Array.isArray(value));
  return value as Record<string, JsonValue>;
}

function createForcedDropConflict(
  tick: number,
  attackerAgentId = 'attacker',
  targetAgentId = 'carrier'
): Extract<WorldConflict, { kind: 'forced_drop' }> {
  return {
    id: `drop:${attackerAgentId}:${targetAgentId}:${tick}`,
    kind: 'forced_drop',
    attackerAgentId,
    targetAgentId,
    objectId: 'banana',
  };
}

function populateForcedDropWorld(state: MutableWorldState): void {
  state.agents.push(
    createAgent('carrier', 'Carrier', { x: 0, z: 0 }, {
      status: 'carrying the banana to home base',
      holding: 'banana',
      intent: { kind: 'go_to_object', objectId: 'home', emotion: 'alert' },
      goal: { kind: 'deliver', objectId: 'home', label: 'run to home base' },
    }),
    createAgent('attacker', 'Attacker', { x: 0.5, z: 0 }, {
      status: 'lining up a bump',
      intent: { kind: 'sabotage', agentId: 'carrier', method: 'bump', emotion: 'alert' },
      goal: { kind: 'sabotage_agent', agentId: 'carrier', method: 'bump', label: 'bump Carrier' },
    })
  );
  state.objects.push(
    createObject('banana', 'banana', { x: 0, z: 0 }, {
      label: 'banana',
      contested: true,
      heldBy: 'carrier',
      tags: ['food'],
      affordances: [{ kind: 'pick_up', label: 'grab banana' }],
    }),
    createObject('home', 'goal', { x: 5, z: 5 }, {
      label: 'home base',
      tags: ['goal', 'score'],
    })
  );
  state.game.score.deliveries.carrier = 0;
  state.game.score.deliveries.attacker = 0;
  state.game.score.forcedDrops.carrier = 0;
  state.game.score.forcedDrops.attacker = 0;
}

function timeoutAfter(ms: number, message: string): Promise<never> {
  return new Promise((_, reject) => {
    setTimeout(() => reject(new Error(message)), ms);
  });
}

test('SimulationRuntime applies deterministic referee fallback after timeout', async () => {
  const engine = createTimeoutEngine();
  const director = new DirectorRuntime(engine, DIRECTOR_CONFIG);
  const bus = new SimulationBus();
  const events: SimulationEvent[] = [];
  bus.onAny((event) => {
    events.push(event);
  });

  const runtime = new SimulationRuntime(director, {
    bus,
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home',
      respawnRules: [{ objectId: 'banana', delayTicks: 1, spawnPoints: [{ x: -4, z: -4 }] }],
    },
    refereeTimeoutMs: 5,
  });

  const internals = runtime as unknown as MutableRuntimeInternals;
  internals.refereeTimeoutMs = 5;
  const state = internals.state;
  state.agents.push(
    createAgent('carrier', 'Carrier', { x: 0, z: 0 }, {
      status: 'carrying the banana to home base',
      holding: 'banana',
      intent: { kind: 'go_to_object', objectId: 'home', emotion: 'alert' },
      goal: { kind: 'deliver', objectId: 'home', label: 'run to home base' },
    }),
    createAgent('attacker', 'Attacker', { x: 0.5, z: 0 }, {
      status: 'lining up a bump',
      intent: { kind: 'sabotage', agentId: 'carrier', method: 'bump', emotion: 'alert' },
      goal: { kind: 'sabotage_agent', agentId: 'carrier', method: 'bump', label: 'bump Carrier' },
    })
  );
  state.objects.push(
    createObject('banana', 'banana', { x: 0, z: 0 }, {
      label: 'banana',
      contested: true,
      heldBy: 'carrier',
      tags: ['food'],
      affordances: [{ kind: 'pick_up', label: 'grab banana' }],
    }),
    createObject('home', 'goal', { x: 5, z: 5 }, {
      label: 'home base',
      tags: ['goal', 'score'],
    })
  );
  state.game.score.deliveries.carrier = 0;
  state.game.score.deliveries.attacker = 0;
  state.game.score.forcedDrops.carrier = 0;
  state.game.score.forcedDrops.attacker = 0;

  try {
    await Promise.race([
      runtime.step(0.1),
      timeoutAfter(500, 'Simulation step hung after the director timed out.'),
    ]);

    await Promise.race([
      runtime.waitForIdle(),
      timeoutAfter(500, 'Simulation never returned to idle after the timed out ruling.'),
    ]);

    const snapshot = runtime.getSnapshot();
    assert.equal(snapshot.game.referee.status, 'idle');
    assert.equal(runtime.isBusy(), false);
    assert.equal(snapshot.game.score.forcedDrops.attacker, 1);
    assert.equal(snapshot.objects.find((object) => object.id === 'banana')?.heldBy, null);
    assert.equal(snapshot.agents.find((agent) => agent.id === 'carrier')?.holding, null);
    assert.deepEqual(engine.cancelCalls, [1]);
    assert.ok(
      events.some(
        (event) =>
          event.kind === 'runtime-error' &&
          event.severity === 'warning' &&
          event.source === 'referee' &&
          event.message.includes('fallback')
      )
    );
    assert.equal(events.some((event) => event.kind === 'director-decision'), true);
    assert.equal(events.some((event) => event.kind === 'game-event' && event.event.kind === 'forced_drop'), true);
  } finally {
    await runtime.dispose();
  }
});

test('SimulationRuntime maps referee selection choices to director resolutions', async () => {
  const engine = createOutputEngine('pickup:beck');
  const director = new DirectorRuntime(engine, DIRECTOR_CONFIG);
  const bus = new SimulationBus();
  const events: SimulationEvent[] = [];
  bus.onAny((event) => {
    events.push(event);
  });

  const runtime = new SimulationRuntime(director, {
    bus,
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home',
      respawnRules: [{ objectId: 'banana', delayTicks: 1, spawnPoints: [{ x: -4, z: -4 }] }],
    },
  });

  const state = (runtime as unknown as MutableRuntimeInternals).state;
  state.agents.push(
    createAgent('aria', 'Aria', { x: 0, z: 0 }, {
      intent: { kind: 'pick_up', objectId: 'banana', emotion: 'happy' },
      goal: {
        kind: 'object_action',
        objectId: 'banana',
        affordance: { kind: 'pick_up', label: 'grab banana' },
        label: 'grab banana',
      },
    }),
    createAgent('beck', 'Beck', { x: 0.1, z: 0 }, {
      intent: { kind: 'pick_up', objectId: 'banana', emotion: 'happy' },
      goal: {
        kind: 'object_action',
        objectId: 'banana',
        affordance: { kind: 'pick_up', label: 'grab banana' },
        label: 'grab banana',
      },
    })
  );
  state.objects.push(
    createObject('banana', 'banana', { x: 0, z: 0 }, {
      label: 'banana',
      contested: true,
      tags: ['food'],
      affordances: [{ kind: 'pick_up', label: 'grab banana' }],
    }),
    createObject('home', 'goal', { x: 5, z: 5 }, {
      label: 'home base',
      tags: ['goal', 'score'],
    })
  );

  try {
    await Promise.race([
      runtime.step(0.1),
      timeoutAfter(500, 'Simulation step hung while applying selected referee ruling.'),
    ]);
    await Promise.race([
      runtime.waitForIdle(),
      timeoutAfter(500, 'Simulation never returned to idle after selected referee ruling.'),
    ]);

    const snapshot = runtime.getSnapshot();
    assert.equal(snapshot.objects.find((object) => object.id === 'banana')?.heldBy, 'beck');
    assert.equal(snapshot.agents.find((agent) => agent.id === 'beck')?.holding, 'banana');
    assert.ok(engine.grammar?.includes('"pickup:beck"'));
    assert.ok(engine.grammar?.includes('"deny"'));
    assert.equal(events.some((event) => event.kind === 'director-decision'), true);
    assert.equal(events.some((event) => event.kind === 'runtime-error'), false);
  } finally {
    await runtime.dispose();
  }
});

test('SimulationRuntime dispose leaves shared bus listeners intact', async () => {
  const director = new DirectorRuntime(createOutputEngine('hold'), DIRECTOR_CONFIG);
  const bus = new SimulationBus();
  const events: SimulationEvent[] = [];
  bus.onAny((event) => {
    events.push(event);
  });

  const runtime = new SimulationRuntime(director, {
    bus,
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home',
      respawnRules: [{ objectId: 'banana', delayTicks: 1, spawnPoints: [{ x: -4, z: -4 }] }],
    },
  });

  try {
    await runtime.dispose();
    bus.emit({ kind: 'tick-start', tick: 99, timeSeconds: 9.9 });
    assert.equal(events.at(-1)?.kind, 'tick-start');
  } finally {
    await runtime.dispose();
  }
});

test('forced-drop referee choices expose policy through grammar without leaking payload fields', () => {
  const state = createWorldState();
  state.tick = 8;
  populateForcedDropWorld(state);
  const conflict = createForcedDropConflict(state.tick);
  const director = new DirectorRuntime(createOutputEngine('drop'), DIRECTOR_CONFIG);
  const payload = expectJsonObject(buildRefereePayload(state, conflict));
  const refereeEvent = expectJsonObject(payload.referee_event);
  const attempt = expectJsonObject(refereeEvent.attempt);
  const recentHistory = expectJsonObject(refereeEvent.recent_history);
  const rulingPolicy = expectJsonObject(refereeEvent.ruling_policy);

  assert.deepEqual(payload.scoreboard, { carrier: 0, attacker: 0 });
  assert.equal(attempt.attackerName, 'Attacker');
  assert.equal(attempt.targetName, 'Carrier');
  assert.equal(attempt.currentHolder, 'carrier');
  assert.equal(attempt.distance, 0.5);
  assert.equal(attempt.attackerIntentAgeTicks, 8);
  assert.equal(attempt.targetDistanceToGoal, 7.07);
  assert.deepEqual(recentHistory.same_pair, []);
  assert.deepEqual(recentHistory.recent_forced_drops, []);
  assert.deepEqual(rulingPolicy.availableOutcomes, ['drop', 'hold', 'attacker_fumbles']);
  assert.deepEqual(rulingPolicy.suppressedOutcomes, []);
  assert.equal(rulingPolicy.fallbackOutcome, 'drop');
  assert.match(String(rulingPolicy.varietyNote), /occasional/);

  const freshChoices = buildRefereeChoices(state, conflict);
  assert.deepEqual(freshChoices.map((choice) => choice.id), ['drop', 'hold', 'attacker_fumbles']);
  const freshRequest = {
    inputs: payload as Record<string, JsonValue>,
    choices: freshChoices,
  };
  const freshGrammar = director.getTaskGrammar<DirectorResolution>('resolve_referee_event', {
    ...freshRequest,
  });
  assert.match(freshGrammar ?? '', /"drop"/);
  assert.match(freshGrammar ?? '', /"hold"/);
  assert.match(freshGrammar ?? '', /"attacker_fumbles"/);

  const prompt = director.getTaskPrompt<DirectorResolution>('resolve_referee_event', freshRequest);
  assert.match(prompt.userPrompt, /recent_history/);
  assert.match(prompt.userPrompt, /ruling_policy/);
  assert.match(prompt.userPrompt, /currentHolder/);
  assert.match(prompt.userPrompt, /fallbackOutcome/);
  assert.doesNotMatch(prompt.userPrompt, /winnerAgentId/);

  state.game.refereeMemory.forcedDrops.push({
    tick: 7,
    attackerAgentId: 'attacker',
    targetAgentId: 'carrier',
    objectId: 'banana',
    outcome: 'attacker_fumbles',
  });
  const suppressedPayload = expectJsonObject(buildRefereePayload(state, conflict));
  const suppressedRefereeEvent = expectJsonObject(suppressedPayload.referee_event);
  const suppressedRecentHistory = expectJsonObject(suppressedRefereeEvent.recent_history);
  const suppressedPolicy = expectJsonObject(suppressedRefereeEvent.ruling_policy);
  const suppressedChoices = buildRefereeChoices(state, conflict);
  assert.deepEqual(suppressedChoices.map((choice) => choice.id), ['drop', 'hold']);
  const suppressedGrammar = director.getTaskGrammar<DirectorResolution>('resolve_referee_event', {
    inputs: suppressedPayload as Record<string, JsonValue>,
    choices: suppressedChoices,
  });
  assert.match(suppressedGrammar ?? '', /"drop"/);
  assert.match(suppressedGrammar ?? '', /"hold"/);
  assert.doesNotMatch(suppressedGrammar ?? '', /"attacker_fumbles"/);
  assert.deepEqual(suppressedRecentHistory.same_pair, [
    {
      tick: 7,
      attackerAgentId: 'attacker',
      targetAgentId: 'carrier',
      objectId: 'banana',
      outcome: 'attacker_fumbles',
    },
  ]);
  assert.deepEqual(suppressedPolicy.suppressedOutcomes, ['attacker_fumbles']);
  assert.equal(suppressedPolicy.fallbackOutcome, 'drop');
  assert.match(String(suppressedPolicy.varietyNote), /fair variation/);
});

test('forced-drop policy suppresses three repeated global outcomes when alternatives remain', () => {
  const state = createWorldState();
  state.tick = 20;
  populateForcedDropWorld(state);
  state.game.refereeMemory.forcedDrops.push(
    { tick: 17, attackerAgentId: 'a', targetAgentId: 'b', objectId: 'banana', outcome: 'drop' },
    { tick: 18, attackerAgentId: 'c', targetAgentId: 'd', objectId: 'banana', outcome: 'drop' },
    { tick: 19, attackerAgentId: 'e', targetAgentId: 'f', objectId: 'banana', outcome: 'drop' }
  );

  const conflict = createForcedDropConflict(state.tick);
  const choices = buildRefereeChoices(state, conflict);
  assert.deepEqual(choices.map((choice) => choice.id), ['hold', 'attacker_fumbles']);
  const payload = expectJsonObject(buildRefereePayload(state, conflict));
  const refereeEvent = expectJsonObject(payload.referee_event);
  const rulingPolicy = expectJsonObject(refereeEvent.ruling_policy);
  assert.deepEqual(rulingPolicy.suppressedOutcomes, ['drop']);

  const director = new DirectorRuntime(createOutputEngine('hold'), DIRECTOR_CONFIG);
  const grammar = director.getTaskGrammar<DirectorResolution>('resolve_referee_event', {
    inputs: payload as Record<string, JsonValue>,
    choices,
  });
  assert.doesNotMatch(grammar ?? '', /"drop"/);
  assert.match(grammar ?? '', /"hold"/);
  assert.match(grammar ?? '', /"attacker_fumbles"/);
});

test('forced-drop rulings record referee history and apply sabotage cooldown', () => {
  const state = createWorldState();
  state.tick = 10;
  populateForcedDropWorld(state);
  const conflict = createForcedDropConflict(state.tick);
  state.game.referee = { status: 'ruling', conflict, startedAtTick: state.tick };

  applyDirectorDecision(state, {
    note: 'The referee rules that the attacker overcommits.',
    resolutions: [
      {
        conflictId: conflict.id,
        objectId: 'banana',
        winnerAgentId: null,
        outcome: 'attacker_fumbles',
      },
    ],
  });

  const attacker = state.agents.find((agent) => agent.id === 'attacker');
  assert.equal(attacker?.cooldowns.sabotageUntilTick, 13);
  assert.deepEqual(state.game.refereeMemory.forcedDrops, [
    {
      tick: 10,
      attackerAgentId: 'attacker',
      targetAgentId: 'carrier',
      objectId: 'banana',
      outcome: 'attacker_fumbles',
    },
  ]);
});

test('agent decision context hides bump option while sabotage is cooling down', () => {
  const state = createWorldState();
  populateForcedDropWorld(state);
  const attacker = state.agents.find((agent) => agent.id === 'attacker')!;
  attacker.cooldowns.sabotageUntilTick = 12;
  const carrier = state.agents.find((agent) => agent.id === 'carrier')!;
  const banana = state.objects.find((object) => object.id === 'banana')!;
  const home = state.objects.find((object) => object.id === 'home')!;
  const perception: AgentPerception = {
    self: attacker,
    nearbyAgents: [
      {
        id: carrier.id,
        name: carrier.name,
        distance: 1.6,
        direction: { x: -1, z: 0 },
        emotion: carrier.emotion,
        status: carrier.status,
        holding: carrier.holding,
        powerUp: carrier.powerUp?.kind ?? null,
        frozenUntilTick: carrier.frozenUntilTick,
      },
    ],
    nearbyObjects: [
      {
        id: banana.id,
        kind: banana.kind,
        label: banana.label,
        description: banana.description,
        distance: 1.6,
        direction: { x: -1, z: 0 },
        active: banana.active,
        heldBy: banana.heldBy,
        contested: banana.contested,
        affordances: banana.affordances,
        tags: banana.tags,
        blocksMovement: banana.blocksMovement,
        collisionRadius: banana.collisionRadius,
      },
      {
        id: home.id,
        kind: home.kind,
        label: home.label,
        description: home.description,
        distance: 6,
        direction: { x: 1, z: 0 },
        active: home.active,
        heldBy: home.heldBy,
        contested: home.contested,
        affordances: home.affordances,
        tags: home.tags,
        blocksMovement: home.blocksMovement,
        collisionRadius: home.collisionRadius,
      },
    ],
    tick: 10,
    bounds: state.bounds,
    directorNote: null,
    game: state.game,
  };

  const decision = buildDecisionContext(perception);
  assert.equal(decision.options.some((option) => option.label === 'bump Carrier'), false);
  assert.equal(decision.options.some((option) => option.label === 'chase Carrier'), false);
  assert.equal(decision.options.some((option) => option.label === 'push Carrier'), true);
  assert.equal(decision.options.some((option) => option.label === 'wait'), false);
  assert.match(decision.prompt, /already in contact range/);
  assert.match(decision.prompt, /Close agents/);

  attacker.cooldowns.sabotageUntilTick = 0;
  const readyDecision = buildDecisionContext(perception);
  assert.equal(readyDecision.options.some((option) => option.label === 'bump Carrier'), true);
  assert.equal(readyDecision.options.some((option) => option.label === 'push Carrier'), false);
});

test('push relocates a nearby carrier without dropping the banana', () => {
  const state = createWorldState();
  state.tick = 10;
  state.agents.push(
    createAgent('carrier', 'Carrier', { x: 0, z: 0 }, {
      speed: 1.2,
      status: 'carrying the banana to home base',
      holding: 'banana',
      intent: { kind: 'go_to_object', objectId: 'home', emotion: 'alert' },
      goal: { kind: 'deliver', objectId: 'home', label: 'run to home base' },
    }),
    createAgent('pusher', 'Pusher', { x: 0.5, z: 0 }, {
      status: 'pushing Carrier',
      intent: { kind: 'push', agentId: 'carrier', emotion: 'alert' },
      goal: { kind: 'push_agent', agentId: 'carrier', label: 'push Carrier' },
    })
  );
  state.objects.push(
    createObject('banana', 'banana', { x: 0, z: 0 }, {
      label: 'banana',
      contested: true,
      heldBy: 'carrier',
      tags: ['food'],
      affordances: [{ kind: 'pick_up', label: 'grab banana' }],
    }),
    createObject('home', 'goal', { x: 5, z: 5 }, {
      label: 'home base',
      tags: ['goal', 'score'],
    })
  );
  state.game.score.forcedDrops.pusher = 0;

  const result = applyTickFirstPass(state, 0.1);
  const carrier = state.agents.find((agent) => agent.id === 'carrier')!;
  const pusher = state.agents.find((agent) => agent.id === 'pusher')!;
  const banana = state.objects.find((object) => object.id === 'banana')!;
  const pushEvent = result.events.find((event) => event.kind === 'push');

  assert.ok(pushEvent && pushEvent.kind === 'push');
  assert.equal(result.events.some((event) => event.kind === 'drop'), false);
  assert.equal(carrier.holding, 'banana');
  assert.equal(banana.heldBy, 'carrier');
  assert.deepEqual(carrier.position, pushEvent.to);
  assert.deepEqual(banana.position, pushEvent.to);
  assert.equal(pusher.intent, null);
  assert.equal(pusher.goal, null);
  assert.equal(carrier.intent, null);
  assert.equal(carrier.goal, null);
  assert.ok(Math.hypot(pushEvent.to.x - pushEvent.from.x, pushEvent.to.z - pushEvent.from.z) > 0.5);
  assert.equal(state.game.score.forcedDrops.pusher, 0);
});

test('invalid repeated forced-drop fumble falls back through policy without pausing', async () => {
  const engine = createOutputEngine('attacker_fumbles');
  const director = new DirectorRuntime(engine, DIRECTOR_CONFIG);
  const bus = new SimulationBus();
  const events: SimulationEvent[] = [];
  bus.onAny((event) => {
    events.push(event);
  });

  const runtime = new SimulationRuntime(director, {
    bus,
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home',
      respawnRules: [{ objectId: 'banana', delayTicks: 1, spawnPoints: [{ x: -4, z: -4 }] }],
    },
  });

  const state = (runtime as unknown as MutableRuntimeInternals).state;
  populateForcedDropWorld(state);
  state.game.refereeMemory.forcedDrops.push({
    tick: 0,
    attackerAgentId: 'attacker',
    targetAgentId: 'carrier',
    objectId: 'banana',
    outcome: 'attacker_fumbles',
  });

  try {
    await Promise.race([
      runtime.step(0.1),
      timeoutAfter(500, 'Simulation step hung while applying forced-drop policy fallback.'),
    ]);
    await Promise.race([
      runtime.waitForIdle(),
      timeoutAfter(500, 'Simulation never returned to idle after forced-drop policy fallback.'),
    ]);

    const snapshot = runtime.getSnapshot();
    const attacker = snapshot.agents.find((agent) => agent.id === 'attacker');
    assert.equal(runtime.isBusy(), false);
    assert.equal(snapshot.objects.find((object) => object.id === 'banana')?.heldBy, null);
    assert.equal(snapshot.game.score.forcedDrops.attacker, 1);
    assert.equal(attacker?.cooldowns.sabotageUntilTick, 4);
    assert.equal(snapshot.game.refereeMemory.forcedDrops.length, 2);
    assert.equal(snapshot.game.refereeMemory.forcedDrops.at(-1)?.outcome, 'drop');
    assert.match(engine.grammar ?? '', /"drop"/);
    assert.match(engine.grammar ?? '', /"hold"/);
    assert.doesNotMatch(engine.grammar ?? '', /"attacker_fumbles"/);
    assert.match(engine.promptText ?? '', /recent_history/);
    assert.match(engine.promptText ?? '', /ruling_policy/);
    assert.doesNotMatch(engine.promptText ?? '', /winnerAgentId/);
    assert.equal(
      events.some(
        (event) =>
          event.kind === 'game-event' &&
          event.event.kind === 'forced_drop' &&
          event.event.outcome === 'drop'
      ),
      true
    );
    assert.equal(
      events.some(
        (event) =>
          event.kind === 'game-event' &&
          event.event.kind === 'forced_drop' &&
          event.event.outcome === 'attacker_fumbles'
      ),
      false
    );
    assert.equal(
      events.some(
        (event) =>
          event.kind === 'runtime-error' &&
          event.severity === 'warning' &&
          event.source === 'referee' &&
          event.message.includes('did not match any available choice')
      ),
      true
    );
  } finally {
    await runtime.dispose();
  }
});

test('forced drops throw the banana away from the carrier scrum', () => {
  const state = createWorldState();
  state.tick = 10;
  state.agents.push(
    createAgent('carrier', 'Carrier', { x: 0, z: 0 }, {
      heading: 0,
      status: 'carrying the banana to home base',
      holding: 'banana',
      intent: { kind: 'go_to_object', objectId: 'home', emotion: 'alert' },
      goal: { kind: 'deliver', objectId: 'home', label: 'run to home base' },
    }),
    createAgent('attacker', 'Attacker', { x: 0.45, z: 0 }, {
      status: 'lining up a bump',
      intent: { kind: 'sabotage', agentId: 'carrier', method: 'bump', emotion: 'alert' },
      goal: { kind: 'sabotage_agent', agentId: 'carrier', method: 'bump', label: 'bump Carrier' },
    }),
    createAgent('shadow-1', 'Shadow 1', { x: -0.55, z: 0.15 }),
    createAgent('shadow-2', 'Shadow 2', { x: 0.2, z: 0.75 })
  );
  state.objects.push(
    createObject('banana', 'banana', { x: 0, z: 0 }, {
      label: 'banana',
      description: 'The bright yellow banana every agent is scrambling to carry home.',
      contested: true,
      heldBy: 'carrier',
      tags: ['food'],
      affordances: [{ kind: 'pick_up', label: 'grab banana' }],
      collisionRadius: 0.2,
    }),
    createObject('home', 'goal', { x: 0, z: -6 }, {
      label: 'home base',
      description: 'The shared scoring circle where a carrier cashes in the banana.',
      tags: ['goal', 'score'],
      collisionRadius: 1.2,
    }),
    createObject('rock', 'rock', { x: 2.8, z: -0.6 }, {
      label: 'rock',
      description: 'A blocking rock.',
      blocksMovement: true,
      collisionRadius: 0.55,
      tags: ['obstacle'],
    })
  );
  state.game.score.deliveries.carrier = 0;
  state.game.score.forcedDrops.attacker = 0;
  state.game.score.forcedDrops.carrier = 0;
  state.game.referee = {
    status: 'ruling',
    startedAtTick: state.tick,
    conflict: {
      id: 'drop:attacker:carrier:10',
      kind: 'forced_drop',
      attackerAgentId: 'attacker',
      targetAgentId: 'carrier',
      objectId: 'banana',
    },
  };

  const events = applyDirectorDecision(state, {
    note: 'The referee rules that the hit jars the banana loose.',
    resolutions: [
      {
        conflictId: 'drop:attacker:carrier:10',
        objectId: 'banana',
        winnerAgentId: null,
        outcome: 'drop',
      },
    ],
  });

  const banana = state.objects.find((object) => object.id === 'banana');
  const carrier = state.agents.find((agent) => agent.id === 'carrier');
  assert.ok(banana);
  assert.ok(carrier);
  assert.equal(banana.heldBy, null);
  assert.equal(carrier.holding, null);
  assert.equal(state.game.score.forcedDrops.attacker, 1);

  const carrierDistance = Math.hypot(banana.position.x - carrier.position.x, banana.position.z - carrier.position.z);
  assert.ok(carrierDistance >= 2.4, `expected banana to land well clear of the carrier scrum, got ${carrierDistance.toFixed(2)}`);

  const nearestAgentDistance = Math.min(
    ...state.agents.map((agent) => Math.hypot(banana.position.x - agent.position.x, banana.position.z - agent.position.z))
  );
  assert.ok(nearestAgentDistance >= 0.9, `expected banana to avoid immediate re-capture, got nearest agent distance ${nearestAgentDistance.toFixed(2)}`);

  const dropEvent = events.find((event) => event.kind === 'drop');
  const forcedDropEvent = events.find((event) => event.kind === 'forced_drop');
  assert.ok(dropEvent && dropEvent.kind === 'drop');
  assert.ok(forcedDropEvent && forcedDropEvent.kind === 'forced_drop');
});
