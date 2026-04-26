import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DirectorRuntime,
  parseDirectorConfig,
  type DirectorRuntimeEngine,
  type JsonValue,
} from '@noumena-labs/cogent-engine/orchestrator';

import { SimulationBus, type SimulationEvent } from './src/runtime/bus.ts';
import {
  applyDirectorDecision,
  applyTickFirstPass,
  BANANA_CARRIER_SPEED_MULTIPLIER,
  type MutableWorldState,
} from './src/runtime/reducer.ts';
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
    narration_brief: { kind: 'text', description: 'Observation facts for play-by-play.' },
  },
  tasks: {
    resolve_referee_event: {
      inputs: ['referee_event', 'scoreboard', 'scene_summary'],
      output: { shape: 'select_one', choices: 'runtime' },
    },
    narrate_scene: {
      purpose: 'Call these observations as one fun, witty old-timey radio line.',
      instructions: ['Use the observations only.', 'Name a player, the live action, and the stakes.'],
      inputs: ['narration_brief'],
      output: { shape: 'text' },
    },
  },
});

interface MutableRuntimeInternals {
  state: MutableWorldState;
  refereeTimeoutMs: number;
}

interface StubChooser {
  query(perception: AgentPerception, options?: { signal?: AbortSignal; timeoutMs?: number }): Promise<{
    goal: null;
    status: 'aborted';
    rawText: string;
  }>;
}

function createTimeoutEngine(): DirectorRuntimeEngine & { cancelCalls: number[] } {
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

function createOutputEngine(outputText: string): DirectorRuntimeEngine & { grammar?: string; promptText?: string } {
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

function createAbortedChooser(): StubChooser {
  return {
    async query(_perception, options = {}) {
      if (options.signal && !options.signal.aborted) {
        await new Promise<void>((resolve) => {
          options.signal?.addEventListener('abort', () => resolve(), { once: true });
        });
      }
      return { goal: null, status: 'aborted', rawText: '' };
    },
  };
}

function timeoutAfter(ms: number, message: string): Promise<never> {
  return new Promise((_, reject) => {
    setTimeout(() => reject(new Error(message)), ms);
  });
}

test('narration prompt includes recent beats and rejects repeated previous calls', async () => {
  const engine = createOutputEngine('one race');
  const director = new DirectorRuntime(engine, DIRECTOR_CONFIG);
  const bus = new SimulationBus();
  const notes: string[] = [];
  bus.on('world-note', (event) => {
    notes.push(event.note);
  });

  const runtime = new SimulationRuntime(director, {
    bus,
    directorCadenceTicks: 1,
    initialDirectorNote: 'one race',
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home',
      respawnRules: [{ objectId: 'banana', delayTicks: 1, spawnPoints: [{ x: -4, z: -4 }] }],
    },
  });

  try {
    const internals = runtime as unknown as MutableRuntimeInternals & {
      simulationAgents: Map<string, StubChooser>;
      recentNarrationEvents: Array<{ tick: number; text: string }>;
      maybeStartNarration(): void;
      narrationInFlight: Promise<void> | null;
    };
    internals.simulationAgents.set('aria', createAbortedChooser());
    internals.state.tick = 12;
    internals.state.game.score.deliveries.aria = 1;
    internals.state.game.score.deliveries.mira = 0;
    internals.state.agents.push(
      createAgent('aria', 'Aria', { x: 0, z: 0 }, {
        holding: 'banana',
        powerUp: { kind: 'ice_cube', objectId: 'ice' },
        status: 'carrying the banana to home base',
        goal: { kind: 'deliver', objectId: 'home', label: 'run to home base' },
        intent: { kind: 'deliver', objectId: 'home', emotion: 'alert' },
      }),
      createAgent('mira', 'Mira', { x: 0.8, z: 0 }, {
        powerUp: { kind: 'bat', objectId: 'bat' },
        status: 'lining up a bat bonk',
        goal: { kind: 'sabotage_agent', agentId: 'aria', method: 'bat', label: 'bonk Aria' },
        intent: { kind: 'sabotage', agentId: 'aria', method: 'bat', emotion: 'alert' },
      })
    );
    internals.state.objects.push(
      createObject('banana', 'banana', { x: 0, z: 0 }, {
        label: 'banana',
        contested: true,
        heldBy: 'aria',
        tags: ['food', 'score'],
        affordances: [{ kind: 'pick_up', label: 'grab banana' }],
      }),
      createObject('home', 'goal', { x: 1, z: 0 }, {
        label: 'home base',
        tags: ['goal', 'score'],
      })
    );
    internals.recentNarrationEvents.push({ tick: 11, text: 'Aria grabbed the banana and turned for home' });

    internals.maybeStartNarration();
    await internals.narrationInFlight;

    assert.match(engine.promptText ?? '', /Call these observations as one fun, witty old-timey radio line/);
    assert.match(engine.promptText ?? '', /Response:\nWrite only the final answer\./);
    assert.match(engine.promptText ?? '', /Write exactly one complete sentence as an old-timey sports caller at an active game\./);
    assert.match(engine.promptText ?? '', /include at least one player name, describe live action, and mention the stakes/);
    assert.match(engine.promptText ?? '', /Do not answer with only a player name, label, list, fragment, or JSON\./);
    assert.match(engine.promptText ?? '', /Use 8 to 24 words\./);
    assert.match(engine.promptText ?? '', /- Previous call to avoid: one race\./);
    assert.match(engine.promptText ?? '', /- Aria has the banana\./);
    assert.match(engine.promptText ?? '', /- Aria is on the doorstep of home base\./);
    assert.match(engine.promptText ?? '', /- Mira is trying to smack Aria with the bat and knock the banana loose\./);
    assert.match(engine.promptText ?? '', /- Recent event: Aria grabbed the banana and turned for home\./);
    assert.doesNotMatch(engine.promptText ?? '', /Output shape:/);
    assert.doesNotMatch(engine.promptText ?? '', /Description: Play-by-play/);
    assert.doesNotMatch(engine.promptText ?? '', /Play:/);
    assert.doesNotMatch(engine.promptText ?? '', /Threats:/);
    assert.equal(notes.length, 0);
    assert.equal(internals.state.directorNote, 'one race');
  } finally {
    await runtime.dispose();
  }
});

test('SimulationRuntime emits raw trace when narration is rejected', async () => {
  const engine = createOutputEngine('Sol');
  const director = new DirectorRuntime(engine, DIRECTOR_CONFIG);
  const bus = new SimulationBus();
  const events: SimulationEvent[] = [];
  bus.onAny((event) => {
    events.push(event);
  });

  const runtime = new SimulationRuntime(director, {
    bus,
    directorCadenceTicks: 1,
    initialDirectorNote: 'The banana race is already rolling.',
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home',
      respawnRules: [{ objectId: 'banana', delayTicks: 1, spawnPoints: [{ x: -4, z: -4 }] }],
    },
  });

  try {
    const internals = runtime as unknown as MutableRuntimeInternals & {
      simulationAgents: Map<string, StubChooser>;
      maybeStartNarration(): void;
      narrationInFlight: Promise<void> | null;
    };
    internals.simulationAgents.set('sol', createAbortedChooser());
    internals.state.tick = 12;
    internals.state.game.score.deliveries.sol = 0;
    internals.state.agents.push(
      createAgent('sol', 'Sol', { x: 0, z: 0 }, {
        status: 'charging toward the banana',
        intent: { kind: 'go_to_object', objectId: 'banana', emotion: 'alert' },
      })
    );
    internals.state.objects.push(
      createObject('banana', 'banana', { x: 1, z: 0 }, {
        label: 'banana',
        contested: true,
        tags: ['food', 'score'],
        affordances: [{ kind: 'pick_up', label: 'grab banana' }],
      }),
      createObject('home', 'goal', { x: 5, z: 0 }, {
        label: 'home base',
        tags: ['goal', 'score'],
      })
    );

    internals.maybeStartNarration();
    await internals.narrationInFlight;

    const trace = events.find((event) => event.kind === 'director-narration-trace');
    assert.ok(trace);
    assert.equal(trace.rawText, 'Sol');
    assert.equal(trace.parsedText, 'Sol');
    assert.equal(trace.accepted, false);
    assert.equal(trace.reason, 'too short');
    assert.equal(events.some((event) => event.kind === 'world-note'), false);
    assert.equal(internals.state.directorNote, 'The banana race is already rolling.');
  } finally {
    await runtime.dispose();
  }
});

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
  assert.match(decision.prompt, /while sabotage is cooling down/);

  attacker.cooldowns.sabotageUntilTick = 0;
  const readyDecision = buildDecisionContext(perception);
  assert.equal(readyDecision.options.some((option) => option.label === 'bump Carrier'), true);
  assert.equal(readyDecision.options.some((option) => option.label === 'push Carrier'), false);
});

test('agent decision context keeps chase first for chase-leaning agents', () => {
  const state = createWorldState();
  const seeker = createAgent('aria', 'Aria', { x: 0, z: 0 }, {
    archetype: 'aria',
  });
  const carrier = createAgent('carrier', 'Carrier', { x: 0, z: 4 }, {
    status: 'carrying the banana to home base',
    holding: 'banana',
  });
  const perception: AgentPerception = {
    self: seeker,
    nearbyAgents: [
      {
        id: carrier.id,
        name: carrier.name,
        distance: 4,
        direction: { x: 0, z: 1 },
        emotion: carrier.emotion,
        status: carrier.status,
        holding: carrier.holding,
        powerUp: carrier.powerUp?.kind ?? null,
        frozenUntilTick: carrier.frozenUntilTick,
      },
    ],
    nearbyObjects: [
      {
        id: 'banana',
        kind: 'banana',
        label: 'banana',
        description: 'banana',
        distance: 4,
        direction: { x: 0, z: 1 },
        active: true,
        heldBy: 'carrier',
        contested: true,
        affordances: [{ kind: 'pick_up', label: 'grab banana' }],
        tags: ['food'],
        blocksMovement: false,
        collisionRadius: 0.2,
      },
      {
        id: 'bat',
        kind: 'bat',
        label: 'baseball bat',
        description: 'bat',
        distance: 1.6,
        direction: { x: 1, z: 0 },
        active: true,
        heldBy: null,
        contested: true,
        affordances: [{ kind: 'pick_up', label: 'grab baseball bat' }],
        tags: ['power_up'],
        blocksMovement: false,
        collisionRadius: 0.24,
      },
      {
        id: 'home',
        kind: 'goal',
        label: 'home base',
        description: 'home base',
        distance: 6,
        direction: { x: 0, z: -1 },
        active: true,
        heldBy: null,
        contested: false,
        affordances: [],
        tags: ['goal'],
        blocksMovement: false,
        collisionRadius: 1.2,
      },
    ],
    tick: 10,
    bounds: state.bounds,
    directorNote: null,
    game: state.game,
  };

  const decision = buildDecisionContext(perception);

  assert.equal(decision.options[0]?.label, 'chase Carrier');
  assert.equal(decision.options.some((option) => option.label === 'go get the baseball bat'), true);
  assert.match(decision.prompt, /weigh direct pressure on Carrier against any nearby power-up/);
  assert.match(decision.prompt, /Power-ups are setup plays only when they help you grab it faster/);
});

test('agent decision context still surfaces nearby power-ups as the top setup for power-up-leaning agents', () => {
  const state = createWorldState();
  const seeker = createAgent('mira', 'Mira', { x: 0, z: 0 }, {
    archetype: 'mira',
  });
  const carrier = createAgent('carrier', 'Carrier', { x: 0, z: 4 }, {
    status: 'carrying the banana to home base',
    holding: 'banana',
  });
  const perception: AgentPerception = {
    self: seeker,
    nearbyAgents: [
      {
        id: carrier.id,
        name: carrier.name,
        distance: 4,
        direction: { x: 0, z: 1 },
        emotion: carrier.emotion,
        status: carrier.status,
        holding: carrier.holding,
        powerUp: carrier.powerUp?.kind ?? null,
        frozenUntilTick: carrier.frozenUntilTick,
      },
    ],
    nearbyObjects: [
      {
        id: 'banana',
        kind: 'banana',
        label: 'banana',
        description: 'banana',
        distance: 4,
        direction: { x: 0, z: 1 },
        active: true,
        heldBy: 'carrier',
        contested: true,
        affordances: [{ kind: 'pick_up', label: 'grab banana' }],
        tags: ['food'],
        blocksMovement: false,
        collisionRadius: 0.2,
      },
      {
        id: 'ice-power-up',
        kind: 'ice_cube',
        label: 'ice cube',
        description: 'ice cube',
        distance: 1.2,
        direction: { x: 1, z: 0 },
        active: true,
        heldBy: null,
        contested: true,
        affordances: [{ kind: 'pick_up', label: 'grab ice cube' }],
        tags: ['power_up'],
        blocksMovement: false,
        collisionRadius: 0.22,
      },
      {
        id: 'home',
        kind: 'goal',
        label: 'home base',
        description: 'home base',
        distance: 6,
        direction: { x: 0, z: -1 },
        active: true,
        heldBy: null,
        contested: false,
        affordances: [],
        tags: ['goal'],
        blocksMovement: false,
        collisionRadius: 1.2,
      },
    ],
    tick: 10,
    bounds: state.bounds,
    directorNote: null,
    game: state.game,
  };

  const decision = buildDecisionContext(perception);

  assert.equal(decision.options[0]?.label, 'go get the ice cube');
  assert.equal(decision.options.some((option) => option.label === 'chase Carrier'), true);
});

test('agent decision context allows tactical sabotage only against the immediate loose-banana threat', () => {
  const state = createWorldState();
  const seeker = createAgent('mira', 'Mira', { x: 0, z: 0 }, {
    archetype: 'mira',
    powerUp: { kind: 'ice_cube', objectId: 'ice-power-up' },
  });
  const threat = createAgent('threat', 'Threat', { x: 0.8, z: 0 }, {
    status: 'rushing banana',
  });
  const drifter = createAgent('drifter', 'Drifter', { x: 0, z: 2.5 }, {
    status: 'wandering wide',
  });
  const perception: AgentPerception = {
    self: seeker,
    nearbyAgents: [
      {
        id: threat.id,
        name: threat.name,
        distance: 0.8,
        direction: { x: 1, z: 0 },
        emotion: threat.emotion,
        status: threat.status,
        holding: threat.holding,
        powerUp: threat.powerUp?.kind ?? null,
        frozenUntilTick: threat.frozenUntilTick,
      },
      {
        id: drifter.id,
        name: drifter.name,
        distance: 2.5,
        direction: { x: 0, z: 1 },
        emotion: drifter.emotion,
        status: drifter.status,
        holding: drifter.holding,
        powerUp: drifter.powerUp?.kind ?? null,
        frozenUntilTick: drifter.frozenUntilTick,
      },
    ],
    nearbyObjects: [
      {
        id: 'banana',
        kind: 'banana',
        label: 'banana',
        description: 'banana',
        distance: 1.1,
        direction: { x: 1, z: 0 },
        active: true,
        heldBy: null,
        contested: true,
        affordances: [{ kind: 'pick_up', label: 'grab banana' }],
        tags: ['food'],
        blocksMovement: false,
        collisionRadius: 0.2,
      },
      {
        id: 'home',
        kind: 'goal',
        label: 'home base',
        description: 'home base',
        distance: 6,
        direction: { x: 0, z: -1 },
        active: true,
        heldBy: null,
        contested: false,
        affordances: [],
        tags: ['goal'],
        blocksMovement: false,
        collisionRadius: 1.2,
      },
    ],
    tick: 10,
    bounds: state.bounds,
    directorNote: null,
    game: state.game,
  };

  const decision = buildDecisionContext(perception);

  assert.equal(decision.options[0]?.label, 'rush banana');
  assert.equal(decision.options.some((option) => option.label === 'throw the ice cube at Threat'), true);
  assert.equal(decision.options.some((option) => option.label === 'throw the ice cube at Drifter'), false);
  assert.equal(decision.options.some((option) => option.label === 'push Threat'), false);

  seeker.cooldowns.sabotageUntilTick = 12;
  const coolingDecision = buildDecisionContext(perception);
  assert.equal(coolingDecision.options[0]?.label, 'rush banana');
  assert.equal(coolingDecision.options.some((option) => option.label === 'throw the ice cube at Threat'), false);
  assert.equal(coolingDecision.options.some((option) => option.label === 'push Threat'), false);
  assert.match(coolingDecision.prompt, /banana is loose, so keep racing it/);
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

test('banana carriers move at a conservative slowdown while holding the banana', () => {
  const state = createWorldState();
  state.agents.push(
    createAgent('carrier', 'Carrier', { x: -2, z: 0 }, {
      speed: 1.35,
      holding: 'banana',
      intent: { kind: 'go_to_object', objectId: 'home', emotion: 'alert' },
      goal: { kind: 'deliver', objectId: 'home', label: 'run to home base' },
    }),
    createAgent('runner', 'Runner', { x: 2, z: 0 }, {
      speed: 1.35,
      intent: { kind: 'go_to_object', objectId: 'bat', emotion: 'alert' },
      goal: { kind: 'go_to_object', objectId: 'bat', label: 'go get the baseball bat' },
    })
  );
  state.objects.push(
    createObject('banana', 'banana', { x: -2, z: 0 }, {
      contested: true,
      heldBy: 'carrier',
      affordances: [{ kind: 'pick_up', label: 'grab banana' }],
    }),
    createObject('home', 'goal', { x: -2, z: 6 }),
    createObject('bat', 'bat', { x: 2, z: 6 })
  );

  const carrierStart = { x: -2, z: 0 };
  const runnerStart = { x: 2, z: 0 };
  applyTickFirstPass(state, 1);

  const carrier = state.agents.find((agent) => agent.id === 'carrier')!;
  const runner = state.agents.find((agent) => agent.id === 'runner')!;
  const banana = state.objects.find((object) => object.id === 'banana')!;
  const carrierDistance = Math.hypot(carrier.position.x - carrierStart.x, carrier.position.z - carrierStart.z);
  const runnerDistance = Math.hypot(runner.position.x - runnerStart.x, runner.position.z - runnerStart.z);

  assert.ok(carrierDistance < runnerDistance);
  assert.ok(Math.abs(carrierDistance - 1.35 * BANANA_CARRIER_SPEED_MULTIPLIER) < 1e-9);
  assert.ok(Math.abs(runnerDistance - 1.35) < 1e-9);
  assert.deepEqual(banana.position, carrier.position);
});

test('stale go_to_object goals are cleared after the reevaluation cap', async () => {
  const director = new DirectorRuntime(createOutputEngine('drop'), DIRECTOR_CONFIG);
  const runtime = new SimulationRuntime(director, {
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home',
      respawnRules: [{ objectId: 'banana', delayTicks: 1, spawnPoints: [{ x: -4, z: -4 }] }],
    },
    maxMoveTicksBeforeReevaluation: 3,
  });

  try {
    const internals = runtime as unknown as MutableRuntimeInternals & { simulationAgents: Map<string, StubChooser> };
    internals.simulationAgents.set('runner', createAbortedChooser());
    internals.state.tick = 2;
    internals.state.agents.push(
      createAgent('runner', 'Runner', { x: -3, z: 0 }, {
        speed: 0,
        status: 'rushing banana',
        goal: { kind: 'go_to_object', objectId: 'banana', label: 'rush banana' },
        intent: { kind: 'go_to_object', objectId: 'banana', emotion: 'alert' },
        intentIssuedAtTick: 0,
      })
    );
    internals.state.objects.push(
      createObject('banana', 'banana', { x: 0, z: 0 }, {
        contested: true,
        affordances: [{ kind: 'pick_up', label: 'grab banana' }],
      }),
      createObject('home', 'goal', { x: 5, z: 5 })
    );

    await runtime.step(0.1);

    const runner = runtime.getSnapshot().agents.find((agent) => agent.id === 'runner');
    assert.ok(runner);
    assert.equal(runner.goal, null);
    assert.equal(runner.intent, null);
    assert.equal(runner.status, 'reconsidering the route');
  } finally {
    await runtime.dispose();
  }
});

test('stale go_to_agent goals are cleared after the reevaluation cap', async () => {
  const director = new DirectorRuntime(createOutputEngine('drop'), DIRECTOR_CONFIG);
  const runtime = new SimulationRuntime(director, {
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home',
      respawnRules: [{ objectId: 'banana', delayTicks: 1, spawnPoints: [{ x: -4, z: -4 }] }],
    },
    maxMoveTicksBeforeReevaluation: 4,
  });

  try {
    const internals = runtime as unknown as MutableRuntimeInternals & { simulationAgents: Map<string, StubChooser> };
    internals.simulationAgents.set('chaser', createAbortedChooser());
    internals.state.tick = 3;
    internals.state.agents.push(
      createAgent('chaser', 'Chaser', { x: -4, z: 0 }, {
        speed: 0,
        status: 'chasing carrier',
        goal: { kind: 'go_to_agent', agentId: 'carrier', label: 'chase Carrier' },
        intent: { kind: 'approach_agent', agentId: 'carrier', emotion: 'curious' },
        intentIssuedAtTick: 0,
      }),
      createAgent('carrier', 'Carrier', { x: 4, z: 0 })
    );
    internals.state.objects.push(
      createObject('banana', 'banana', { x: 0, z: 0 }, {
        contested: true,
        affordances: [{ kind: 'pick_up', label: 'grab banana' }],
      }),
      createObject('home', 'goal', { x: 5, z: 5 })
    );

    await runtime.step(0.1);

    const chaser = runtime.getSnapshot().agents.find((agent) => agent.id === 'chaser');
    assert.ok(chaser);
    assert.equal(chaser.goal, null);
    assert.equal(chaser.intent, null);
    assert.equal(chaser.status, 'reconsidering the route');
  } finally {
    await runtime.dispose();
  }
});

test('deliver goals are not cleared by the reevaluation cap', async () => {
  const director = new DirectorRuntime(createOutputEngine('drop'), DIRECTOR_CONFIG);
  const runtime = new SimulationRuntime(director, {
    game: {
      title: 'Banana Dash',
      bananaObjectId: 'banana',
      goalObjectId: 'home',
      respawnRules: [{ objectId: 'banana', delayTicks: 1, spawnPoints: [{ x: -4, z: -4 }] }],
    },
    maxMoveTicksBeforeReevaluation: 2,
  });

  try {
    const internals = runtime as unknown as MutableRuntimeInternals;
    internals.state.tick = 2;
    internals.state.agents.push(
      createAgent('carrier', 'Carrier', { x: -3, z: 0 }, {
        speed: 0,
        status: 'running home',
        holding: 'banana',
        goal: { kind: 'deliver', objectId: 'home', label: 'run to home base' },
        intent: { kind: 'deliver', objectId: 'home', emotion: 'alert' },
        intentIssuedAtTick: 0,
      })
    );
    internals.state.objects.push(
      createObject('banana', 'banana', { x: -3, z: 0 }, {
        contested: true,
        heldBy: 'carrier',
        affordances: [{ kind: 'pick_up', label: 'grab banana' }],
      }),
      createObject('home', 'goal', { x: 5, z: 5 })
    );

    await runtime.step(0.1);

    const carrier = runtime.getSnapshot().agents.find((agent) => agent.id === 'carrier');
    assert.ok(carrier);
    assert.deepEqual(carrier.goal, { kind: 'deliver', objectId: 'home', label: 'run to home base' });
    assert.deepEqual(carrier.intent, { kind: 'deliver', objectId: 'home', emotion: 'alert' });
  } finally {
    await runtime.dispose();
  }
});

test('forced banana drops clear active banana-contest goals that are still in flight', () => {
  const state = createWorldState();
  state.tick = 10;
  state.agents.push(
    createAgent('carrier', 'Carrier', { x: 0, z: 0 }, {
      status: 'dropping the banana',
      holding: 'banana',
      goal: { kind: 'drop', label: 'drop banana' },
      intent: { kind: 'drop', emotion: 'alert' },
    }),
    createAgent('banana-runner', 'Banana Runner', { x: -2, z: 0 }, {
      status: 'rushing banana',
      goal: { kind: 'go_to_object', objectId: 'banana', label: 'rush banana' },
      intent: { kind: 'go_to_object', objectId: 'banana', emotion: 'alert' },
    }),
    createAgent('scout', 'Scout', { x: 2, z: 0 }, {
      status: 'heading to bat',
      goal: { kind: 'go_to_object', objectId: 'bat', label: 'go get the baseball bat' },
      intent: { kind: 'go_to_object', objectId: 'bat', emotion: 'alert' },
      navigation: {
        detourTarget: { x: 1.5, z: 0.2 },
        blockedTicks: 2,
        obstacleId: 'rock',
      },
    }),
    createAgent('bruiser', 'Bruiser', { x: 0.6, z: 0 }, {
      status: 'shoving scout',
      goal: { kind: 'push_agent', agentId: 'scout', label: 'push Scout' },
      intent: { kind: 'push', agentId: 'scout', emotion: 'alert' },
    }),
    createAgent('attacker', 'Attacker', { x: 0.4, z: 0 }, {
      status: 'lining up a bump',
      goal: { kind: 'sabotage_agent', agentId: 'carrier', method: 'bump', label: 'bump Carrier' },
      intent: { kind: 'sabotage', agentId: 'carrier', method: 'bump', emotion: 'alert' },
    })
  );
  state.objects.push(
    createObject('banana', 'banana', { x: 0, z: 0 }, {
      contested: true,
      heldBy: 'carrier',
      affordances: [{ kind: 'pick_up', label: 'grab banana' }],
    }),
    createObject('home', 'goal', { x: 5, z: 5 }),
    createObject('bat', 'bat', { x: 3, z: 0 }, {
      contested: true,
      affordances: [{ kind: 'pick_up', label: 'grab baseball bat' }],
    })
  );

  const bus = new SimulationBus();
  const director = new DirectorRuntime(createOutputEngine('drop'), DIRECTOR_CONFIG);
  const runtime = new SimulationRuntime(director, {
    bus,
    game: state.game,
  });
  const internals = runtime as unknown as MutableRuntimeInternals;
  internals.state.tick = state.tick;
  internals.state.timeSeconds = state.timeSeconds;
  internals.state.bounds = state.bounds;
  internals.state.agents = state.agents;
  internals.state.objects = state.objects;

  const emitGameEvents = (runtime as unknown as { emitGameEvents(events: readonly unknown[]): void }).emitGameEvents;
  emitGameEvents.call(runtime, [
    { kind: 'drop', agentId: 'carrier', objectId: 'banana', from: { x: 0, z: 0 }, to: { x: 0.8, z: 0 }, cause: 'bump' },
    {
      kind: 'forced_drop',
      attackerAgentId: 'attacker',
      targetAgentId: 'carrier',
      objectId: 'banana',
      position: { x: 0, z: 0 },
      outcome: 'drop',
    },
  ]);

  const bananaRunner = internals.state.agents.find((agent) => agent.id === 'banana-runner');
  const scout = internals.state.agents.find((agent) => agent.id === 'scout');
  const bruiser = internals.state.agents.find((agent) => agent.id === 'bruiser');
  const attacker = internals.state.agents.find((agent) => agent.id === 'attacker');

  assert.equal(bananaRunner?.goal, null);
  assert.equal(bananaRunner?.intent, null);
  assert.equal(bananaRunner?.status, 'banana is loose; changing plans');
  assert.equal(scout?.goal, null);
  assert.equal(scout?.intent, null);
  assert.equal(scout?.status, 'banana is loose; changing plans');
  assert.equal(scout?.navigation.detourTarget, null);
  assert.equal(scout?.navigation.blockedTicks, 0);
  assert.equal(scout?.navigation.obstacleId, null);
  assert.equal(bruiser?.goal, null);
  assert.equal(bruiser?.intent, null);
  assert.equal(bruiser?.status, 'banana is loose; changing plans');
  assert.equal(attacker?.goal, null);
  assert.equal(attacker?.intent, null);
  assert.equal(attacker?.status, 'banana is loose; changing plans');
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
