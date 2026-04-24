import assert from 'node:assert/strict';
import test from 'node:test';

import type { CharacterAgentEngine } from 'cogent-engine/character';
import { DirectorRuntime, parseDirectorConfig } from 'cogent-engine/orchestrator';

import { SimulationBus, type SimulationEvent } from './src/runtime/bus.ts';
import { applyDirectorDecision, type MutableWorldState } from './src/runtime/reducer.ts';
import { SimulationRuntime } from './src/runtime/simulation-runtime.ts';
import type { SimulationAgentState, SimulationObjectState, Vec2 } from './src/runtime/types.ts';

const DIRECTOR_CONFIG = parseDirectorConfig({
  id: 'banana-dash-referee',
  director: {
    role: 'Banana Dash referee',
    instructions: ['Resolve conflicts quickly using only the supplied scene state.'],
  },
  queries: {
    resolve_referee_event: {
      response: {
        type: 'object',
        properties: {
          note: { type: 'string', maxLength: 160 },
          resolutions: {
            type: 'array',
            maxItems: 4,
            items: {
              type: 'object',
              properties: {
                conflictId: { type: 'string', maxLength: 64 },
                objectId: { type: 'string', nullable: true, maxLength: 64 },
                winnerAgentId: { type: 'string', nullable: true, maxLength: 64 },
                outcome: {
                  type: 'string',
                  enum: ['pickup', 'deny', 'drop', 'hold', 'attacker_fumbles'],
                },
                note: { type: 'string', nullable: true, maxLength: 160 },
              },
            },
          },
        },
      },
    },
  },
});

interface MutableRuntimeState {
  agents: SimulationAgentState[];
  objects: SimulationObjectState[];
  game: {
    score: {
      deliveries: Record<string, number>;
      forcedDrops: Record<string, number>;
    };
  };
}

interface MutableRuntimeInternals {
  state: MutableRuntimeState;
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
        throw new Error('Expected an abortable director query.');
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
    intentIssuedAtTick: 0,
    thinking: false,
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
      bananaSpawnPoints: [{ x: -4, z: -4 }],
      score: {
        deliveries: {},
        forcedDrops: {},
      },
      referee: { status: 'idle' },
      pendingRespawn: null,
      nextSpawnIndex: 0,
    },
  };
}

function timeoutAfter(ms: number, message: string): Promise<never> {
  return new Promise((_, reject) => {
    setTimeout(() => reject(new Error(message)), ms);
  });
}

test('SimulationRuntime reports referee timeout without forcing a fallback ruling', async () => {
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
      bananaSpawnPoints: [{ x: -4, z: -4 }],
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
      intent: { kind: 'sabotage', agentId: 'carrier', emotion: 'alert' },
      goal: { kind: 'sabotage_agent', agentId: 'carrier', label: 'bump Carrier' },
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
    assert.equal(snapshot.game.score.forcedDrops.attacker, 0);
    assert.equal(snapshot.objects.find((object) => object.id === 'banana')?.heldBy, 'carrier');
    assert.equal(snapshot.agents.find((agent) => agent.id === 'carrier')?.holding, 'banana');
    assert.deepEqual(engine.cancelCalls, [1]);
    assert.ok(
      events.some(
        (event) =>
          event.kind === 'runtime-error' &&
          event.severity === 'critical' &&
          event.source === 'referee' &&
          event.message.includes('timed out')
      )
    );
    assert.equal(events.some((event) => event.kind === 'director-decision'), false);
    assert.equal(events.some((event) => event.kind === 'game-event' && event.event.kind === 'forced_drop'), false);
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
      intent: { kind: 'sabotage', agentId: 'carrier', emotion: 'alert' },
      goal: { kind: 'sabotage_agent', agentId: 'carrier', label: 'bump Carrier' },
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
