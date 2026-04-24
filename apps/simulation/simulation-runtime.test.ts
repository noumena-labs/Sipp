import assert from 'node:assert/strict';
import test from 'node:test';

import type { CharacterAgentEngine } from 'cogent-engine/character';
import { DirectorRuntime, parseDirectorConfig } from 'cogent-engine/orchestrator';

import { SimulationBus, type SimulationEvent } from './src/runtime/bus.ts';
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
