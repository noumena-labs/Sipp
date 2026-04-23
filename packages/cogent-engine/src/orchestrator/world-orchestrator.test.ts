//////////////////////////////////////////////////////////////////////////////
//
// world-orchestrator.test.ts
//
// - Integration test over WorldOrchestrator with a scripted fake engine.
//   Verifies that `step()`:
//     * queries one agent per tick round-robin,
//     * applies the returned intent to state,
//     * triggers the director on a conflict and applies the resolution,
//     * runs cadence narration when no conflict.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import type {
  CharacterAgentEngine,
} from '../character/character-agent.js';
import type {
  GenerateRequestId,
  GenerateResponse,
  PromptOptions,
} from '../core/inference-types.js';
import type { CharacterConfig } from '../character/character-config.js';
import { SIMULATION_ACTION_NAMES } from './simulation-character-actions.js';
import { SimulationAgent } from './simulation-agent.js';
import { WorldDirector } from './world-director.js';
import { WorldOrchestrator } from './world-orchestrator.js';

interface Scripted {
  readonly contextKeyContains: string;
  readonly output: string;
}

function buildSimulationConfig(id: string, name: string): CharacterConfig {
  return {
    id,
    persona: { name, summary: `${name} lives in the courtyard.` },
    actions: {
      actions: SIMULATION_ACTION_NAMES.map((n) => ({ name: n, description: n })),
    },
  };
}

class FakeEngine implements CharacterAgentEngine {
  public readonly scripts: Scripted[] = [];
  public readonly queueCalls: Array<{ contextKey: string; options: PromptOptions }> = [];
  private pending: Array<{
    id: GenerateRequestId;
    contextKey: string;
    options: PromptOptions;
  }> = [];
  private nextId = 1;

  public enqueue(contextKeyContains: string, output: string): void {
    this.scripts.push({ contextKeyContains, output });
  }

  public async applyChatTemplate(
    messages: Array<{ role: string; content: string }>,
    _add: boolean
  ): Promise<string> {
    return messages.map((m) => `${m.role}: ${m.content}`).join('\n');
  }

  public async queuePrompt(
    contextKey: string,
    _prompt: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestId> {
    const opts = (typeof options === 'object' && options) || {};
    this.queueCalls.push({ contextKey, options: opts as PromptOptions });
    const id = this.nextId++;
    this.pending.push({ id, contextKey, options: opts as PromptOptions });
    return id;
  }

  public async runQueuedRequest(
    requestId: GenerateRequestId,
    _opts?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    const entry = this.pending.find((p) => p.id === requestId);
    if (!entry) {
      return {
        requestId,
        completed: false,
        failed: true,
        cancelled: false,
        outputText: '',
        errorMessage: 'no pending',
      };
    }
    this.pending = this.pending.filter((p) => p.id !== requestId);
    const scriptIdx = this.scripts.findIndex((s) =>
      entry.contextKey.includes(s.contextKeyContains)
    );
    if (scriptIdx < 0) {
      return {
        requestId,
        completed: true,
        failed: false,
        cancelled: false,
        outputText: '',
      };
    }
    const script = this.scripts.splice(scriptIdx, 1)[0]!;
    return {
      requestId,
      completed: true,
      failed: false,
      cancelled: false,
      outputText: script.output,
    };
  }

  public async cancelQueuedRequest(_id: GenerateRequestId): Promise<boolean> {
    return true;
  }

  public getChatTemplate(): string | null {
    return null;
  }

  public getBosText(): string {
    return '';
  }

  public getEosText(): string {
    return '';
  }
}

test('WorldOrchestrator step queries one agent and applies its intent', async () => {
  const engine = new FakeEngine();
  const director = new WorldDirector('w', engine);
  const orchestrator = new WorldOrchestrator(director, {
    id: 'w',
    bounds: { halfExtent: 8 },
    tickHz: 2,
    directorCadenceTicks: 1000, // effectively disable cadence for this test
  });

  const aria = new SimulationAgent(
    'aria',
    engine,
    buildSimulationConfig('aria-arch', 'Aria')
  );
  orchestrator.addAgent(aria, { id: 'aria', name: 'Aria', position: { x: 0, z: 0 } });

  engine.enqueue(
    'agent:aria',
    '{"intent":{"kind":"move_to","target":{"x":3,"z":0},"emotion":"curious"},"status":"heading east"}'
  );

  await orchestrator.step();

  const snap = orchestrator.getSnapshot();
  const agentState = snap.agents.find((a) => a.id === 'aria')!;
  assert.ok(agentState);
  assert.equal(agentState.status, 'heading east');
  assert.equal(agentState.emotion, 'curious');
  // Intent remains active (move_to doesn't clear on this tick).
  assert.ok(agentState.intent);
  // Agent has moved toward target from (0,0) toward (3,0).
  assert.ok(agentState.position.x > 0);
  await orchestrator.dispose();
});

test('WorldOrchestrator triggers director on contested pick_up', async () => {
  const engine = new FakeEngine();
  const director = new WorldDirector('w', engine);
  const orchestrator = new WorldOrchestrator(director, {
    id: 'w',
    tickHz: 2,
    directorCadenceTicks: 1000,
  });

  const aria = new SimulationAgent('aria', engine, buildSimulationConfig('aria-arch', 'Aria'));
  const beck = new SimulationAgent('beck', engine, buildSimulationConfig('beck-arch', 'Beck'));
  // Agents begin adjacent; banana is placed far away so their pick_up
  // intents cannot resolve until we teleport the banana between them,
  // guaranteeing the first-pass sees a contested object on the same tick.
  orchestrator.addAgent(aria, { id: 'aria', name: 'Aria', position: { x: -0.3, z: 0 } });
  orchestrator.addAgent(beck, { id: 'beck', name: 'Beck', position: { x: 0.3, z: 0 } });
  orchestrator.upsertObject({ id: 'banana_a', kind: 'banana', position: { x: 50, z: 50 } });

  // Tick 1: aria is queried and commits to pick_up (banana is far away, so
  // no resolution this tick — intent persists).
  engine.enqueue(
    'agent:aria',
    '{"intent":{"kind":"pick_up","objectId":"banana_a","emotion":"happy"},"status":"mine"}'
  );
  await orchestrator.step();

  // Tick 2: beck is queried and also commits to pick_up.
  engine.enqueue(
    'agent:beck',
    '{"intent":{"kind":"pick_up","objectId":"banana_a","emotion":"alert"},"status":"mine"}'
  );
  await orchestrator.step();

  // Now teleport the banana so that both agents are within interaction
  // range on the very next tick, triggering the contested-object path.
  orchestrator.upsertObject({ id: 'banana_a', kind: 'banana', position: { x: 0, z: 0 } });

  engine.enqueue(
    'director:w',
    '{"note":"aria arrived first","resolutions":[{"objectId":"banana_a","winnerAgentId":"aria","note":"first"}]}'
  );

  const conflictEvents: unknown[] = [];
  const decisionEvents: unknown[] = [];
  orchestrator.bus.on('director-conflict', (e) => conflictEvents.push(e));
  orchestrator.bus.on('director-decision', (e) => decisionEvents.push(e));

  await orchestrator.step();

  const snap = orchestrator.getSnapshot();
  const banana = snap.objects.find((o) => o.id === 'banana_a');
  assert.ok(banana);
  assert.equal(banana!.heldBy, 'aria');
  assert.equal(conflictEvents.length, 1);
  assert.equal(decisionEvents.length, 1);

  await orchestrator.dispose();
});

test('WorldOrchestrator runs cadence narration when no conflict', async () => {
  const engine = new FakeEngine();
  const director = new WorldDirector('w', engine);
  const orchestrator = new WorldOrchestrator(director, {
    id: 'w',
    tickHz: 2,
    directorCadenceTicks: 1,
  });

  const aria = new SimulationAgent('aria', engine, buildSimulationConfig('aria-arch', 'Aria'));
  orchestrator.addAgent(aria, { id: 'aria', name: 'Aria', position: { x: 0, z: 0 } });

  engine.enqueue(
    'agent:aria',
    '{"intent":{"kind":"wait","emotion":"thinking","reason":"pausing"},"status":"idle"}'
  );
  engine.enqueue('director:w', '{"note":"the fountain burbles."}');

  const notes: string[] = [];
  orchestrator.bus.on('world-note', (e) => notes.push(e.note));

  await orchestrator.step();

  assert.deepEqual(notes, ['the fountain burbles.']);
  await orchestrator.dispose();
});

test('WorldOrchestrator addAgent / removeAgent / upsertObject / removeObject', async () => {
  const engine = new FakeEngine();
  const orchestrator = new WorldOrchestrator(null, { id: 'w', tickHz: 2 });
  const aria = new SimulationAgent('aria', engine, buildSimulationConfig('aria-arch', 'Aria'));
  orchestrator.addAgent(aria, { id: 'aria', name: 'Aria', position: { x: 0, z: 0 } });
  orchestrator.upsertObject({ id: 'banana', kind: 'banana', position: { x: 1, z: 1 } });
  assert.equal(orchestrator.getSnapshot().agents.length, 1);
  assert.equal(orchestrator.getSnapshot().objects.length, 1);
  orchestrator.upsertObject({ id: 'banana', kind: 'banana', position: { x: 2, z: 2 } });
  assert.equal(orchestrator.getSnapshot().objects[0]!.position.x, 2);
  orchestrator.removeAgent('aria');
  orchestrator.removeObject('banana');
  assert.equal(orchestrator.getSnapshot().agents.length, 0);
  assert.equal(orchestrator.getSnapshot().objects.length, 0);
  await orchestrator.dispose();
});

test('WorldOrchestrator tolerates invalid agent JSON by falling back', async () => {
  const engine = new FakeEngine();
  const orchestrator = new WorldOrchestrator(null, { id: 'w', tickHz: 2 });
  const aria = new SimulationAgent('aria', engine, buildSimulationConfig('aria-arch', 'Aria'));
  orchestrator.addAgent(aria, { id: 'aria', name: 'Aria', position: { x: 0, z: 0 } });
  engine.enqueue('agent:aria', 'not json at all');
  await orchestrator.step();
  const snap = orchestrator.getSnapshot();
  const a = snap.agents[0]!;
  assert.equal(a.emotion, 'confused');
  await orchestrator.dispose();
});
