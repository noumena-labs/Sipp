//////////////////////////////////////////////////////////////////////////////
//
// world-reducer.test.ts
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  INTERACTION_RADIUS,
  applyDirectorDecision,
  applyTickFirstPass,
  stepMovement,
  type MutableWorldState,
} from './world-reducer.js';
import type {
  AgentIntent,
  SimulationAgentState,
  SimulationObjectState,
  WorldBounds,
} from './simulation-types.js';

const BOUNDS: WorldBounds = { halfExtent: 8 };

function agent(
  id: string,
  x: number,
  z: number,
  intent: AgentIntent | null = null
): SimulationAgentState {
  return {
    id,
    name: id,
    position: { x, z },
    heading: 0,
    speed: 2,
    emotion: null,
    status: '',
    intent,
    holding: null,
    intentIssuedAtTick: -1,
  };
}

function obj(
  id: string,
  x: number,
  z: number,
  contested = false
): SimulationObjectState {
  return {
    id,
    kind: 'banana',
    position: { x, z },
    contested,
    heldBy: null,
    tags: [],
  };
}

function state(
  agents: SimulationAgentState[],
  objects: SimulationObjectState[] = []
): MutableWorldState {
  return {
    tick: 0,
    timeSeconds: 0,
    bounds: BOUNDS,
    agents,
    objects,
    directorNote: null,
  };
}

test('stepMovement moves agent toward move_to target at bounded speed', () => {
  const a = agent('a', 0, 0, { kind: 'move_to', target: { x: 10, z: 0 }, emotion: 'curious' });
  const { position } = stepMovement(a, [], [a], 0.5, BOUNDS);
  // speed 2, dt 0.5 => moves 1 unit
  assert.ok(Math.abs(position.x - 1) < 1e-6);
  assert.equal(position.z, 0);
});

test('stepMovement does nothing for wait intent', () => {
  const a = agent('a', 3, 3, { kind: 'wait', emotion: 'thinking' });
  const { position } = stepMovement(a, [], [a], 1, BOUNDS);
  assert.equal(position.x, 3);
  assert.equal(position.z, 3);
});

test('stepMovement clamps to world bounds', () => {
  const a = agent('a', 7.9, 0, {
    kind: 'move_to',
    target: { x: 100, z: 0 },
    emotion: 'curious',
  });
  a.speed = 50;
  const { position } = stepMovement(a, [], [a], 1, BOUNDS);
  assert.equal(position.x, BOUNDS.halfExtent);
});

test('applyTickFirstPass resolves uncontested pick_up', () => {
  const a = agent('a', 0, 0, { kind: 'pick_up', objectId: 'banana_a', emotion: 'happy' });
  const banana = obj('banana_a', 0, 0);
  const s = state([a], [banana]);
  const result = applyTickFirstPass(s, 1 / 60);
  assert.equal(result.conflicts.length, 0);
  assert.equal(s.objects[0]!.heldBy, 'a');
  assert.equal(s.agents[0]!.holding, 'banana_a');
  assert.equal(s.agents[0]!.intent, null);
});

test('applyTickFirstPass surfaces conflicts for multiple contenders', () => {
  const a = agent('a', 0, 0, { kind: 'pick_up', objectId: 'banana_a', emotion: 'happy' });
  const b = agent('b', 0.1, 0, { kind: 'pick_up', objectId: 'banana_a', emotion: 'alert' });
  const banana = obj('banana_a', 0, 0);
  const s = state([a, b], [banana]);
  const result = applyTickFirstPass(s, 1 / 60);
  assert.equal(result.conflicts.length, 1);
  assert.equal(result.conflicts[0]!.objectId, 'banana_a');
  assert.deepEqual([...result.conflicts[0]!.contenderAgentIds].sort(), ['a', 'b']);
  assert.equal(s.objects[0]!.heldBy, null);
});

test('applyTickFirstPass ignores pick_up when out of range', () => {
  const a = agent('a', 0, 0, { kind: 'pick_up', objectId: 'banana_a', emotion: 'happy' });
  const banana = obj('banana_a', 5, 0);
  const s = state([a], [banana]);
  const result = applyTickFirstPass(s, 1 / 60);
  assert.equal(result.conflicts.length, 0);
  assert.equal(s.objects[0]!.heldBy, null);
  // Intent remains active because pick_up target is still too far.
  assert.ok(s.agents[0]!.intent);
});

test('applyTickFirstPass drops a held object and clears holding', () => {
  const a = agent('a', 0, 0, { kind: 'drop', emotion: 'confused' });
  a.holding = 'banana_a';
  const banana = obj('banana_a', 0, 0);
  banana.heldBy = 'a';
  const s = state([a], [banana]);
  applyTickFirstPass(s, 1 / 60);
  assert.equal(s.agents[0]!.holding, null);
  assert.equal(s.objects[0]!.heldBy, null);
});

test('applyTickFirstPass clears wait intents so agent can re-query', () => {
  const a = agent('a', 0, 0, { kind: 'wait', emotion: 'thinking' });
  const s = state([a]);
  applyTickFirstPass(s, 1 / 60);
  assert.equal(s.agents[0]!.intent, null);
});

test('applyDirectorDecision assigns winner and clears loser intent', () => {
  const a = agent('a', 0, 0, { kind: 'pick_up', objectId: 'banana_a', emotion: 'happy' });
  const b = agent('b', 0.1, 0, { kind: 'pick_up', objectId: 'banana_a', emotion: 'alert' });
  const banana = obj('banana_a', 0, 0);
  const s = state([a, b], [banana]);
  applyDirectorDecision(s, {
    note: 'aria gets it',
    resolutions: [{ objectId: 'banana_a', winnerAgentId: 'a' }],
  });
  assert.equal(s.objects[0]!.heldBy, 'a');
  assert.equal(s.agents[0]!.holding, 'banana_a');
  assert.equal(s.agents[1]!.intent, null);
});

test('applyDirectorDecision null winner denies everyone', () => {
  const a = agent('a', 0, 0, { kind: 'pick_up', objectId: 'banana_a', emotion: 'happy' });
  const b = agent('b', 0.1, 0, { kind: 'pick_up', objectId: 'banana_a', emotion: 'alert' });
  const banana = obj('banana_a', 0, 0);
  const s = state([a, b], [banana]);
  applyDirectorDecision(s, {
    note: 'nobody',
    resolutions: [{ objectId: 'banana_a', winnerAgentId: null }],
  });
  assert.equal(s.objects[0]!.heldBy, null);
  assert.equal(s.agents[0]!.intent, null);
  assert.equal(s.agents[1]!.intent, null);
});

test('INTERACTION_RADIUS is a sensible small number', () => {
  assert.ok(INTERACTION_RADIUS > 0 && INTERACTION_RADIUS < 3);
});
