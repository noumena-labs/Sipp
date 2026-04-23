//////////////////////////////////////////////////////////////////////////////
//
// sensing.test.ts
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildPerception,
  clampToBounds,
  vec2Direction,
  vec2Distance,
} from './sensing.js';
import type {
  SimulationAgentState,
  SimulationObjectState,
  WorldBounds,
} from './simulation-types.js';

function agent(
  id: string,
  x: number,
  z: number,
  overrides: Partial<SimulationAgentState> = {}
): SimulationAgentState {
  return {
    id,
    name: id,
    position: { x, z },
    heading: 0,
    speed: 1,
    emotion: null,
    status: '',
    intent: null,
    holding: null,
    intentIssuedAtTick: -1,
    ...overrides,
  };
}

function obj(id: string, x: number, z: number): SimulationObjectState {
  return {
    id,
    kind: 'banana',
    position: { x, z },
    contested: false,
    heldBy: null,
    tags: [],
  };
}

const BOUNDS: WorldBounds = { halfExtent: 8 };

test('vec2Distance is euclidean', () => {
  assert.equal(vec2Distance({ x: 0, z: 0 }, { x: 3, z: 4 }), 5);
});

test('vec2Direction returns zero for coincident points', () => {
  const d = vec2Direction({ x: 1, z: 1 }, { x: 1, z: 1 });
  assert.equal(d.x, 0);
  assert.equal(d.z, 0);
});

test('clampToBounds clamps to halfExtent on both axes', () => {
  const c = clampToBounds({ x: 99, z: -99 }, { halfExtent: 4 });
  assert.equal(c.x, 4);
  assert.equal(c.z, -4);
});

test('buildPerception excludes self and sorts by distance', () => {
  const me = agent('me', 0, 0);
  const others = [agent('far', 7, 0), agent('close', 1, 0), agent('mid', 3, 0)];
  const perception = buildPerception(me, [me, ...others], [], 1, BOUNDS, null);
  assert.equal(perception.nearbyAgents.length, 3);
  assert.equal(perception.nearbyAgents[0]!.id, 'close');
  assert.equal(perception.nearbyAgents[1]!.id, 'mid');
  assert.equal(perception.nearbyAgents[2]!.id, 'far');
});

test('buildPerception respects sight radius', () => {
  const me = agent('me', 0, 0);
  const others = [agent('far', 12, 0), agent('close', 1, 0)];
  const perception = buildPerception(me, [me, ...others], [], 1, BOUNDS, null, {
    agentSightRadius: 3,
  });
  assert.equal(perception.nearbyAgents.length, 1);
  assert.equal(perception.nearbyAgents[0]!.id, 'close');
});

test('buildPerception limits max neighbours', () => {
  const me = agent('me', 0, 0);
  const objects = Array.from({ length: 20 }, (_, i) => obj(`o${i}`, i * 0.1, 0));
  const perception = buildPerception(me, [me], objects, 1, BOUNDS, null, {
    maxNeighbours: 4,
  });
  assert.equal(perception.nearbyObjects.length, 4);
});

test('buildPerception propagates directorNote', () => {
  const me = agent('me', 0, 0);
  const perception = buildPerception(me, [me], [], 7, BOUNDS, 'hello');
  assert.equal(perception.directorNote, 'hello');
  assert.equal(perception.tick, 7);
});
