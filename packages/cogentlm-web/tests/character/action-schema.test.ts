//////////////////////////////////////////////////////////////////////////////
//
// action-schema.test.ts
//
// - Exercises validateActionSchema, expandActionCues, and
//   renderActionCueList for the flat action model.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import type { ActionSchema } from '../../src/character/action-schema.js';
import {
  ActionSchemaError,
  assertValidActionSchema,
  expandActionCues,
  renderActionCueList,
  summarizeActionCues,
  validateActionSchema,
} from '../../src/character/action-schema.js';

const SCHEMA: ActionSchema = [
  {
    id: 'wave',
    description: 'Wave a hand.',
    usageHint: 'greeting or saying goodbye',
  },
  {
    id: 'look_at_you',
    cue: 'look at you',
    description: 'Turn attention toward the user.',
  },
  {
    id: 'shake_head',
    description: 'Shake the head side to side.',
  },
];

test('validateActionSchema accepts a well-formed schema', () => {
  assert.equal(validateActionSchema(SCHEMA), null);
});

test('validateActionSchema accepts an empty schema for choose-only characters', () => {
  assert.equal(validateActionSchema([]), null);
});

test('validateActionSchema rejects duplicate action ids', () => {
  const error = validateActionSchema([{ id: 'wave' }, { id: 'wave' }]);
  assert.match(String(error), /Duplicate action id/);
});

test('validateActionSchema rejects invalid identifiers', () => {
  const error = validateActionSchema([{ id: '1bad' }]);
  assert.match(String(error), /Invalid action id/);
});

test('validateActionSchema rejects empty cue overrides', () => {
  const error = validateActionSchema([{ id: 'wave', cue: '   ' }]);
  assert.match(String(error), /invalid cue label/i);
});

test('validateActionSchema rejects cue label collisions', () => {
  const error = validateActionSchema([
    { id: 'look_at_you', cue: 'look at you' },
    { id: 'look_at_you_again', cue: 'look at you' },
  ]);
  assert.match(String(error), /collision/i);
});

test('assertValidActionSchema throws ActionSchemaError on invalid input', () => {
  assert.throws(
    () => assertValidActionSchema([{ id: 'bad id' }]),
    (error) => error instanceof ActionSchemaError
  );
});

test('expandActionCues produces one cue per action in declaration order', () => {
  const cues = expandActionCues(SCHEMA);
  assert.deepEqual(cues, [
    { label: 'wave', id: 'wave' },
    { label: 'look at you', id: 'look_at_you' },
    { label: 'shake head', id: 'shake_head' },
  ]);
});

test('summarizeActionCues preserves cue labels and metadata', () => {
  const cues = summarizeActionCues(SCHEMA);
  assert.deepEqual(cues, [
    {
      label: 'wave',
      id: 'wave',
      description: 'Wave a hand.',
      usageHint: 'greeting or saying goodbye',
    },
    {
      label: 'look at you',
      id: 'look_at_you',
      description: 'Turn attention toward the user.',
      usageHint: undefined,
    },
    {
      label: 'shake head',
      id: 'shake_head',
      description: 'Shake the head side to side.',
      usageHint: undefined,
    },
  ]);
});

test('renderActionCueList emits bracketed, comma-separated labels', () => {
  const text = renderActionCueList(SCHEMA);
  assert.equal(text, '[wave], [look at you], [shake head]');
});
