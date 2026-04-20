//////////////////////////////////////////////////////////////////////////////
//
// action-schema.test.ts
//
// - Exercises validateActionSchema and renderActionSchemaForPrompt.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import type { ActionSchema } from './action-schema.js';
import { renderActionSchemaForPrompt, validateActionSchema } from './action-schema.js';

const WAVE_SCHEMA: ActionSchema = {
  actions: [
    {
      name: 'wave',
      description: 'Wave a hand.',
      args: [
        { name: 'duration_ms', type: 'number', description: 'Wave length in ms.' },
      ],
    },
    {
      name: 'set_mood',
      args: [
        { name: 'mood', type: 'enum', values: ['happy', 'sad'] },
        { name: 'persist', type: 'boolean' },
      ],
    },
    {
      name: 'idle',
      args: [],
    },
  ],
};

test('validateActionSchema accepts a well-formed schema', () => {
  assert.equal(validateActionSchema(WAVE_SCHEMA), null);
});

test('validateActionSchema rejects an empty schema', () => {
  assert.match(
    String(validateActionSchema({ actions: [] })),
    /at least one action/
  );
});

test('validateActionSchema rejects duplicate action names', () => {
  const error = validateActionSchema({
    actions: [
      { name: 'wave', args: [] },
      { name: 'wave', args: [] },
    ],
  });
  assert.match(String(error), /Duplicate action name/);
});

test('validateActionSchema rejects invalid identifiers', () => {
  const error = validateActionSchema({
    actions: [{ name: '1bad', args: [] }],
  });
  assert.match(String(error), /Invalid action name/);
});

test('validateActionSchema rejects enum args without values', () => {
  const error = validateActionSchema({
    actions: [
      { name: 'x', args: [{ name: 'mode', type: 'enum' }] },
    ],
  });
  assert.match(String(error), /non-empty values/);
});

test('renderActionSchemaForPrompt produces a readable listing', () => {
  const prose = renderActionSchemaForPrompt(WAVE_SCHEMA);
  assert.match(prose, /wave\(duration_ms: number\) — Wave a hand\./);
  assert.match(prose, /set_mood\(mood: "happy" \| "sad", persist: boolean\)/);
  assert.match(prose, /idle\(\)/);
});
