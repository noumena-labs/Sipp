//////////////////////////////////////////////////////////////////////////////
//
// action-schema.test.ts
//
// - Exercises validateActionSchema, expandActionCues, and
//   renderActionCueList.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import type { ActionSchema } from './action-schema.js';
import {
  expandActionCues,
  renderActionCueList,
  validateActionSchema,
} from './action-schema.js';

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
      name: 'shake_head',
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

test('expandActionCues produces one cue per enum value and a bare cue otherwise', () => {
  const cues = expandActionCues(WAVE_SCHEMA);
  // wave: first arg is a non-enum number → bare `[wave]` with empty args.
  // set_mood: first arg is enum with two values → two cues.
  // shake_head: no args → bare `[shake head]` with underscore converted.
  assert.deepEqual(cues.map((cue) => cue.label), [
    'wave',
    'mood: happy',
    'mood: sad',
    'shake head',
  ]);
  const happy = cues.find((cue) => cue.label === 'mood: happy');
  assert.ok(happy);
  assert.equal(happy.name, 'set_mood');
  assert.deepEqual(happy.args, { mood: 'happy' });
});

test('expandActionCues respects cueLabel and cueLabels overrides', () => {
  const cues = expandActionCues({
    actions: [
      {
        name: 'wave',
        cueLabel: 'greet',
        args: [],
      },
      {
        name: 'intensity',
        args: [
          {
            name: 'level',
            type: 'enum',
            values: ['low', 'high'],
            cueLabels: { low: 'wave softly', high: 'wave energetically' },
          },
        ],
      },
    ],
  });
  assert.deepEqual(cues.map((cue) => cue.label), [
    'greet',
    'wave softly',
    'wave energetically',
  ]);
});

test('expandActionCues throws when two cues collapse to the same label', () => {
  assert.throws(
    () =>
      expandActionCues({
        actions: [
          { name: 'wave', args: [] },
          { name: 'wave_', cueLabel: 'wave', args: [] },
        ],
      }),
    /collision/i
  );
});

test('renderActionCueList emits bracketed, comma-separated labels', () => {
  const text = renderActionCueList(WAVE_SCHEMA);
  assert.equal(text, '[wave], [mood: happy], [mood: sad], [shake head]');
});
