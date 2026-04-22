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

import type { ActionSchema } from './action-schema.js';
import {
  expandActionCues,
  findCanonicalActionCue,
  renderActionCapabilityList,
  renderActionCueList,
  summarizeActionCues,
  validateActionSchema,
} from './action-schema.js';

const SCHEMA: ActionSchema = {
  actions: [
    {
      name: 'wave',
      description: 'Wave a hand.',
      usageHint: 'greeting or saying goodbye',
    },
    {
      name: 'look_at_you',
      cue: 'look at you',
      description: 'Turn attention toward the user.',
    },
    {
      name: 'shake_head',
      description: 'Shake the head side to side.',
    },
  ],
};

test('validateActionSchema accepts a well-formed schema', () => {
  assert.equal(validateActionSchema(SCHEMA), null);
});

test('validateActionSchema rejects an empty schema', () => {
  assert.match(String(validateActionSchema({ actions: [] })), /at least one action/);
});

test('validateActionSchema rejects duplicate action names', () => {
  const error = validateActionSchema({
    actions: [{ name: 'wave' }, { name: 'wave' }],
  });
  assert.match(String(error), /Duplicate action name/);
});

test('validateActionSchema rejects invalid identifiers', () => {
  const error = validateActionSchema({
    actions: [{ name: '1bad' }],
  });
  assert.match(String(error), /Invalid action name/);
});

test('validateActionSchema rejects empty cue overrides', () => {
  const error = validateActionSchema({
    actions: [{ name: 'wave', cue: '   ' }],
  });
  assert.match(String(error), /invalid cue label/i);
});

test('validateActionSchema rejects cue label collisions', () => {
  const error = validateActionSchema({
    actions: [
      { name: 'look_at_you', cue: 'look at you' },
      { name: 'look_at_you_again', cue: 'look at you' },
    ],
  });
  assert.match(String(error), /collision/i);
});

test('expandActionCues produces one cue per action in declaration order', () => {
  const cues = expandActionCues(SCHEMA);
  assert.deepEqual(cues, [
    { label: 'wave', name: 'wave' },
    { label: 'look at you', name: 'look_at_you' },
    { label: 'shake head', name: 'shake_head' },
  ]);
});

test('summarizeActionCues preserves cue labels and metadata', () => {
  const cues = summarizeActionCues(SCHEMA);
  assert.deepEqual(cues, [
    {
      label: 'wave',
      name: 'wave',
      description: 'Wave a hand.',
      usageHint: 'greeting or saying goodbye',
    },
    {
      label: 'look at you',
      name: 'look_at_you',
      description: 'Turn attention toward the user.',
      usageHint: undefined,
    },
    {
      label: 'shake head',
      name: 'shake_head',
      description: 'Shake the head side to side.',
      usageHint: undefined,
    },
  ]);
});

test('renderActionCueList emits bracketed, comma-separated labels', () => {
  const text = renderActionCueList(SCHEMA);
  assert.equal(text, '[wave], [look at you], [shake head]');
});

test('renderActionCapabilityList ties visible cues back to flat runtime actions', () => {
  const text = renderActionCapabilityList(SCHEMA);
  assert.equal(
    text,
    [
      '- [wave] -> wave: Wave a hand.; use when greeting or saying goodbye',
      '- [look at you] -> look_at_you: Turn attention toward the user.',
      '- [shake head] -> shake_head: Shake the head side to side.',
    ].join('\n')
  );
});

test('findCanonicalActionCue resolves runtime actions back to primary cue labels', () => {
  const cue = findCanonicalActionCue(
    {
      actions: [{ name: 'look_at_you', cue: 'look at you' }],
    },
    'look_at_you'
  );

  assert.ok(cue);
  assert.equal(cue?.label, 'look at you');
});
