//////////////////////////////////////////////////////////////////////////////
//
// action-grammar.test.ts
//
// - Verifies the compiled GBNF grammar has the expected shape, is within
//   the bridge's 64 KiB cap, and rejects malformed schemas.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import { MAX_GRAMMAR_BYTES } from '../wasm/wasm-bridge.js';
import { ActionSchemaError, compileActionGrammar } from './action-grammar.js';
import type { ActionSchema } from './action-schema.js';

const SCHEMA: ActionSchema = {
  actions: [
    {
      name: 'wave',
      args: [],
    },
    {
      name: 'set_mood',
      args: [
        { name: 'mood', type: 'enum', values: ['happy', 'sad'] },
      ],
    },
    {
      name: 'shake_head',
      args: [],
    },
  ],
};

test('compileActionGrammar throws ActionSchemaError on invalid input', () => {
  assert.throws(
    () => compileActionGrammar({ actions: [] }),
    (error) => error instanceof ActionSchemaError
  );
});

test('compileActionGrammar emits a root rule that requires at least one atom', () => {
  const grammar = compileActionGrammar(SCHEMA);
  // Kleene-plus at root: grammar stack stays shallow (single frame per
  // iteration) while still forbidding the zero-length match that caused a
  // sampler deadlock with LFM2 earlier.
  assert.match(grammar, /^root ::= \( action-cue \| prose-char \)\+/m);
});

test('compileActionGrammar allows broad prose while reserving [ for cues', () => {
  const grammar = compileActionGrammar(SCHEMA);
  // Negated class matches any single codepoint except `[`. This replaced a
  // four-alternation positive-range rule that caused large grammar-stack
  // fanout during sampling.
  assert.match(grammar, /^prose-char ::= \[\^\[\]/m);
});

test('compileActionGrammar wraps cue labels in square brackets and restricts alternatives', () => {
  const grammar = compileActionGrammar(SCHEMA);
  assert.match(grammar, /action-cue ::= "\[" cue-label "\]"/);
  assert.match(
    grammar,
    /cue-label ::= "wave" \| "mood: happy" \| "mood: sad" \| "shake head"/
  );
});

test('compileActionGrammar includes aliases even when prompt rendering stays canonical', () => {
  const grammar = compileActionGrammar({
    actions: [
      {
        name: 'set_mood',
        args: [
          {
            name: 'mood',
            type: 'enum',
            values: ['happy'],
            cueLabels: { happy: 'smile' },
            cueAliases: { happy: ['mood: happy', 'be happy'] },
          },
        ],
      },
    ],
  });

  assert.match(grammar, /cue-label ::= "smile" \| "mood: happy" \| "be happy"/);
});

test('compileActionGrammar rejects schemas with colliding cue labels', () => {
  assert.throws(
    () =>
      compileActionGrammar({
        actions: [
          { name: 'wave', args: [] },
          { name: 'wave_', cueLabel: 'wave', args: [] },
        ],
      }),
    (error) => error instanceof ActionSchemaError
  );
});

test('compileActionGrammar stays well below the bridge grammar size cap', () => {
  const grammar = compileActionGrammar(SCHEMA);
  const byteLength = new TextEncoder().encode(grammar).byteLength;
  assert.ok(byteLength < MAX_GRAMMAR_BYTES, `grammar was ${byteLength} bytes`);
  // Realistic action grammars should be tiny — a few hundred bytes at most.
  assert.ok(byteLength < 4096, `grammar unexpectedly large: ${byteLength} bytes`);
});

test('compileActionGrammar handles schemas with only argless actions', () => {
  const grammar = compileActionGrammar({
    actions: [
      { name: 'idle', args: [] },
      { name: 'stop', args: [] },
    ],
  });
  assert.match(grammar, /cue-label ::= "idle" \| "stop"/);
});
