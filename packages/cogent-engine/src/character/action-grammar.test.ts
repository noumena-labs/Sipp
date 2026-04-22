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
    { name: 'wave' },
    { name: 'smile' },
    { name: 'shake_head' },
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
  assert.match(grammar, /^root ::= \( action-cue \| prose-char \)\+/m);
});

test('compileActionGrammar allows broad prose while reserving [ for cues', () => {
  const grammar = compileActionGrammar(SCHEMA);
  assert.match(grammar, /^prose-char ::= \[\^\[/m);
});

test('compileActionGrammar wraps cue labels in square brackets and restricts alternatives', () => {
  const grammar = compileActionGrammar(SCHEMA);
  assert.match(grammar, /action-cue ::= "\[" cue-label "\]"/);
  assert.match(grammar, /cue-label ::= "wave" \| "smile" \| "shake head"/);
});

test('compileActionGrammar uses custom cue labels when provided', () => {
  const grammar = compileActionGrammar({
    actions: [
      { name: 'look_at_you', cue: 'look at you' },
      { name: 'look_down', cue: 'look down' },
    ],
  });

  assert.match(grammar, /cue-label ::= "look at you" \| "look down"/);
});

test('compileActionGrammar rejects schemas with colliding cue labels', () => {
  assert.throws(
    () =>
      compileActionGrammar({
        actions: [
          { name: 'wave' },
          { name: 'wave_again', cue: 'wave' },
        ],
      }),
    (error) => error instanceof ActionSchemaError
  );
});

test('compileActionGrammar stays well below the bridge grammar size cap', () => {
  const grammar = compileActionGrammar(SCHEMA);
  const byteLength = new TextEncoder().encode(grammar).byteLength;
  assert.ok(byteLength < MAX_GRAMMAR_BYTES, `grammar was ${byteLength} bytes`);
  assert.ok(byteLength < 4096, `grammar unexpectedly large: ${byteLength} bytes`);
});
