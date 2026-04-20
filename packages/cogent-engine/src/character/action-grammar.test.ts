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
      args: [{ name: 'duration_ms', type: 'number' }],
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

test('compileActionGrammar throws ActionSchemaError on invalid input', () => {
  assert.throws(
    () => compileActionGrammar({ actions: [] }),
    (error) => error instanceof ActionSchemaError
  );
});

test('compileActionGrammar emits a root rule that alternates prose and tags', () => {
  const grammar = compileActionGrammar(SCHEMA);
  assert.match(grammar, /^root ::= \(prose-char \| action-tag\)\*/m);
});

test('compileActionGrammar encodes action names as bare literals', () => {
  const grammar = compileActionGrammar(SCHEMA);
  // The action-name alternation must contain each action's literal.
  assert.match(grammar, /action-name ::= "wave" \| "set_mood" \| "idle"/);
});

test('compileActionGrammar renders per-action args rules in declaration order', () => {
  const grammar = compileActionGrammar(SCHEMA);
  // wave has a single numeric arg.
  assert.match(grammar, /action-args-wave ::= "\{" "\\"duration_ms\\"" ":" arg-wave-duration_ms "\}"/);
  assert.match(grammar, /arg-wave-duration_ms ::= json-number/);
  // set_mood has an enum followed by a boolean.
  assert.match(
    grammar,
    /action-args-set_mood ::= "\{" "\\"mood\\"" ":" arg-set_mood-mood "," "\\"persist\\"" ":" arg-set_mood-persist "\}"/
  );
  assert.match(grammar, /arg-set_mood-mood ::= "\\"happy\\"" \| "\\"sad\\""/);
  assert.match(grammar, /arg-set_mood-persist ::= json-bool/);
});

test('compileActionGrammar omits args part when all actions are argless', () => {
  const grammar = compileActionGrammar({
    actions: [
      { name: 'idle', args: [] },
      { name: 'stop', args: [] },
    ],
  });
  assert.match(grammar, /action-args-part ::= ""/);
  // No per-action args rules should be emitted when no action declares args.
  assert.ok(!/action-args-/.test(grammar.replaceAll('action-args-part', '')));
});

test('compileActionGrammar stays well below the bridge grammar size cap', () => {
  const grammar = compileActionGrammar(SCHEMA);
  const byteLength = new TextEncoder().encode(grammar).byteLength;
  assert.ok(byteLength < MAX_GRAMMAR_BYTES, `grammar was ${byteLength} bytes`);
  // Realistic action grammars should be tiny — a few hundred bytes.
  assert.ok(byteLength < 4096, `grammar unexpectedly large: ${byteLength} bytes`);
});
