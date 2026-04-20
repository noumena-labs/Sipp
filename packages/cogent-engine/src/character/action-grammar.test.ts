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
  // wave has a single numeric arg. Note that rule-name fragments are
  // sanitised (_ -> -) even though the on-wire action/arg names retain `_`.
  assert.match(grammar, /action-args-wave ::= "\{" "\\"duration_ms\\"" ":" arg-wave-duration-ms "\}"/);
  assert.match(grammar, /arg-wave-duration-ms ::= json-number/);
  // set_mood has an enum followed by a boolean.
  assert.match(
    grammar,
    /action-args-set-mood ::= "\{" "\\"mood\\"" ":" arg-set-mood-mood "," "\\"persist\\"" ":" arg-set-mood-persist "\}"/
  );
  assert.match(grammar, /arg-set-mood-mood ::= "\\"happy\\"" \| "\\"sad\\""/);
  assert.match(grammar, /arg-set-mood-persist ::= json-bool/);
});

test('compileActionGrammar emits only GBNF-legal rule names', () => {
  const grammar = compileActionGrammar(SCHEMA);
  // llama.cpp's GBNF parser accepts only [a-zA-Z0-9-] in rule names. An
  // earlier version of the compiler interpolated user identifiers verbatim,
  // producing rules like `action-args-set_mood` that the parser rejected
  // with `expecting newline or end at _mood | ...`. Guard that regression
  // by scanning every rule LHS.
  const ruleLhsRe = /^([A-Za-z][A-Za-z0-9-]*)\s*::=/;
  const violations: string[] = [];
  for (const line of grammar.split('\n')) {
    const trimmed = line.trimStart();
    if (trimmed === '' || !trimmed.includes('::=')) {
      continue;
    }
    const match = ruleLhsRe.exec(trimmed);
    if (match == null) {
      violations.push(`Illegal rule LHS: ${JSON.stringify(trimmed.slice(0, 80))}`);
    }
  }
  assert.deepEqual(violations, []);
});

test('compileActionGrammar rejects schemas whose names collide after sanitisation', () => {
  assert.throws(
    () =>
      compileActionGrammar({
        actions: [
          { name: 'set_mood', args: [] },
          { name: 'set-mood', args: [] },
        ],
      }),
    (error) => error instanceof ActionSchemaError
  );
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
