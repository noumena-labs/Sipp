//////////////////////////////////////////////////////////////////////////////
//
// agent-grammar.test.ts
//
// - Validation coverage for parseAgentOutput.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  defaultAgentOutput,
  getAgentGrammar,
  parseAgentOutput,
} from './agent-grammar.js';

test('parseAgentOutput accepts well-formed move_to', () => {
  const out = parseAgentOutput(
    '{"intent":{"kind":"move_to","target":{"x":1.5,"z":-2.25},"emotion":"curious"},"status":"heading to the bench"}'
  );
  assert.ok(out);
  assert.equal(out!.intent.kind, 'move_to');
  if (out!.intent.kind === 'move_to') {
    assert.equal(out!.intent.target.x, 1.5);
    assert.equal(out!.intent.target.z, -2.25);
    assert.equal(out!.intent.emotion, 'curious');
  }
  assert.equal(out!.status, 'heading to the bench');
});

test('parseAgentOutput accepts pick_up', () => {
  const out = parseAgentOutput(
    '{"intent":{"kind":"pick_up","objectId":"banana_a","emotion":"happy"},"status":"mine!"}'
  );
  assert.ok(out);
  if (out!.intent.kind === 'pick_up') {
    assert.equal(out!.intent.objectId, 'banana_a');
    assert.equal(out!.intent.emotion, 'happy');
  }
});

test('parseAgentOutput rejects malformed JSON', () => {
  assert.equal(parseAgentOutput('not json'), null);
  assert.equal(parseAgentOutput(''), null);
  assert.equal(parseAgentOutput('{"intent":{}}'), null);
});

test('parseAgentOutput falls back emotion when unknown', () => {
  const out = parseAgentOutput(
    '{"intent":{"kind":"wait","emotion":"giddy","reason":"idle"},"status":""}'
  );
  assert.ok(out);
  // unknown emotion defaults to 'confused'
  assert.equal(out!.intent.emotion, 'confused');
});

test('parseAgentOutput rejects pick_up without objectId', () => {
  assert.equal(
    parseAgentOutput('{"intent":{"kind":"pick_up","emotion":"happy"},"status":""}'),
    null
  );
});

test('parseAgentOutput rejects move_to with non-finite numbers', () => {
  assert.equal(
    parseAgentOutput(
      '{"intent":{"kind":"move_to","target":{"x":"nope","z":0},"emotion":"curious"},"status":""}'
    ),
    null
  );
});

test('defaultAgentOutput yields wait + confused', () => {
  const out = defaultAgentOutput('test');
  assert.equal(out.intent.kind, 'wait');
  assert.equal(out.intent.emotion, 'confused');
});

test('getAgentGrammar returns non-empty GBNF', () => {
  const grammar = getAgentGrammar();
  assert.ok(grammar.includes('root ::='));
  assert.ok(grammar.includes('intent-kind'));
  assert.ok(grammar.includes('"\\"curious\\""'));
});
