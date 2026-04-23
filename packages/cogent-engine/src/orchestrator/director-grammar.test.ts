//////////////////////////////////////////////////////////////////////////////
//
// director-grammar.test.ts
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import { getDirectorGrammar, parseDirectorOutput } from './director-grammar.js';

test('parseDirectorOutput accepts narration-only payload', () => {
  const out = parseDirectorOutput('{"note":"the courtyard hums."}');
  assert.ok(out);
  assert.equal(out!.note, 'the courtyard hums.');
  assert.equal(out!.resolutions.length, 0);
});

test('parseDirectorOutput accepts resolution payload', () => {
  const out = parseDirectorOutput(
    '{"note":"aria gets it","resolutions":[{"objectId":"banana_a","winnerAgentId":"aria","note":"closer"}]}'
  );
  assert.ok(out);
  assert.equal(out!.resolutions.length, 1);
  assert.equal(out!.resolutions[0]!.objectId, 'banana_a');
  assert.equal(out!.resolutions[0]!.winnerAgentId, 'aria');
});

test('parseDirectorOutput accepts null winner', () => {
  const out = parseDirectorOutput(
    '{"note":"nobody","resolutions":[{"objectId":"banana_a","winnerAgentId":null}]}'
  );
  assert.ok(out);
  assert.equal(out!.resolutions[0]!.winnerAgentId, null);
});

test('parseDirectorOutput rejects malformed JSON', () => {
  assert.equal(parseDirectorOutput('not-json'), null);
  assert.equal(parseDirectorOutput(''), null);
});

test('getDirectorGrammar contains expected rules', () => {
  const grammar = getDirectorGrammar();
  assert.ok(grammar.includes('resolutions'));
  assert.ok(grammar.includes('winnerAgentId'));
});
