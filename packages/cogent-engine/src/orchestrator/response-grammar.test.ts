import assert from 'node:assert/strict';
import test from 'node:test';

import { compileResponseGrammar } from './response-grammar.js';
import { renderResponseSchemaSummary, validateResponseValue } from './response-schema.js';
import type { ResponseSchema } from './director-types.js';

const SCHEMA: ResponseSchema = {
  type: 'object',
  properties: {
    note: { type: 'string', maxLength: 80 },
    winnerAgentId: { type: 'string', nullable: true, maxLength: 32 },
    confidence: { type: 'number' },
  },
};

test('compileResponseGrammar emits JSON object rules', () => {
  const grammar = compileResponseGrammar(SCHEMA);
  assert.ok(grammar.includes('root ::='));
  assert.ok(grammar.includes('winnerAgentId'));
  assert.ok(grammar.includes('number ::='));
});

test('validateResponseValue accepts matching values', () => {
  const error = validateResponseValue(
    { note: 'Aria gets there first.', winnerAgentId: 'aria', confidence: 0.82 },
    SCHEMA
  );
  assert.equal(error, null);
});

test('validateResponseValue rejects missing keys and wrong types', () => {
  assert.match(
    validateResponseValue({ note: 'x', confidence: 1 }, SCHEMA) ?? '',
    /winnerAgentId is required/
  );
  assert.match(
    validateResponseValue(
      { note: 'x', winnerAgentId: null, confidence: 'high' as unknown as number },
      SCHEMA
    ) ?? '',
    /must be a finite number/
  );
});

test('renderResponseSchemaSummary describes nested contracts', () => {
  const summary = renderResponseSchemaSummary(SCHEMA);
  assert.ok(summary.includes('object'));
  assert.ok(summary.includes('winnerAgentId'));
});
