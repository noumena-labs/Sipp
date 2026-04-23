import assert from 'node:assert/strict';
import test from 'node:test';

import { DirectorConfigError, parseDirectorConfig } from './director-config.js';

test('parseDirectorConfig accepts a valid multi-query config', () => {
  const config = parseDirectorConfig({
    id: 'courtyard-director',
    scenario: { name: 'Courtyard', summary: 'Snack-time courtyard.' },
    director: {
      role: 'Scenario director',
      objective: 'Keep the scene coherent.',
      instructions: ['Only use the provided state.'],
    },
    hooks: {
      conflict: 'Conflict detector output.',
      state_snapshot: 'Current world state.',
    },
    queries: {
      resolve_conflict: {
        description: 'Resolve a conflict.',
        hooks: ['conflict', 'state_snapshot'],
        instructions: ['Pick a winner.'],
        response: {
          type: 'object',
          properties: {
            winnerAgentId: { type: 'string', nullable: true, maxLength: 32 },
            note: { type: 'string', maxLength: 120 },
          },
        },
      },
      narrate: {
        response: {
          type: 'object',
          properties: {
            note: { type: 'string', maxLength: 120 },
          },
        },
      },
    },
  });

  assert.equal(config.id, 'courtyard-director');
  assert.equal(config.director.role, 'Scenario director');
  assert.ok(config.queries.resolve_conflict);
  assert.ok(config.queries.narrate);
});

test('parseDirectorConfig rejects unknown query hook references', () => {
  assert.throws(
    () =>
      parseDirectorConfig({
        id: 'd',
        director: { role: 'role' },
        hooks: { known: 'known hook' },
        queries: {
          q: {
            hooks: ['missing'],
            response: { type: 'object', properties: { note: { type: 'string' } } },
          },
        },
      }),
    (error: unknown) =>
      error instanceof DirectorConfigError && /unknown hook/.test(error.message)
  );
});

test('parseDirectorConfig rejects invalid response schema objects', () => {
  assert.throws(
    () =>
      parseDirectorConfig({
        id: 'd',
        director: { role: 'role' },
        queries: {
          q: {
            response: { type: 'object', properties: {} },
          },
        },
      }),
    (error: unknown) =>
      error instanceof DirectorConfigError && /must define at least one field/.test(error.message)
  );
});
