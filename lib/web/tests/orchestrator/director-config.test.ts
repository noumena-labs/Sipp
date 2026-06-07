import assert from 'node:assert/strict';
import test from 'node:test';

import { DirectorConfigError, parseDirectorConfig } from '../../src/orchestrator/director-config.js';

test('parseDirectorConfig accepts shape-driven tasks', () => {
  const config = parseDirectorConfig({
    id: 'courtyard-director',
    scenario: { name: 'Courtyard', summary: 'Snack-time courtyard.' },
    director: {
      role: 'Scenario director',
      objective: 'Keep the scene coherent.',
      instructions: ['Only use the provided state.'],
    },
    inputs: {
      conflict: { kind: 'data', description: 'Conflict detector output.' },
      screenshot: { kind: 'image', description: 'Current app screenshot.' },
    },
    tasks: {
      resolve_conflict: {
        purpose: 'Resolve a conflict.',
        inputs: ['conflict'],
        instructions: ['Pick a ruling.'],
        output: { shape: 'select_one', choices: 'runtime' },
      },
      inspect_screen: {
        inputs: ['screenshot'],
        output: {
          shape: 'text_with_directives',
          maxDirectives: 1,
          directives: [
            { id: 'nav.billing', label: 'Open billing' },
          ],
        },
      },
    },
  });

  assert.equal(config.id, 'courtyard-director');
  assert.equal(config.director.role, 'Scenario director');
  assert.equal(config.inputs?.screenshot?.kind, 'image');
  assert.equal(config.tasks.resolve_conflict?.output.shape, 'select_one');
  assert.equal(config.tasks.inspect_screen?.output.shape, 'text_with_directives');
});

test('parseDirectorConfig rejects unknown task input references', () => {
  assert.throws(
    () =>
      parseDirectorConfig({
        id: 'd',
        director: { role: 'role' },
        inputs: { known: { kind: 'text', description: 'known input' } },
        tasks: {
          q: {
            inputs: ['missing'],
            output: { shape: 'text' },
          },
        },
      }),
    (error: unknown) =>
      error instanceof DirectorConfigError && /unknown input/.test(error.message)
  );
});

test('parseDirectorConfig rejects invalid output shapes', () => {
  assert.throws(
    () =>
      parseDirectorConfig({
        id: 'd',
        director: { role: 'role' },
        tasks: {
          q: {
            output: { shape: 'json', response: { type: 'object' } },
          },
        },
      }),
    (error: unknown) =>
      error instanceof DirectorConfigError && /Unsupported output shape/.test(error.message)
  );
});

test('parseDirectorConfig rejects duplicate static choice ids', () => {
  assert.throws(
    () =>
      parseDirectorConfig({
        id: 'd',
        director: { role: 'role' },
        tasks: {
          q: {
            output: {
              shape: 'select_one',
              choices: [
                { id: 'yes' },
                { id: 'yes' },
              ],
            },
          },
        },
      }),
    (error: unknown) =>
      error instanceof DirectorConfigError && /duplicate choice id/.test(error.message)
  );
});
