//////////////////////////////////////////////////////////////////////////////
//
// character-config.test.ts
//
// - Covers parseCharacterConfig validation rules and resolveMaxMemoryTurns
//   defaulting behaviour.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import type { ActionSchema } from './action-schema.js';
import {
  CharacterConfigError,
  DEFAULT_MEMORY_MAX_TURNS,
  parseCharacterConfig,
  resolveMaxMemoryTurns,
  type CharacterConfig,
} from './character-config.js';

const validActions: ActionSchema = {
  actions: [
    { name: 'wave', description: 'wave hello', args: [] },
  ],
};

function buildValid(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 'aria-01',
    persona: { name: 'Aria', description: 'A friendly guide.' },
    actions: validActions,
    ...overrides,
  };
}

test('parseCharacterConfig accepts a minimal valid config and round-trips fields', () => {
  const raw = buildValid({
    assets: { vrm: '/models/aria.vrm' },
    memory: { maxTurns: 4 },
    persona: {
      name: 'Aria',
      description: 'A friendly guide.',
      dialogExamples: [
        { user: ' hello ', assistant: ' [wave] Hi there! ' },
      ],
    },
  });
  const config = parseCharacterConfig(raw);
  assert.equal(config.id, 'aria-01');
  assert.equal(config.persona.name, 'Aria');
  assert.equal(config.actions.actions.length, 1);
  assert.equal(config.assets?.vrm, '/models/aria.vrm');
  assert.equal(config.memory?.maxTurns, 4);
  assert.deepEqual(config.persona.dialogExamples, [
    { user: 'hello', assistant: '[wave] Hi there!' },
  ]);
});

test('parseCharacterConfig rejects non-object input', () => {
  assert.throws(() => parseCharacterConfig(null), CharacterConfigError);
  assert.throws(() => parseCharacterConfig('nope' as unknown), CharacterConfigError);
  assert.throws(() => parseCharacterConfig(42 as unknown), CharacterConfigError);
});

test('parseCharacterConfig rejects missing or invalid id', () => {
  const missing = buildValid();
  delete (missing as { id?: unknown }).id;
  assert.throws(() => parseCharacterConfig(missing), /id/);

  const bad = buildValid({ id: 'has spaces!' });
  assert.throws(() => parseCharacterConfig(bad), /id/);

  const empty = buildValid({ id: '' });
  assert.throws(() => parseCharacterConfig(empty), /id/);
});

test('parseCharacterConfig rejects missing persona.name', () => {
  const raw = buildValid({ persona: { description: 'nameless' } });
  assert.throws(() => parseCharacterConfig(raw), /persona\.name/);
});

test('parseCharacterConfig requires persona to be an object', () => {
  const raw = buildValid({ persona: 'Aria' });
  assert.throws(() => parseCharacterConfig(raw), /persona/);
});

test('parseCharacterConfig surfaces action-schema error messages', () => {
  const raw = buildValid({
    actions: { actions: [{ name: 'bad id', description: 'x', args: [] }] },
  });
  assert.throws(
    () => parseCharacterConfig(raw),
    (err: unknown) =>
      err instanceof CharacterConfigError && /Invalid actions schema/.test(err.message)
  );
});

test('parseCharacterConfig validates memory.maxTurns', () => {
  const float = buildValid({ memory: { maxTurns: 2.5 } });
  assert.throws(() => parseCharacterConfig(float), /maxTurns/);

  const negative = buildValid({ memory: { maxTurns: -1 } });
  assert.throws(() => parseCharacterConfig(negative), /maxTurns/);

  const wrong = buildValid({ memory: { maxTurns: 'lots' } });
  assert.throws(() => parseCharacterConfig(wrong), /maxTurns/);

  const zero = parseCharacterConfig(buildValid({ memory: { maxTurns: 0 } }));
  assert.equal(zero.memory?.maxTurns, 0);
});

test('parseCharacterConfig rejects non-object assets/memory', () => {
  assert.throws(() => parseCharacterConfig(buildValid({ assets: 'nope' })), /assets/);
  assert.throws(() => parseCharacterConfig(buildValid({ memory: 7 })), /memory/);
});

test('parseCharacterConfig validates persona notes and dialogExamples', () => {
  assert.throws(
    () => parseCharacterConfig(buildValid({ persona: { name: 'Aria', notes: 'nope' } })),
    /persona\.notes/
  );
  assert.throws(
    () => parseCharacterConfig(buildValid({ persona: { name: 'Aria', dialogExamples: 'nope' } })),
    /persona\.dialogExamples/
  );
  assert.throws(
    () =>
      parseCharacterConfig(
        buildValid({
          persona: {
            name: 'Aria',
            dialogExamples: [{ user: 'hello', assistant: '' }],
          },
        })
      ),
    /persona\.dialogExamples/
  );
});

test('resolveMaxMemoryTurns returns configured value or default', () => {
  const withConfig: CharacterConfig = parseCharacterConfig(
    buildValid({ memory: { maxTurns: 3 } })
  );
  assert.equal(resolveMaxMemoryTurns(withConfig), 3);

  const noMemory: CharacterConfig = parseCharacterConfig(buildValid());
  assert.equal(resolveMaxMemoryTurns(noMemory), DEFAULT_MEMORY_MAX_TURNS);

  const zero: CharacterConfig = parseCharacterConfig(
    buildValid({ memory: { maxTurns: 0 } })
  );
  assert.equal(resolveMaxMemoryTurns(zero), 0);
});
