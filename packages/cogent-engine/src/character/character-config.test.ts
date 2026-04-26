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

const validActions: ActionSchema = [{ id: 'wave', description: 'wave hello' }];

function buildValid(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 'aria-01',
    persona: { name: 'Aria', summary: 'A friendly guide.' },
    actions: validActions,
    ...overrides,
  };
}

test('parseCharacterConfig accepts a valid simplified persona and trims fields', () => {
  const raw = buildValid({
    memory: { maxTurns: 4 },
    persona: {
      name: ' Aria ',
      summary: ' A friendly guide. ',
      role: ' community coordinator ',
      backstory: ' Grew up in a stationery shop. ',
      currentLife: {
        description: ' Runs a shared studio. ',
      },
      personality: {
        traits: [' warm ', ' curious '],
        description: ' Notices little details. ',
      },
      anchorExamples: [{ user: ' who are you? ', assistant: ' [wave] I am Aria. ' }],
      dialogExamples: [{ user: ' hello ', assistant: ' [wave] Hi there! ' }],
    },
  });
  const config = parseCharacterConfig(raw);
  assert.equal(config.id, 'aria-01');
  assert.equal(config.persona.name, 'Aria');
  assert.equal(config.persona.summary, 'A friendly guide.');
  assert.equal(config.persona.role, 'community coordinator');
  assert.equal(config.persona.backstory, 'Grew up in a stationery shop.');
  assert.equal(config.persona.currentLife?.description, 'Runs a shared studio.');
  assert.deepEqual(config.persona.personality?.traits, ['warm', 'curious']);
  assert.equal(config.persona.personality?.description, 'Notices little details.');
  assert.equal(config.actions.length, 1);
  assert.equal(config.memory?.maxTurns, 4);
  assert.deepEqual(config.persona.anchorExamples, [{ user: 'who are you?', assistant: '[wave] I am Aria.' }]);
  assert.deepEqual(config.persona.dialogExamples, [{ user: 'hello', assistant: '[wave] Hi there!' }]);
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
  const raw = buildValid({ persona: { summary: 'nameless' } });
  assert.throws(() => parseCharacterConfig(raw), /persona\.name/);
});

test('parseCharacterConfig requires persona to be an object', () => {
  const raw = buildValid({ persona: 'Aria' });
  assert.throws(() => parseCharacterConfig(raw), /persona/);
});

test('parseCharacterConfig surfaces action-schema error messages', () => {
  const raw = buildValid({
    actions: [{ id: 'bad id', description: 'x' }],
  });
  assert.throws(
    () => parseCharacterConfig(raw),
    (err: unknown) => err instanceof CharacterConfigError && /Invalid actions schema/.test(err.message)
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

test('parseCharacterConfig rejects renderer-owned assets and non-object memory', () => {
  assert.throws(() => parseCharacterConfig(buildValid({ assets: {} })), /assets/);
  assert.throws(() => parseCharacterConfig(buildValid({ memory: 7 })), /memory/);
});

test('parseCharacterConfig validates persona notes, anchorExamples, and dialogExamples', () => {
  assert.throws(
    () => parseCharacterConfig(buildValid({ persona: { name: 'Aria', notes: 'nope' } })),
    /persona\.notes/
  );
  assert.throws(
    () => parseCharacterConfig(buildValid({ persona: { name: 'Aria', anchorExamples: 'nope' } })),
    /persona\.anchorExamples/
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
            anchorExamples: [{ user: 'hello', assistant: '' }],
          },
        })
      ),
    /persona\.anchorExamples/
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

test('parseCharacterConfig validates currentLife and simplified personality', () => {
  assert.throws(
    () => parseCharacterConfig(buildValid({ persona: { name: 'Aria', currentLife: 'nope' } })),
    /persona\.currentLife/
  );
  assert.throws(
    () => parseCharacterConfig(buildValid({ persona: { name: 'Aria', personality: 'nope' } })),
    /persona\.personality/
  );
  assert.throws(
    () =>
      parseCharacterConfig(
        buildValid({
          persona: { name: 'Aria', currentLife: { description: ['open shop'] } },
        })
      ),
    /persona\.currentLife\.description/
  );
  assert.throws(
    () =>
      parseCharacterConfig(
        buildValid({
          persona: { name: 'Aria', personality: { traits: 'warm' } },
        })
      ),
    /persona\.personality\.traits/
  );
  assert.throws(
    () =>
      parseCharacterConfig(
        buildValid({
          persona: { name: 'Aria', personality: { description: ['quirky'] } },
        })
      ),
    /persona\.personality\.description/
  );
});

test('resolveMaxMemoryTurns returns configured value or default', () => {
  const withConfig: CharacterConfig = parseCharacterConfig(buildValid({ memory: { maxTurns: 3 } }));
  assert.equal(resolveMaxMemoryTurns(withConfig), 3);

  const noMemory: CharacterConfig = parseCharacterConfig(buildValid());
  assert.equal(resolveMaxMemoryTurns(noMemory), DEFAULT_MEMORY_MAX_TURNS);

  const zero: CharacterConfig = parseCharacterConfig(buildValid({ memory: { maxTurns: 0 } }));
  assert.equal(resolveMaxMemoryTurns(zero), 0);
});
