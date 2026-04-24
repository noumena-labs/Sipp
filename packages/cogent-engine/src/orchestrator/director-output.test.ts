import assert from 'node:assert/strict';
import test from 'node:test';

import {
  compileDirectorOutputGrammar,
  DirectorOutputError,
  parseDirectorOutput,
  resolveDirectorChoices,
} from './director-output.js';
import type { DirectorOutputConfig } from './director-types.js';

test('select_many parser enforces min, max, uniqueness, and legal choices', () => {
  const output: DirectorOutputConfig = {
    shape: 'select_many',
    choices: 'runtime',
    min: 1,
    max: 2,
  };
  const resolved = resolveDirectorChoices(output, {
    choices: [{ id: 'a' }, { id: 'b' }, { id: 'c' }],
  });

  assert.deepEqual(parseDirectorOutput('a\nc', output, resolved).selections.map((selection) => selection.id), ['a', 'c']);
  assert.throws(() => parseDirectorOutput('', output, resolved), DirectorOutputError);
  assert.throws(() => parseDirectorOutput('a\na', output, resolved), DirectorOutputError);
  assert.throws(() => parseDirectorOutput('a\nb\nc', output, resolved), DirectorOutputError);
  assert.throws(() => parseDirectorOutput('z', output, resolved), DirectorOutputError);
});

test('select_many grammar emits one-choice-per-line rules', () => {
  const output: DirectorOutputConfig = {
    shape: 'select_many',
    choices: 'runtime',
    min: 0,
    max: 2,
  };
  const resolved = resolveDirectorChoices(output, {
    choices: [{ id: 'alpha' }, { id: 'beta' }],
  });
  const grammar = compileDirectorOutputGrammar(output, resolved);

  assert.match(grammar ?? '', /selection-line ::= "alpha" \| "beta"/);
  assert.match(grammar ?? '', /linebreak ::= "\\n"/);
});

test('selection grammars preserve punctuation-safe runtime ids', () => {
  const output: DirectorOutputConfig = {
    shape: 'select_one',
    choices: 'runtime',
  };
  const resolved = resolveDirectorChoices(output, {
    choices: [
      { id: 'nav.billing' },
      { id: 'pickup:agent_1' },
      { id: 'call-referee' },
    ],
  });
  const grammar = compileDirectorOutputGrammar(output, resolved);

  assert.equal(grammar, 'root ::= "nav.billing" | "pickup:agent_1" | "call-referee"\n');
});

test('select_slots parser enforces required slots and per-slot choices', () => {
  const output: DirectorOutputConfig = {
    shape: 'select_slots',
    slots: [
      { name: 'intent', choices: 'runtime' },
      { name: 'tone', choices: [{ id: 'brief' }, { id: 'friendly' }] },
    ],
  };
  const resolved = resolveDirectorChoices(output, {
    slotChoices: {
      intent: [{ id: 'advise' }, { id: 'navigate' }],
    },
  });

  const parsed = parseDirectorOutput('intent=advise\ntone=friendly', output, resolved);
  assert.deepEqual(
    parsed.selections.map((selection) => `${selection.slot}:${selection.id}`),
    ['intent:advise', 'tone:friendly']
  );
  assert.throws(() => parseDirectorOutput('intent=advise', output, resolved), DirectorOutputError);
  assert.throws(() => parseDirectorOutput('intent=unknown\ntone=brief', output, resolved), DirectorOutputError);
});

test('text_with_directives extracts grounded directives and keeps prose', () => {
  const output: DirectorOutputConfig = {
    shape: 'text_with_directives',
    directives: 'runtime',
    maxDirectives: 1,
    maxLength: 120,
  };
  const resolved = resolveDirectorChoices(output, {
    directives: [
      { id: 'nav.billing', label: 'Open billing', payload: { route: '/billing' } },
    ],
  });

  const parsed = parseDirectorOutput('Open billing next. [nav.billing]', output, resolved);

  assert.equal(parsed.text, 'Open billing next.');
  assert.equal(parsed.selections[0]?.id, 'nav.billing');
  assert.deepEqual(parsed.selections[0]?.payload, { route: '/billing' });
  assert.throws(() => parseDirectorOutput('Go now. [nav.unknown]', output, resolved), DirectorOutputError);
  assert.throws(() => parseDirectorOutput('Go now. [unknown cue]', output, resolved), DirectorOutputError);
});

test('text_with_directives grammar reserves brackets for known directives', () => {
  const output: DirectorOutputConfig = {
    shape: 'text_with_directives',
    directives: 'runtime',
    maxDirectives: 2,
    maxLength: 120,
  };
  const resolved = resolveDirectorChoices(output, {
    directives: [
      { id: 'nav.billing' },
      { id: 'inspect.menu' },
    ],
  });
  const grammar = compileDirectorOutputGrammar(output, resolved);

  assert.match(grammar ?? '', /^root ::= \( directive-cue \| prose-char \)\+/m);
  assert.match(grammar ?? '', /^prose-char ::= \[\^\[\]/m);
  assert.match(grammar ?? '', /^directive-cue ::= "\[" directive-id "\]"/m);
  assert.match(grammar ?? '', /^directive-id ::= "nav.billing" \| "inspect.menu"/m);
});
