import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ChoiceGrammarError,
  compileChoiceGrammar,
  parseChoiceOutput,
} from '../../src/character/choice-grammar.js';

test('compileChoiceGrammar emits a literal alternation root rule', () => {
  const grammar = compileChoiceGrammar(['yes', 'no', 'approach:aria']);
  assert.equal(
    grammar,
    [
      'root ::= ws? choice ws?',
      'choice ::= "yes" | "no" | "approach:aria"',
      'ws ::= (" " | "\\t" | "\\r" | "\\n")+',
      '',
    ].join('\n')
  );
});

test('parseChoiceOutput accepts only explicit choice ids', () => {
  assert.equal(parseChoiceOutput(' yes ', ['yes', 'no']), 'yes');
  assert.equal(parseChoiceOutput(' yes. ', ['yes', 'no']), null);
  assert.equal(parseChoiceOutput('option yes', ['yes', 'no']), null);
  assert.equal(parseChoiceOutput('maybe', ['yes', 'no']), null);
  assert.equal(parseChoiceOutput('', ['yes', 'no']), null);
});

test('compileChoiceGrammar rejects empty or duplicate choices', () => {
  assert.throws(() => compileChoiceGrammar([]), ChoiceGrammarError);
  assert.throws(() => compileChoiceGrammar(['yes', ' yes ']), ChoiceGrammarError);
  assert.throws(() => compileChoiceGrammar(['ok', '']), ChoiceGrammarError);
});
