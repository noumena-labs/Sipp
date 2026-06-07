//////////////////////////////////////////////////////////////////////////////
//
// choice-grammar.ts
//
// - Tiny helper for strict literal choice queries.
// - Used by `CharacterRuntime.choose()` to constrain the model to emit exactly
//   one of the provided choices and nothing else.
//
//////////////////////////////////////////////////////////////////////////////

import { assertGrammarByteSize, literalAlternation } from '../utils/grammar.js';

export class ChoiceGrammarError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = 'ChoiceGrammarError';
  }
}

export function compileChoiceGrammar(choices: readonly string[]): string {
  const normalized = normalizeChoices(choices);
  const grammar = `root ::= ${literalAlternation(normalized)}\n`;
  assertGrammarByteSize(grammar, { label: 'choice grammar' });
  return grammar;
}

export function parseChoiceOutput(raw: string, choices: readonly string[]): string | null {
  const normalized = normalizeChoices(choices);
  const trimmed = raw.trim();
  return normalized.includes(trimmed) ? trimmed : null;
}

function normalizeChoices(choices: readonly string[]): readonly string[] {
  if (choices.length === 0) {
    throw new ChoiceGrammarError('choices must contain at least one option.');
  }
  const normalized = choices.map((choice, index) => {
    if (typeof choice !== 'string') {
      throw new ChoiceGrammarError(`choice at index ${index} must be a string.`);
    }
    const trimmed = choice.trim();
    if (trimmed.length === 0) {
      throw new ChoiceGrammarError(`choice at index ${index} must be non-empty.`);
    }
    return trimmed;
  });
  const unique = new Set(normalized);
  if (unique.size !== normalized.length) {
    throw new ChoiceGrammarError('choices must be unique after trimming.');
  }
  return normalized;
}
