//////////////////////////////////////////////////////////////////////////////
//
// action-grammar.ts
//
// - Compiles an ActionSchema into a GBNF grammar that constrains the model's
//   output to interleaved prose and bracketed action cues.
//
// Wire format the grammar produces:
//
//     Hello there! [wave] all done. [mood: happy]
//
// - Cues are short natural-language labels wrapped in square brackets. The
//   grammar restricts the allowed labels to the exact set enumerated by
//   {@link expandActionCues}, so the model cannot invent new cue text.
//
// - Square brackets were chosen over angle-bracketed XML tags to keep the
//   surface stylistically compatible with prose. Earlier versions used a
//   typed `<action name="..." args={...}/>` form whose code-shape caused
//   small models to mimic function-signature boilerplate in dialog.
//
//////////////////////////////////////////////////////////////////////////////

import { expandActionCues, validateActionSchema } from './action-schema.js';
import type { ActionSchema } from './action-schema.js';

/**
 * Thrown when a supplied ActionSchema fails structural validation. Exposed
 * as a named class so callers can distinguish schema errors from other
 * errors.
 */
export class ActionSchemaError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = 'ActionSchemaError';
  }
}

/**
 * Minimal diagnostic grammar used to prove whether grammar-constrained
 * decoding works at all through the runtime path.
 */
export const MINIMAL_TEST_GRAMMAR_SOURCE = 'root ::= "yes" | "no"\n';

/**
 * Generates a GBNF grammar that:
 *   - always starts at `root`;
 *   - accepts any interleaving of prose characters and bracketed cues;
 *   - restricts cue labels to the declared alternation set.
 *
 * The returned source is guaranteed to be <= the bridge's 64 KiB cap for
 * any reasonable schema.
 */
export function compileActionGrammar(schema: ActionSchema): string {
  const validationError = validateActionSchema(schema);
  if (validationError != null) {
    throw new ActionSchemaError(validationError);
  }

  let cues;
  try {
    cues = expandActionCues(schema);
  } catch (error) {
    throw new ActionSchemaError((error as Error).message);
  }

  if (cues.length === 0) {
    throw new ActionSchemaError('Action schema produced no cues.');
  }

  const rules: string[] = [];

  // `root` is one or more atoms, where an atom is either a bracketed action
  // cue or a single prose character. Using `(alt)+` directly (instead of an
  // `atom atom*` pair with an intermediate rule) keeps the grammar stack
  // shallow during sampling: one fewer rule layer, single stack frame per
  // iteration. Requiring `+` (one-or-more) keeps the zero-length deadlock
  // fix we need with stochastic samplers.
  rules.push('root ::= ( action-cue | prose-char )+');

  // Prose is any single codepoint except `[`, which is reserved for the
  // opening bracket of an action cue. The negated class `[^[]` compiles to
  // a single stack frame per prose char, matching what llama.cpp's upstream
  // example grammars use. This replaces an earlier four-alternation positive
  // range rule that, while semantically equivalent, created a large stack
  // fanout per sampled token piece.
  rules.push('prose-char ::= [^[]');

  // Each cue is `[` + literal-label + `]`. The label alternation enumerates
  // the exact legal labels so the model cannot invent unknown cues.
  rules.push('action-cue ::= "[" cue-label "]"');

  const labelAlts = cues.map((cue) => gbnfStringLiteral(cue.label)).join(' | ');
  rules.push(`cue-label ::= ${labelAlts}`);

  return rules.join('\n') + '\n';
}

/**
 * Escapes a string as a GBNF string literal. GBNF uses JSON-style escapes
 * for `"` and `\`. Our labels are short ASCII phrases so no further
 * escaping is required, but we remain defensive in case authors introduce
 * special characters via `cueLabel` / `cueLabels`.
 */
function gbnfStringLiteral(source: string): string {
  const escaped = source.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  return `"${escaped}"`;
}
