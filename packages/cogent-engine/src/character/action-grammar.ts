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

  // `root` is one or more atoms; an atom is either a prose character or a
  // bracketed cue. Requiring at least one atom avoids the zero-length root
  // deadlock we observed with LFM2 under grammar-constrained sampling.
  rules.push('root ::= atom atom*');
  rules.push('atom ::= prose-char | action-cue');

  // Prose is expressed as explicit positive ranges rather than a negated
  // catch-all like `[^]` / `[^]` / `[^\[]`. The negated form
  // parsed correctly, but under llama.cpp's grammar sampler it could still
  // collapse the grammar stack to empty on accepted token pieces, which then
  // aborts on the next sampler apply. These ranges keep ordinary dialog free
  // while reserving `[` exclusively for action cues.
  //
  // Allowed prose:
  //   - ASCII whitespace: space, tab, CR, LF
  //   - Printable ASCII before `[` : U+0021..U+005A
  //   - Printable ASCII after `[`  : U+005C..U+007E
  //   - All non-ASCII Unicode code points
  rules.push('prose-char ::= [ \\t\\n\\r] | [!-Z] | [\\\\-~] | [\\x80-\\U0010FFFF]');

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
