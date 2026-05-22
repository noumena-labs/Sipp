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
//   typed `<action id="..." args={...}/>` form whose code-shape caused
//   small models to mimic function-signature boilerplate in dialog.
//
//////////////////////////////////////////////////////////////////////////////

import {
  assertValidActionSchema,
  expandActionCues,
} from './action-schema.js';
import type { ActionSchema } from './action-schema.js';
import { compileBracketCueGrammar, compileBracketProseGrammar } from '../core/grammar-fragments.js';
import { assertGrammarByteSize } from '../utils/grammar.js';

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
  assertValidActionSchema(schema);
  const cues = expandActionCues(schema);
  const grammar = cues.length === 0
    ? compileBracketProseGrammar()
    : compileBracketCueGrammar({
        cueRuleName: 'action-cue',
        labelRuleName: 'cue-label',
        labels: cues.map((cue) => cue.label),
      });
  assertGrammarByteSize(grammar);
  return grammar;
}
