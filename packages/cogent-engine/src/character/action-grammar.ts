//////////////////////////////////////////////////////////////////////////////
//
// action-grammar.ts
//
// - Compiles an ActionSchema into a GBNF grammar that constrains the model's
//   output to interleaved prose and well-typed action tags.
//
// Wire format the grammar produces:
//
//     Hello there!<action name="wave" args={"duration_ms":800}/> done.
//
// - The <action .../> tag is a self-closing XML-style envelope so the text
//   stream remains trivially splittable into prose + action tokens.
//
// - Argument payload is JSON (an object literal) so it parses cleanly on the
//   TS side and we reuse the existing JSON-shaped grammar rules.
//
//////////////////////////////////////////////////////////////////////////////

import type { ActionArgSpec, ActionSchema, ActionSpec } from './action-schema.js';
import { validateActionSchema } from './action-schema.js';

/**
 * Thrown when a supplied ActionSchema fails structural validation. Exposed as
 * a named class so callers can distinguish schema errors from other errors.
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
 *   - accepts any interleaving of prose characters and action tags;
 *   - restricts action names to the declared set;
 *   - restricts each argument's value kind to its declared type;
 *   - restricts enum args to the declared set of string literals.
 *
 * The grammar is intentionally permissive about prose: any UTF-8 byte that is
 * NOT `<` is allowed. This lets the model stream natural language freely and
 * switch into an action tag only when it chooses the `<action ` prefix.
 *
 * The returned source is guaranteed to be <= the bridge's 64 KiB cap for any
 * reasonable schema (our largest realistic schema is well under 4 KiB).
 */
export function compileActionGrammar(schema: ActionSchema): string {
  const validationError = validateActionSchema(schema);
  if (validationError != null) {
    throw new ActionSchemaError(validationError);
  }

  const rules: string[] = [];

  // `root` is prose interleaved with zero or more action tags.
  rules.push('root ::= (prose-char | action-tag)*');

  // Prose = any single byte that is not the `<` that would begin a tag.
  // We enumerate by excluding `<` (0x3C). GBNF supports character classes.
  rules.push('prose-char ::= [^<]');

  // `action-tag` begins with the literal `<action name="` followed by a name,
  // then an optional args object, then `/>`.
  rules.push('action-tag ::= "<action name=\\"" action-name "\\"" action-args-part "/>"');

  // action-name is the alternation of the declared identifiers (bare,
  // without surrounding quotes — the quotes are emitted by the parent rule).
  const actionNameAlts = schema.actions
    .map((action) => rawStringLiteral(action.name))
    .join(' | ');
  rules.push(`action-name ::= ${actionNameAlts}`);

  // action-args-part is either empty (when no action declares args) or a
  // space + `args=` + per-action payload alternation.
  const anyActionHasArgs = schema.actions.some((action) => action.args.length > 0);
  if (!anyActionHasArgs) {
    rules.push('action-args-part ::= ""');
  } else {
    rules.push('action-args-part ::= "" | ws "args=" action-args');
    const actionArgsAlts = schema.actions
      .filter((action) => action.args.length > 0)
      .map((action) => `action-args-${action.name}`)
      .join(' | ');
    rules.push(`action-args ::= ${actionArgsAlts}`);
    for (const action of schema.actions) {
      if (action.args.length === 0) {
        continue;
      }
      rules.push(renderActionArgsRule(action));
      for (const arg of action.args) {
        rules.push(renderArgValueRule(action, arg));
      }
    }
  }

  // Shared JSON-ish primitives.
  rules.push('ws ::= " "');
  rules.push('json-number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?');
  rules.push('json-string ::= "\\"" json-string-char* "\\""');
  // Permit any character except `"` and `\` inside strings; escape sequences
  // are out of scope for v1 — our argument strings are ASCII labels.
  rules.push('json-string-char ::= [^"\\\\]');
  rules.push('json-bool ::= "true" | "false"');

  return rules.join('\n') + '\n';
}

function renderActionArgsRule(action: ActionSpec): string {
  const parts: string[] = ['"{"'];
  for (let index = 0; index < action.args.length; index += 1) {
    const arg = action.args[index];
    if (index > 0) {
      parts.push('","');
    }
    parts.push(jsonStringLiteral(arg.name));
    parts.push('":"');
    parts.push(`arg-${action.name}-${arg.name}`);
  }
  parts.push('"}"');
  return `action-args-${action.name} ::= ${parts.join(' ')}`;
}

function renderArgValueRule(action: ActionSpec, arg: ActionArgSpec): string {
  const ruleName = `arg-${action.name}-${arg.name}`;
  switch (arg.type) {
    case 'string':
      return `${ruleName} ::= json-string`;
    case 'number':
      return `${ruleName} ::= json-number`;
    case 'boolean':
      return `${ruleName} ::= json-bool`;
    case 'enum': {
      const alts = (arg.values ?? []).map((value) => jsonStringLiteral(value)).join(' | ');
      return `${ruleName} ::= ${alts}`;
    }
    default: {
      // Exhaustive guard — all cases are handled above.
      const exhaustive: never = arg.type;
      throw new ActionSchemaError(`Unsupported arg type: ${String(exhaustive)}`);
    }
  }
}

/**
 * Escapes a JavaScript string as a GBNF string literal. GBNF uses the same
 * escape conventions as JSON for `"` and `\`, which covers every identifier
 * we accept (validated to match `[A-Za-z_][A-Za-z0-9_]*`) and every enum
 * value we permit (arbitrary ASCII strings).
 */
function jsonStringLiteral(source: string): string {
  const escaped = source.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  return `"\\"${escaped}\\""`;
}

/**
 * Produces a GBNF literal that matches the raw source string *without*
 * surrounding JSON quotes. Used when the parent rule has already emitted the
 * opening/closing `"` characters (e.g. inside `name="..."`).
 */
function rawStringLiteral(source: string): string {
  const escaped = source.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  return `"${escaped}"`;
}
