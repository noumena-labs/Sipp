//////////////////////////////////////////////////////////////////////////////
//
// action-schema.ts
//
// - Declarative schema describing the set of actions a character can emit.
// - Renderer-agnostic: the schema captures action names, argument types, and
//   human-readable descriptions. Bindings (three.js, DOM, etc.) translate
//   emitted actions into runtime effects.
//
//////////////////////////////////////////////////////////////////////////////

/**
 * Primitive argument types supported by the action grammar generator.
 *
 * The types intentionally mirror the JSON value kinds that GBNF can express
 * naturally: strings, numbers, booleans, and string enumerations. Complex
 * nested payloads are discouraged for v1 because they dramatically increase
 * grammar size and sampling cost.
 */
export type ActionArgType = 'string' | 'number' | 'boolean' | 'enum';

export interface ActionArgSpec {
  /** Argument identifier — must match the JSON key emitted by the model. */
  readonly name: string;
  /** Primitive value kind. */
  readonly type: ActionArgType;
  /** Allowed values when {@link type} is `'enum'`. Ignored otherwise. */
  readonly values?: readonly string[];
  /** Optional human-readable description (included in prompt text only). */
  readonly description?: string;
  /**
   * Numeric bounds — informational only (not enforced by the grammar itself,
   * which always accepts any JSON number). Bindings can clamp at dispatch.
   */
  readonly min?: number;
  readonly max?: number;
}

export interface ActionSpec {
  /**
   * Action identifier. Emitted verbatim as the JSON `name` field so bindings
   * can dispatch with a simple switch. Keep identifiers in snake_case.
   */
  readonly name: string;
  /** Short description of the action, surfaced to the model in the prompt. */
  readonly description?: string;
  /** Ordered list of argument specs. Order is significant for readability. */
  readonly args: readonly ActionArgSpec[];
}

export interface ActionSchema {
  /** The registered actions the model is allowed to emit. */
  readonly actions: readonly ActionSpec[];
}

/**
 * Validates that an action schema is well-formed. Returns an error message on
 * the first problem detected; returns null on success.
 *
 * The checks are intentionally conservative:
 *   - every action/arg name must be a non-empty identifier matching the
 *     pattern `[A-Za-z_][A-Za-z0-9_]*`;
 *   - action names must be unique;
 *   - arg names within an action must be unique;
 *   - enum args must declare at least one allowed value;
 *   - non-enum args must NOT declare `values`.
 */
const IDENTIFIER_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

export function validateActionSchema(schema: ActionSchema): string | null {
  if (!schema || !Array.isArray(schema.actions)) {
    return 'ActionSchema.actions must be an array.';
  }
  if (schema.actions.length === 0) {
    return 'ActionSchema.actions must contain at least one action.';
  }

  const seenActionNames = new Set<string>();
  for (const action of schema.actions) {
    if (!action || typeof action.name !== 'string' || !IDENTIFIER_RE.test(action.name)) {
      return `Invalid action name: ${JSON.stringify(action?.name)}`;
    }
    if (seenActionNames.has(action.name)) {
      return `Duplicate action name: ${action.name}`;
    }
    seenActionNames.add(action.name);

    if (!Array.isArray(action.args)) {
      return `Action "${action.name}" must declare an args array (use [] for no args).`;
    }

    const seenArgNames = new Set<string>();
    for (const arg of action.args) {
      if (!arg || typeof arg.name !== 'string' || !IDENTIFIER_RE.test(arg.name)) {
        return `Invalid arg name in action "${action.name}": ${JSON.stringify(arg?.name)}`;
      }
      if (seenArgNames.has(arg.name)) {
        return `Duplicate arg "${arg.name}" in action "${action.name}".`;
      }
      seenArgNames.add(arg.name);

      if (arg.type === 'enum') {
        if (!Array.isArray(arg.values) || arg.values.length === 0) {
          return `Enum arg "${arg.name}" in action "${action.name}" requires a non-empty values[].`;
        }
        for (const value of arg.values) {
          if (typeof value !== 'string' || value.length === 0) {
            return `Enum arg "${arg.name}" in action "${action.name}" has an invalid value.`;
          }
        }
      } else if (arg.values != null) {
        return `Non-enum arg "${arg.name}" in action "${action.name}" must not declare values[].`;
      }
    }
  }

  return null;
}

/**
 * Renders the schema as human-readable prose for inclusion in the system
 * prompt. The prose complements the GBNF grammar by explaining each action's
 * intent to the model — the grammar constrains syntax, the prose teaches
 * semantics.
 */
export function renderActionSchemaForPrompt(schema: ActionSchema): string {
  const lines: string[] = [];
  for (const action of schema.actions) {
    const argsSummary = action.args
      .map((arg) => {
        if (arg.type === 'enum' && arg.values != null) {
          return `${arg.name}: ${arg.values.map((value) => `"${value}"`).join(' | ')}`;
        }
        return `${arg.name}: ${arg.type}`;
      })
      .join(', ');
    const descriptionSuffix = action.description ? ` — ${action.description}` : '';
    lines.push(`- ${action.name}(${argsSummary})${descriptionSuffix}`);
    for (const arg of action.args) {
      if (arg.description) {
        lines.push(`    · ${arg.name}: ${arg.description}`);
      }
    }
  }
  return lines.join('\n');
}
