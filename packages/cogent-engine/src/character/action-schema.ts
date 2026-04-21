//////////////////////////////////////////////////////////////////////////////
//
// action-schema.ts
//
// - Declarative schema describing the set of actions a character can emit.
// - Renderer-agnostic: the schema captures action names, argument types, and
//   human-readable descriptions. Bindings (three.js, DOM, etc.) translate
//   emitted actions into runtime effects.
//
// - The model-facing surface is a flat list of natural-language "cues"
//   wrapped in square brackets (e.g. `[wave]`, `[wave softly]`,
//   `[mood: happy]`). Cues collapse enum-valued arguments into one cue per
//   enum value so the model never sees code-shaped meta-syntax it might
//   echo back into dialog. Internally each cue round-trips to the original
//   (name, args) pair via a lookup table built at schema registration.
//
//////////////////////////////////////////////////////////////////////////////

/**
 * Primitive argument types supported by the action grammar generator.
 *
 * For v1 the model-facing cue surface only exposes argless actions and
 * enum-valued actions. `string`, `number`, and `boolean` args remain in the
 * schema vocabulary for future use but are currently not projected into
 * cues — an action that declares a non-enum arg is represented as a bare
 * `[name]` cue with empty args.
 */
export type ActionArgType = 'string' | 'number' | 'boolean' | 'enum';

export interface ActionArgSpec {
  /** Argument identifier — matches the key on the emitted args object. */
  readonly name: string;
  /** Primitive value kind. */
  readonly type: ActionArgType;
  /** Allowed values when {@link type} is `'enum'`. Ignored otherwise. */
  readonly values?: readonly string[];
  /** Optional human-readable description (not surfaced to the model). */
  readonly description?: string;
  /**
   * Numeric bounds — informational only (not enforced by the grammar itself).
   * Bindings can clamp at dispatch.
   */
  readonly min?: number;
  readonly max?: number;
  /**
   * Optional per-enum-value cue label override. When present, keys are enum
   * values and the associated strings are used as the cue suffix instead of
   * the default `"{argName}: {value}"` form. Lets authors write
   * `[wave softly]` instead of `[wave intensity: low]`.
   */
  readonly cueLabels?: Readonly<Record<string, string>>;
}

export interface ActionSpec {
  /**
   * Action identifier. Emitted verbatim as the `name` field on parsed action
   * events so bindings can dispatch with a simple switch. Keep identifiers
   * in snake_case.
   */
  readonly name: string;
  /** Short description of the action (not surfaced to the model). */
  readonly description?: string;
  /** Ordered list of argument specs. Order is significant for readability. */
  readonly args: readonly ActionArgSpec[];
  /**
   * Optional override for the bare `[label]` shown when this action has no
   * enum args. Defaults to the action's snake_case name with underscores
   * converted to spaces (e.g. `shake_head` → `[shake head]`).
   */
  readonly cueLabel?: string;
}

export interface ActionSchema {
  /** The registered actions the model is allowed to emit. */
  readonly actions: readonly ActionSpec[];
}

/**
 * Validates that an action schema is well-formed. Returns an error message on
 * the first problem detected; returns null on success.
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
 * A single bracketed cue that the model can emit and the parser recognises.
 *
 * `label` is the verbatim text that appears between the square brackets —
 * e.g. `wave`, `wave softly`, `mood: happy`. `name` and `args` are the
 * (action, args) tuple the cue dispatches to.
 */
export interface ActionCue {
  readonly label: string;
  readonly name: string;
  readonly args: Readonly<Record<string, string>>;
}

/**
 * Default cue label for an argless action. Converts snake_case to space-
 * separated words so identifiers read naturally in dialog (`shake_head` →
 * `shake head`). Authors can override via {@link ActionSpec.cueLabel}.
 */
function defaultActionLabel(action: ActionSpec): string {
  return action.cueLabel ?? action.name.replace(/_/g, ' ');
}

/**
 * Default cue label for an enum-valued action variant. The format is
 * `{argName}: {value}` which reads naturally for mood-like axes
 * (e.g. `mood: happy`) while still being distinct from general prose.
 * Overridable per-value via {@link ActionArgSpec.cueLabels}.
 */
function defaultEnumLabel(action: ActionSpec, arg: ActionArgSpec, value: string): string {
  const override = arg.cueLabels?.[value];
  if (override != null) {
    return override;
  }
  return `${arg.name.replace(/_/g, ' ')}: ${value}`;
}

/**
 * Expands an action schema into the flat list of cues the model is allowed
 * to emit. The ordering is deterministic (schema declaration order, then
 * enum value order) so prompt-cache keys remain stable across rebuilds.
 *
 * Rules:
 *   - Actions with no args produce a single `[actionLabel]` cue that
 *     dispatches to `{name, args:{}}`.
 *   - Actions whose first arg is an enum produce one cue per enum value,
 *     dispatching to `{name, args:{argName: value}}`. Additional args on
 *     such actions are dropped from the cue surface (there is no
 *     multi-enum cross-product in v1).
 *   - Actions whose first arg is a non-enum primitive (`string`, `number`,
 *     `boolean`) collapse to a single bare `[actionLabel]` cue with empty
 *     args. The typed arg is not exposed to the model.
 *
 * Throws if the schema is malformed or if two different cues collapse to
 * the same label (rare; authors can disambiguate via `cueLabels`).
 */
export function expandActionCues(schema: ActionSchema): readonly ActionCue[] {
  const validationError = validateActionSchema(schema);
  if (validationError != null) {
    throw new Error(validationError);
  }

  const cues: ActionCue[] = [];
  const seen = new Set<string>();

  const push = (cue: ActionCue): void => {
    if (seen.has(cue.label)) {
      throw new Error(
        `Cue label collision: "[${cue.label}]" is produced by more than one action. ` +
          `Disambiguate via cueLabel / cueLabels in the schema.`
      );
    }
    seen.add(cue.label);
    cues.push(cue);
  };

  for (const action of schema.actions) {
    const firstArg = action.args[0];
    if (firstArg != null && firstArg.type === 'enum' && firstArg.values != null) {
      for (const value of firstArg.values) {
        push({
          label: defaultEnumLabel(action, firstArg, value),
          name: action.name,
          args: { [firstArg.name]: value },
        });
      }
    } else {
      push({
        label: defaultActionLabel(action),
        name: action.name,
        args: {},
      });
    }
  }

  return cues;
}

/**
 * Renders the schema as a flat, comma-separated list of bracketed cues for
 * inclusion in the system prompt. The prose complements the GBNF grammar
 * by enumerating the exact vocabulary — the grammar constrains syntax, the
 * prose lists the allowed cues.
 */
export function renderActionCueList(schema: ActionSchema): string {
  const cues = expandActionCues(schema);
  return cues.map((cue) => `[${cue.label}]`).join(', ');
}
