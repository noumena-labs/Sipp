//////////////////////////////////////////////////////////////////////////////
//
// action-schema.ts
//
// - Declarative schema describing the set of actions a character can emit.
// - Renderer-agnostic: each declared action is one model-facing cue and one
//   runtime action name. Bindings (three.js, DOM, etc.) translate emitted
//   action names into runtime effects.
//
// - The model-facing surface is intentionally flat: one bracketed cue such as
//   `[wave]` or `[look at you]` maps directly to one runtime action name.
//   This keeps character authoring, prompt rendering, parsing, and dispatch
//   aligned 1:1 with no hidden arg expansion or enum projection.
//
//////////////////////////////////////////////////////////////////////////////

export interface ActionSpec {
  /**
   * Action identifier. Emitted verbatim as the `name` field on parsed action
   * events so bindings can dispatch with a simple switch. Keep identifiers
   * in snake_case.
   */
  readonly name: string;
  /**
   * Optional override for the bracketed cue shown to the model. Defaults to
   * the action's snake_case name with underscores converted to spaces.
   */
  readonly cue?: string;
  /** Short description of the action for prompts and tooling. */
  readonly description?: string;
  /**
   * Optional short usage hint shown to the model in prompts. Helps nudge when
   * the action feels natural without hardcoding character-specific logic.
   */
  readonly usageHint?: string;
}

export interface ActionSchema {
  /** The registered actions the model is allowed to emit. */
  readonly actions: readonly ActionSpec[];
}

export interface ActionCue {
  /** The exact label that appears between square brackets. */
  readonly label: string;
  /** The runtime action name the cue dispatches to. */
  readonly name: string;
}

export interface ActionCueSummary {
  readonly label: string;
  readonly name: string;
  readonly description?: string;
  readonly usageHint?: string;
}

const IDENTIFIER_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

function defaultActionLabel(action: ActionSpec): string {
  return action.cue?.trim() || action.name.replace(/_/g, ' ');
}

/**
 * Validates that an action schema is well-formed. Returns an error message on
 * the first problem detected; returns null on success.
 */
export function validateActionSchema(schema: ActionSchema): string | null {
  if (!schema || !Array.isArray(schema.actions)) {
    return 'ActionSchema.actions must be an array.';
  }
  if (schema.actions.length === 0) {
    return 'ActionSchema.actions must contain at least one action.';
  }

  const seenActionNames = new Set<string>();
  const seenCueLabels = new Set<string>();
  for (const action of schema.actions) {
    if (!action || typeof action.name !== 'string' || !IDENTIFIER_RE.test(action.name)) {
      return `Invalid action name: ${JSON.stringify(action?.name)}`;
    }
    if (seenActionNames.has(action.name)) {
      return `Duplicate action name: ${action.name}`;
    }
    seenActionNames.add(action.name);

    if (action.cue != null && (typeof action.cue !== 'string' || action.cue.trim().length === 0)) {
      return `Action "${action.name}" has an invalid cue label.`;
    }

    const label = defaultActionLabel(action);
    if (seenCueLabels.has(label)) {
      return `Cue label collision: "[${label}]" is produced by more than one action.`;
    }
    seenCueLabels.add(label);
  }

  return null;
}

/**
 * Expands an action schema into the flat list of cues the model is allowed
 * to emit. Ordering is deterministic (schema declaration order) so prompt-
 * cache keys remain stable across rebuilds.
 */
export function expandActionCues(schema: ActionSchema): readonly ActionCue[] {
  const validationError = validateActionSchema(schema);
  if (validationError != null) {
    throw new Error(validationError);
  }

  return schema.actions.map((action) => ({
    label: defaultActionLabel(action),
    name: action.name,
  }));
}

/**
 * Returns one canonical prompt-facing cue per runtime action.
 */
export function summarizeActionCues(schema: ActionSchema): readonly ActionCueSummary[] {
  const validationError = validateActionSchema(schema);
  if (validationError != null) {
    throw new Error(validationError);
  }

  return schema.actions.map((action) => ({
    label: defaultActionLabel(action),
    name: action.name,
    description: action.description,
    usageHint: action.usageHint,
  }));
}

/** Resolves a runtime action name back to its canonical prompt-facing cue. */
export function findCanonicalActionCue(
  schema: ActionSchema,
  name: string
): ActionCueSummary | null {
  for (const summary of summarizeActionCues(schema)) {
    if (summary.name === name) {
      return summary;
    }
  }
  return null;
}

/**
 * Renders the schema as a flat, comma-separated list of bracketed cues for
 * inclusion in the system prompt. The prose complements the GBNF grammar
 * by enumerating the exact vocabulary — the grammar constrains syntax, the
 * prose lists the allowed cues.
 */
export function renderActionCueList(schema: ActionSchema): string {
  const cues = summarizeActionCues(schema);
  return cues.map((cue) => `[${cue.label}]`).join(', ');
}

/**
 * Renders a deterministic, model-facing capability list derived from the
 * action schema. Each line ties the visible cue surface back to the runtime
 * action name so prompts can describe capabilities without hardcoding any
 * particular character.json shape.
 */
export function renderActionCapabilityList(schema: ActionSchema): string {
  const cues = summarizeActionCues(schema);

  return cues
    .map((cue) => {
      const description = cue.description?.trim();
      const usageHint = cue.usageHint?.trim();
      const detailParts = [
        description && description.length > 0 ? description : null,
        usageHint && usageHint.length > 0 ? `use when ${usageHint}` : null,
      ].filter((part): part is string => part != null);
      return detailParts.length === 0
        ? `- [${cue.label}] -> ${cue.name}`
        : `- [${cue.label}] -> ${cue.name}: ${detailParts.join('; ')}`;
    })
    .join('\n');
}
