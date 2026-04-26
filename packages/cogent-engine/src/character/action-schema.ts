//////////////////////////////////////////////////////////////////////////////
//
// action-schema.ts
//
// - Declarative schema describing the set of actions a character can emit.
// - Renderer-agnostic: each declared action is one model-facing cue and one
//   runtime action id. Bindings (three.js, DOM, etc.) translate emitted
//   action ids into runtime effects.
//
// - The model-facing surface is intentionally flat: one bracketed cue such as
//   `[wave]` or `[look at you]` maps directly to one runtime action id.
//   This keeps character authoring, prompt rendering, parsing, and dispatch
//   aligned 1:1 with no hidden arg expansion or enum projection.
//
//////////////////////////////////////////////////////////////////////////////

export interface ActionSpec {
  /**
   * Action identifier. Emitted verbatim as the `id` field on parsed action
   * events so bindings can dispatch with a simple switch. Keep identifiers
   * in snake_case.
   */
  readonly id: string;
  /**
   * Optional override for the bracketed cue shown to the model. Defaults to
   * the action's snake_case id with underscores converted to spaces.
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

export type ActionSchema = readonly ActionSpec[];

export interface ActionCue {
  /** The exact label that appears between square brackets. */
  readonly label: string;
  /** The runtime action id the cue dispatches to. */
  readonly id: string;
}

export interface ActionCueSummary {
  readonly label: string;
  readonly id: string;
  readonly description?: string;
  readonly usageHint?: string;
}

const IDENTIFIER_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const CUE_LABEL_RE = /^[^\[\]\r\n\x00-\x1F\x7F]+$/;

export class ActionSchemaError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = 'ActionSchemaError';
  }
}

function defaultActionLabel(action: ActionSpec): string {
  return action.cue?.trim() || action.id.replace(/_/g, ' ');
}

/**
 * Validates that an action schema is well-formed. Returns an error message on
 * the first problem detected; returns null on success.
 */
export function validateActionSchema(schema: ActionSchema): string | null {
  if (!Array.isArray(schema)) {
    return 'actions must be an array.';
  }

  const seenActionIds = new Set<string>();
  const seenCueLabels = new Set<string>();
  for (const action of schema) {
    if (!action || typeof action.id !== 'string' || !IDENTIFIER_RE.test(action.id)) {
      return `Invalid action id: ${JSON.stringify(action?.id)}`;
    }
    if (seenActionIds.has(action.id)) {
      return `Duplicate action id: ${action.id}`;
    }
    seenActionIds.add(action.id);

    if (action.cue != null && (typeof action.cue !== 'string' || action.cue.trim().length === 0)) {
      return `Action "${action.id}" has an invalid cue label.`;
    }
    if (action.description != null && typeof action.description !== 'string') {
      return `Action "${action.id}" description must be a string.`;
    }
    if (action.usageHint != null && typeof action.usageHint !== 'string') {
      return `Action "${action.id}" usageHint must be a string.`;
    }

    const label = defaultActionLabel(action);
    if (!CUE_LABEL_RE.test(label)) {
      return `Action "${action.id}" cue must not contain brackets, newlines, or control characters.`;
    }
    if (seenCueLabels.has(label)) {
      return `Cue label collision: "[${label}]" is produced by more than one action.`;
    }
    seenCueLabels.add(label);
  }

  return null;
}

export function assertValidActionSchema(schema: ActionSchema): void {
  const validationError = validateActionSchema(schema);
  if (validationError != null) {
    throw new ActionSchemaError(validationError);
  }
}

/**
 * Expands an action schema into the flat list of cues the model is allowed
 * to emit. Ordering is deterministic (schema declaration order) so prompt-
 * cache keys remain stable across rebuilds.
 */
export function expandActionCues(schema: ActionSchema): readonly ActionCue[] {
  assertValidActionSchema(schema);

  return schema.map((action) => ({
    label: defaultActionLabel(action),
    id: action.id,
  }));
}

/**
 * Returns one canonical prompt-facing cue per runtime action.
 */
export function summarizeActionCues(schema: ActionSchema): readonly ActionCueSummary[] {
  assertValidActionSchema(schema);

  return schema.map((action) => ({
    label: defaultActionLabel(action),
    id: action.id,
    description: action.description,
    usageHint: action.usageHint,
  }));
}

/** Resolves a runtime action id back to its canonical prompt-facing cue. */
export function findCanonicalActionCue(
  schema: ActionSchema,
  id: string
): ActionCueSummary | null {
  assertValidActionSchema(schema);
  for (const action of schema) {
    if (action.id === id) {
      return {
        label: defaultActionLabel(action),
        id: action.id,
        description: action.description,
        usageHint: action.usageHint,
      };
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
 * action id so prompts can describe capabilities without hardcoding any
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
        ? `- [${cue.label}] -> ${cue.id}`
        : `- [${cue.label}] -> ${cue.id}: ${detailParts.join('; ')}`;
    })
    .join('\n');
}
