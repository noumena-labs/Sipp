//////////////////////////////////////////////////////////////////////////////
//
// persona.ts
//
// - Turns a CharacterConfig persona section plus the action schema into a
//   final system prompt that the engine can feed into the model.
//
// - Kept deliberately declarative so the same persona text works across
//   multiple chat templates (the engine applies the model-specific chat
//   template on top; this file only concerns itself with the payload).
//
//////////////////////////////////////////////////////////////////////////////

import type { ActionSchema } from './action-schema.js';
import { renderActionCueList, summarizeActionCues } from './action-schema.js';

export interface PersonaDialogExample {
  readonly user: string;
  readonly assistant: string;
}

export interface PersonaSpec {
  /** Display name of the character (injected into the system prompt). */
  readonly name: string;
  /** One-line persona summary (e.g. "A cheerful robotics assistant."). */
  readonly summary?: string;
  /**
   * Longer, freeform persona description. May include multiple paragraphs.
   * Rendered verbatim into the system prompt after the summary line.
   */
  readonly description?: string;
  /**
   * Static notes that become part of the system prompt every turn. Useful
   * for world-state, safety rules, or authoring-time instructions.
   */
  readonly notes?: readonly string[];
  /**
   * Optional style guidelines (tone, pacing, formatting preferences).
   */
  readonly style?: string;
  /**
   * Optional few-shot examples that demonstrate how the configured character
   * should respond. These are prompt examples only; they are not replayed as
   * conversation history.
   */
  readonly dialogExamples?: readonly PersonaDialogExample[];
}

/**
 * Renders the persona + action schema into a single system prompt string.
 * The prompt is deterministic given the same inputs, which lets the runtime
 * key the prefix KV cache on the character id and reuse it across turns.
 */
export function renderSystemPrompt(persona: PersonaSpec, schema: ActionSchema): string {
  const sections: string[] = [];
  const cueSummary = summarizeActionCues(schema);
  const cueList = renderActionCueList(schema);

  sections.push(`You are ${persona.name}, and only ${persona.name}.`);
  if (persona.summary) {
    sections.push(persona.summary.trim());
  }
  if (persona.description) {
    sections.push(persona.description.trim());
  }
  if (persona.notes && persona.notes.length > 0) {
    sections.push('Notes:\n' + persona.notes.map((note) => `- ${note.trim()}`).join('\n'));
  }
  if (persona.style) {
    sections.push(`Voice: ${persona.style.trim()}`);
  }

  sections.push(
    'Speak in first person and stay in character throughout.'
  );
  sections.push(
    `Stay within this persona and the supported cues below. Do not invent other identities, training, tools, or abilities.`
  );
  sections.push(
    'Keep replies natural, brief, and in character. Most replies should be 1-2 short sentences.'
  );
  sections.push(
    'Use at most one brief bracketed cue when it fits the moment or when the user directly asks for it; do the cue instead of explaining it.'
  );
  sections.push(
    'Stay with the immediate moment; react to what the user says instead of offering generic advice or lists.'
  );
  sections.push(
    'Supported cues: ' + cueList + '.'
  );
  const usageGuide = renderUsageHintGuide(cueSummary);
  if (usageGuide.length > 0) {
    sections.push(usageGuide);
  }

  return sections.join('\n\n');
}

function renderUsageHintGuide(cues: ReturnType<typeof summarizeActionCues>): string {
  const hints = Array.from(
    new Map(
      cues
        .filter((cue) => cue.usageHint != null && cue.usageHint.trim().length > 0)
        .map((cue) => [cue.label, cue.usageHint!.trim()])
    )
  ).slice(0, cues.length);

  if (hints.length === 0) {
    return '';
  }

  return (
    'Cue moments: ' +
    hints.map(([label, hint]) => `[${label}] for ${hint}`).join('; ') +
    '.'
  );
}
