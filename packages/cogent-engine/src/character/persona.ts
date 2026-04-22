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

  sections.push(`You are ${persona.name}.`);
  if (persona.summary) {
    sections.push(persona.summary.trim());
  }
  if (persona.description) {
    sections.push(persona.description.trim());
  }
  if (persona.style) {
    sections.push(`Style: ${persona.style.trim()}`);
  }
  if (persona.notes && persona.notes.length > 0) {
    sections.push('Notes:\n' + persona.notes.map((note) => `- ${note.trim()}`).join('\n'));
  }

  sections.push(
    `Your only name is ${persona.name}; you have no last name, alternate identity, or other persona. Speak in first person and stay fully in character.`
  );
  sections.push(
    `Your capabilities are exactly the persona and supported cues below. Stay inside that scope and do not invent other identities, developers, training history, or unsupported abilities.`
  );
  sections.push(
    'Reply style: brief, natural, and in character. Most replies should be 1-2 short sentences. Avoid numbered lists, headings, canned assistant phrasing, and long explanations unless the user asks for them.'
  );
  sections.push(
    'Embodied cues: use at most one short bracketed cue when it naturally fits the moment. If a supported cue is directly requested, do it briefly instead of explaining it. Do not explain the cue system unless the user asks.'
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
  ).slice(0, 4);

  if (hints.length === 0) {
    return '';
  }

  return (
    'Use cues naturally in social moments: ' +
    hints.map(([label, hint]) => `[${label}] for ${hint}`).join('; ') +
    '.'
  );
}
