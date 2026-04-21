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
import { renderActionCueList } from './action-schema.js';

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
}

/**
 * Renders the persona + action schema into a single system prompt string.
 * The prompt is deterministic given the same inputs, which lets the runtime
 * key the prefix KV cache on the character id and reuse it across turns.
 */
export function renderSystemPrompt(persona: PersonaSpec, schema: ActionSchema): string {
  const sections: string[] = [];

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

//  sections.push(
//     'You can express yourself through both spoken text and embedded action tags. ' +
//       'To trigger a behavior, embed a tag of the form ' +
//       '`<action name="<name>" args={<json>}/>` anywhere in your reply. ' +
//       'Only the listed actions are permitted; the output is grammar-constrained.'

  const cueList = renderActionCueList(schema);
  sections.push(
    'You can express physical gestures and mood shifts by placing short cues in square brackets inline with your dialog. ' +
      'Only use cues from this exact list: ' +
      cueList +
      '. ' +
      'Cues are optional — emit one only when it genuinely fits what you are saying. Write normally in the voice of the character; never invent new cues or reproduce this instruction. ' +
      'Example: [wave] Hi there! [mood: happy] It\u2019s nice to meet you.'
  );

  return sections.join('\n\n');
}
