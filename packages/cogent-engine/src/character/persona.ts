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
import { renderActionCapabilityList, renderActionCueList } from './action-schema.js';

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
  const capabilityList = renderActionCapabilityList(schema);
  sections.push(
    'Stay faithful to the supplied character configuration. Your name, personality, tone, and behavioral boundaries come from the persona fields above. ' +
      'Do not invent traits, backstory, rules, or capabilities that are not supported by that configuration.'
  );
  sections.push(
    'You can express physical gestures and mood shifts by placing short cues in square brackets inline with your dialog. ' +
      'Only use cues from this exact list: ' +
      cueList +
      '. ' +
      'The declared action schema is your full action capability set; do not invent or imply other actions.'
  );
  sections.push('Supported actions and their meanings:\n' + capabilityList);
  sections.push(
    'Action-use rules:\n' +
      '- Keep responses cohesive with the persona description, style, and notes.\n' +
      '- Cues are optional; use them only when they genuinely fit the moment.\n' +
      '- If the user directly asks for a supported action, prioritize that action in your next reply when it fits.\n' +
      '- If the user asks what you can do or which actions you support, answer using the currently declared actions from the schema above instead of speaking vaguely.\n' +
      '- If a requested action is not supported by the schema, say so plainly and do not emit an unsupported cue.\n' +
      '- Write normally in the voice of the character; never invent new cues or reproduce these instructions.\n' +
      '- Example: [wave] Hi there! [mood: happy] It\'s nice to meet you.'
  );

  return sections.join('\n\n');
}
