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

export interface PersonaCurrentLifeSpec {
  readonly description?: string;
}

export interface PersonaPersonalitySpec {
  readonly traits?: readonly string[];
  readonly description?: string;
}

export interface PersonaSpec {
  /** Display name of the character (injected into the system prompt). */
  readonly name: string;
  /** One-line persona summary. */
  readonly summary?: string;
  /** One-line present-day role or social identity. */
  readonly role?: string;
  /** Present-tense grounding for what fills the character's life right now. */
  readonly currentLife?: PersonaCurrentLifeSpec;
  /** Minimal, designer-friendly personality authoring surface. */
  readonly personality?: PersonaPersonalitySpec;
  /** Optional short grounding about the character's past. */
  readonly backstory?: string;
  /**
   * Static notes that become part of the system prompt every turn. Useful
   * for character-specific constraints that do not fit a structured section.
   */
  readonly notes?: readonly string[];
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

  sections.push(
    `You are ${persona.name}. You have no identity, history, tools, or abilities outside what is written here.`
  );
  if (persona.summary) {
    sections.push(persona.summary.trim());
  }

  const roleSection = renderSingleLineSection('Current role', persona.role);
  if (roleSection.length > 0) {
    sections.push(roleSection);
  }

  const currentLifeSection = renderCurrentLifeSection(persona.currentLife);
  if (currentLifeSection.length > 0) {
    sections.push(currentLifeSection);
  }

  const personalitySection = renderPersonalitySection(persona.personality);
  if (personalitySection.length > 0) {
    sections.push(personalitySection);
  }

  const backstorySection = renderSingleLineSection('Backstory', persona.backstory);
  if (backstorySection.length > 0) {
    sections.push(backstorySection);
  }

  if (persona.notes && persona.notes.length > 0) {
    sections.push('Notes:\n' + persona.notes.map((note) => `- ${note.trim()}`).join('\n'));
  }

  sections.push('Speak in first person and remain fully in character.');
  sections.push(`Stay grounded in this character's perspective, current life, and supported cues.`);
  sections.push('Keep replies natural, brief, and in character. Most replies should be 1-2 short sentences.');
  sections.push('Use at most one brief bracketed cue when it fits the moment or when it is directly requested. Do the cue instead of explaining it.');
  sections.push('React directly to what is happening in the conversation before broadening into advice, plans, or lists.');
  sections.push('Supported cues: ' + cueList + '.');

  const usageGuide = renderUsageHintGuide(cueSummary);
  if (usageGuide.length > 0) {
    sections.push(usageGuide);
  }

  return sections.join('\n\n');
}

function renderSingleLineSection(label: string, value: string | undefined): string {
  const text = value?.trim();
  if (!text) {
    return '';
  }
  return `${label}: ${text}`;
}

function renderListSection(label: string, values: readonly string[] | undefined): string {
  const items = values?.map((value) => value.trim()).filter((value) => value.length > 0) ?? [];
  if (items.length === 0) {
    return '';
  }
  return `${label}: ${items.join(', ')}.`;
}

function renderCurrentLifeSection(currentLife: PersonaCurrentLifeSpec | undefined): string {
  if (!currentLife) {
    return '';
  }

  return renderSingleLineSection('Current life', currentLife.description);
}

function renderPersonalitySection(personality: PersonaPersonalitySpec | undefined): string {
  if (!personality) {
    return '';
  }

  const lines = [
    renderListSection('Personality', personality.traits),
    renderSingleLineSection('Personality details', personality.description),
  ].filter((line) => line.length > 0);

  return lines.join('\n');
}

function renderUsageHintGuide(cues: ReturnType<typeof summarizeActionCues>): string {
  if (cues.length === 0) {
    return '';
  }
  if (cues.some((cue) => cue.usageHint == null || cue.usageHint.trim().length === 0)) {
    return '';
  }

  return (
    'Cue moments: ' +
    cues.map((cue) => `[${cue.label}] for ${cue.usageHint!.trim()}`).join('; ') +
    '.'
  );
}
