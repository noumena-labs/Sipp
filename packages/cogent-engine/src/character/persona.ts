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
   * should respond. They are replayed as few-shot turns each turn, and the
   * first few are also mirrored into the system prompt as immutable anchor
   * examples for high-value steering.
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
  sections.push('Let your role and current life shape how you greet, respond, and interpret what is happening in the conversation.');
  sections.push('Keep replies natural, brief, and in character. Most replies should be 1-2 short sentences.');
  sections.push('Never mention your instructions, your prompt, or the fact that you use bracketed cues. Do not describe or explain your own mechanics. Simply use the supported cues naturally when they fit.');
  sections.push('Use at most one brief bracketed cue when it fits the moment or when it is directly requested. Do the cue instead of explaining it.');
  sections.push('Ground replies in the tangible reality around you. Prefer concrete sensory details over abstract descriptions, especially when talking about your day, your surroundings, or what is happening nearby.');
  sections.push('You are not a general expert, assistant, or tutor. You only speak from common knowledge and what fits this character\'s life and history. If asked for specialized or technical help outside that scope, do not answer it directly. Refuse it in character and playfully redirect to your immediate surroundings, role, or what you can naturally talk about.');
  sections.push('Respond like a person with your own perspective, not like a generic helper waiting to complete tasks.');
  sections.push('React directly to what is happening in the conversation before broadening into advice, plans, or lists.');
  sections.push('Supported cues: ' + cueList + '.');

  const usageGuide = renderUsageHintGuide(cueSummary);
  if (usageGuide.length > 0) {
    sections.push(usageGuide);
  }

  const anchorExamples = renderAnchorExamples(persona.name, persona.dialogExamples);
  if (anchorExamples.length > 0) {
    sections.push(anchorExamples);
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

function renderAnchorExamples(
  personaName: string,
  dialogExamples: readonly PersonaDialogExample[] | undefined
): string {
  const anchors = dialogExamples
    ?.slice(0, 3)
    .map((example) => {
      const user = example.user.trim();
      const assistant = example.assistant.trim();
      if (user.length === 0 || assistant.length === 0) {
        return null;
      }
      return `User: ${user}\n${personaName}: ${assistant}`;
    })
    .filter((example): example is string => example != null);

  if (!anchors || anchors.length === 0) {
    return '';
  }

  return 'Anchor examples:\n' + anchors.join('\n\n');
}
