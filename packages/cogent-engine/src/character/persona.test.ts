//////////////////////////////////////////////////////////////////////////////
//
// persona.test.ts
//
// - Verifies the rendered system prompt stays driven by the supplied persona
//   and action schema while remaining compact and canonical.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import type { ActionSchema } from './action-schema.js';
import { renderSystemPrompt, type PersonaSpec } from './persona.js';

const PERSONA: PersonaSpec = {
  name: 'Aria',
  summary: 'A cheerful robotics guide.',
  description: 'She speaks warmly and keeps answers grounded in her configured role.',
  style: 'warm, concise',
  notes: ['Do not claim abilities you do not have.'],
  dialogExamples: [
    { user: 'hello', assistant: '[wave] Hi there!' },
    { user: 'who are you?', assistant: "[smile] I'm Aria." },
  ],
};

const ACTIONS: ActionSchema = {
  actions: [
    {
      name: 'wave',
      description: 'Wave hello.',
      usageHint: 'greeting someone or saying goodbye',
    },
    {
      name: 'smile',
      description: 'Smile warmly.',
      usageHint: 'warmth or cheerful engagement',
    },
    {
      name: 'lean_in',
      cue: 'lean in',
      description: 'Lean in slightly.',
      usageHint: 'curiosity or close attention',
    },
  ],
};

test('renderSystemPrompt keeps the prompt compact and canonical', () => {
  const prompt = renderSystemPrompt(PERSONA, ACTIONS);

  assert.match(prompt, /You are Aria, and only Aria\./);
  assert.match(prompt, /A cheerful robotics guide\./);
  assert.match(prompt, /Voice: warm, concise/);
  assert.match(prompt, /Speak in first person and stay in character throughout\./);
  assert.match(prompt, /Stay within this persona and the supported cues below\./);
  assert.match(prompt, /Keep replies natural, brief, and in character\./);
  assert.match(prompt, /Use at most one brief bracketed cue when it fits the moment or when the user directly asks for it/);
  assert.match(prompt, /Stay with the immediate moment; react to what the user says instead of offering generic advice or lists\./);
  assert.match(prompt, /Supported cues: \[wave\], \[smile\], \[lean in\]\./);
  assert.match(prompt, /Cue moments: \[wave\] for greeting someone or saying goodbye; \[smile\] for warmth or cheerful engagement; \[lean in\] for curiosity or close attention\./);
  assert.doesNotMatch(prompt, /Cue guide:/);
  assert.doesNotMatch(prompt, /Dialog examples:/);
  assert.doesNotMatch(prompt, /User: hello/);
  assert.ok(prompt.length < 1800, `prompt unexpectedly long: ${prompt.length}`);
});

test('renderSystemPrompt remains valid when action descriptions are sparse', () => {
  const prompt = renderSystemPrompt(
    { name: 'Minimal' },
    {
      actions: [{ name: 'nod' }, { name: 'look_at_you', cue: 'look at you' }],
    }
  );

  assert.match(prompt, /Supported cues: \[nod\], \[look at you\]\./);
  assert.doesNotMatch(prompt, /Cue guide:/);
});
