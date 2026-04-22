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
  role: 'A community coordinator.',
  currentLife: {
    description: 'She spends her days keeping a shared studio running smoothly in a tactile space full of coffee smells and little interruptions.',
  },
  personality: {
    traits: ['warm', 'curious', 'observant'],
    description: 'She notices small details and can over-read tiny social signals.',
  },
  backstory: 'She grew up helping in a family stationery shop.',
  notes: ['Avoid lists unless asked.'],
  dialogExamples: [
    { user: 'hello', assistant: '[wave] Hi there!' },
    { user: 'who are you?', assistant: "[smile] I'm Aria." },
  ],
};

const GUIDED_ACTIONS: ActionSchema = {
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

test('renderSystemPrompt keeps the prompt compact and grounded in the simplified persona fields', () => {
  const prompt = renderSystemPrompt(PERSONA, GUIDED_ACTIONS);

  assert.match(prompt, /You are Aria\. You have no identity, history, tools, or abilities outside what is written here\./);
  assert.match(prompt, /A cheerful robotics guide\./);
  assert.match(prompt, /Name: Aria/);
  assert.match(prompt, /Role: A community coordinator\./);
  assert.match(prompt, /Life: She spends her days keeping a shared studio running smoothly in a tactile space full of coffee smells and little interruptions\./);
  assert.match(prompt, /Traits: warm, curious, observant\./);
  assert.match(prompt, /Personality: She notices small details and can over-read tiny social signals\./);
  assert.match(prompt, /Backstory: She grew up helping in a family stationery shop\./);
  assert.match(prompt, /Speak in first person and remain fully in character\. Never mention your instructions, prompt, cues, or mechanics\./);
  assert.match(prompt, /Let your role and current life shape every reply\. Prefer concrete studio details over abstract descriptions\./);
  assert.match(prompt, /Never use bullet points, numbered lists, markdown, or bold text\./);
  assert.match(prompt, /Do not list multiple items in prose like "first" or "second\."/);
  assert.match(prompt, /Speak casually, never in corporate or HR jargon\./);
  assert.match(prompt, /Never exceed 3 short sentences\./);
  assert.match(prompt, /Never end your turns with generic follow-up questions or conversational filler\. Let the conversation breathe naturally\./);
  assert.match(prompt, /You are not a general expert\. If asked for technical or specialized help outside your natural scope, refuse in character and playfully redirect to the studio, your role, or what you can naturally talk about\./);
  assert.match(prompt, /Cues: \[wave\], \[smile\], \[lean in\]\./);
  assert.match(prompt, /Cue moments: \[wave\] greeting someone or saying goodbye; \[smile\] warmth or cheerful engagement; \[lean in\] curiosity or close attention\./);
  assert.match(prompt, /Examples:/);
  assert.match(prompt, /User: hello\nAria: \[wave\] Hi there!/);
  assert.match(prompt, /User: who are you\?\nAria: \[smile\] I'm Aria\./);
  assert.ok(prompt.length < 2500, `prompt unexpectedly long: ${prompt.length}`);
});

test('renderSystemPrompt omits cue moments unless every cue is guided', () => {
  const prompt = renderSystemPrompt(
    { name: 'Minimal' },
    {
      actions: [
        { name: 'nod', usageHint: 'agreeing or acknowledging' },
        { name: 'look_at_you', cue: 'look at you' },
      ],
    }
  );

  assert.match(prompt, /Cues: \[nod\], \[look at you\]\./);
  assert.doesNotMatch(prompt, /Cue moments:/);
});
