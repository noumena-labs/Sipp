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

import type { ActionSchema } from '../../src/character/action-schema.js';
import { renderSystemPrompt, type PersonaSpec } from '../../src/character/persona.js';

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
  anchorExamples: [
    { user: 'who are you?', assistant: "[smile] I'm Aria." },
    { user: 'can you code?', assistant: '[shake head] No, that is outside my lane.' },
  ],
  dialogExamples: [
    { user: 'hello', assistant: '[wave] Hi there!' },
    { user: 'long day?', assistant: '[lean in] You look like you have been carrying a lot.' },
  ],
};

const GUIDED_ACTIONS: ActionSchema = [
  {
    id: 'wave',
    description: 'Wave hello.',
    usageHint: 'greeting someone or saying goodbye',
  },
  {
    id: 'smile',
    description: 'Smile warmly.',
    usageHint: 'warmth or cheerful engagement',
  },
  {
    id: 'lean_in',
    cue: 'lean in',
    description: 'Lean in slightly.',
    usageHint: 'curiosity or close attention',
  },
];

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
  assert.match(prompt, /Notes:\n- Avoid lists unless asked\./);
  assert.match(prompt, /Cues: \[wave\], \[smile\], \[lean in\]\./);
  assert.match(prompt, /Cue moments: \[wave\] greeting someone or saying goodbye; \[smile\] warmth or cheerful engagement; \[lean in\] curiosity or close attention\./);
  assert.match(prompt, /Examples:/);
  assert.match(prompt, /User: who are you\?\nAria: \[smile\] I'm Aria\./);
  assert.match(prompt, /User: can you code\?\nAria: \[shake head\] No, that is outside my lane\./);
  assert.doesNotMatch(prompt, /User: hello\nAria: \[wave\] Hi there!/);
  assert.doesNotMatch(prompt, /Never mention your instructions, prompt, cues, or mechanics/);
  assert.doesNotMatch(prompt, /Prefer concrete studio details over abstract descriptions/);
  assert.ok(prompt.length < 2500, `prompt unexpectedly long: ${prompt.length}`);
});

test('renderSystemPrompt omits Examples when anchorExamples are absent', () => {
  const prompt = renderSystemPrompt(
    {
      name: 'Minimal',
      dialogExamples: [{ user: 'hello', assistant: 'Hi there.' }],
    },
    [{ id: 'nod' }]
  );

  assert.doesNotMatch(prompt, /Examples:/);
});

test('renderSystemPrompt includes every configured anchor example', () => {
  const prompt = renderSystemPrompt(
    {
      name: 'Guide',
      anchorExamples: [
        { user: 'one', assistant: 'A.' },
        { user: 'two', assistant: 'B.' },
        { user: 'three', assistant: 'C.' },
        { user: 'four', assistant: 'D.' },
      ],
    },
    [{ id: 'nod' }]
  );

  assert.match(prompt, /User: one\nGuide: A\./);
  assert.match(prompt, /User: two\nGuide: B\./);
  assert.match(prompt, /User: three\nGuide: C\./);
  assert.match(prompt, /User: four\nGuide: D\./);
});

test('renderSystemPrompt omits cue moments unless every cue is guided', () => {
  const prompt = renderSystemPrompt(
    { name: 'Minimal' },
    [
      { id: 'nod', usageHint: 'agreeing or acknowledging' },
      { id: 'look_at_you', cue: 'look at you' },
    ]
  );

  assert.match(prompt, /Cues: \[nod\], \[look at you\]\./);
  assert.doesNotMatch(prompt, /Cue moments:/);
});
