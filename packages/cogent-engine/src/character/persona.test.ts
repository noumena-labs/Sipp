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
      args: [],
    },
    {
      name: 'set_mood',
      description: 'Shift facial expression to match a mood.',
      usageHint: 'the emotional tone of the reply naturally fits the mood',
      args: [
        {
          name: 'mood',
          type: 'enum',
          values: ['happy', 'curious'],
          cueLabels: { happy: 'smile', curious: 'lean in' },
          cueAliases: { happy: ['mood: happy'] },
        },
      ],
    },
  ],
};

test('renderSystemPrompt keeps the prompt compact and canonical', () => {
  const prompt = renderSystemPrompt(PERSONA, ACTIONS);

  assert.match(prompt, /You are Aria\./);
  assert.match(prompt, /A cheerful robotics guide\./);
  assert.match(prompt, /Style: warm, concise/);
  assert.match(prompt, /Your only name is Aria; you have no last name, alternate identity, or other persona/);
  assert.match(prompt, /Your capabilities are exactly the persona and supported cues below/);
  assert.match(prompt, /Reply style: brief, natural, and in character/);
  assert.match(prompt, /Supported cues: \[wave\], \[smile\], \[lean in\]\./);
  assert.match(prompt, /Use cues naturally in social moments: \[wave\] for greeting someone or saying goodbye; \[smile\] for the emotional tone of the reply naturally fits the mood; \[lean in\] for the emotional tone of the reply naturally fits the mood\./);
  assert.doesNotMatch(prompt, /Cue guide:/);
  assert.doesNotMatch(prompt, /Dialog examples:/);
  assert.doesNotMatch(prompt, /User: hello/);
  assert.ok(prompt.length < 1800, `prompt unexpectedly long: ${prompt.length}`);
});

test('renderSystemPrompt remains valid when action descriptions are sparse', () => {
  const prompt = renderSystemPrompt(
    { name: 'Minimal' },
    {
      actions: [
        { name: 'nod', args: [] },
        {
          name: 'look_at',
          args: [
            {
              name: 'target',
              type: 'enum',
              values: ['camera'],
              cueLabels: { camera: 'look at you' },
              cueAliases: { camera: ['target: camera'] },
            },
          ],
        },
      ],
    }
  );

  assert.match(prompt, /Supported cues: \[nod\], \[look at you\]\./);
  assert.doesNotMatch(prompt, /Cue guide:/);
});
