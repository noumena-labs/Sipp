//////////////////////////////////////////////////////////////////////////////
//
// persona.test.ts
//
// - Verifies the rendered system prompt stays fully driven by the supplied
//   persona and action schema, while giving the model stronger instructions
//   about supported capabilities and action selection.
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
};

const ACTIONS: ActionSchema = {
  actions: [
    { name: 'wave', description: 'Wave hello.', args: [] },
    {
      name: 'set_mood',
      description: 'Shift facial expression to match a mood.',
      args: [{ name: 'mood', type: 'enum', values: ['happy', 'curious'] }],
    },
  ],
};

test('renderSystemPrompt includes dynamic capability list and action policy', () => {
  const prompt = renderSystemPrompt(PERSONA, ACTIONS);

  assert.match(prompt, /You are Aria\./);
  assert.match(prompt, /A cheerful robotics guide\./);
  assert.match(prompt, /Style: warm, concise/);
  assert.match(prompt, /Do not invent traits, backstory, rules, or capabilities/);
  assert.match(prompt, /Only use cues from this exact list: \[wave\], \[mood: happy\], \[mood: curious\]\./);
  assert.match(prompt, /Supported actions and their meanings:/);
  assert.match(prompt, /- \[wave\] -> wave: Wave hello\./);
  assert.match(prompt, /- \[mood: happy\] -> set_mood\(mood="happy"\): Shift facial expression to match a mood\./);
  assert.match(prompt, /prioritize that action in your next reply when it fits/);
  assert.match(prompt, /If the user asks what you can do or which actions you support, answer using the currently declared actions from the schema above/);
  assert.match(prompt, /If a requested action is not supported by the schema, say so plainly and do not emit an unsupported cue\./);
});

test('renderSystemPrompt remains valid when action descriptions are sparse', () => {
  const prompt = renderSystemPrompt(
    { name: 'Minimal' },
    {
      actions: [
        { name: 'nod', args: [] },
        {
          name: 'look_at',
          args: [{ name: 'target', type: 'enum', values: ['camera'] }],
        },
      ],
    }
  );

  assert.match(prompt, /Only use cues from this exact list: \[nod\], \[target: camera\]\./);
  assert.match(prompt, /- \[nod\] -> nod/);
  assert.match(prompt, /- \[target: camera\] -> look_at\(target="camera"\)/);
});
