//////////////////////////////////////////////////////////////////////////////
//
// custom-template.test.ts
//
// - Golden tests for each supported ChatFormat.
// - Verifies that the system turn is rendered (the whole point of this
//   refactor — llama.cpp's native template dropped it).
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildChatPrompt,
  sniffChatFormat,
  type ChatFormat,
} from './custom-template.js';
import type { ChatMessage } from '../core/inference-types.js';

const SAMPLE: ChatMessage[] = [
  { role: 'system', content: 'You are Aria.' },
  { role: 'user', content: 'Hi' },
];

test('chatml renders system + user + assistant priming', () => {
  const out = buildChatPrompt({
    format: 'chatml',
    messages: SAMPLE,
    bosText: '',
  });
  assert.equal(
    out,
    '<|im_start|>system\nYou are Aria.<|im_end|>\n' +
      '<|im_start|>user\nHi<|im_end|>\n' +
      '<|im_start|>assistant\n'
  );
});

test('chatml emits BOS text before first turn', () => {
  const out = buildChatPrompt({
    format: 'chatml',
    messages: SAMPLE,
    bosText: '<s>',
  });
  assert.ok(out.startsWith('<s><|im_start|>system'));
});

test('llama3 renders all three turns with EOT terminators', () => {
  const out = buildChatPrompt({
    format: 'llama3',
    messages: SAMPLE,
    bosText: '<|begin_of_text|>',
  });
  assert.equal(
    out,
    '<|begin_of_text|>' +
      '<|start_header_id|>system<|end_header_id|>\n\nYou are Aria.<|eot_id|>' +
      '<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>' +
      '<|start_header_id|>assistant<|end_header_id|>\n\n'
  );
});

test('llama2 folds system into <<SYS>> block of first user turn', () => {
  const out = buildChatPrompt({
    format: 'llama2',
    messages: SAMPLE,
    bosText: '<s>',
  });
  assert.equal(
    out,
    '<s>[INST] <<SYS>>\nYou are Aria.\n<</SYS>>\n\nHi [/INST]'
  );
});

test('mistral prepends system to first user turn', () => {
  const out = buildChatPrompt({
    format: 'mistral',
    messages: SAMPLE,
    bosText: '<s>',
  });
  assert.equal(out, '<s>[INST] You are Aria.\n\nHi [/INST]');
});

test('gemma merges system into first user turn and uses model role', () => {
  const out = buildChatPrompt({
    format: 'gemma',
    messages: SAMPLE,
    bosText: '<bos>',
  });
  assert.equal(
    out,
    '<bos><start_of_turn>user\nYou are Aria.\n\nHi<end_of_turn>\n' +
      '<start_of_turn>model\n'
  );
});

test('phi3 renders system/user/assistant priming', () => {
  const out = buildChatPrompt({
    format: 'phi3',
    messages: SAMPLE,
    bosText: '',
  });
  assert.equal(
    out,
    '<|system|>\nYou are Aria.<|end|>\n' +
      '<|user|>\nHi<|end|>\n' +
      '<|assistant|>\n'
  );
});

test('addGenerationPrompt: false omits trailing assistant marker (chatml)', () => {
  const out = buildChatPrompt({
    format: 'chatml',
    messages: SAMPLE,
    bosText: '',
    addGenerationPrompt: false,
  });
  assert.ok(!out.endsWith('<|im_start|>assistant\n'));
  assert.ok(out.endsWith('<|im_end|>\n'));
});

test('multi-turn chatml renders every turn in order', () => {
  const out = buildChatPrompt({
    format: 'chatml',
    messages: [
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'u1' },
      { role: 'assistant', content: 'a1' },
      { role: 'user', content: 'u2' },
    ],
    bosText: '',
  });
  const expected =
    '<|im_start|>system\nsys<|im_end|>\n' +
    '<|im_start|>user\nu1<|im_end|>\n' +
    '<|im_start|>assistant\na1<|im_end|>\n' +
    '<|im_start|>user\nu2<|im_end|>\n' +
    '<|im_start|>assistant\n';
  assert.equal(out, expected);
});

// -----------------------------------------------------------------------------
// sniffChatFormat
// -----------------------------------------------------------------------------
const SNIFF_CASES: Array<[ChatFormat, string]> = [
  ['chatml', "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}"],
  [
    'llama3',
    "<|start_header_id|>{{role}}<|end_header_id|>\n\n{{content}}<|eot_id|>",
  ],
  ['gemma', '<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n'],
  ['phi3', '<|user|>\n{{content}}<|end|>\n<|assistant|>\n'],
  ['llama2', '<s>[INST] <<SYS>>\n{{sys}}\n<</SYS>>\n\n{{u}} [/INST]'],
  ['mistral', '<s>[INST] {{u}} [/INST]'],
];

for (const [expected, template] of SNIFF_CASES) {
  test(`sniffChatFormat identifies ${expected}`, () => {
    assert.equal(sniffChatFormat(template), expected);
  });
}

test('sniffChatFormat returns null for unknown template', () => {
  assert.equal(sniffChatFormat('<|totally_made_up|>'), null);
});

test('sniffChatFormat returns null for empty/undefined input', () => {
  assert.equal(sniffChatFormat(null), null);
  assert.equal(sniffChatFormat(''), null);
});
