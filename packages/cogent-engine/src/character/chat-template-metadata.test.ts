import assert from 'node:assert/strict';
import test from 'node:test';

import type { ChatTemplateMessage } from '../wasm/wasm-bridge.js';
import {
  buildAppliedChatTemplateContext,
  probeChatTemplateBoundaryInfo,
} from './chat-template-metadata.js';

function createProvider(formatMessage: (messages: ChatTemplateMessage[], addAssistant: boolean) => string) {
  const calls: Array<{ messages: ChatTemplateMessage[]; addAssistant: boolean }> = [];
  return {
    calls,
    applyChatTemplate(messages: ChatTemplateMessage[], addAssistant: boolean): Promise<string> {
      calls.push({ messages, addAssistant });
      return Promise.resolve(formatMessage(messages, addAssistant));
    },
    getChatTemplate(): string {
      return 'fake-template';
    },
    getEosText(): string {
      return '</s>';
    },
  };
}

test('buildAppliedChatTemplateContext uses applied template output as prompt text', async () => {
  const provider = createProvider((messages, addAssistant) => {
    const parts = messages
      .map((message) => `<${message.role}>\n${message.content}</${message.role}>\n`)
      .join('');
    return `${parts}${addAssistant ? '<assistant>\n' : ''}`;
  });

  const context = await buildAppliedChatTemplateContext(provider, [
    { role: 'system', content: 'You are Aria.' },
    { role: 'user', content: 'Hello' },
  ]);

  assert.equal(
    context.promptText,
    '<system>\nYou are Aria.</system>\n<user>\nHello</user>\n<assistant>\n'
  );
  assert.deepEqual(context.boundaryMarkers, ['</assistant>\n', '<system>\n', '<user>\n', '<assistant>\n', '</s>']);
  assert.equal(context.templateSource, 'fake-template');
});

test('buildAppliedChatTemplateContext derives assistant end marker from full-history template', async () => {
  const provider = createProvider((messages, addAssistant) => {
    const turns = messages
      .map((message) => `<<${message.role}>>\n${message.content}<</${message.role}>>\n`)
      .join('');
    if (addAssistant) {
      return `${turns}<<assistant>>\n`;
    }
    return turns;
  });

  const context = await buildAppliedChatTemplateContext(provider, [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'hi' },
  ]);

  assert.ok(context.boundaryMarkers.includes('<<assistant>>\n'));
  assert.ok(context.boundaryMarkers.includes('<<user>>\n'));
});

test('buildAppliedChatTemplateContext accepts cached boundary metadata to avoid reprobes', async () => {
  const provider = createProvider((messages, addAssistant) => {
    const parts = messages
      .map((message) => `<${message.role}>\n${message.content}</${message.role}>\n`)
      .join('');
    return `${parts}${addAssistant ? '<assistant>\n' : ''}`;
  });

  const boundaryInfo = await probeChatTemplateBoundaryInfo(provider);
  assert.equal(provider.calls.length, 5);

  const context = await buildAppliedChatTemplateContext(
    provider,
    [
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'hi' },
    ],
    boundaryInfo
  );

  assert.equal(provider.calls.length, 6);
  assert.equal(context.promptText, '<system>\nsys</system>\n<user>\nhi</user>\n<assistant>\n');
});
