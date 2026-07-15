import assert from 'node:assert/strict';
import test from 'node:test';

import {
  formatRequestStats,
  toChatMessages,
  type ConversationMessage,
} from './src/chat-state.ts';
import {
  getCuratedModel,
  projectorRequirementMessage,
  resolveModelSelection,
} from './src/model-registry.ts';

test('text model selection never includes a projector', () => {
  const resolved = resolveModelSelection({
    kind: 'curated',
    modelId: 'qwen2.5-0.5b-instruct',
  });

  assert.equal(resolved.capability, 'text');
  assert.equal(resolved.source.kind, 'remote');
  assert.equal(resolved.source.modelUrls.length, 1);
});

test('curated vision selection owns its hidden projector source', () => {
  const resolved = resolveModelSelection({
    kind: 'curated',
    modelId: 'lfm2.5-vl-450m',
  });

  assert.equal(resolved.capability, 'vision');
  assert.equal(resolved.source.kind, 'remote');
  assert.match(resolved.source.projectorUrl ?? '', /mmproj-LFM2\.5-VL-450m-F16\.gguf$/);
});

test('custom URL selection remains model-only after curated vision selection', () => {
  const vision = getCuratedModel('lfm2.5-vl-450m');
  assert.equal(vision.source.kind, 'remote');

  const custom = resolveModelSelection({
    kind: 'custom-url',
    url: 'https://models.example.test/custom.gguf',
  });

  assert.equal(custom.capability, 'text');
  assert.deepEqual(custom.source, {
    kind: 'remote',
    modelUrls: ['https://models.example.test/custom.gguf'],
  });
  assert.equal(custom.custom, true);
});

test('custom file selection remains model-only', () => {
  const file = new File(['gguf'], 'local-model.gguf');
  const resolved = resolveModelSelection({ kind: 'custom-file', file });

  assert.deepEqual(resolved.source, { kind: 'local', modelFiles: [file] });
  assert.equal(resolved.capability, 'text');
});

test('custom vision imports receive curated guidance', () => {
  const resolved = resolveModelSelection({
    kind: 'custom-url',
    url: 'https://models.example.test/custom-vision.gguf',
  });

  assert.match(
    projectorRequirementMessage(resolved),
    /choose a curated vision model/i
  );
});

test('chat serialization preserves complete history and excludes pending output', () => {
  const messages: ConversationMessage[] = [
    {
      id: '1',
      role: 'user',
      text: 'First question',
      status: 'complete',
    },
    {
      id: '2',
      role: 'assistant',
      text: 'First answer',
      status: 'complete',
    },
    {
      id: '3',
      role: 'user',
      text: 'Follow-up',
      status: 'complete',
    },
    {
      id: '4',
      role: 'assistant',
      text: 'Partial answer',
      status: 'streaming',
    },
  ];

  assert.deepEqual(toChatMessages(messages), [
    { role: 'user', content: 'First question' },
    { role: 'assistant', content: 'First answer' },
    { role: 'user', content: 'Follow-up' },
  ]);
});

test('request metrics remain compact', () => {
  const text = formatRequestStats({
    inputTokens: 12,
    outputTokens: 24,
    cacheMode: 'live_slot_prefix',
    cacheSource: 'live',
    cacheHits: 8,
    prefillTokens: 4,
    ttftMs: 123.6,
    interTokenMs: 20,
    e2eMs: 700,
    decodeTokensPerSecond: 48.25,
    e2eTokensPerSecond: 34.2,
    prefillTokensPerSecond: 100,
    prefillMs: 40,
    decodeMs: 500,
  });

  assert.equal(text, '48.3 tok/s | 124 ms TTFT | 24 tokens');
});
