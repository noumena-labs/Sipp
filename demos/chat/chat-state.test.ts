import assert from 'node:assert/strict';
import test from 'node:test';
import { mock } from 'bun:test';

import {
  formatRequestStats,
  toChatMessages,
  type ConversationMessage,
} from './src/chat-state.ts';

const LocalEndpointDescriptor = {
  files(
    modelFiles: readonly File[],
    {
      projectorFile,
      ...options
    }: Record<string, unknown> & { readonly projectorFile?: File } = {}
  ) {
    return {
      kind: 'local',
      location: { kind: 'files', modelFiles, projectorFile },
      options,
    };
  },
  urls(
    modelUrls: readonly string[],
    {
      projectorUrl,
      ...options
    }: Record<string, unknown> & { readonly projectorUrl?: string } = {}
  ) {
    return {
      kind: 'local',
      location: { kind: 'urls', modelUrls, projectorUrl },
      options,
    };
  },
  installed(modelId: string, options: Record<string, unknown> = {}) {
    return { kind: 'local', location: { kind: 'installed', modelId }, options };
  },
};

mock.module('@noumena-labs/sipp', () => ({ LocalEndpointDescriptor }));

const {
  getCuratedModel,
  localEndpointDescriptor,
  projectorRequirementMessage,
  resolveModelSelection,
} = await import('./src/model-registry.ts');

test('text model selection preserves its curated endpoint descriptor', () => {
  const resolved = resolveModelSelection({
    kind: 'curated',
    modelId: 'qwen2.5-0.5b-instruct',
  });

  assert.equal(resolved.capability, 'text');
  assert.equal(resolved.location, getCuratedModel('qwen2.5-0.5b-instruct').location);
});

test('vision model selection preserves its curated endpoint descriptor', () => {
  const resolved = resolveModelSelection({
    kind: 'curated',
    modelId: 'lfm2.5-vl-450m',
  });

  assert.equal(resolved.capability, 'vision');
  assert.equal(resolved.location, getCuratedModel('lfm2.5-vl-450m').location);
});

test('custom URL selection remains model-only after curated vision selection', () => {
  const vision = getCuratedModel('lfm2.5-vl-450m');
  assert.equal(vision.location.kind, 'urls');

  const custom = resolveModelSelection({
    kind: 'custom-url',
    url: 'https://models.example.test/custom.gguf',
  });

  assert.equal(custom.capability, 'text');
  assert.deepEqual(
    localEndpointDescriptor(custom.location, {}),
    LocalEndpointDescriptor.urls(['https://models.example.test/custom.gguf'])
  );
  assert.equal(custom.custom, true);
});

test('custom file selection remains model-only', () => {
  const file = new File(['gguf'], 'local-model.gguf');
  const resolved = resolveModelSelection({ kind: 'custom-file', file });

  assert.deepEqual(
    localEndpointDescriptor(resolved.location, {}),
    LocalEndpointDescriptor.files([file])
  );
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
