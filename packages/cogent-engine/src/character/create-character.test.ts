import assert from 'node:assert/strict';
import test from 'node:test';

import { ActionBus } from './action-bus.js';
import type { CharacterAgentEngine } from './character-agent.js';
import { createCharacterFromConfigUrl } from './create-character.js';

function createEngineStub(): CharacterAgentEngine {
  return {
    async queuePrompt() {
      return 1;
    },
    async runQueuedRequest() {
      return {
        requestId: 1,
        completed: true,
        failed: false,
        cancelled: false,
        outputText: '',
      };
    },
    async applyChatTemplate() {
      return '';
    },
  };
}

test('createCharacterFromConfigUrl fetches, parses, and builds an agent', async () => {
  const bus = new ActionBus();
  const fetchCalls: Array<{ url: string; signal?: AbortSignal }> = [];
  const engine = createEngineStub();

  const result = await createCharacterFromConfigUrl({
    configUrl: '/characters/aria/character.json',
    engine,
    bus,
    fetch: async (url, init) => {
      fetchCalls.push({ url: String(url), signal: init?.signal ?? undefined });
      return {
        ok: true,
        status: 200,
        async json() {
          return {
            id: 'aria',
            persona: { name: 'Aria' },
            actions: { actions: [{ name: 'wave' }] },
          };
        },
      } as Response;
    },
  });

  assert.equal(fetchCalls.length, 1);
  assert.equal(fetchCalls[0].url, '/characters/aria/character.json');
  assert.equal(result.config.id, 'aria');
  assert.equal(result.agent.bus, bus);
});

test('createCharacterFromConfigUrl surfaces HTTP errors', async () => {
  await assert.rejects(
    () =>
      createCharacterFromConfigUrl({
        configUrl: '/missing.json',
        engine: createEngineStub(),
        fetch: async () =>
          ({
            ok: false,
            status: 404,
            async json() {
              return null;
            },
          }) as Response,
      }),
    /character\.json HTTP 404/
  );
});
