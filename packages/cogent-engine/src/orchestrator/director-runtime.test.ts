import assert from 'node:assert/strict';
import test from 'node:test';

import type { CharacterAgentEngine } from '../character/character-agent.js';
import type {
  GenerateRequestId,
  GenerateResponse,
  PromptOptions,
} from '../core/inference-types.js';
import { DirectorRuntime } from './director-runtime.js';
import { parseDirectorConfig } from './director-config.js';

class FakeEngine implements CharacterAgentEngine {
  public outputText = '';
  public fail = false;
  public grammar: string | undefined;

  public async applyChatTemplate(
    messages: Array<{ role: string; content: string }>,
    _addAssistant: boolean
  ): Promise<string> {
    return messages.map((message) => `${message.role}: ${message.content}`).join('\n');
  }

  public async queuePrompt(
    _contextKey: string,
    _prompt: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestId> {
    if (typeof options === 'object' && options) {
      this.grammar = options.grammar;
    }
    return 1;
  }

  public async runQueuedRequest(
    requestId: GenerateRequestId,
    _options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    return {
      requestId,
      completed: true,
      failed: this.fail,
      cancelled: false,
      outputText: this.outputText,
      ...(this.fail ? { errorMessage: 'boom' } : {}),
    };
  }

  public async cancelQueuedRequest(_requestId: GenerateRequestId): Promise<boolean> {
    return true;
  }

  public getChatTemplate(): string | null {
    return null;
  }

  public getBosText(): string {
    return '';
  }

  public getEosText(): string {
    return '';
  }
}

const CONFIG = parseDirectorConfig({
  id: 'courtyard-director',
  director: {
    role: 'Scenario director',
    instructions: ['Only use supplied state.'],
  },
  queries: {
    resolve_conflict: {
      response: {
        type: 'object',
        properties: {
          note: { type: 'string', maxLength: 120 },
          winnerAgentId: { type: 'string', nullable: true, maxLength: 32 },
        },
      },
    },
  },
});

test('DirectorRuntime returns parsed validated data', async () => {
  const engine = new FakeEngine();
  engine.outputText = '{"note":"Aria wins","winnerAgentId":"aria"}';
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.query('resolve_conflict', {
    state: { tick: 3 },
    conflict: { contenders: ['aria', 'beck'] },
  });

  assert.deepEqual(result.data, { note: 'Aria wins', winnerAgentId: 'aria' });
  assert.equal(result.cancelled, false);
  assert.ok(engine.grammar?.includes('winnerAgentId'));
});

test('DirectorRuntime surfaces validation failure as null data', async () => {
  const engine = new FakeEngine();
  engine.outputText = '{"note":"ok"}';
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.query('resolve_conflict', { state: { tick: 1 } });

  assert.equal(result.data, null);
  assert.match(result.errorMessage ?? '', /winnerAgentId is required/);
});

test('DirectorRuntime surfaces engine failure', async () => {
  const engine = new FakeEngine();
  engine.fail = true;
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.query('resolve_conflict', { state: { tick: 1 } });

  assert.equal(result.data, null);
  assert.equal(result.errorMessage, 'boom');
});
