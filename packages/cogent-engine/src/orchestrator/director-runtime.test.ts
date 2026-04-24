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
  public waitForAbort = false;
  public cancelCalls: GenerateRequestId[] = [];

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
    options?: { signal?: AbortSignal }
  ): Promise<GenerateResponse> {
    if (this.waitForAbort) {
      await new Promise<void>((resolve) => {
        const signal = options?.signal;
        if (!signal) {
          return;
        }
        if (signal.aborted) {
          resolve();
          return;
        }
        signal.addEventListener('abort', () => resolve(), { once: true });
      });
      return {
        requestId,
        completed: false,
        failed: false,
        cancelled: true,
        outputText: '',
      };
    }
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
    this.cancelCalls.push(_requestId);
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
  assert.equal(result.status, 'ok');
  assert.ok(engine.grammar?.includes('winnerAgentId'));
});

test('DirectorRuntime surfaces validation failure as null data', async () => {
  const engine = new FakeEngine();
  engine.outputText = '{"note":"ok"}';
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.query('resolve_conflict', { state: { tick: 1 } });

  assert.equal(result.data, null);
  assert.equal(result.status, 'invalid_response');
  assert.match(result.errorMessage ?? '', /winnerAgentId is required/);
});

test('DirectorRuntime surfaces engine failure', async () => {
  const engine = new FakeEngine();
  engine.fail = true;
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.query('resolve_conflict', { state: { tick: 1 } });

  assert.equal(result.data, null);
  assert.equal(result.status, 'failed');
  assert.equal(result.errorMessage, 'boom');
});

test('DirectorRuntime returns timed_out on timeout and cancels the queued request', async () => {
  const engine = new FakeEngine();
  engine.waitForAbort = true;
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.query('resolve_conflict', { state: { tick: 1 } }, { timeoutMs: 1 });

  assert.equal(result.data, null);
  assert.equal(result.status, 'timed_out');
  assert.equal(result.errorMessage, 'Director query timed out.');
  assert.deepEqual(engine.cancelCalls, [1]);
});
