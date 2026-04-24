import assert from 'node:assert/strict';
import test from 'node:test';

import type { CharacterAgentEngine } from '../character/character-agent.js';
import type {
  GenerateRequestId,
  GenerateResponse,
  PromptOptions,
} from '../core/inference-types.js';
import { parseDirectorConfig } from './director-config.js';
import { DirectorRuntime } from './director-runtime.js';

class FakeEngine implements CharacterAgentEngine {
  public outputText = '';
  public fail = false;
  public grammar: string | undefined;
  public media: Uint8Array[] | undefined;
  public prompt = '';
  public waitForAbort = false;
  public mediaMarker: string | null = '<image>';
  public queueCalls = 0;
  public cancelCalls: GenerateRequestId[] = [];

  public async applyChatTemplate(
    messages: Array<{ role: string; content: string }>,
    _addAssistant: boolean
  ): Promise<string> {
    this.prompt = messages.map((message) => `${message.role}: ${message.content}`).join('\n');
    return this.prompt;
  }

  public async queuePrompt(
    _contextKey: string,
    _prompt: string,
    options?: number | PromptOptions
  ): Promise<GenerateRequestId> {
    this.queueCalls += 1;
    if (typeof options === 'object' && options) {
      this.grammar = options.grammar;
      this.media = options.media;
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
        if (!signal) return;
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

  public async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
    this.cancelCalls.push(requestId);
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

  public getMediaMarker(): string | null {
    return this.mediaMarker;
  }
}

const CONFIG = parseDirectorConfig({
  id: 'courtyard-director',
  director: {
    role: 'Scenario director',
    instructions: ['Only use supplied state.'],
  },
  inputs: {
    state: { kind: 'data', description: 'Current world state.' },
    screenshot: { kind: 'image', description: 'Current app screenshot.' },
  },
  tasks: {
    resolve_conflict: {
      purpose: 'Resolve a conflict.',
      inputs: ['state'],
      output: { shape: 'select_one', choices: 'runtime' },
    },
    narrate: {
      inputs: ['state'],
      output: { shape: 'text', maxLength: 80 },
    },
    inspect_screen: {
      inputs: ['screenshot'],
      output: { shape: 'text', maxLength: 80 },
    },
    choose_many: {
      inputs: ['state'],
      output: { shape: 'select_many', choices: 'runtime', min: 1, max: 2 },
    },
    choose_slots: {
      inputs: ['state'],
      output: {
        shape: 'select_slots',
        slots: [
          { name: 'intent', choices: 'runtime' },
          { name: 'tone', choices: [{ id: 'brief' }, { id: 'friendly' }] },
        ],
      },
    },
    assist_with_directives: {
      inputs: ['state'],
      output: { shape: 'text_with_directives', directives: 'runtime', maxDirectives: 1, maxLength: 120 },
    },
  },
});

test('DirectorRuntime returns a selected runtime choice with hidden payload', async () => {
  const engine = new FakeEngine();
  engine.outputText = 'winner:aria';
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.run('resolve_conflict', {
    inputs: { state: { tick: 3 } },
    choices: [
      {
        id: 'winner:aria',
        label: 'Aria wins',
        description: 'Award the contested object to Aria.',
        payload: { winnerAgentId: 'aria', secret: 'not shown' },
      },
      { id: 'deny', label: 'Deny pickup', payload: { winnerAgentId: null } },
    ],
  });

  assert.equal(result.status, 'ok');
  assert.equal(result.selections[0]?.id, 'winner:aria');
  assert.deepEqual(result.selections[0]?.payload, { winnerAgentId: 'aria', secret: 'not shown' });
  assert.ok(engine.grammar?.includes('"winner:aria"'));
  assert.ok(engine.prompt.includes('winner:aria - Aria wins'));
  assert.equal(engine.prompt.includes('not shown'), false);
  assert.equal(engine.prompt.includes('Never output JSON'), true);
});

test('DirectorRuntime exposes task grammar and prompt for inspection', () => {
  const engine = new FakeEngine();
  const runtime = new DirectorRuntime(engine, CONFIG);
  const request = {
    inputs: { state: { tick: 3 } },
    choices: [
      { id: 'winner:aria', label: 'Aria wins', payload: { secret: 'hidden' } },
      { id: 'deny' },
    ],
  };

  const grammar = runtime.getTaskGrammar('resolve_conflict', request);
  const prompt = runtime.getTaskPrompt('resolve_conflict', request);

  assert.equal(grammar, prompt.grammar);
  assert.ok(grammar?.includes('"winner:aria"'));
  assert.ok(prompt.userPrompt.includes('winner:aria - Aria wins'));
  assert.equal(prompt.userPrompt.includes('hidden'), false);
  assert.equal(prompt.media.length, 0);
});

test('DirectorRuntime threads grammar for select_many and select_slots', async () => {
  const engine = new FakeEngine();
  engine.outputText = 'alpha\nbeta';
  const runtime = new DirectorRuntime(engine, CONFIG);
  const many = await runtime.run('choose_many', {
    inputs: { state: { tick: 2 } },
    choices: [{ id: 'alpha' }, { id: 'beta' }],
  });

  assert.equal(many.status, 'ok');
  assert.ok(engine.grammar?.includes('selection-line ::= "alpha" | "beta"'));

  engine.outputText = 'intent=advise\ntone=brief';
  const slots = await runtime.run('choose_slots', {
    inputs: { state: { tick: 2 } },
    slotChoices: { intent: [{ id: 'advise' }, { id: 'navigate' }] },
  });

  assert.equal(slots.status, 'ok');
  assert.ok(engine.grammar?.includes('"intent="'));
  assert.ok(engine.grammar?.includes('slot0-choice ::= "advise" | "navigate"'));
});

test('DirectorRuntime threads directive grammar for text_with_directives', async () => {
  const engine = new FakeEngine();
  engine.outputText = 'Open billing next. [nav.billing]';
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.run('assist_with_directives', {
    inputs: { state: { tick: 2 } },
    directives: [
      { id: 'nav.billing', label: 'Open billing', payload: { route: '/billing', secret: 'hidden' } },
      { id: 'inspect.menu' },
    ],
  });

  assert.equal(result.status, 'ok');
  assert.equal(result.text, 'Open billing next.');
  assert.equal(result.selections[0]?.id, 'nav.billing');
  assert.deepEqual(result.selections[0]?.payload, { route: '/billing', secret: 'hidden' });
  assert.ok(engine.grammar?.includes('directive-cue ::= "[" directive-id "]"'));
  assert.ok(engine.grammar?.includes('"nav.billing" | "inspect.menu"'));
  assert.equal(engine.prompt.includes('hidden'), false);
});

test('DirectorRuntime rejects malformed directive cues after generation', async () => {
  const engine = new FakeEngine();
  engine.outputText = 'Try this [unknown cue]';
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.run('assist_with_directives', {
    inputs: { state: { tick: 2 } },
    directives: [{ id: 'nav.billing' }],
  });

  assert.equal(result.status, 'invalid_response');
  assert.match(result.errorMessage ?? '', /unknown or malformed directive/);
});

test('DirectorRuntime reports oversized grammars before queueing generation', async () => {
  const engine = new FakeEngine();
  const runtime = new DirectorRuntime(engine, CONFIG);
  const choices = Array.from({ length: 9000 }, (_value, index) => ({ id: `choice-${index}` }));

  const result = await runtime.run('resolve_conflict', {
    inputs: { state: { tick: 1 } },
    choices,
  });

  assert.equal(result.status, 'invalid_response');
  assert.match(result.errorMessage ?? '', /grammar exceeds maximum size/);
  assert.equal(engine.queueCalls, 0);
});

test('DirectorRuntime returns invalid_response for unknown selections', async () => {
  const engine = new FakeEngine();
  engine.outputText = 'winner:mira';
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.run('resolve_conflict', {
    inputs: { state: { tick: 1 } },
    choices: [{ id: 'winner:aria' }],
  });

  assert.equal(result.status, 'invalid_response');
  assert.match(result.errorMessage ?? '', /did not match any available choice/);
});

test('DirectorRuntime returns text task output without JSON parsing', async () => {
  const engine = new FakeEngine();
  engine.outputText = 'Aria sprints toward home base.';
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.run('narrate', { inputs: { state: { tick: 4 } } });

  assert.equal(result.status, 'ok');
  assert.equal(result.text, 'Aria sprints toward home base.');
  assert.deepEqual(result.selections, []);
  assert.equal(engine.grammar, undefined);
});

test('DirectorRuntime renders image inputs through media markers', async () => {
  const engine = new FakeEngine();
  engine.outputText = 'The settings page is open.';
  const runtime = new DirectorRuntime(engine, CONFIG);
  const image = Uint8Array.from([1, 2, 3]);

  const result = await runtime.run('inspect_screen', {
    inputs: {
      screenshot: { kind: 'image', media: image, description: 'Browser screenshot.' },
    },
  });

  assert.equal(result.status, 'ok');
  assert.equal(result.text, 'The settings page is open.');
  assert.deepEqual(engine.media, [image]);
  assert.ok(engine.prompt.includes('<image>'));
  assert.ok(engine.prompt.includes('Browser screenshot.'));
});

test('DirectorRuntime surfaces engine failure', async () => {
  const engine = new FakeEngine();
  engine.fail = true;
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.run('narrate', { inputs: { state: { tick: 1 } } });

  assert.equal(result.status, 'failed');
  assert.equal(result.errorMessage, 'boom');
});

test('DirectorRuntime returns timed_out on timeout and cancels the queued request', async () => {
  const engine = new FakeEngine();
  engine.waitForAbort = true;
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.run('narrate', { inputs: { state: { tick: 1 } }, timeoutMs: 1 });

  assert.equal(result.status, 'timed_out');
  assert.equal(result.errorMessage, 'Director task timed out.');
  assert.deepEqual(engine.cancelCalls, [1]);
});
