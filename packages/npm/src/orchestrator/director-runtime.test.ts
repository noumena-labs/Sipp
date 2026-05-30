import assert from 'node:assert/strict';
import test from 'node:test';

import type {
  BrowserTextRun,
  ChatInput,
  ChatOptions,
  GenerationResult,
  TokenBatch,
} from '../models/types.js';
import { parseDirectorConfig } from './director-config.js';
import { DirectorRuntime, type DirectorRuntimeEngine } from './director-runtime.js';

class FakeEngine implements DirectorRuntimeEngine {
  public outputText = '';
  public fail = false;
  public grammar: string | undefined;
  public media: Uint8Array[] | undefined;
  public prompt = '';
  public waitForAbort = false;
  public mediaMarker: string | null = '<image>';
  public queryCalls = 0;

  public readonly models = {
    current: () => ({ mediaMarker: this.mediaMarker }),
  };

  public chat(
    input: ChatInput,
    options?: ChatOptions
  ): BrowserTextRun {
    this.queryCalls += 1;
    if (typeof options === 'object' && options) {
      this.grammar = options.grammar;
    }
    const messages = Array.isArray(input) ? input : input.messages;
    this.media = Array.isArray(input) ? undefined : input.media;
    this.prompt = messages.map((message) => `${message.role}: ${message.content}`).join('\n');

    if (this.waitForAbort) {
      const response = new Promise<GenerationResult>((_resolve, reject) => {
        const signal = options?.signal;
        if (!signal) return;
        if (signal.aborted) {
          reject(new DOMException('Operation aborted.', 'AbortError'));
          return;
        }
        signal.addEventListener(
          'abort',
          () => reject(new DOMException('Operation aborted.', 'AbortError')),
          { once: true }
        );
      });
      return textRun(response);
    }

    if (this.fail) {
      return textRun(Promise.reject(new Error('boom')));
    }

    const safeText = sanitizeFakeChatOutput(this.outputText);
    return textRun(Promise.resolve(generationResult(safeText)), [tokenBatch(safeText)]);
  }
}

function generationResult(text: string): GenerationResult {
  return {
    id: 'test',
    text,
    finishReason: 'stop',
    stats: {
      inputTokens: 1,
      outputTokens: 1,
      cacheHits: 0,
      prefillMs: 0,
      decodeMs: 0,
    },
  };
}

function tokenBatch(text: string): TokenBatch {
  return {
    requestId: 'test',
    streamId: 1,
    sequenceStart: 0,
    text,
    frameCount: 1,
    byteCount: new TextEncoder().encode(text).byteLength,
    stats: {
      framesSent: 1,
      bytesSent: new TextEncoder().encode(text).byteLength,
      framesDropped: 0,
      batchesSent: 1,
    },
  };
}

function textRun(
  response: Promise<GenerationResult>,
  batches: readonly TokenBatch[] = []
): BrowserTextRun {
  return {
    response,
    tokens: {
      async *[Symbol.asyncIterator](): AsyncIterator<TokenBatch> {
        for (const batch of batches) {
          yield batch;
        }
      },
    },
    cancel: () => {},
  };
}

function sanitizeFakeChatOutput(text: string): string {
  const markers = ['</s>', '<user>', '<assistant>', '<system>'];
  let index = -1;
  for (const marker of markers) {
    const markerIndex = text.indexOf(marker);
    if (markerIndex >= 0 && (index < 0 || markerIndex < index)) {
      index = markerIndex;
    }
  }
  return (index >= 0 ? text.slice(0, index) : text).trim();
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
      output: { shape: 'text' },
    },
    inspect_screen: {
      inputs: ['screenshot'],
      output: { shape: 'text' },
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
      output: { shape: 'text_with_directives', directives: 'runtime', maxDirectives: 1 },
    },
  },
});

test('DirectorRuntime returns a selected runtime choice with hidden payload', async () => {
  const engine = new FakeEngine();
  engine.outputText = 'winner:aria';
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.run('resolve_conflict', {
    inputs: { state: { tick: 3 }, extra: 'not for this task' },
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
  assert.equal(engine.prompt.includes('not for this task'), false);
  assert.equal(engine.prompt.includes('Never output JSON'), true);
});

test('DirectorRuntime parses sanitized assistant text from engine chat', async () => {
  const engine = new FakeEngine();
  engine.outputText = 'Aria sprints toward home base.</s><user>ignored</user>';
  const runtime = new DirectorRuntime(engine, CONFIG);

  const result = await runtime.run('narrate', { inputs: { state: { tick: 4 } } });

  assert.equal(result.status, 'ok');
  assert.equal(result.text, 'Aria sprints toward home base.');
  assert.equal(result.rawText, 'Aria sprints toward home base.');
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
  assert.ok(engine.prompt.includes('Response:\nWrite only the final answer.'));
  assert.ok(engine.prompt.includes('Available directives:'));
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

  assert.equal(result.status, 'invalid_request');
  assert.match(result.errorMessage ?? '', /grammar exceeds maximum size/);
  assert.equal(engine.queryCalls, 0);
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
  assert.ok(engine.prompt.includes('Task:\nComplete task narrate.'));
  assert.ok(engine.prompt.includes('Response:\nWrite only the final answer.'));
  assert.equal(engine.prompt.includes('Output shape:'), false);
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
});
