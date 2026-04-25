//////////////////////////////////////////////////////////////////////////////
//
// character-agent.test.ts
//
// - Exercises CharacterAgent with a fake engine that captures the onToken
//   callback from queuePrompt options, then scripts token emission during
//   runQueuedRequest. Covers the turn-event stream, memory accounting,
//   cancellation, error handling, and bus mirroring.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import type {
  GenerateRequestId,
  GenerateResponse,
  PromptOptions,
} from '../core/inference-types.js';
import { ActionBus, type CharacterEvent } from './action-bus.js';
import {
  CharacterAgent,
  type CharacterAgentEngine,
  type ChatEvent,
} from './character-agent.js';
import type { CharacterConfig } from './character-config.js';

function buildConfig(overrides: Partial<CharacterConfig> = {}): CharacterConfig {
  return {
    id: 'aria-01',
    persona: { name: 'Aria', summary: 'A friendly guide.' },
    actions: {
      actions: [
        {
          name: 'wave',
          description: 'wave hello',
        },
        {
          name: 'smile',
          description: 'adjust expression',
        },
      ],
    },
    ...overrides,
  };
}

interface ScriptedResponse {
  readonly tokens: readonly string[];
  readonly response?: Partial<GenerateResponse>;
  readonly throwOnRun?: Error;
  readonly waitBeforeTokens?: Promise<void>;
}

interface FakeEngine extends CharacterAgentEngine {
  readonly queuePromptCalls: Array<{
    contextKey: string;
    promptText: string;
    options: PromptOptions | number | undefined;
  }>;
  readonly applyChatTemplateCalls: Array<{
    messages: Array<{ role: string; content: string }>;
    addAssistant: boolean;
  }>;
  readonly runCalls: GenerateRequestId[];
  readonly cancelCalls: GenerateRequestId[];
  enqueue(script: ScriptedResponse): void;
  waitForRunCount(count: number): Promise<void>;
}

function createFakeEngine(): FakeEngine {
  const queuePromptCalls: FakeEngine['queuePromptCalls'] = [];
  const applyChatTemplateCalls: FakeEngine['applyChatTemplateCalls'] = [];
  const runCalls: GenerateRequestId[] = [];
  const cancelCalls: GenerateRequestId[] = [];
  const scripts: ScriptedResponse[] = [];
  const pendingCallbacks: Array<((token: string) => void) | undefined> = [];
  const runWaiters: Array<{ count: number; resolve: () => void }> = [];
  let nextId = 1;

  const flushRunWaiters = (): void => {
    for (let index = runWaiters.length - 1; index >= 0; index -= 1) {
      if (runCalls.length >= runWaiters[index].count) {
        const waiter = runWaiters.splice(index, 1)[0];
        waiter.resolve();
      }
    }
  };

  return {
    queuePromptCalls,
    applyChatTemplateCalls,
    runCalls,
    cancelCalls,
    enqueue(script: ScriptedResponse) {
      scripts.push(script);
    },
    waitForRunCount(count: number): Promise<void> {
      if (runCalls.length >= count) {
        return Promise.resolve();
      }
      return new Promise<void>((resolve) => {
        runWaiters.push({ count, resolve });
      });
    },
    getChatTemplate(): string | null {
      return 'fake-template';
    },
    getBosText(): string {
      return '';
    },
    getEosText(): string {
      return '</s>';
    },
    async applyChatTemplate(
      messages: Array<{ role: string; content: string }>,
      addAssistant: boolean
    ): Promise<string> {
      applyChatTemplateCalls.push({ messages, addAssistant });
      const rendered = messages
        .map((message) => `<${message.role}>\n${message.content}</${message.role}>\n`)
        .join('');
      return `${rendered}${addAssistant ? '<assistant>\n' : ''}`;
    },
    async queuePrompt(
      contextKey: string,
      promptText: string,
      options?: number | PromptOptions
    ): Promise<GenerateRequestId> {
      queuePromptCalls.push({ contextKey, promptText, options });
      const id = nextId++;
      const callback =
        typeof options === 'object' && options != null ? options.onToken : undefined;
      pendingCallbacks.push(callback);
      return id;
    },
    async runQueuedRequest(
      requestId: GenerateRequestId,
      runOptions: { signal?: AbortSignal } = {}
    ): Promise<GenerateResponse> {
      runCalls.push(requestId);
      flushRunWaiters();
      const script = scripts.shift();
      const onToken = pendingCallbacks.shift();
      if (!script) {
        throw new Error(`No scripted response enqueued for request ${requestId}`);
      }
      if (script.throwOnRun) {
        throw script.throwOnRun;
      }
      if (script.waitBeforeTokens) {
        await waitForScriptRelease(runOptions.signal, script.waitBeforeTokens);
        if (runOptions.signal?.aborted) {
          return {
            requestId,
            completed: false,
            failed: false,
            cancelled: true,
            outputText: '',
          };
        }
      }
      let output = '';
      for (const token of script.tokens) {
        if (runOptions.signal?.aborted) {
          return {
            requestId,
            completed: false,
            failed: false,
            cancelled: true,
            outputText: output,
          };
        }
        output += token;
        onToken?.(token);
      }
      return {
        requestId,
        completed: true,
        failed: false,
        cancelled: false,
        outputText: output,
        ...script.response,
      };
    },
    async cancelQueuedRequest(requestId: GenerateRequestId): Promise<boolean> {
      cancelCalls.push(requestId);
      return true;
    },
  };
}

async function collectEvents(iter: AsyncIterable<ChatEvent>): Promise<ChatEvent[]> {
  const out: ChatEvent[] = [];
  for await (const event of iter) {
    out.push(event);
  }
  return out;
}

function waitForScriptRelease(
  signal: AbortSignal | undefined,
  gate: Promise<void>
): Promise<void> {
  if (!signal) {
    return gate;
  }
  if (signal.aborted) {
    return Promise.resolve();
  }

  return new Promise<void>((resolve, reject) => {
    const onAbort = (): void => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    };

    signal.addEventListener('abort', onAbort, { once: true });
    gate.then(
      () => {
        signal.removeEventListener('abort', onAbort);
        resolve();
      },
      (error) => {
        signal.removeEventListener('abort', onAbort);
        reject(error);
      }
    );
  });
}

test('CharacterAgent exposes stable systemPrompt and grammarSource', () => {
  const engine = createFakeEngine();
  const agent = new CharacterAgent(engine, buildConfig());
  const prompt1 = agent.getSystemPrompt();
  const prompt2 = agent.getSystemPrompt();
  assert.equal(prompt1, prompt2);
  assert.ok(prompt1.includes('Aria'));

  const grammar = agent.getGrammarSource();
  assert.ok(grammar.length > 0);
  assert.ok(grammar.includes('root'));
});

test('chat() yields turn-start, prose, action, turn-end in order', async () => {
  const engine = createFakeEngine();
  engine.enqueue({
    tokens: ['Hello ', 'there. ', '[wave]', ' Bye.'],
  });
  const agent = new CharacterAgent(engine, buildConfig());
  const events = await collectEvents(agent.chat('hi'));

  assert.equal(events[0].kind, 'turn-start');
  const kinds = events.map((e) => e.kind);
  assert.ok(kinds.includes('prose'));
  assert.ok(kinds.includes('action'));
  assert.equal(events[events.length - 1].kind, 'turn-end');

  const action = events.find((e) => e.kind === 'action');
  assert.ok(action && action.kind === 'action');
  assert.equal(action.name, 'wave');

  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.cancelled, false);
  assert.ok(end.finalText.includes('Hello there.'));
});

test('chat() threads grammar and maxOutputTokens into queuePrompt options', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['ok'] });
  const agent = new CharacterAgent(engine, buildConfig(), { maxOutputTokens: 42 });
  await collectEvents(agent.chat('hi'));

  assert.equal(engine.queuePromptCalls.length, 1);
  const call = engine.queuePromptCalls[0];
  assert.equal(call.contextKey, 'aria-01');
  assert.ok(typeof call.options === 'object' && call.options != null);
  const opts = call.options as PromptOptions;
  assert.equal(opts.nTokens, 42);
  assert.equal(typeof opts.grammar, 'string');
  assert.ok(opts.grammar && opts.grammar.includes('root'));
  assert.equal(typeof opts.onToken, 'function');
});

test('chat() reuses a stable contextKey for each turn', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['first reply'] });
  engine.enqueue({ tokens: ['second reply'] });
  const agent = new CharacterAgent(engine, buildConfig());

  await collectEvents(agent.chat('first'));
  await collectEvents(agent.chat('second'));

  assert.equal(engine.queuePromptCalls.length, 2);
  assert.equal(engine.queuePromptCalls[0].contextKey, 'aria-01');
  assert.equal(engine.queuePromptCalls[1].contextKey, 'aria-01');
});

test('chat() probes chat template boundaries once per agent and reuses them across turns', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['first reply'] });
  engine.enqueue({ tokens: ['second reply'] });
  const agent = new CharacterAgent(engine, buildConfig());

  await collectEvents(agent.chat('first'));
  await collectEvents(agent.chat('second'));

  assert.equal(engine.applyChatTemplateCalls.length, 7);
});

test('successful turns commit user+assistant pairs to memory', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['hi there'] });
  const agent = new CharacterAgent(engine, buildConfig());
  await collectEvents(agent.chat('hello'));

  const memory = agent.getMemory();
  assert.equal(memory.length, 2);
  assert.equal(memory[0].role, 'user');
  assert.equal(memory[0].content, 'hello');
  assert.equal(memory[1].role, 'assistant');
  assert.equal(memory[1].content, 'hi there');
});

test('assistant memory preserves cues when actions are interleaved', async () => {
  const engine = createFakeEngine();
  engine.enqueue({
    tokens: ['Hello ', '[wave]', ' there.'],
  });
  const agent = new CharacterAgent(engine, buildConfig());

  const events = await collectEvents(agent.chat('hello'));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.finalText, 'Hello  there.');

  const memory = agent.getMemory();
  assert.equal(memory.length, 2);
  assert.equal(memory[1].role, 'assistant');
  assert.equal(memory[1].content, 'Hello [wave] there.');
});

test('assistant memory keeps multiple cues inline for later turns', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['[smile] Hello ', '[wave]', ' again.'] });
  const agent = new CharacterAgent(engine, buildConfig());

  await collectEvents(agent.chat('hello'));

  const memory = agent.getMemory();
  assert.equal(memory[1].content, '[smile] Hello [wave] again.');
});

test('chat() stops before leaked next-turn chat template markers', async () => {
  const engine = createFakeEngine();
  engine.enqueue({
    tokens: ['Hello there.', '</assistant>\n<user>\nignored'],
  });
  const agent = new CharacterAgent(engine, buildConfig());

  const events = await collectEvents(agent.chat('hello'));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.finalText, 'Hello there.');
  assert.ok(engine.cancelCalls.length >= 1);

  const memory = agent.getMemory();
  assert.equal(memory[1].content, 'Hello there.');
});

test('chat() trims partial boundary prefixes that arrive at end of stream', async () => {
  const engine = createFakeEngine();
  engine.enqueue({
    tokens: ['Hello there.', '</assist'],
  });
  const agent = new CharacterAgent(engine, buildConfig());

  const events = await collectEvents(agent.chat('hello'));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.finalText, 'Hello there.');

  const memory = agent.getMemory();
  assert.equal(memory[1].content, 'Hello there.');
});

test('errored turns do not commit to memory and surface errorMessage', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: [], throwOnRun: new Error('boom') });
  const agent = new CharacterAgent(engine, buildConfig());
  const events = await collectEvents(agent.chat('hello'));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.errorMessage, 'boom');
  assert.equal(agent.getMemory().length, 0);
  assert.equal(engine.cancelCalls.length, 1);
});

test('cancelled turns do not commit to memory', async () => {
  const engine = createFakeEngine();
  const controller = new AbortController();
  controller.abort();
  engine.enqueue({
    tokens: [],
    response: { completed: false, cancelled: true, outputText: '' },
  });
  const agent = new CharacterAgent(engine, buildConfig());
  const events = await collectEvents(agent.chat('hi', { signal: controller.signal }));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.cancelled, true);
  assert.equal(agent.getMemory().length, 0);
});

test('bus emits mirror the async iterator events', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['hey'] });
  const bus = new ActionBus();
  const busEvents: CharacterEvent[] = [];
  bus.onAny((event) => {
    busEvents.push(event);
  });
  const agent = new CharacterAgent(engine, buildConfig(), { bus });
  const iterEvents = await collectEvents(agent.chat('yo'));
  assert.equal(busEvents.length, iterEvents.length);
  for (let i = 0; i < busEvents.length; i += 1) {
    assert.equal(busEvents[i].kind, iterEvents[i].kind);
  }
});

test('maxTurns: 0 disables memory retention', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['one'] });
  engine.enqueue({ tokens: ['two'] });
  const agent = new CharacterAgent(engine, buildConfig({ memory: { maxTurns: 0 } }));
  await collectEvents(agent.chat('a'));
  await collectEvents(agent.chat('b'));
  assert.equal(agent.getMemory().length, 0);
});

test('memory sliding window drops oldest pairs past maxTurns', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['r1'] });
  engine.enqueue({ tokens: ['r2'] });
  engine.enqueue({ tokens: ['r3'] });
  const agent = new CharacterAgent(engine, buildConfig({ memory: { maxTurns: 2 } }));
  await collectEvents(agent.chat('m1'));
  await collectEvents(agent.chat('m2'));
  await collectEvents(agent.chat('m3'));

  const memory = agent.getMemory();
  assert.equal(memory.length, 4);
  assert.equal(memory[0].content, 'm2');
  assert.equal(memory[1].content, 'r2');
  assert.equal(memory[2].content, 'm3');
  assert.equal(memory[3].content, 'r3');
});

test('clearMemory empties the sliding window', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['hi'] });
  const agent = new CharacterAgent(engine, buildConfig());
  await collectEvents(agent.chat('hello'));
  assert.equal(agent.getMemory().length, 2);
  agent.clearMemory();
  assert.equal(agent.getMemory().length, 0);
});

test('choose() threads literal-choice grammar into queuePrompt options', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['wait'] });
  const agent = new CharacterAgent(engine, buildConfig());

  const result = await agent.choose('What should you do?', {
    choices: ['wait', 'wander', 'approach:aria'],
  });

  assert.equal(result.choice, 'wait');
  assert.equal(result.status, 'ok');
  assert.equal(engine.queuePromptCalls.length, 1);
  const call = engine.queuePromptCalls[0];
  assert.ok(typeof call.options === 'object' && call.options != null);
  const opts = call.options as PromptOptions;
  assert.equal(opts.promptFormat, 'raw');
  assert.equal(opts.nTokens, 24);
  assert.ok(typeof opts.grammar === 'string' && opts.grammar.includes('approach:aria'));
  assert.ok(call.promptText.includes('Choose exactly one of the following options and output only that option text:'));
});

test('choose() is stateless and does not write memory', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['chat reply'] });
  engine.enqueue({ tokens: ['wander'] });
  const agent = new CharacterAgent(engine, buildConfig());

  await collectEvents(agent.chat('hello'));
  assert.equal(agent.getMemory().length, 2);

  const result = await agent.choose('Pick one.', {
    choices: ['wait', 'wander'],
  });

  assert.equal(result.choice, 'wander');
  assert.equal(agent.getMemory().length, 2);
  const rendered = engine.queuePromptCalls[1]!.promptText;
  assert.doesNotMatch(rendered, /chat reply/);
  assert.doesNotMatch(rendered, /<user>\nhello<\/user>/);
});

test('choose() returns null on invalid model output', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['something else entirely'] });
  const agent = new CharacterAgent(engine, buildConfig());

  const result = await agent.choose('Pick one.', {
    choices: ['yes', 'no'],
  });

  assert.equal(result.choice, null);
  assert.equal(result.status, 'invalid_response');
});

test('choose() surfaces engine failure', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: [], throwOnRun: new Error('boom') });
  const agent = new CharacterAgent(engine, buildConfig());

  const result = await agent.choose('Pick one.', {
    choices: ['yes', 'no'],
  });

  assert.equal(result.choice, null);
  assert.equal(result.status, 'failed');
  assert.equal(result.errorMessage, 'boom');
  assert.equal(engine.cancelCalls.length, 1);
});

test('choose() returns cancelled when aborted', async () => {
  const engine = createFakeEngine();
  const controller = new AbortController();
  controller.abort();
  engine.enqueue({ tokens: [], response: { completed: false, cancelled: true, outputText: '' } });
  const agent = new CharacterAgent(engine, buildConfig());

  const result = await agent.choose('Pick one.', {
    choices: ['yes', 'no'],
    signal: controller.signal,
  });

  assert.equal(result.choice, null);
  assert.equal(result.status, 'aborted');
});

test('choose() returns cancelled on timeout and cancels the queued request', async () => {
  const engine = createFakeEngine();
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  engine.enqueue({ tokens: ['yes'], waitBeforeTokens: gate });
  const agent = new CharacterAgent(engine, buildConfig());

  const result = await agent.choose('Pick one.', {
    choices: ['yes', 'no'],
    timeoutMs: 1,
  });
  release();

  assert.equal(result.choice, null);
  assert.equal(result.status, 'timed_out');
  assert.equal(result.errorMessage, 'Choice timed out.');
  assert.deepEqual(engine.cancelCalls, [1]);
});

test('overlapping chat() auto-cancels the prior turn and commits only the replacement turn', async () => {
  const engine = createFakeEngine();
  let releaseFirst!: () => void;
  const firstGate = new Promise<void>((resolve) => {
    releaseFirst = resolve;
  });
  engine.enqueue({ tokens: ['first reply'], waitBeforeTokens: firstGate });
  engine.enqueue({ tokens: ['second reply'] });
  const agent = new CharacterAgent(engine, buildConfig());

  const firstEventsPromise = collectEvents(agent.chat('first'));
  await engine.waitForRunCount(1);
  const secondEventsPromise = collectEvents(agent.chat('second'));
  releaseFirst();

  const [firstEvents, secondEvents] = await Promise.all([
    firstEventsPromise,
    secondEventsPromise,
  ]);

  const firstEnd = firstEvents[firstEvents.length - 1];
  assert.ok(firstEnd.kind === 'turn-end');
  assert.equal(firstEnd.cancelled, true);

  const secondEnd = secondEvents[secondEvents.length - 1];
  assert.ok(secondEnd.kind === 'turn-end');
  assert.equal(secondEnd.cancelled, false);
  assert.equal(secondEnd.finalText, 'second reply');

  const memory = agent.getMemory();
  assert.equal(memory.length, 2);
  assert.equal(memory[0].content, 'second');
  assert.equal(memory[1].content, 'second reply');
});

test('chat() passes a rendered raw prompt to queuePrompt', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['hi'] });
  const agent = new CharacterAgent(engine, buildConfig());
  await collectEvents(agent.chat('hello'));

  const call = engine.queuePromptCalls[0];
  assert.ok(typeof call.promptText === 'string' && call.promptText.length > 0);
  assert.ok(call.promptText.includes('<system>\n'));
  assert.ok(call.promptText.includes('Aria'));
  assert.ok(call.promptText.includes('<user>\n'));
  assert.ok(call.promptText.includes('hello'));
  assert.ok(call.promptText.endsWith('<assistant>\n'));

  assert.ok(typeof call.options === 'object' && call.options != null);
  const opts = call.options as PromptOptions;
  assert.equal(opts.promptFormat, 'raw');
});

test('chat() includes prior turn history in the rendered prompt', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['first reply'] });
  engine.enqueue({ tokens: ['second reply'] });
  const agent = new CharacterAgent(engine, buildConfig());
  await collectEvents(agent.chat('first question'));
  await collectEvents(agent.chat('second question'));

  const secondCall = engine.queuePromptCalls[1];
  const rendered = secondCall.promptText;
  const idxSystem = rendered.indexOf('<system>\n');
  const idxFirstUser = rendered.indexOf('first question');
  const idxAssistant = rendered.indexOf('first reply');
  const idxSecondUser = rendered.indexOf('second question');
  assert.ok(idxSystem >= 0 && idxFirstUser > idxSystem);
  assert.ok(idxAssistant > idxFirstUser);
  assert.ok(idxSecondUser > idxAssistant);
});

test('chat() keeps the user message literal in the rendered prompt', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['[wave] Hi!'] });
  const agent = new CharacterAgent(engine, buildConfig());

  await collectEvents(agent.chat('hi there'));

  const call = engine.queuePromptCalls[0];
  assert.match(call.promptText, /<user>\nhi there<\/user>/);
  assert.doesNotMatch(call.promptText, /reply briefly and warmly/);
});

test('chat() injects persona dialog examples as few-shot chat turns', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['[wave] hi'] });
  const agent = new CharacterAgent(
    engine,
      buildConfig({
        persona: {
          name: 'Mira',
          summary: 'An observant companion.',
          dialogExamples: [
            { user: 'hello', assistant: '[wave] Hello there.' },
            { user: 'are you okay?', assistant: '[settle] I am here with you.' },
        ],
      },
      actions: {
        actions: [
          { name: 'wave', description: 'wave hello' },
          { name: 'settle', description: 'adjust expression' },
        ],
      },
    })
  );

  await collectEvents(agent.chat('hello'));

  assert.equal(engine.queuePromptCalls.length, 1);
  const call = engine.queuePromptCalls[0];
  assert.doesNotMatch(call.promptText, /Dialog examples:/);
  assert.match(call.promptText, /<user>\nhello<\/user>\n<assistant>\n\[wave\] Hello there\.<\/assistant>/);
  assert.match(call.promptText, /<user>\nare you okay\?<\/user>\n<assistant>\n\[settle\] I am here with you\.<\/assistant>/);
  const firstExampleIndex = call.promptText.indexOf('<user>\nhello</user>');
  const liveUserIndex = call.promptText.lastIndexOf('<user>\nhello</user>');
  assert.ok(firstExampleIndex >= 0 && liveUserIndex > firstExampleIndex);
});

test('chat() does not inject anchorExamples as few-shot chat turns', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['[wave] hi'] });
  const agent = new CharacterAgent(
    engine,
    buildConfig({
      persona: {
        name: 'Mira',
        summary: 'An observant companion.',
        anchorExamples: [
          { user: 'who are you?', assistant: '[wave] I am Mira.' },
          { user: 'can you code?', assistant: '[settle] No, that is outside my lane.' },
        ],
        dialogExamples: [{ user: 'hello', assistant: '[wave] Hello there.' }],
      },
      actions: {
        actions: [
          { name: 'wave', description: 'wave hello' },
          { name: 'settle', description: 'adjust expression' },
        ],
      },
    })
  );

  await collectEvents(agent.chat('hello'));

  const call = engine.queuePromptCalls[0];
  assert.match(call.promptText, /<system>[\s\S]*Examples:\n\n?User: who are you\?/);
  assert.match(call.promptText, /<user>\nhello<\/user>\n<assistant>\n\[wave\] Hello there\.<\/assistant>/);
  assert.doesNotMatch(call.promptText, /<user>\nwho are you\?<\/user>/);
  assert.doesNotMatch(call.promptText, /<user>\ncan you code\?<\/user>/);
});

test('chat() preserves non-exact user-prefix text in assistant output', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['hello there, friend'] });
  const agent = new CharacterAgent(engine, buildConfig());

  const events = await collectEvents(agent.chat('hello'));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.finalText, 'hello there, friend');

  const memory = agent.getMemory();
  assert.equal(memory[1].content, 'hello there, friend');
});

test('chat() strips only an exact leading user-message echo before storing assistant output', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ["Tell me what's your name?\n\nI'm Aria."] });
  const agent = new CharacterAgent(engine, buildConfig());

  const events = await collectEvents(agent.chat("Tell me what's your name?"));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.finalText, "I'm Aria.");

  const memory = agent.getMemory();
  assert.equal(memory[1].content, "I'm Aria.");
});
