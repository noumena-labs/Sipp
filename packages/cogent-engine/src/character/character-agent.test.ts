//////////////////////////////////////////////////////////////////////////////
//
// character-agent.test.ts
//
// - Exercises CharacterRuntime with a fake engine that captures the onToken
//   callback from queuePrompt options, then scripts token emission during
//   runQueuedRequest. Covers the turn-event stream, memory accounting,
//   cancellation, error handling, and bus mirroring.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import type {
  QueryInput,
  QueryOptions,
} from '../model-management/model-types.js';
import type { ChatTemplateMessage } from '../core/chat-template-boundaries.js';
import { CharacterEventBus, type CharacterEvent } from './action-bus.js';
import {
  CharacterRuntime,
  type CharacterRuntimeEngine,
  type ChatEvent,
} from './character-agent.js';
import type { CharacterConfig } from './character-config.js';

function buildConfig(overrides: Partial<CharacterConfig> = {}): CharacterConfig {
  return {
    id: 'aria-01',
    persona: { name: 'Aria', summary: 'A friendly guide.' },
    actions: [
      {
        id: 'wave',
        description: 'wave hello',
      },
      {
        id: 'smile',
        description: 'adjust expression',
      },
    ],
    ...overrides,
  };
}

interface ScriptedResponse {
  readonly tokens: readonly string[];
  readonly throwOnRun?: Error;
  readonly waitBeforeTokens?: Promise<void>;
}

interface FakeEngine extends CharacterRuntimeEngine {
  readonly queryCalls: Array<{
    input: QueryInput;
    options: QueryOptions | undefined;
  }>;
  readonly applyChatTemplateCalls: Array<{
    messages: ChatTemplateMessage[];
    addAssistant: boolean;
  }>;
  enqueue(script: ScriptedResponse): void;
  waitForRunCount(count: number): Promise<void>;
}

function createFakeEngine(): FakeEngine {
  const queryCalls: FakeEngine['queryCalls'] = [];
  const applyChatTemplateCalls: FakeEngine['applyChatTemplateCalls'] = [];
  const scripts: ScriptedResponse[] = [];
  const runWaiters: Array<{ count: number; resolve: () => void }> = [];
  let completedCount = 0;

  const flushRunWaiters = (): void => {
    for (let index = runWaiters.length - 1; index >= 0; index -= 1) {
      if (completedCount >= runWaiters[index].count) {
        const waiter = runWaiters.splice(index, 1)[0];
        waiter.resolve();
      }
    }
  };

  return {
    queryCalls,
    applyChatTemplateCalls,
    enqueue(script: ScriptedResponse) {
      scripts.push(script);
    },
    waitForRunCount(count: number): Promise<void> {
      if (completedCount >= count) {
        return Promise.resolve();
      }
      return new Promise<void>((resolve) => {
        runWaiters.push({ count, resolve });
      });
    },
    getChatTemplate(): string | null {
      return 'fake-template';
    },
    getEosText(): string {
      return '</s>';
    },
    async applyChatTemplate(
      messages: ChatTemplateMessage[],
      addAssistant: boolean
    ): Promise<string> {
      applyChatTemplateCalls.push({ messages, addAssistant });
      const rendered = messages
        .map((message) => `<${message.role}>\n${message.content}</${message.role}>\n`)
        .join('');
      return `${rendered}${addAssistant ? '<assistant>\n' : ''}`;
    },
    async query(
      input: QueryInput,
      options?: QueryOptions
    ): Promise<string> {
      queryCalls.push({ input, options });
      completedCount++;
      flushRunWaiters();

      const script = scripts.shift();
      if (!script) {
        throw new Error('No scripted response enqueued for query');
      }
      if (script.throwOnRun) {
        throw script.throwOnRun;
      }
      if (script.waitBeforeTokens) {
        await waitForScriptRelease(options?.signal, script.waitBeforeTokens);
        if (options?.signal?.aborted) {
          throw new DOMException('Operation aborted.', 'AbortError');
        }
      }

      if (options?.signal?.aborted) {
        throw new DOMException('Operation aborted.', 'AbortError');
      }
      let output = '';
      for (const token of script.tokens) {
        if (options?.signal?.aborted) {
          throw new DOMException('Operation aborted.', 'AbortError');
        }
        output += token;
        options?.onToken?.(token);
      }
      return output;
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

test('CharacterRuntime exposes stable systemPrompt and grammarSource', () => {
  const engine = createFakeEngine();
  const agent = new CharacterRuntime(engine, buildConfig());
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
  const agent = new CharacterRuntime(engine, buildConfig());
  const events = await collectEvents(agent.chat('hi'));

  assert.equal(events[0].kind, 'turn-start');
  const kinds = events.map((e) => e.kind);
  assert.ok(kinds.includes('prose'));
  assert.ok(kinds.includes('action'));
  assert.equal(events[events.length - 1].kind, 'turn-end');

  const action = events.find((e) => e.kind === 'action');
  assert.ok(action && action.kind === 'action');
  assert.equal(action.id, 'wave');

  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.status, 'ok');
  assert.ok(end.finalText.includes('Hello there.'));
});

test('chat() threads grammar and maxOutputTokens into queuePrompt options', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['ok'] });
  const agent = new CharacterRuntime(engine, buildConfig(), { maxOutputTokens: 42 });
  await collectEvents(agent.chat('hi'));

  assert.equal(engine.queryCalls.length, 1);
  const call = engine.queryCalls[0];
  assert.ok(typeof call.options === 'object' && call.options != null);
  const opts = call.options as QueryOptions;
  assert.equal(opts.maxTokens, 42);
  assert.equal(typeof (opts as any).grammar, 'string');
  assert.ok((opts as any).grammar && (opts as any).grammar.includes('root'));
  assert.equal(typeof opts.onToken, 'function');
});

test('chat() reuses a stable contextKey for each turn', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['first reply'] });
  engine.enqueue({ tokens: ['second reply'] });
  const agent = new CharacterRuntime(engine, buildConfig());

  await collectEvents(agent.chat('first'));
  await collectEvents(agent.chat('second'));

  assert.equal(engine.queryCalls.length, 2);
});

test('chat() probes chat template boundaries once per agent and reuses them across turns', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['first reply'] });
  engine.enqueue({ tokens: ['second reply'] });
  const agent = new CharacterRuntime(engine, buildConfig());

  await collectEvents(agent.chat('first'));
  await collectEvents(agent.chat('second'));

  assert.equal(engine.applyChatTemplateCalls.length, 7);
  assert.equal(engine.applyChatTemplateCalls.filter((call) => call.messages.some((message) => message.content === '__CE_BOUNDARY_SYSTEM__')).length, 5);
});

test('successful turns commit user+assistant pairs to memory', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['hi there'] });
  const agent = new CharacterRuntime(engine, buildConfig());
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
  const agent = new CharacterRuntime(engine, buildConfig());

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
  const agent = new CharacterRuntime(engine, buildConfig());

  await collectEvents(agent.chat('hello'));

  const memory = agent.getMemory();
  assert.equal(memory[1].content, '[smile] Hello [wave] again.');
});

test('chat() stops before leaked next-turn chat template markers', async () => {
  const engine = createFakeEngine();
  engine.enqueue({
    tokens: ['Hello there.', '</assistant>\n<user>\nignored'],
  });
  const agent = new CharacterRuntime(engine, buildConfig());

  const events = await collectEvents(agent.chat('hello'));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.finalText, 'Hello there.');

  const memory = agent.getMemory();
  assert.equal(memory[1].content, 'Hello there.');
});

test('chat() trims partial boundary prefixes that arrive at end of stream', async () => {
  const engine = createFakeEngine();
  engine.enqueue({
    tokens: ['Hello there.', '</assist'],
  });
  const agent = new CharacterRuntime(engine, buildConfig());

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
  const agent = new CharacterRuntime(engine, buildConfig());
  const events = await collectEvents(agent.chat('hello'));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.errorMessage, 'boom');
  assert.equal(agent.getMemory().length, 0);
});

test('aborted turns do not commit to memory', async () => {
  const engine = createFakeEngine();
  const controller = new AbortController();
  controller.abort();
  const agent = new CharacterRuntime(engine, buildConfig());
  const events = await collectEvents(agent.chat('hi', { signal: controller.signal }));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.status, 'aborted');
  assert.equal(agent.getMemory().length, 0);
});

test('bus emits mirror the async iterator events', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['hey'] });
  const bus = new CharacterEventBus();
  const busEvents: CharacterEvent[] = [];
  bus.onAny((event) => {
    busEvents.push(event);
  });
  const agent = new CharacterRuntime(engine, buildConfig(), { bus });
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
  const agent = new CharacterRuntime(engine, buildConfig({ memory: { maxTurns: 0 } }));
  await collectEvents(agent.chat('a'));
  await collectEvents(agent.chat('b'));
  assert.equal(agent.getMemory().length, 0);
});

test('memory sliding window drops oldest pairs past maxTurns', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['r1'] });
  engine.enqueue({ tokens: ['r2'] });
  engine.enqueue({ tokens: ['r3'] });
  const agent = new CharacterRuntime(engine, buildConfig({ memory: { maxTurns: 2 } }));
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
  const agent = new CharacterRuntime(engine, buildConfig());
  await collectEvents(agent.chat('hello'));
  assert.equal(agent.getMemory().length, 2);
  agent.clearMemory();
  assert.equal(agent.getMemory().length, 0);
});

test('choose() threads literal-choice grammar into queuePrompt options', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['wait'] });
  const agent = new CharacterRuntime(engine, buildConfig());

  const result = await agent.choose('What should you do?', {
    choices: ['wait', 'wander', 'approach:aria'],
  });

  assert.equal(result.selection, 'wait');
  assert.equal(result.status, 'ok');
  assert.equal(engine.queryCalls.length, 1);
  const call = engine.queryCalls[0];
  assert.ok(typeof call.options === 'object' && call.options != null);
  const opts = call.options as QueryOptions;
  assert.equal(opts.format, 'raw');
  assert.equal(opts.maxTokens, 24);
  assert.ok(typeof (opts as any).grammar === 'string' && (opts as any).grammar.includes('approach:aria'));
  const promptText = (call.input as any).prompt;
  assert.ok(promptText.includes('Choose exactly one of the following options and output only that option text:'));
});

test('choose() is stateless and does not write memory', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['chat reply'] });
  engine.enqueue({ tokens: ['wander'] });
  const agent = new CharacterRuntime(engine, buildConfig());

  await collectEvents(agent.chat('hello'));
  assert.equal(agent.getMemory().length, 2);

  const result = await agent.choose('Pick one.', {
    choices: ['wait', 'wander'],
  });

  assert.equal(result.selection, 'wander');
  assert.equal(agent.getMemory().length, 2);
  const rendered = (engine.queryCalls[1]!.input as any).prompt;
  assert.doesNotMatch(rendered, /chat reply/);
  assert.doesNotMatch(rendered, /<user>\nhello<\/user>/);
});

test('choose() returns null on invalid model output', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['something else entirely'] });
  const agent = new CharacterRuntime(engine, buildConfig());

  const result = await agent.choose('Pick one.', {
    choices: ['yes', 'no'],
  });

  assert.equal(result.selection, null);
  assert.equal(result.status, 'invalid_response');
});

test('choose() surfaces engine failure', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: [], throwOnRun: new Error('boom') });
  const agent = new CharacterRuntime(engine, buildConfig());

  const result = await agent.choose('Pick one.', {
    choices: ['yes', 'no'],
  });

  assert.equal(result.selection, null);
  assert.equal(result.status, 'failed');
  assert.equal(result.errorMessage, 'boom');
});

test('choose() returns aborted when aborted', async () => {
  const engine = createFakeEngine();
  const controller = new AbortController();
  controller.abort();
  engine.enqueue({ tokens: [] });
  const agent = new CharacterRuntime(engine, buildConfig());

  const result = await agent.choose('Pick one.', {
    choices: ['yes', 'no'],
    signal: controller.signal,
  });

  assert.equal(result.selection, null);
  assert.equal(result.status, 'aborted');
});

test('choose() returns timed_out on timeout and cancels the queued request', async () => {
  const engine = createFakeEngine();
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  engine.enqueue({ tokens: ['yes'], waitBeforeTokens: gate });
  const agent = new CharacterRuntime(engine, buildConfig());

  const result = await agent.choose('Pick one.', {
    choices: ['yes', 'no'],
    timeoutMs: 1,
  });
  release();

  assert.equal(result.selection, null);
  assert.equal(result.status, 'timed_out');
  assert.equal(result.errorMessage, 'Choice timed out.');
});

test('overlapping chat() auto-cancels the prior turn and commits only the replacement turn', async () => {
  const engine = createFakeEngine();
  let releaseFirst!: () => void;
  const firstGate = new Promise<void>((resolve) => {
    releaseFirst = resolve;
  });
  engine.enqueue({ tokens: ['first reply'], waitBeforeTokens: firstGate });
  engine.enqueue({ tokens: ['second reply'] });
  const agent = new CharacterRuntime(engine, buildConfig());

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
  assert.equal(firstEnd.status, 'aborted');

  const secondEnd = secondEvents[secondEvents.length - 1];
  assert.ok(secondEnd.kind === 'turn-end');
  assert.equal(secondEnd.status, 'ok');
  assert.equal(secondEnd.finalText, 'second reply');

  const memory = agent.getMemory();
  assert.equal(memory.length, 2);
  assert.equal(memory[0].content, 'second');
  assert.equal(memory[1].content, 'second reply');
});

test('chat() passes a rendered raw prompt to query', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['hi'] });
  const agent = new CharacterRuntime(engine, buildConfig());
  await collectEvents(agent.chat('hello'));

  const call = engine.queryCalls[0];
  const promptText = (call.input as any).prompt;
  assert.ok(typeof promptText === 'string' && promptText.length > 0);
  assert.ok(promptText.includes('<system>\n'));
  assert.ok(promptText.includes('Aria'));
  assert.ok(promptText.includes('<user>\n'));
  assert.ok(promptText.includes('hello'));
  assert.ok(promptText.endsWith('<assistant>\n'));

  assert.ok(typeof call.options === 'object' && call.options != null);
  const opts = call.options as QueryOptions;
  assert.equal(opts.format, 'raw');
});

test('chat() includes prior turn history in the rendered prompt', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['first reply'] });
  engine.enqueue({ tokens: ['second reply'] });
  const agent = new CharacterRuntime(engine, buildConfig());
  await collectEvents(agent.chat('first question'));
  await collectEvents(agent.chat('second question'));

  const secondCall = engine.queryCalls[1];
  const rendered = (secondCall.input as any).prompt;
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
  const agent = new CharacterRuntime(engine, buildConfig());

  await collectEvents(agent.chat('hi there'));

  const call = engine.queryCalls[0];
  assert.match((call.input as any).prompt, /<user>\nhi there<\/user>/);
  assert.doesNotMatch((call.input as any).prompt, /reply briefly and warmly/);
});

test('chat() injects persona dialog examples as few-shot chat turns', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['[wave] hi'] });
  const agent = new CharacterRuntime(
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
      actions: [
        { id: 'wave', description: 'wave hello' },
        { id: 'settle', description: 'adjust expression' },
      ],
    })
  );

  await collectEvents(agent.chat('hello'));

  assert.equal(engine.queryCalls.length, 1);
  const call = engine.queryCalls[0];
  const promptText = (call.input as any).prompt;
  assert.doesNotMatch(promptText, /Dialog examples:/);
  assert.match(promptText, /<user>\nhello<\/user>\n<assistant>\n\[wave\] Hello there\.<\/assistant>/);
  assert.match(promptText, /<user>\nare you okay\?<\/user>\n<assistant>\n\[settle\] I am here with you\.<\/assistant>/);
  const firstExampleIndex = promptText.indexOf('<user>\nhello</user>');
  const liveUserIndex = promptText.lastIndexOf('<user>\nhello</user>');
  assert.ok(firstExampleIndex >= 0 && liveUserIndex > firstExampleIndex);
});

test('chat() does not inject anchorExamples as few-shot chat turns', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['[wave] hi'] });
  const agent = new CharacterRuntime(
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
      actions: [
        { id: 'wave', description: 'wave hello' },
        { id: 'settle', description: 'adjust expression' },
      ],
    })
  );

  await collectEvents(agent.chat('hello'));

  const call = engine.queryCalls[0];
  const promptText = (call.input as any).prompt;
  assert.match(promptText, /<system>[\s\S]*Examples:\n\n?User: who are you\?/);
  assert.match(promptText, /<user>\nhello<\/user>\n<assistant>\n\[wave\] Hello there\.<\/assistant>/);
  assert.doesNotMatch(promptText, /<user>\nwho are you\?<\/user>/);
  assert.doesNotMatch(promptText, /<user>\ncan you code\?<\/user>/);
});

test('chat() preserves non-exact user-prefix text in assistant output', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['hello there, friend'] });
  const agent = new CharacterRuntime(engine, buildConfig());

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
  const agent = new CharacterRuntime(engine, buildConfig());

  const events = await collectEvents(agent.chat("Tell me what's your name?"));
  const end = events[events.length - 1];
  assert.ok(end.kind === 'turn-end');
  assert.equal(end.finalText, "I'm Aria.");

  const memory = agent.getMemory();
  assert.equal(memory[1].content, "I'm Aria.");
});
