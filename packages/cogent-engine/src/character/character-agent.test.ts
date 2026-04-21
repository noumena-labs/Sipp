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

const baseActions = {
  actions: [
    { name: 'wave', description: 'wave hello', args: [] },
  ],
} as const;

function buildConfig(overrides: Partial<CharacterConfig> = {}): CharacterConfig {
  return {
    id: 'aria-01',
    persona: { name: 'Aria', description: 'A friendly guide.' },
    actions: baseActions,
    ...overrides,
  };
}

interface ScriptedResponse {
  readonly tokens: readonly string[];
  readonly response?: Partial<GenerateResponse>;
  readonly throwOnRun?: Error;
  readonly abortBeforeRun?: AbortController;
}

interface FakeEngine extends CharacterAgentEngine {
  readonly queuePromptCalls: Array<{
    contextKey: string;
    promptText: string;
    options: PromptOptions | number | undefined;
  }>;
  readonly runCalls: GenerateRequestId[];
  readonly cancelCalls: GenerateRequestId[];
  enqueue(script: ScriptedResponse): void;
}

function createFakeEngine(): FakeEngine {
  const queuePromptCalls: FakeEngine['queuePromptCalls'] = [];
  const runCalls: GenerateRequestId[] = [];
  const cancelCalls: GenerateRequestId[] = [];
  const scripts: ScriptedResponse[] = [];
  const pendingCallbacks: Array<((token: string) => void) | undefined> = [];
  let nextId = 1;

  return {
    queuePromptCalls,
    runCalls,
    cancelCalls,
    enqueue(script: ScriptedResponse) {
      scripts.push(script);
    },
    getChatTemplate(): string | null {
      // Minimal ChatML signature so sniffChatFormat() returns 'chatml'.
      return '<|im_start|>system<|im_end|>';
    },
    getBosText(): string {
      return '';
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
      const script = scripts.shift();
      const onToken = pendingCallbacks.shift();
      if (!script) {
        throw new Error(`No scripted response enqueued for request ${requestId}`);
      }
      if (script.throwOnRun) {
        throw script.throwOnRun;
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
  assert.equal(call.contextKey, 'aria-01::turn-1');
  assert.ok(typeof call.options === 'object' && call.options != null);
  const opts = call.options as PromptOptions;
  assert.equal(opts.nTokens, 42);
  // Grammar is compiled from the action schema and forwarded to the engine.
  assert.equal(typeof opts.grammar, 'string');
  assert.ok(opts.grammar && opts.grammar.includes('root'));
  assert.equal(typeof opts.onToken, 'function');
});

test('chat() uses a fresh contextKey for each turn', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['first reply'] });
  engine.enqueue({ tokens: ['second reply'] });
  const agent = new CharacterAgent(engine, buildConfig());

  await collectEvents(agent.chat('first'));
  await collectEvents(agent.chat('second'));

  assert.equal(engine.queuePromptCalls.length, 2);
  assert.equal(engine.queuePromptCalls[0].contextKey, 'aria-01::turn-1');
  assert.equal(engine.queuePromptCalls[1].contextKey, 'aria-01::turn-2');
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

test('assistant memory stores prose only when actions are interleaved', async () => {
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
  assert.equal(memory[1].content, 'Hello  there.');
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
  // Best-effort cancel should have been attempted.
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
  const agent = new CharacterAgent(
    engine,
    buildConfig({ memory: { maxTurns: 0 } })
  );
  await collectEvents(agent.chat('a'));
  await collectEvents(agent.chat('b'));
  assert.equal(agent.getMemory().length, 0);
});

test('memory sliding window drops oldest pairs past maxTurns', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['r1'] });
  engine.enqueue({ tokens: ['r2'] });
  engine.enqueue({ tokens: ['r3'] });
  const agent = new CharacterAgent(
    engine,
    buildConfig({ memory: { maxTurns: 2 } })
  );
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

test('chat() passes a rendered raw prompt to queuePrompt', async () => {
  const engine = createFakeEngine();
  engine.enqueue({ tokens: ['hi'] });
  const agent = new CharacterAgent(engine, buildConfig());
  await collectEvents(agent.chat('hello'));

  const call = engine.queuePromptCalls[0];
  // Should be the ChatML-rendered conversation, not empty/delta.
  assert.ok(typeof call.promptText === 'string' && call.promptText.length > 0);
  assert.ok(call.promptText.includes('<|im_start|>system'));
  assert.ok(call.promptText.includes('Aria'));
  assert.ok(call.promptText.includes('<|im_start|>user'));
  assert.ok(call.promptText.includes('hello'));
  // Trailing assistant header should be present to cue generation.
  assert.ok(call.promptText.endsWith('<|im_start|>assistant\n'));

  assert.ok(typeof call.options === 'object' && call.options != null);
  const opts = call.options as PromptOptions;
  // Messages path is gone; runtime must receive 'raw' so it does not double-wrap.
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
  // System + first user + first assistant + second user, in that order.
  const idxSystem = rendered.indexOf('<|im_start|>system');
  const idxFirstUser = rendered.indexOf('first question');
  const idxAssistant = rendered.indexOf('first reply');
  const idxSecondUser = rendered.indexOf('second question');
  assert.ok(idxSystem >= 0 && idxFirstUser > idxSystem);
  assert.ok(idxAssistant > idxFirstUser);
  assert.ok(idxSecondUser > idxAssistant);
});
