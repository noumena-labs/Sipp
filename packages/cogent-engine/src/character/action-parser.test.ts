//////////////////////////////////////////////////////////////////////////////
//
// action-parser.test.ts
//
// - Exercises the streaming parser: chunk-boundary robustness, prose
//   coalescing, unknown-cue handling.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import type { ActionSchema } from './action-schema.js';
import { expandActionCues } from './action-schema.js';
import {
  ActionParseError,
  parseActionCue,
  StreamingActionParser,
  type ParsedEvent,
} from './action-parser.js';

const SCHEMA: ActionSchema = {
  actions: [
    { name: 'wave', args: [] },
    { name: 'shake_head', args: [] },
    {
      name: 'set_mood',
      args: [{ name: 'mood', type: 'enum', values: ['happy', 'sad'] }],
    },
  ],
};

const CUES = expandActionCues(SCHEMA);

function drainAll(parser: StreamingActionParser, chunks: string[]): ParsedEvent[] {
  const events: ParsedEvent[] = [];
  for (const chunk of chunks) {
    events.push(...parser.consume(chunk));
  }
  events.push(...parser.flush());
  return events;
}

test('parseActionCue resolves a bare cue label', () => {
  const event = parseActionCue('[wave]', CUES);
  assert.equal(event.kind, 'action');
  assert.equal(event.name, 'wave');
  assert.deepEqual(event.args, {});
});

test('parseActionCue resolves an enum-valued cue label', () => {
  const event = parseActionCue('[mood: happy]', CUES);
  assert.equal(event.name, 'set_mood');
  assert.deepEqual(event.args, { mood: 'happy' });
});

test('parseActionCue rejects unknown labels', () => {
  assert.throws(
    () => parseActionCue('[look at moon]', CUES),
    (error) => error instanceof ActionParseError
  );
});

test('parseActionCue rejects malformed envelopes', () => {
  assert.throws(
    () => parseActionCue('wave', CUES),
    (error) => error instanceof ActionParseError
  );
});

test('StreamingActionParser emits prose and action events in stream order', () => {
  const parser = new StreamingActionParser(SCHEMA);
  const events = drainAll(parser, [
    'Hello ',
    'there!',
    '[wave]',
    ' all done.',
  ]);
  const action = events.find((event) => event.kind === 'action');
  assert.ok(action && action.kind === 'action' && action.name === 'wave');

  const actionIndex = events.indexOf(action);
  const leading = events
    .slice(0, actionIndex)
    .filter((event) => event.kind === 'prose')
    .map((event) => (event.kind === 'prose' ? event.text : ''))
    .join('');
  const trailing = events
    .slice(actionIndex + 1)
    .filter((event) => event.kind === 'prose')
    .map((event) => (event.kind === 'prose' ? event.text : ''))
    .join('');
  assert.equal(leading, 'Hello there!');
  assert.equal(trailing, ' all done.');
});

test('StreamingActionParser resolves enum-valued cues', () => {
  const parser = new StreamingActionParser(SCHEMA);
  const events = drainAll(parser, ['Hi! [mood: happy] nice to meet you.']);
  const action = events.find((event) => event.kind === 'action');
  assert.ok(action && action.kind === 'action');
  assert.equal(action.name, 'set_mood');
  assert.deepEqual(action.args, { mood: 'happy' });
});

test('StreamingActionParser coalesces contiguous prose within a single chunk', () => {
  const parser = new StreamingActionParser(SCHEMA);
  const events = drainAll(parser, ['abc']);
  assert.equal(events.length, 1);
  assert.deepEqual(events[0], { kind: 'prose', text: 'abc' });
});

test('StreamingActionParser preserves stream order across chunks without losing prose', () => {
  const parser = new StreamingActionParser(SCHEMA);
  const events = drainAll(parser, ['a', 'b', 'c']);
  // Each chunk may flush as its own prose event — the invariant is that
  // the concatenated prose equals the full input and ordering is preserved.
  const joined = events
    .filter((event) => event.kind === 'prose')
    .map((event) => (event.kind === 'prose' ? event.text : ''))
    .join('');
  assert.equal(joined, 'abc');
  assert.ok(events.every((event) => event.kind === 'prose'));
});

test('StreamingActionParser tolerates cue boundaries split across chunks', () => {
  const parser = new StreamingActionParser(SCHEMA);
  const source = 'hello[mood: sad]world';
  const chunks: string[] = [];
  for (let index = 0; index < source.length; index += 3) {
    chunks.push(source.slice(index, index + 3));
  }
  const events = drainAll(parser, chunks);
  const action = events.find((event) => event.kind === 'action');
  assert.ok(action && action.kind === 'action');
  assert.equal(action.name, 'set_mood');
  assert.deepEqual(action.args, { mood: 'sad' });
  const prose = events
    .filter((event) => event.kind === 'prose')
    .map((event) => (event.kind === 'prose' ? event.text : ''))
    .join('');
  assert.equal(prose, 'helloworld');
});

test('StreamingActionParser surfaces unterminated cue as prose on flush', () => {
  const parser = new StreamingActionParser(SCHEMA);
  const events = drainAll(parser, ['pre [wave']);
  const joined = events
    .filter((event) => event.kind === 'prose')
    .map((event) => (event.kind === 'prose' ? event.text : ''))
    .join('');
  assert.equal(joined, 'pre [wave');
});

test('StreamingActionParser surfaces unknown cues as prose verbatim', () => {
  const parser = new StreamingActionParser(SCHEMA);
  const events = drainAll(parser, ['hi [look at moon] there']);
  const actions = events.filter((event) => event.kind === 'action');
  assert.equal(actions.length, 0);
  const joined = events
    .filter((event) => event.kind === 'prose')
    .map((event) => (event.kind === 'prose' ? event.text : ''))
    .join('');
  assert.equal(joined, 'hi [look at moon] there');
});

test('StreamingActionParser defers bytes that might start a cue', () => {
  const parser = new StreamingActionParser(SCHEMA);
  // The trailing `[w` is an unfinished cue — the parser must hold those
  // bytes until the closing `]` arrives (or flush is called).
  const first = parser.consume('hi [w');
  const actions = first.filter((event) => event.kind === 'action');
  assert.equal(actions.length, 0);
  const rest = parser.consume('ave] done');
  rest.push(...parser.flush());
  const action = rest.find((event) => event.kind === 'action');
  assert.ok(action && action.kind === 'action');
  assert.equal(action.name, 'wave');
});
