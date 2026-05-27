//////////////////////////////////////////////////////////////////////////////
//
// action-parser.test.ts
//
// - Exercises the streaming parser: chunk-boundary robustness, prose
//   coalescing, unknown-cue handling for the flat action model.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import type { ActionSchema } from './action-schema.js';
import {
  StreamingActionParser,
  type ParsedEvent,
} from './action-parser.js';

const SCHEMA: ActionSchema = [
  { id: 'wave' },
  { id: 'shake_head' },
  { id: 'look_at_you', cue: 'look at you' },
];

function drainAll(parser: StreamingActionParser, chunks: string[]): ParsedEvent[] {
  const events: ParsedEvent[] = [];
  for (const chunk of chunks) {
    events.push(...parser.consume(chunk));
  }
  events.push(...parser.flush());
  return events;
}

test('StreamingActionParser emits prose and action events in stream order', () => {
  const parser = new StreamingActionParser(SCHEMA);
  const events = drainAll(parser, ['Hello ', 'there!', '[wave]', ' all done.']);
  const action = events.find((event) => event.kind === 'action');
  assert.ok(action && action.kind === 'action' && action.id === 'wave');

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

test('StreamingActionParser resolves custom cue labels', () => {
  const parser = new StreamingActionParser(SCHEMA);
  const events = drainAll(parser, ['Hi! [look at you] nice to meet you.']);
  const action = events.find((event) => event.kind === 'action');
  assert.ok(action && action.kind === 'action');
  assert.equal(action.id, 'look_at_you');
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
  const joined = events
    .filter((event) => event.kind === 'prose')
    .map((event) => (event.kind === 'prose' ? event.text : ''))
    .join('');
  assert.equal(joined, 'abc');
  assert.ok(events.every((event) => event.kind === 'prose'));
});

test('StreamingActionParser tolerates cue boundaries split across chunks', () => {
  const parser = new StreamingActionParser(SCHEMA);
  const source = 'hello[look at you]world';
  const chunks: string[] = [];
  for (let index = 0; index < source.length; index += 3) {
    chunks.push(source.slice(index, index + 3));
  }
  const events = drainAll(parser, chunks);
  const action = events.find((event) => event.kind === 'action');
  assert.ok(action && action.kind === 'action');
  assert.equal(action.id, 'look_at_you');
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
  const first = parser.consume('hi [w');
  const actions = first.filter((event) => event.kind === 'action');
  assert.equal(actions.length, 0);
  const rest = parser.consume('ave] done');
  rest.push(...parser.flush());
  const action = rest.find((event) => event.kind === 'action');
  assert.ok(action && action.kind === 'action');
  assert.equal(action.id, 'wave');
});
