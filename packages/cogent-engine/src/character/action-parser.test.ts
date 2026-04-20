//////////////////////////////////////////////////////////////////////////////
//
// action-parser.test.ts
//
// - Exercises the streaming parser: chunk-boundary robustness, prose
//   coalescing, malformed-tag handling.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ActionParseError,
  parseActionTag,
  StreamingActionParser,
  type ParsedEvent,
} from './action-parser.js';

function drainAll(parser: StreamingActionParser, chunks: string[]): ParsedEvent[] {
  const events: ParsedEvent[] = [];
  for (const chunk of chunks) {
    events.push(...parser.consume(chunk));
  }
  events.push(...parser.flush());
  return events;
}

test('parseActionTag parses an argless tag', () => {
  const event = parseActionTag('<action name="idle"/>');
  assert.equal(event.kind, 'action');
  assert.equal(event.name, 'idle');
  assert.deepEqual(event.args, {});
});

test('parseActionTag parses a tag with a JSON args payload', () => {
  const event = parseActionTag('<action name="wave" args={"duration_ms":800}/>');
  assert.equal(event.name, 'wave');
  assert.deepEqual(event.args, { duration_ms: 800 });
});

test('parseActionTag rejects malformed tags', () => {
  assert.throws(() => parseActionTag('<action />'), (error) => error instanceof ActionParseError);
});

test('StreamingActionParser emits prose and action events in stream order', () => {
  const parser = new StreamingActionParser();
  const events = drainAll(parser, [
    'Hello ',
    'there!',
    '<action name="wave" args={"duration_ms":800}/>',
    ' all done.',
  ]);
  // Prose is streamed as it becomes unambiguous; we assert on ordering and
  // full reconstruction rather than exact chunking.
  const action = events.find((event) => event.kind === 'action');
  assert.ok(action && action.kind === 'action' && action.name === 'wave');

  // The action must appear after every byte of the leading prose and before
  // every byte of the trailing prose.
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

test('StreamingActionParser coalesces contiguous prose across chunks', () => {
  const parser = new StreamingActionParser();
  const events = drainAll(parser, ['a', 'b', 'c']);
  assert.equal(events.length, 1);
  assert.deepEqual(events[0], { kind: 'prose', text: 'abc' });
});

test('StreamingActionParser tolerates tag boundaries split across chunks', () => {
  const parser = new StreamingActionParser();
  const source = 'hello<action name="wave" args={"duration_ms":800}/>world';
  // Chunk every 3 characters to force boundaries inside the tag.
  const chunks: string[] = [];
  for (let index = 0; index < source.length; index += 3) {
    chunks.push(source.slice(index, index + 3));
  }
  const events = drainAll(parser, chunks);
  const action = events.find((event) => event.kind === 'action');
  assert.ok(action, 'action event must be emitted');
  if (action && action.kind === 'action') {
    assert.equal(action.name, 'wave');
    assert.deepEqual(action.args, { duration_ms: 800 });
  }
  const prose = events.filter((event) => event.kind === 'prose').map((event) =>
    event.kind === 'prose' ? event.text : ''
  );
  assert.equal(prose.join(''), 'helloworld');
});

test('StreamingActionParser surfaces unterminated tag as prose on flush', () => {
  const parser = new StreamingActionParser();
  const events = drainAll(parser, ['pre <action name="wave"']);
  const joined = events
    .filter((event) => event.kind === 'prose')
    .map((event) => (event.kind === 'prose' ? event.text : ''))
    .join('');
  assert.equal(joined, 'pre <action name="wave"');
});

test('StreamingActionParser defers prose that might start a tag', () => {
  const parser = new StreamingActionParser();
  // The trailing `<act` is ambiguous; the parser must hold back those bytes.
  const first = parser.consume('hi <act');
  const proseSoFar = first
    .filter((event) => event.kind === 'prose')
    .map((event) => (event.kind === 'prose' ? event.text : ''))
    .join('');
  // `<action` prefix length is 7, so up to 6 trailing bytes are deferred.
  // 'hi <act' is 7 bytes → only the first byte is safely emitted.
  assert.ok(proseSoFar.length < 'hi <act'.length, 'parser must hold back ambiguous bytes');
  assert.ok('hi <act'.startsWith(proseSoFar));
  // Completing with the rest of a valid tag resolves the ambiguity.
  const rest = parser.consume('ion name="idle"/>');
  const actions = rest.filter((event) => event.kind === 'action');
  assert.equal(actions.length, 1);
  const action = actions[0];
  if (action && action.kind === 'action') {
    assert.equal(action.name, 'idle');
  }
});
