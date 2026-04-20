//////////////////////////////////////////////////////////////////////////////
//
// action-bus.test.ts
//
// - Verifies typed dispatch, wildcard listeners, disposer semantics, and
//   that a throwing listener doesn't prevent siblings from running.
//
//////////////////////////////////////////////////////////////////////////////

import assert from 'node:assert/strict';
import test from 'node:test';

import { ActionBus, type CharacterEvent } from './action-bus.js';

test('ActionBus delivers events to the correct typed listener', () => {
  const bus = new ActionBus();
  const seen: string[] = [];
  bus.on('prose', (event) => {
    seen.push(`prose:${event.text}`);
  });
  bus.on('action', (event) => {
    seen.push(`action:${event.name}`);
  });

  bus.emit({ kind: 'prose', text: 'hi' });
  bus.emit({ kind: 'action', name: 'wave', args: {}, raw: '<action name="wave"/>' });

  assert.deepEqual(seen, ['prose:hi', 'action:wave']);
});

test('ActionBus wildcard listener receives every event', () => {
  const bus = new ActionBus();
  const events: CharacterEvent[] = [];
  bus.onAny((event) => {
    events.push(event);
  });
  bus.emit({ kind: 'turn-start', userMessage: 'hello' });
  bus.emit({ kind: 'turn-end', finalText: 'hi', cancelled: false });
  assert.equal(events.length, 2);
  assert.equal(events[0].kind, 'turn-start');
  assert.equal(events[1].kind, 'turn-end');
});

test('ActionBus disposer removes the listener', () => {
  const bus = new ActionBus();
  let count = 0;
  const dispose = bus.on('prose', () => {
    count += 1;
  });
  bus.emit({ kind: 'prose', text: 'a' });
  dispose();
  bus.emit({ kind: 'prose', text: 'b' });
  assert.equal(count, 1);
});

test('ActionBus continues dispatching when one listener throws', () => {
  const originalError = console.error;
  const errors: string[] = [];
  console.error = (...args) => {
    errors.push(args.join(' '));
  };
  try {
    const bus = new ActionBus();
    const events: string[] = [];
    bus.on('prose', () => {
      throw new Error('boom');
    });
    bus.on('prose', (event) => {
      events.push(event.text);
    });
    bus.emit({ kind: 'prose', text: 'ok' });
    assert.deepEqual(events, ['ok']);
    assert.ok(errors.some((entry) => entry.includes('boom')));
  } finally {
    console.error = originalError;
  }
});

test('ActionBus clear() removes every listener', () => {
  const bus = new ActionBus();
  let hits = 0;
  bus.on('prose', () => {
    hits += 1;
  });
  bus.onAny(() => {
    hits += 1;
  });
  bus.clear();
  bus.emit({ kind: 'prose', text: 'x' });
  assert.equal(hits, 0);
});
