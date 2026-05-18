import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createStreamingRingBuffer,
  StreamingRingReader,
  StreamingRingWriter,
} from './streaming-ring.js';

test('StreamingRingReader can drain one message at a time', () => {
  const sab = createStreamingRingBuffer(1024);
  const writer = new StreamingRingWriter(sab);
  const reader = new StreamingRingReader(sab);

  assert.equal(writer.tryWriteString(7, 'a'), true);
  assert.equal(writer.tryWriteString(7, 'b'), true);

  const first = reader.drain(1);
  assert.equal(first.length, 1);
  assert.equal(first[0].requestId, 7);
  assert.equal(first[0].sequence, 0);
  assert.equal(first[0].text, 'a');

  const second = reader.drain(1);
  assert.equal(second.length, 1);
  assert.equal(second[0].requestId, 7);
  assert.equal(second[0].sequence, 1);
  assert.equal(second[0].text, 'b');
});
