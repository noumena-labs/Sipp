import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SharedTokenRingReader,
  type SharedTokenRingDescriptor,
} from '../../src/runtime/shared-token-ring.js';

const HEADER_INTS = 8;
const HEADER_BYTES = HEADER_INTS * 4;
const HEADER_WRITE_INDEX = 0;
const HEADER_READ_INDEX = 1;
const HEADER_CAPACITY = 2;
const HEADER_DROP_COUNT = 3;
const RECORD_HEADER_BYTES = 16;

test('SharedTokenRingReader drains native token records', () => {
  const ring = createTestRing(128);
  const reader = new SharedTokenRingReader(ring.descriptor);

  writeRecord(ring, 7, 0, 2, 'hi');
  writeRecord(ring, 7, 2, 1, '!');

  const records = collectRecords(reader);
  assert.deepEqual(records, [
    {
      streamId: 7,
      sequenceStart: 0,
      frameCount: 2,
      byteCount: 2,
      text: 'hi',
    },
    {
      streamId: 7,
      sequenceStart: 2,
      frameCount: 1,
      byteCount: 1,
      text: '!',
    },
  ]);
});

test('SharedTokenRingReader decodes shared payloads through normal array buffers', () => {
  const OriginalTextDecoder = globalThis.TextDecoder;
  class BrowserStrictTextDecoder extends OriginalTextDecoder {
    public override decode(
      input?: Parameters<TextDecoder['decode']>[0],
      options?: TextDecodeOptions
    ): string {
      if (
        input instanceof Uint8Array &&
        typeof SharedArrayBuffer !== 'undefined' &&
        input.buffer instanceof SharedArrayBuffer
      ) {
        throw new TypeError('Shared payload view passed to TextDecoder');
      }
      return super.decode(input, options);
    }
  }

  globalThis.TextDecoder = BrowserStrictTextDecoder;
  try {
    const ring = createTestRing(128);
    const reader = new SharedTokenRingReader(ring.descriptor);

    writeRecord(ring, 9, 0, 1, 'shared');

    assert.equal(collectRecords(reader)[0].text, 'shared');
  } finally {
    globalThis.TextDecoder = OriginalTextDecoder;
  }
});

test('SharedTokenRingReader drains records that wrap the body', () => {
  const ring = createTestRing(40);
  const reader = new SharedTokenRingReader(ring.descriptor);

  writeRecord(ring, 1, 0, 1, 'abcdefghij');
  assert.equal(collectRecords(reader)[0].text, 'abcdefghij');
  writeRecord(ring, 1, 1, 1, 'klmnopqrst');

  const records = collectRecords(reader);
  assert.equal(records.length, 1);
  assert.equal(records[0].sequenceStart, 1);
  assert.equal(records[0].text, 'klmnopqrst');
});

test('SharedTokenRingReader reports drop deltas', () => {
  const ring = createTestRing(64);
  const reader = new SharedTokenRingReader(ring.descriptor);

  Atomics.add(ring.header, HEADER_DROP_COUNT, 2);
  assert.equal(reader.consumeDropDelta(), 2);
  assert.equal(reader.consumeDropDelta(), 0);
});

interface TestRing {
  readonly descriptor: SharedTokenRingDescriptor;
  readonly header: Int32Array;
  readonly body: Uint8Array;
}

function collectRecords(reader: SharedTokenRingReader) {
  const records: Array<{
    streamId: number;
    sequenceStart: number;
    frameCount: number;
    byteCount: number;
    text: string;
  }> = [];
  reader.drain((streamId, sequenceStart, frameCount, byteCount, text) => {
    records.push({ streamId, sequenceStart, frameCount, byteCount, text });
  });
  return records;
}

function createTestRing(bodyCapacity: number): TestRing {
  const buffer = new SharedArrayBuffer(HEADER_BYTES + bodyCapacity);
  const header = new Int32Array(buffer, 0, HEADER_INTS);
  Atomics.store(header, HEADER_CAPACITY, bodyCapacity);
  return {
    descriptor: {
      buffer,
      headerOffset: 0,
      bodyOffset: HEADER_BYTES,
      bodyCapacity,
    },
    header,
    body: new Uint8Array(buffer, HEADER_BYTES, bodyCapacity),
  };
}

function writeRecord(
  ring: TestRing,
  streamId: number,
  sequenceStart: number,
  frameCount: number,
  text: string
): void {
  const bytes = new TextEncoder().encode(text);
  const writeIndex = Atomics.load(ring.header, HEADER_WRITE_INDEX);
  const offset = positiveModulo(writeIndex, ring.body.byteLength);
  writeU32(ring.body, offset, streamId);
  writeU32(ring.body, offset + 4, sequenceStart);
  writeU32(ring.body, offset + 8, frameCount);
  writeU32(ring.body, offset + 12, bytes.byteLength);
  writeBytes(ring.body, offset + RECORD_HEADER_BYTES, bytes);
  Atomics.store(
    ring.header,
    HEADER_WRITE_INDEX,
    (writeIndex + RECORD_HEADER_BYTES + bytes.byteLength) | 0
  );
}

function writeU32(body: Uint8Array, offset: number, value: number): void {
  const capacity = body.byteLength;
  const index = positiveModulo(offset, capacity);
  body[index] = value & 0xff;
  body[(index + 1) % capacity] = (value >>> 8) & 0xff;
  body[(index + 2) % capacity] = (value >>> 16) & 0xff;
  body[(index + 3) % capacity] = (value >>> 24) & 0xff;
}

function writeBytes(body: Uint8Array, offset: number, bytes: Uint8Array): void {
  const capacity = body.byteLength;
  const index = positiveModulo(offset, capacity);
  const tail = capacity - index;
  if (bytes.byteLength <= tail) {
    body.set(bytes, index);
    return;
  }
  body.set(bytes.subarray(0, tail), index);
  body.set(bytes.subarray(tail), 0);
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}
