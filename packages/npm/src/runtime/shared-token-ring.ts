const HEADER_INTS = 8;
const HEADER_WRITE_INDEX = 0;
const HEADER_READ_INDEX = 1;
const HEADER_CAPACITY = 2;
const HEADER_DROP_COUNT = 3;
const RECORD_HEADER_BYTES = 16;

export interface SharedTokenRingDescriptor {
  readonly buffer: SharedArrayBuffer | ArrayBuffer;
  readonly headerOffset: number;
  readonly bodyOffset: number;
  readonly bodyCapacity: number;
}

export type SharedTokenRingRecordConsumer = (
  streamId: number,
  sequenceStart: number,
  frameCount: number,
  byteCount: number,
  text: string
) => void;

export class SharedTokenRingReader {
  private readonly header: Int32Array;
  private readonly body: Uint8Array;
  private readonly isShared: boolean;
  private readonly decoder = new TextDecoder('utf-8', { fatal: false });
  private lastDropCount = 0;

  public constructor(descriptor: SharedTokenRingDescriptor) {
    this.header = new Int32Array(descriptor.buffer, descriptor.headerOffset, HEADER_INTS);
    this.body = new Uint8Array(
      descriptor.buffer,
      descriptor.bodyOffset,
      descriptor.bodyCapacity
    );
    this.isShared =
      typeof SharedArrayBuffer !== 'undefined' &&
      descriptor.buffer instanceof SharedArrayBuffer;
  }

  public drain(
    consume: SharedTokenRingRecordConsumer,
    maxRecords: number = Number.POSITIVE_INFINITY
  ): number {
    const writeIndex = this.loadHeader(HEADER_WRITE_INDEX);
    const readIndex = this.loadHeader(HEADER_READ_INDEX);
    const available = (writeIndex - readIndex) | 0;
    if (available <= 0) {
      return 0;
    }

    let cursor = readIndex;
    let consumed = 0;
    let recordCount = 0;
    while (consumed < available && recordCount < maxRecords) {
      if (available - consumed < RECORD_HEADER_BYTES) {
        break;
      }

      const offset = positiveModulo(cursor, this.body.byteLength);
      const streamId = this.readU32(offset);
      const sequenceStart = this.readU32(offset + 4);
      const frameCount = this.readU32(offset + 8);
      const byteCount = this.readU32(offset + 12);
      const recordSize = RECORD_HEADER_BYTES + byteCount;
      if (available - consumed < recordSize) {
        break;
      }

      const payload = this.readBytes(offset + RECORD_HEADER_BYTES, byteCount);
      consume(streamId, sequenceStart, frameCount, byteCount, this.decoder.decode(payload));
      cursor = (cursor + recordSize) | 0;
      consumed += recordSize;
      recordCount += 1;
    }

    if (consumed > 0) {
      this.storeHeader(HEADER_READ_INDEX, cursor);
    }
    return recordCount;
  }

  public consumeDropDelta(): number {
    const total = this.loadHeader(HEADER_DROP_COUNT);
    const delta = (total - this.lastDropCount) | 0;
    this.lastDropCount = total;
    return delta;
  }

  public capacity(): number {
    return this.loadHeader(HEADER_CAPACITY);
  }

  private loadHeader(index: number): number {
    return this.isShared ? Atomics.load(this.header, index) : this.header[index];
  }

  private storeHeader(index: number, value: number): void {
    if (this.isShared) {
      Atomics.store(this.header, index, value);
      return;
    }
    this.header[index] = value;
  }

  private readU32(offset: number): number {
    const capacity = this.body.byteLength;
    const index = positiveModulo(offset, capacity);
    return (
      this.body[index] |
      (this.body[(index + 1) % capacity] << 8) |
      (this.body[(index + 2) % capacity] << 16) |
      (this.body[(index + 3) % capacity] << 24)
    ) >>> 0;
  }

  private readBytes(offset: number, length: number): Uint8Array {
    const capacity = this.body.byteLength;
    const index = positiveModulo(offset, capacity);
    const tail = capacity - index;
    if (length <= tail) {
      return this.body.subarray(index, index + length);
    }

    const out = new Uint8Array(length);
    out.set(this.body.subarray(index), 0);
    out.set(this.body.subarray(0, length - tail), tail);
    return out;
  }
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}
