// SAB-backed ring buffer for worker→main token streaming.
//
// Wire: [u32 LE requestId | u32 LE textLength | utf8 bytes].  Messages can
// straddle the body wrap.  Header (8×i32 = 32B):
//   [0] writeIndex (monotonic, atomic)
//   [1] readIndex  (monotonic, atomic)
//   [2] capacity   (= body byte length, constant)
//   [3] dropCount  (atomic; bumped when a record won't fit)
// Indices are interpreted modulo capacity.  `(writeIndex - readIndex) | 0`
// gives in-flight bytes (capacity ≪ 2^31, so wrap is safe).

const HEADER_INTS = 8;
const HEADER_BYTES = HEADER_INTS * 4;
const HEADER_WRITE_INDEX = 0;
const HEADER_READ_INDEX = 1;
const HEADER_CAPACITY = 2;
const HEADER_DROP_COUNT = 3;

const MESSAGE_PREFIX_BYTES = 8; // u32 requestId + u32 textLength

export const DEFAULT_STREAMING_RING_CAPACITY = 64 * 1024;

export interface StreamingRingMessage {
  requestId: number;
  text: string;
}

// Allocates a fresh SAB streaming ring; returns the SAB for postMessage.
export function createStreamingRingBuffer(
  bodyCapacityBytes: number = DEFAULT_STREAMING_RING_CAPACITY
): SharedArrayBuffer {
  if (typeof SharedArrayBuffer === 'undefined') {
    throw new Error(
      'SharedArrayBuffer is not available in this context. Streaming requires cross-origin isolation (COOP/COEP).'
    );
  }
  if (!Number.isInteger(bodyCapacityBytes) || bodyCapacityBytes <= 0) {
    throw new RangeError(
      `Streaming ring capacity must be a positive integer, got ${bodyCapacityBytes}.`
    );
  }
  const sab = new SharedArrayBuffer(HEADER_BYTES + bodyCapacityBytes);
  const header = new Int32Array(sab, 0, HEADER_INTS);
  Atomics.store(header, HEADER_CAPACITY, bodyCapacityBytes);
  return sab;
}

// Worker-side writer.  Never blocks; overflow bumps `dropCount`.
export class StreamingRingWriter {
  private readonly header: Int32Array;
  private readonly body: Uint8Array;
  private readonly capacity: number;
  private readonly encoder: TextEncoder;

  public constructor(sab: SharedArrayBuffer) {
    this.header = new Int32Array(sab, 0, HEADER_INTS);
    this.body = new Uint8Array(sab, HEADER_BYTES);
    this.capacity = this.body.byteLength;
    this.encoder = new TextEncoder();
    if (Atomics.load(this.header, HEADER_CAPACITY) !== this.capacity) {
      throw new Error('Streaming ring header capacity does not match body length.');
    }
  }

  // Writes pre-encoded UTF-8 bytes.  Returns false on overflow.
  public tryWriteBytes(requestId: number, payload: Uint8Array): boolean {
    const messageSize = MESSAGE_PREFIX_BYTES + payload.byteLength;
    if (messageSize > this.capacity) {
      Atomics.add(this.header, HEADER_DROP_COUNT, 1);
      return false;
    }
    const writeIndex = Atomics.load(this.header, HEADER_WRITE_INDEX);
    const readIndex = Atomics.load(this.header, HEADER_READ_INDEX);
    if (((writeIndex - readIndex) | 0) + messageSize > this.capacity) {
      Atomics.add(this.header, HEADER_DROP_COUNT, 1);
      return false;
    }
    const offset = ((writeIndex % this.capacity) + this.capacity) % this.capacity;
    this.writeU32(offset, requestId >>> 0);
    this.writeU32((offset + 4) % this.capacity, payload.byteLength >>> 0);
    this.writeBytes((offset + MESSAGE_PREFIX_BYTES) % this.capacity, payload);
    // Single atomic store publishes the new bytes to the reader.
    Atomics.store(
      this.header,
      HEADER_WRITE_INDEX,
      ((writeIndex + messageSize) | 0)
    );
    return true;
  }

  public tryWriteString(requestId: number, text: string): boolean {
    return this.tryWriteBytes(requestId, this.encoder.encode(text));
  }

  public dropCount(): number {
    return Atomics.load(this.header, HEADER_DROP_COUNT);
  }

  private writeU32(offset: number, value: number): void {
    this.body[offset] = value & 0xff;
    this.body[(offset + 1) % this.capacity] = (value >>> 8) & 0xff;
    this.body[(offset + 2) % this.capacity] = (value >>> 16) & 0xff;
    this.body[(offset + 3) % this.capacity] = (value >>> 24) & 0xff;
  }

  private writeBytes(offset: number, src: Uint8Array): void {
    const tail = this.capacity - offset;
    if (src.byteLength <= tail) {
      this.body.set(src, offset);
      return;
    }
    this.body.set(src.subarray(0, tail), offset);
    this.body.set(src.subarray(tail), 0);
  }
}

// Main-side reader.  Non-blocking drain; per-request TextDecoder state
// preserved across calls so multi-byte codepoints stitch correctly.
export class StreamingRingReader {
  private readonly header: Int32Array;
  private readonly body: Uint8Array;
  private readonly capacity: number;
  private readonly decoders = new Map<number, TextDecoder>();
  private lastDropCount = 0;

  public constructor(sab: SharedArrayBuffer) {
    this.header = new Int32Array(sab, 0, HEADER_INTS);
    this.body = new Uint8Array(sab, HEADER_BYTES);
    this.capacity = this.body.byteLength;
    if (Atomics.load(this.header, HEADER_CAPACITY) !== this.capacity) {
      throw new Error('Streaming ring header capacity does not match body length.');
    }
  }

  public drain(): StreamingRingMessage[] {
    const writeIndex = Atomics.load(this.header, HEADER_WRITE_INDEX);
    const readIndex = Atomics.load(this.header, HEADER_READ_INDEX);
    const available = (writeIndex - readIndex) | 0;
    if (available <= 0) {
      return [];
    }
    const messages: StreamingRingMessage[] = [];
    let cursor = readIndex;
    let consumed = 0;
    while (consumed < available) {
      if (available - consumed < MESSAGE_PREFIX_BYTES) {
        break;
      }
      const offset = ((cursor % this.capacity) + this.capacity) % this.capacity;
      const requestId = this.readU32(offset);
      const textLength = this.readU32((offset + 4) % this.capacity);
      const messageSize = MESSAGE_PREFIX_BYTES + textLength;
      if (available - consumed < messageSize) {
        break;
      }
      const payload = this.readBytes(
        (offset + MESSAGE_PREFIX_BYTES) % this.capacity,
        textLength
      );
      // `stream: true` so multi-byte codepoints stitch across drains.
      const text = this.decoderFor(requestId).decode(payload, { stream: true });
      messages.push({ requestId, text });
      cursor = (cursor + messageSize) | 0;
      consumed += messageSize;
    }
    if (consumed > 0) {
      Atomics.store(this.header, HEADER_READ_INDEX, cursor);
    }
    return messages;
  }

  // Returns drops since the last call.  Internal counter is advanced.
  public consumeDropDelta(): number {
    const total = Atomics.load(this.header, HEADER_DROP_COUNT);
    const delta = (total - this.lastDropCount) | 0;
    this.lastDropCount = total;
    return delta;
  }

  // Drops a request's TextDecoder.  Call when the request settles.
  public forgetRequest(requestId: number): void {
    this.decoders.delete(requestId);
  }

  private decoderFor(requestId: number): TextDecoder {
    let decoder = this.decoders.get(requestId);
    if (decoder == null) {
      decoder = new TextDecoder('utf-8', { fatal: false });
      this.decoders.set(requestId, decoder);
    }
    return decoder;
  }

  private readU32(offset: number): number {
    return (
      this.body[offset] |
      (this.body[(offset + 1) % this.capacity] << 8) |
      (this.body[(offset + 2) % this.capacity] << 16) |
      (this.body[(offset + 3) % this.capacity] << 24)
    ) >>> 0;
  }

  private readBytes(offset: number, length: number): Uint8Array {
    // `slice` (not subarray) so the writer wrapping over this region
    // doesn't mutate bytes still held by the TextDecoder.
    const tail = this.capacity - offset;
    if (length <= tail) {
      return this.body.slice(offset, offset + length);
    }
    const out = new Uint8Array(length);
    out.set(this.body.subarray(offset, offset + tail), 0);
    out.set(this.body.subarray(0, length - tail), tail);
    return out;
  }
}
