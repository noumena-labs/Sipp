const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

export interface AssetHashProvider {
  sha256Text(value: string): string;
  sha256Blob(blob: Blob, signal?: AbortSignal): Promise<string>;
}

function rightRotate(value: number, amount: number): number {
  return (value >>> amount) | (value << (32 - amount));
}

function toHex(value: number): string {
  return value.toString(16).padStart(8, '0');
}

export class Sha256 {
  private readonly state = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  private readonly buffer = new Uint8Array(64);
  private readonly words = new Uint32Array(64);
  private bufferLength = 0;
  private bytesHashed = 0;
  private finished = false;

  public update(data: Uint8Array): void {
    if (this.finished) {
      throw new Error('Cannot update a finalized hash.');
    }
    let position = 0;
    this.bytesHashed += data.byteLength;
    while (position < data.byteLength) {
      const take = Math.min(data.byteLength - position, 64 - this.bufferLength);
      this.buffer.set(data.subarray(position, position + take), this.bufferLength);
      this.bufferLength += take;
      position += take;
      if (this.bufferLength === 64) {
        this.processBlock(this.buffer);
        this.bufferLength = 0;
      }
    }
  }

  public digest(): string {
    if (!this.finished) {
      const bytesHashed = this.bytesHashed;
      this.buffer[this.bufferLength++] = 0x80;
      if (this.bufferLength > 56) {
        this.buffer.fill(0, this.bufferLength, 64);
        this.processBlock(this.buffer);
        this.bufferLength = 0;
      }
      this.buffer.fill(0, this.bufferLength, 56);

      const bitLengthHigh = Math.floor(bytesHashed / 0x20000000);
      const bitLengthLow = (bytesHashed << 3) >>> 0;
      this.buffer[56] = bitLengthHigh >>> 24;
      this.buffer[57] = bitLengthHigh >>> 16;
      this.buffer[58] = bitLengthHigh >>> 8;
      this.buffer[59] = bitLengthHigh;
      this.buffer[60] = bitLengthLow >>> 24;
      this.buffer[61] = bitLengthLow >>> 16;
      this.buffer[62] = bitLengthLow >>> 8;
      this.buffer[63] = bitLengthLow;
      this.processBlock(this.buffer);
      this.finished = true;
    }

    return Array.from(this.state, toHex).join('');
  }

  private processBlock(chunk: Uint8Array): void {
    const words = this.words;
    for (let i = 0; i < 16; i++) {
      const j = i * 4;
      words[i] =
        ((chunk[j] << 24) | (chunk[j + 1] << 16) | (chunk[j + 2] << 8) | chunk[j + 3]) >>> 0;
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rightRotate(words[i - 15], 7) ^ rightRotate(words[i - 15], 18) ^ (words[i - 15] >>> 3);
      const s1 = rightRotate(words[i - 2], 17) ^ rightRotate(words[i - 2], 19) ^ (words[i - 2] >>> 10);
      words[i] = (words[i - 16] + s0 + words[i - 7] + s1) >>> 0;
    }

    let a = this.state[0];
    let b = this.state[1];
    let c = this.state[2];
    let d = this.state[3];
    let e = this.state[4];
    let f = this.state[5];
    let g = this.state[6];
    let h = this.state[7];

    for (let i = 0; i < 64; i++) {
      const s1 = rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + s1 + ch + K[i] + words[i]) >>> 0;
      const s0 = rightRotate(a, 2) ^ rightRotate(a, 13) ^ rightRotate(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (s0 + maj) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    this.state[0] = (this.state[0] + a) >>> 0;
    this.state[1] = (this.state[1] + b) >>> 0;
    this.state[2] = (this.state[2] + c) >>> 0;
    this.state[3] = (this.state[3] + d) >>> 0;
    this.state[4] = (this.state[4] + e) >>> 0;
    this.state[5] = (this.state[5] + f) >>> 0;
    this.state[6] = (this.state[6] + g) >>> 0;
    this.state[7] = (this.state[7] + h) >>> 0;
  }
}

export function sha256Text(value: string): string {
  const hash = new Sha256();
  hash.update(new TextEncoder().encode(value));
  return hash.digest();
}

export async function sha256Blob(blob: Blob, signal?: AbortSignal): Promise<string> {
  if (signal?.aborted) {
    throw new DOMException('Hashing aborted.', 'AbortError');
  }
  const hash = new Sha256();
  const reader = blob.stream().getReader();
  try {
    while (true) {
      if (signal?.aborted) {
        throw new DOMException('Hashing aborted.', 'AbortError');
      }
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value != null && value.byteLength > 0) {
        hash.update(value);
      }
    }
  } finally {
    reader.releaseLock();
  }
  return hash.digest();
}
