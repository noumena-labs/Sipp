import test from 'node:test';
import assert from 'node:assert/strict';
import { PairingValidator } from './pairing-validator.js';

const GGUF_MAGIC = 0x46554747;
const GGUF_VERSION = 3;
const GGUF_STRING = 8;
const GGUF_BOOL = 7;

function ggufFile(
  name: string,
  metadata: Record<string, string | boolean>
): File {
  const encoder = new TextEncoder();
  const chunks: Uint8Array[] = [];

  const pushU8 = (value: number): void => {
    chunks.push(Uint8Array.of(value & 0xff));
  };

  const pushU32 = (value: number): void => {
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setUint32(0, value, true);
    chunks.push(bytes);
  };

  const pushU64 = (value: number): void => {
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setBigUint64(0, BigInt(value), true);
    chunks.push(bytes);
  };

  const pushString = (value: string): void => {
    const bytes = encoder.encode(value);
    pushU64(bytes.byteLength);
    chunks.push(bytes);
  };

  pushU32(GGUF_MAGIC);
  pushU32(GGUF_VERSION);
  pushU64(0);
  pushU64(Object.keys(metadata).length);

  for (const [key, value] of Object.entries(metadata)) {
    pushString(key);
    if (typeof value === 'string') {
      pushU32(GGUF_STRING);
      pushString(value);
      continue;
    }
    pushU32(GGUF_BOOL);
    pushU8(value ? 1 : 0);
  }

  return new File(chunks, name);
}

test('PairingValidator accepts explicit projectors for LFM VL base models', async () => {
  const validator = new PairingValidator();
  const base = await validator.classify(
    'asset-model',
    ggufFile('base.gguf', {
      'general.architecture': 'lfm2',
      'clip.has_vision_encoder': true,
    })
  );
  const projector = await validator.classify(
    'asset-projector',
    ggufFile('mmproj.gguf', {
      'general.architecture': 'clip',
      'clip.projector_type': 'lfm2',
      'clip.has_vision_encoder': true,
    })
  );

  const plan = validator.resolve([base, projector], projector.assetId);
  assert.equal(plan.modality, 'vision');
  assert.equal(plan.status, 'ready');
  assert.equal(plan.projectorAssetId, projector.assetId);
});

test('PairingValidator detects LFM VL bases from filename when GGUF metadata is unavailable', async () => {
  const validator = new PairingValidator();
  const base = await validator.classify(
    'asset-model',
    new File(['not-a-gguf'], 'LFM 2.5 VL 1.6B Q8_0.gguf')
  );
  const projector = await validator.classify(
    'asset-projector',
    new File(['not-a-gguf'], 'mmproj-LFM2-VL-1.6B-f16.gguf')
  );

  const plan = validator.resolve([base, projector], projector.assetId);
  assert.equal(plan.modality, 'vision');
  assert.equal(plan.status, 'ready');
});
