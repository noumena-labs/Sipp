import test from 'node:test';
import assert from 'node:assert/strict';
import { ModelAssetClassifier } from './model-asset-classifier.js';

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

test('ModelAssetClassifier detects LFM vision base metadata', async () => {
  const classifier = new ModelAssetClassifier();

  const base = await classifier.classify(
    'asset-model',
    ggufFile('base.gguf', {
      'general.architecture': 'lfm2',
      'clip.has_vision_encoder': true,
    })
  );

  assert.equal(base.inspection.role, 'model');
  assert.equal(base.inspection.visionCapable, true);
  assert.deepEqual(base.inspection.compatibleVisionProjectorTypes, ['lfm2']);
});

test('ModelAssetClassifier detects projector metadata without filename fallback', async () => {
  const classifier = new ModelAssetClassifier();

  const projector = await classifier.classify(
    'asset-projector',
    ggufFile('mmproj.gguf', {
      'general.architecture': 'clip',
      'clip.projector_type': 'lfm2',
      'clip.has_vision_encoder': true,
    })
  );
  const namedProjector = await classifier.classify(
    'asset-named-projector',
    new File(['not-a-gguf'], 'mmproj-LFM2-VL-1.6B-f16.gguf')
  );

  assert.equal(projector.inspection.role, 'projector');
  assert.equal(projector.inspection.providedVisionProjectorType, 'lfm2');
  assert.equal(namedProjector.inspection.role, 'unknown');
});
