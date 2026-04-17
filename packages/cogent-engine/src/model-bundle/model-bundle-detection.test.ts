import assert from 'node:assert/strict';
import test from 'node:test';

import {
  detectModel,
  detectModelFromFilename,
  detectModelFromGgufFile,
  detectModelFromUrl,
  discoverProjector,
  discoverProjectorFromHuggingFace,
  findProjectorFileCandidates,
  parseHuggingFaceUrl,
  resolveLocalModelAndProjectorFiles,
  splitModelAndProjectorFiles,
  validateProjectorUrl,
} from './model-bundle-detection.js';

const textEncoder = new TextEncoder();

enum GgufValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

function encodeUint32(value: number): Uint8Array {
  const buffer = new ArrayBuffer(4);
  new DataView(buffer).setUint32(0, value, true);
  return new Uint8Array(buffer);
}

function encodeUint64(value: number): Uint8Array {
  const buffer = new ArrayBuffer(8);
  new DataView(buffer).setBigUint64(0, BigInt(value), true);
  return new Uint8Array(buffer);
}

function encodeBool(value: boolean): Uint8Array {
  return Uint8Array.from([value ? 1 : 0]);
}

function encodeString(value: string): Uint8Array {
  const bytes = textEncoder.encode(value);
  return concatBytes(encodeUint64(bytes.length), bytes);
}

function encodeField(key: string, type: GgufValueType, value: Uint8Array): Uint8Array {
  return concatBytes(encodeString(key), encodeUint32(type), value);
}

function buildGgufFile(
  fields: Array<{ key: string; type: GgufValueType; value: Uint8Array }>
): Uint8Array {
  return concatBytes(
    encodeUint32(0x46554747),
    encodeUint32(3),
    encodeUint64(0),
    encodeUint64(fields.length),
    ...fields.map((field) => encodeField(field.key, field.type, field.value))
  );
}

function concatBytes(...parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const output = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.byteLength;
  }
  return output;
}

function toBlobPart(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

test('detectModelFromFilename identifies vision models and excludes projector files', () => {
  const vision = detectModelFromFilename('Qwen2-VL-2B-Instruct-Q4_K_M.gguf');
  assert.equal(vision.isVisionModel, true);
  assert.equal(vision.isProjector, false);
  assert.equal(vision.detectionMethod, 'filename');
  assert.equal(vision.modelType, null);
  assert.equal(vision.modelArchitecture, null);

  const projector = detectModelFromFilename('mmproj-Qwen2-VL-2B-Instruct-f16.gguf');
  assert.equal(projector.isVisionModel, false);
  assert.equal(projector.isProjector, true);
  assert.equal(projector.detectionMethod, 'filename');
});

test('detectModelFromUrl derives a suggested HuggingFace projector URL', () => {
  const detection = detectModelFromUrl(
    'https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf'
  );
  assert.equal(detection.isVisionModel, true);
  assert.equal(
    detection.suggestedProjectorUrl,
    'https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-model-f16.gguf'
  );
  assert.equal(detection.detectionMethod, 'url');
});

test('parseHuggingFaceUrl extracts repo coordinates', () => {
  assert.deepEqual(
    parseHuggingFaceUrl(
      'https://huggingface.co/org/repo/resolve/main/model.gguf'
    ),
    {
      org: 'org',
      repo: 'repo',
      ref: 'main',
      filename: 'model.gguf',
      baseUrl: 'https://huggingface.co/org/repo',
    }
  );
});

test('findProjectorFileCandidates and splitModelAndProjectorFiles pair a single local projector', () => {
  const files = [
    { name: 'model-00001-of-00002.gguf' },
    { name: 'model-00002-of-00002.gguf' },
    { name: 'mmproj-model-f16.gguf' },
  ];
  const candidates = findProjectorFileCandidates(files);
  assert.deepEqual(candidates.map((candidate) => candidate.name), ['mmproj-model-f16.gguf']);

  const split = splitModelAndProjectorFiles(files);
  assert.equal(split.errorMessage, null);
  assert.equal(split.projectorFile?.name, 'mmproj-model-f16.gguf');
  assert.deepEqual(
    split.modelFiles.map((file) => file.name),
    ['model-00001-of-00002.gguf', 'model-00002-of-00002.gguf']
  );
});

test('splitModelAndProjectorFiles reports multiple projector candidates', () => {
  const split = splitModelAndProjectorFiles([
    { name: 'model.gguf' },
    { name: 'mmproj-a.gguf' },
    { name: 'projector-b.gguf' },
  ]);
  assert.match(split.errorMessage ?? '', /Multiple projector candidates found/);
  assert.deepEqual(split.candidateFileNames, ['mmproj-a.gguf', 'projector-b.gguf']);
});

test('discoverProjectorFromHuggingFace prefers f16 projectors from sibling metadata', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    assert.equal(
      String(input),
      'https://huggingface.co/api/models/bartowski/Qwen2-VL-2B-Instruct-GGUF'
    );
    return new Response(
      JSON.stringify({
        siblings: [
          { rfilename: 'Qwen2-VL-2B-Instruct-Q4_K_M.gguf' },
          { rfilename: 'mmproj-Qwen2-VL-2B-Instruct-q8_0.gguf' },
          { rfilename: 'mmproj-Qwen2-VL-2B-Instruct-f16.gguf' },
        ],
      }),
      { status: 200 }
    );
  }) as typeof fetch;

  try {
    const result = await discoverProjectorFromHuggingFace(
      'https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf'
    );
    assert.equal(
      result.projectorUrl,
      'https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf'
    );
    assert.equal(result.source, 'hf-api');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('discoverProjector falls back to HEAD probing when the HuggingFace API misses', async () => {
  const originalFetch = globalThis.fetch;
  const calls: Array<{ url: string; method: string }> = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    calls.push({ url, method });

    if (url === 'https://huggingface.co/api/models/org/repo') {
      return new Response(JSON.stringify({ siblings: [] }), { status: 200 });
    }
    if (url.endsWith('/mmproj-model-f16.gguf') && method === 'HEAD') {
      return new Response(null, { status: 200 });
    }
    return new Response(null, { status: 404 });
  }) as typeof fetch;

  try {
    const result = await discoverProjector(
      'https://huggingface.co/org/repo/resolve/main/model.gguf'
    );
    assert.equal(
      result.projectorUrl,
      'https://huggingface.co/org/repo/resolve/main/mmproj-model-f16.gguf'
    );
    assert.equal(result.source, 'head-probe');
    assert.deepEqual(
      calls.map((call) => `${call.method} ${call.url}`),
      [
        'GET https://huggingface.co/api/models/org/repo',
        'HEAD https://huggingface.co/org/repo/resolve/main/mmproj-model-f16.gguf',
      ]
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('validateProjectorUrl returns null when HEAD is not reachable', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(null, { status: 404 })) as typeof fetch;
  try {
    assert.equal(await validateProjectorUrl('https://example.com/mmproj.gguf'), null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('detectModel delegates by source type', () => {
  assert.equal(detectModel('file', 'llava.gguf').isVisionModel, true);
  assert.equal(detectModel('url', 'https://example.com/mmproj.gguf').isProjector, true);
});

test('detectModelFromGgufFile reads projector metadata even when the filename is generic', async () => {
  const projector = new File(
    [
      toBlobPart(buildGgufFile([
        { key: 'general.type', type: GgufValueType.STRING, value: encodeString('mmproj') },
        { key: 'general.architecture', type: GgufValueType.STRING, value: encodeString('clip') },
        {
          key: 'clip.projector_type',
          type: GgufValueType.STRING,
          value: encodeString('qwen2vl_merger'),
        },
      ])),
    ],
    'adapter.gguf'
  );

  const detection = await detectModelFromGgufFile(projector);
  assert.equal(detection.isProjector, true);
  assert.equal(detection.isVisionModel, false);
  assert.equal(detection.detectionMethod, 'gguf-metadata');
  assert.equal(detection.modelType, 'mmproj');
  assert.equal(detection.modelArchitecture, 'clip');
});

test('detectModelFromGgufFile detects qwen2vl metadata without relying on filename patterns', async () => {
  const model = new File(
    [
      toBlobPart(buildGgufFile([
        { key: 'general.type', type: GgufValueType.STRING, value: encodeString('model') },
        {
          key: 'general.architecture',
          type: GgufValueType.STRING,
          value: encodeString('qwen2vl'),
        },
      ])),
    ],
    'custom-model.gguf'
  );

  const detection = await detectModelFromGgufFile(model);
  assert.equal(detection.isProjector, false);
  assert.equal(detection.isVisionModel, true);
  assert.equal(detection.detectionMethod, 'gguf-metadata');
  assert.equal(detection.modelType, 'model');
  assert.equal(detection.modelArchitecture, 'qwen2vl');
});

test('detectModelFromGgufFile falls back to filename heuristics for legacy llama-based VLMs', async () => {
  const model = new File(
    [
      toBlobPart(buildGgufFile([
        { key: 'general.type', type: GgufValueType.STRING, value: encodeString('model') },
        {
          key: 'general.architecture',
          type: GgufValueType.STRING,
          value: encodeString('llama'),
        },
        {
          key: 'clip.has_vision_encoder',
          type: GgufValueType.BOOL,
          value: encodeBool(false),
        },
      ])),
    ],
    'llava-v1.5-7b-q4_k.gguf'
  );

  const detection = await detectModelFromGgufFile(model);
  assert.equal(detection.isProjector, false);
  assert.equal(detection.isVisionModel, true);
  assert.equal(detection.detectionMethod, 'filename');
  assert.equal(detection.modelArchitecture, 'llama');
});

test('resolveLocalModelAndProjectorFiles pairs a metadata-detected projector with a generic filename', async () => {
  const model = new File(
    [
      toBlobPart(buildGgufFile([
        { key: 'general.type', type: GgufValueType.STRING, value: encodeString('model') },
        {
          key: 'general.architecture',
          type: GgufValueType.STRING,
          value: encodeString('qwen2vl'),
        },
      ])),
    ],
    'model.gguf'
  );
  const projector = new File(
    [
      toBlobPart(buildGgufFile([
        { key: 'general.type', type: GgufValueType.STRING, value: encodeString('mmproj') },
        { key: 'general.architecture', type: GgufValueType.STRING, value: encodeString('clip') },
      ])),
    ],
    'helper.gguf'
  );

  const result = await resolveLocalModelAndProjectorFiles([model, projector]);
  assert.equal(result.errorMessage, null);
  assert.equal(result.projectorFile?.name, 'helper.gguf');
  assert.deepEqual(result.modelFiles.map((file) => file.name), ['model.gguf']);
});
