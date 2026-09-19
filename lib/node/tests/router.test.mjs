import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const bindingDir = path.resolve(fileURLToPath(new URL('..', import.meta.url)));
const testBackend = process.env.SIPP_NODE_TEST_BACKEND ?? process.env.SIPP_NODE_BACKEND ?? 'cpu';

test('router imports the selected built binding and exposes backend helpers', () => {
  process.env.SIPP_NODE_BACKEND = testBackend;
  const binding = require('../router.js');

  assert.equal(typeof binding.getActiveBackend, 'function');
  assert.equal(typeof binding.onFallback, 'function');
  assert.equal(binding.getActiveBackend(), testBackend);
  assert.equal(typeof binding.decodeGatewayQueryBody, 'function');
  assert.equal(typeof binding.gatewayTextResponseBody, 'function');
  assert.equal(typeof binding.backendObservabilityJson, 'function');
  assert.equal(typeof binding.backendIsUsable, 'function');
  assert.equal(typeof binding.Endpoint.local, 'function');
  assert.equal(typeof binding.Endpoint.gateway, 'function');
  assert.equal(typeof binding.Endpoint.provider, 'function');
  assert.equal(typeof binding.SippClient.prototype.add, 'function');
  assert.equal(typeof binding.SippClient.prototype.remove, 'function');
  assert.equal(typeof binding.ModelStore.prototype.add, 'function');
  assert.equal(binding.ModelStore.prototype.installFiles, undefined);
  assert.equal(binding.ModelStore.prototype.installUrls, undefined);
  assert.equal(typeof binding.ModelStore.prototype.list, 'function');
  assert.equal(typeof binding.ModelStore.prototype.remove, 'function');
  assert.equal(binding.SippClient.prototype['add' + 'Local'], undefined);
  assert.equal(binding.SippClient.prototype.addHttpEndpoint, undefined);
});

test('public declarations expose one endpoint factory', () => {
  const declarations = readFileSync(path.join(bindingDir, 'index.d.ts'), 'utf8');
  const routerDeclarations = readFileSync(path.join(bindingDir, 'router.d.ts'), 'utf8');

  assert.match(
    declarations,
    /class Endpoint[\s\S]*static local\([\s\S]*static gateway\([\s\S]*static provider\(/
  );
  assert.doesNotMatch(declarations, /EndpointDescriptor|LocalDescriptor|installed\(/);
  assert.doesNotMatch(declarations, /export type ModelSource/);
  assert.doesNotMatch(declarations, /readonly source:/);
  assert.doesNotMatch(declarations, /openai-compatible/);
  assert.doesNotMatch(declarations, /readonly modelPath:/);
  assert.match(declarations, /readonly models: ModelStore/);
  assert.match(
    declarations,
    /interface ListenOptions[\s\S]*readonly maxTokens\?: number/
  );
  assert.match(declarations, /query\(input: QueryInput, options\?: QueryOptions\): SippTextRun/);
  assert.match(declarations, /chat\(input: ChatInput, options\?: QueryOptions\): SippTextRun/);
  assert.match(declarations, /embed\(input: string, options\?: EmbedOptions\): SippEmbeddingRun/);
  assert.match(declarations, /listen\(audio: Buffer, options\?: ListenOptions\): SippTextRun/);
  assert.match(declarations, /speak\(text: string, options\?: SpeakOptions\): SippAudioRun/);
  assert.doesNotMatch(declarations, /interface Sipp(Query|Chat|Embed|Listen|Speak)Request/);
  assert.match(routerDeclarations, /interface FallbackEvent/);
  assert.match(routerDeclarations, /function onFallback/);
});

test('endpoint factories create opaque native endpoint values', () => {
  process.env.SIPP_NODE_BACKEND = testBackend;
  const { Endpoint } = require('../router.js');
  const model = {
    id: 'model-a',
    name: 'Model A',
    bytes: 1,
    modality: 'text',
    status: 'ready',
  };

  assert.ok(Endpoint.local(model) instanceof Endpoint);
  assert.ok(Endpoint.gateway({
    target: 'model-a',
    baseUrl: 'https://gateway.example.test',
  }) instanceof Endpoint);
  assert.ok(Endpoint.provider({
    provider: 'openai',
    model: 'model-a',
    apiKey: 'secret',
  }) instanceof Endpoint);
  assert.throws(
    () => Endpoint.provider({
      provider: 'openai',
      model: 'model-a',
      apiKey: 'secret',
      version: '2023-06-01',
    }),
    /version is not valid for the OpenAI provider/
  );
  assert.equal(Endpoint.installed, undefined);
});

test('same-id endpoint replacement is owned by the native client', async () => {
  process.env.SIPP_NODE_BACKEND = testBackend;
  const { Endpoint, SippClient } = require('../router.js');
  const storageRoot = mkdtempSync(path.join(os.tmpdir(), 'sipp-node-replacement-'));
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      const target = JSON.parse(body).model;
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ text: target, finish_reason: 'stop' }));
    });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));

  try {
    const address = server.address();
    assert.notEqual(address, null);
    assert.equal(typeof address, 'object');
    const baseUrl = `http://127.0.0.1:${address.port}`;
    const client = new SippClient({ storageRoot });
    const original = await client.add('active', Endpoint.gateway({
      target: 'first',
      baseUrl,
    }));
    await client.add('active', Endpoint.gateway({
      target: 'second',
      baseUrl,
    }));

    const response = await client.query('route', { endpoint: original }).response;
    assert.equal(response.text, 'second');
  } finally {
    await new Promise((resolve, reject) => {
      server.close((error) => (error == null ? resolve() : reject(error)));
    });
    rmSync(storageRoot, { recursive: true, force: true });
  }
});

test('remote 503 errors preserve lifecycle metadata after the shared retry policy', async () => {
  process.env.SIPP_NODE_BACKEND = testBackend;
  const binding = require('../router.js');
  const storageRoot = mkdtempSync(path.join(os.tmpdir(), 'sipp-node-remote-'));
  let requestCount = 0;
  const server = createServer((_request, response) => {
    requestCount += 1;
    response.writeHead(503, { 'Retry-After': '0' });
    response.end();
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));

  try {
    const address = server.address();
    assert.notEqual(address, null);
    assert.equal(typeof address, 'object');
    const client = new binding.SippClient({ storageRoot });

    await assert.rejects(
      client.models.add([new URL(`http://127.0.0.1:${address.port}/model.gguf`).href]),
      (error) => {
        assert.equal(error.name, 'ModelLifecycleError');
        assert.equal(error.code, 'REMOTE_METADATA_UNAVAILABLE');
        assert.equal(error.status, 503);
        assert.equal(error.retryAfterMs, 0);
        return true;
      }
    );
    assert.equal(requestCount, 4);
  } finally {
    await new Promise((resolve, reject) => {
      server.close((error) => (error == null ? resolve() : reject(error)));
    });
    rmSync(storageRoot, { recursive: true, force: true });
  }
});

test('router rejects invalid backend names before loading native artifacts', () => {
  const result = spawnSync(
    process.execPath,
    ['-e', "process.env.SIPP_NODE_BACKEND='bogus'; require('./router.js')"],
    {
      cwd: bindingDir,
      encoding: 'utf8',
    }
  );

  assert.notEqual(result.status, 0);
  assert.match(`${result.stdout}\n${result.stderr}`, /Invalid SIPP_NODE_BACKEND=bogus/);
});

test('router augments native run classes with memoized responses and async token iterables', () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), 'sipp-node-router-'));
  const fakeNative = path.join(tempDir, 'fake-native.cjs');
  writeFileSync(
    fakeNative,
    `
class SippTextRun {
  constructor() {
    this.responseCalls = 0;
    this.nextTokenCalls = 0;
  }
  __response() {
    this.responseCalls += 1;
    return Promise.resolve({ text: 'done' });
  }
  async __nextToken() {
    this.nextTokenCalls += 1;
    if (this.nextTokenCalls === 1) return { text: 'a' };
    if (this.nextTokenCalls === 2) return { text: 'b' };
    return null;
  }
}
class SippEmbeddingRun {
  constructor() {
    this.responseCalls = 0;
  }
  __response() {
    this.responseCalls += 1;
    return Promise.resolve({ values: [1, 2, 3] });
  }
}
class SippAudioRun {
  constructor() {
    this.responseCalls = 0;
  }
  __response() {
    this.responseCalls += 1;
    return Promise.resolve({ audio: Buffer.from('RIFF') });
  }
}
class ModelStore {
  add(sources) {
    return sources;
  }
}
module.exports = {
  ModelStore,
  SippTextRun,
  SippEmbeddingRun,
  SippAudioRun,
  backendObservabilityJson() {
    return JSON.stringify({
      compiled: { vulkan: true },
      gpuOffloadSupported: true,
      availableBackends: [{ name: 'vulkan' }],
      devices: [],
    });
  },
};
`,
    'utf8'
  );

  const script = `
const assert = require('node:assert/strict');
  const binding = require('./router.js');
(async () => {
  assert.equal(binding.getActiveBackend(), 'vulkan');
  const textRun = new binding.SippTextRun();
  assert.equal(textRun.response, textRun.response);
  assert.equal(textRun.responseCalls, 1);
  const tokens = [];
  for await (const batch of textRun) tokens.push(batch.text);
  assert.deepEqual(tokens, ['a', 'b']);
  const tokenAccessor = textRun.tokens[Symbol.asyncIterator]();
  assert.equal(typeof tokenAccessor.next, 'function');
  const embeddingRun = new binding.SippEmbeddingRun();
  assert.equal(embeddingRun.response, embeddingRun.response);
  assert.equal(embeddingRun.responseCalls, 1);
  const audioRun = new binding.SippAudioRun();
  assert.equal(audioRun.response, audioRun.response);
  assert.equal(audioRun.responseCalls, 1);
  console.log('ok');
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
`;
  const result = spawnSync(process.execPath, ['-e', script], {
    cwd: bindingDir,
    encoding: 'utf8',
    env: {
      ...process.env,
      SIPP_NODE_BACKEND: 'vulkan',
      NAPI_RS_NATIVE_LIBRARY_PATH: fakeNative,
    },
  });

  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  assert.match(result.stdout, /ok/);
});

test('router emits structured events when automatic backend selection falls back', () => {
  const script = `
const assert = require('node:assert/strict');
const Module = require('node:module');
const originalLoad = Module._load;
const cpuBinding = {};
const acceleratorBinding = { backendIsUsable: () => false };

Module._load = function load(request, parent, isMain) {
  const match = String(request).match(/sipp_node_(cpu|cuda|metal|vulkan)\./);
  if (match != null) {
    return match[1] === 'cpu' ? cpuBinding : acceleratorBinding;
  }
  return originalLoad.call(this, request, parent, isMain);
};

process.env.SIPP_NODE_BACKEND = 'auto';
delete process.env.NAPI_RS_NATIVE_LIBRARY_PATH;
const binding = require('./router.js');
const events = [];
const unsubscribe = binding.onFallback((event) => events.push(event));

setImmediate(() => {
  const skipped = process.platform === 'darwin'
    ? ['metal']
    : ['cuda', 'vulkan'];
  assert.equal(binding.getActiveBackend(), 'cpu');
  assert.deepEqual(events.map((event) => event.type), skipped.map(() => 'fallback-warning'));
  assert.deepEqual(events.map((event) => event.kind), skipped.map(() => 'backend'));
  assert.deepEqual(events.map((event) => event.fallbackTo), skipped.map(() => 'cpu'));
  assert.deepEqual(
    events.map((event) => event.detail),
    skipped.map((backend) =>
      \`\${backend} backend unavailable: \${backend} binding loaded, but no usable \${backend} backend was reported by llama.cpp\`
    ),
  );
  unsubscribe();
  console.log('ok');
});
`;

  const result = spawnSync(process.execPath, ['-e', script], {
    cwd: bindingDir,
    encoding: 'utf8',
    env: { ...process.env },
  });

  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  assert.match(result.stdout, /ok/);
});

test('router preserves the native unusable-backend diagnostic', () => {
  const script = `
const Module = require('node:module');
const originalLoad = Module._load;

Module._load = function load(request, parent, isMain) {
  if (String(request).includes('sipp_node_metal.')) {
    return { backendIsUsable: () => false };
  }
  return originalLoad.call(this, request, parent, isMain);
};

process.env.SIPP_NODE_BACKEND = 'metal';
delete process.env.NAPI_RS_NATIVE_LIBRARY_PATH;
require('./router.js');
`;

  const result = spawnSync(process.execPath, ['-e', script], {
    cwd: bindingDir,
    encoding: 'utf8',
    env: { ...process.env },
  });

  assert.notEqual(result.status, 0);
  assert.match(
    `${result.stdout}\n${result.stderr}`,
    /metal binding loaded, but no usable metal backend was reported by llama\.cpp/,
  );
  assert.doesNotMatch(`${result.stdout}\n${result.stderr}`, /Cannot find module/);
});
