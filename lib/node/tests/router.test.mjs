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
  assert.equal(binding.getActiveBackend(), testBackend);
  assert.equal(typeof binding.decodeGatewayQueryBody, 'function');
  assert.equal(typeof binding.gatewayTextResponseBody, 'function');
  assert.equal(typeof binding.backendObservabilityJson, 'function');
  assert.equal(typeof binding.EndpointDescriptor.local, 'function');
  assert.equal(typeof binding.EndpointDescriptor.gateway, 'function');
  assert.equal(typeof binding.EndpointDescriptor.provider, 'function');
  assert.deepEqual(Object.keys(binding.EndpointDescriptor), [
    'local',
    'gateway',
    'provider',
  ]);
  assert.equal(typeof binding.SippClient.prototype.add, 'function');
  assert.equal(typeof binding.SippClient.prototype.remove, 'function');
  assert.equal(typeof binding.ModelStore.prototype.installFiles, 'function');
  assert.equal(typeof binding.ModelStore.prototype.installUrls, 'function');
  assert.equal(typeof binding.ModelStore.prototype.list, 'function');
  assert.equal(typeof binding.ModelStore.prototype.remove, 'function');
  assert.equal(binding.SippClient.prototype['add' + 'Local'], undefined);
  assert.equal(binding.SippClient.prototype.addHttpEndpoint, undefined);
});

test('public declarations expose one endpoint descriptor factory', () => {
  const declarations = readFileSync(path.join(bindingDir, 'index.d.ts'), 'utf8');

  assert.match(
    declarations,
    /const EndpointDescriptor:[\s\S]*local\([\s\S]*gateway\([\s\S]*provider\(/
  );
  assert.doesNotMatch(declarations, /LocalEndpointDescriptor|installed\(/);
  assert.doesNotMatch(declarations, /export type ModelSource/);
  assert.doesNotMatch(declarations, /readonly source:/);
  assert.doesNotMatch(declarations, /openai-compatible/);
  assert.doesNotMatch(declarations, /readonly modelPath:/);
  assert.match(declarations, /readonly models: ModelStore/);
});

test('endpoint factories create the native descriptor shapes', () => {
  process.env.SIPP_NODE_BACKEND = testBackend;
  const { EndpointDescriptor } = require('../router.js');

  assert.deepEqual(EndpointDescriptor.local('model-a'), {
    kind: 'local',
    modelId: 'model-a',
    config: undefined,
  });
  assert.deepEqual(EndpointDescriptor.gateway({
    target: 'model-a',
    baseUrl: 'https://gateway.example.test',
  }), {
    kind: 'gateway',
    target: 'model-a',
    baseUrl: 'https://gateway.example.test',
    authentication: undefined,
    staticHeaders: undefined,
    timeoutMs: undefined,
    queryRoute: undefined,
    chatRoute: undefined,
    embedRoute: undefined,
    protocolOptions: undefined,
  });
  assert.equal(EndpointDescriptor.provider({
    provider: 'openai',
    model: 'model-a',
  }).kind, 'provider');
  assert.equal(EndpointDescriptor.installed, undefined);
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
      client.models.installUrls(
        [`http://127.0.0.1:${address.port}/model.gguf`],
      ),
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
module.exports = {
  SippTextRun,
  SippEmbeddingRun,
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
