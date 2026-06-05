import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const bindingDir = path.resolve(fileURLToPath(new URL('..', import.meta.url)));
const testBackend = process.env.COGENTLM_NODE_TEST_BACKEND ?? process.env.COGENTLM_NODE_BACKEND ?? 'cpu';

test('router imports the selected built binding and exposes backend helpers', () => {
  process.env.COGENTLM_NODE_BACKEND = testBackend;
  const binding = require('../router.js');

  assert.equal(typeof binding.getActiveBackend, 'function');
  assert.equal(binding.getActiveBackend(), testBackend);
  assert.equal(typeof binding.backendObservabilityJson, 'function');
});

test('router rejects invalid backend names before loading native artifacts', () => {
  const result = spawnSync(
    process.execPath,
    ['-e', "process.env.COGENTLM_NODE_BACKEND='bogus'; require('./router.js')"],
    {
      cwd: bindingDir,
      encoding: 'utf8',
    }
  );

  assert.notEqual(result.status, 0);
  assert.match(`${result.stdout}\n${result.stderr}`, /Invalid COGENTLM_NODE_BACKEND=bogus/);
});

test('router augments native run classes with memoized responses and async token iterables', () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), 'cogentlm-node-router-'));
  const fakeNative = path.join(tempDir, 'fake-native.cjs');
  writeFileSync(
    fakeNative,
    `
class CogentTextRun {
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
class CogentEmbeddingRun {
  constructor() {
    this.responseCalls = 0;
  }
  __response() {
    this.responseCalls += 1;
    return Promise.resolve({ values: [1, 2, 3] });
  }
}
module.exports = {
  CogentTextRun,
  CogentEmbeddingRun,
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
  const textRun = new binding.CogentTextRun();
  assert.equal(textRun.response, textRun.response);
  assert.equal(textRun.responseCalls, 1);
  const tokens = [];
  for await (const batch of textRun) tokens.push(batch.text);
  assert.deepEqual(tokens, ['a', 'b']);
  const tokenAccessor = textRun.tokens[Symbol.asyncIterator]();
  assert.equal(typeof tokenAccessor.next, 'function');
  const embeddingRun = new binding.CogentEmbeddingRun();
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
      COGENTLM_NODE_BACKEND: 'vulkan',
      NAPI_RS_NATIVE_LIBRARY_PATH: fakeNative,
    },
  });

  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  assert.match(result.stdout, /ok/);
});
