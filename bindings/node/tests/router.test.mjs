import assert from 'node:assert/strict';
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
