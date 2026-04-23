import test from 'node:test';
import assert from 'node:assert/strict';
import { CogentEngine } from './cogent-engine.js';
import { QueryError } from './model-management/model-types.js';

test('CogentEngine exposes the minimal root API', async () => {
  const engine = await CogentEngine.create({
    moduleUrl: 'https://example.test/runtime.js',
    wasmUrl: 'https://example.test/runtime.wasm',
    executionMode: 'main-thread',
  });

  assert.equal(typeof engine.models.load, 'function');
  assert.equal(typeof engine.models.current, 'function');
  assert.equal(typeof engine.models.list, 'function');
  assert.equal(typeof engine.models.remove, 'function');
  assert.equal(typeof engine.query, 'function');
  assert.equal(typeof engine.close, 'function');
  assert.deepEqual(Object.keys(engine), ['models']);

  await engine.close();
  assert.throws(
    () => engine.models.current(),
    (error) => error instanceof QueryError && error.code === 'ENGINE_CLOSED'
  );
});
