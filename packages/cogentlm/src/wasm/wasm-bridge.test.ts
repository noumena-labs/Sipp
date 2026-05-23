import test from 'node:test';
import assert from 'node:assert/strict';
import { WasmBridge } from './wasm-bridge.js';
import type { EngineModule } from './engine-module.js';

function createSha256TestModule(updateLengths: number[] = []): EngineModule {
  return {
    FS: {
      analyzePath: () => ({ exists: false }),
      mkdir: () => {},
      writeFile: () => {},
      unlink: () => {},
      mount: () => {},
      unmount: () => {},
    },
    WORKERFS: {},
    HEAP32: new Int32Array(128),
    HEAPF64: new Float64Array(128),
    HEAPU8: new Uint8Array(4096),
    _malloc: () => 32,
    _free: () => {},
    ccall: (ident: string, _returnType: string | null, _argTypes: string[], args: unknown[]) => {
      if (ident === 'CE_Sha256Create') {
        return 1;
      }
      if (ident === 'CE_Sha256Update') {
        updateLengths.push(args[2] as number);
        return 0;
      }
      if (ident === 'CE_Sha256Finalize') {
        return 64;
      }
      if (ident === 'CE_FreeString' || ident === 'CE_Sha256Close') {
        return 0;
      }
      throw new Error(`Unexpected call: ${ident}`);
    },
    UTF8ToString: () => 'a'.repeat(64),
    addFunction: () => 0,
    removeFunction: () => {},
  };
}

test('WasmBridge forwards Rust runtime config JSON without TS-side normalization', async () => {
  const calls: unknown[][] = [];
  const module = {
    ccall: (_ident: string, _returnType: string, _argTypes: string[], args: unknown[]) => {
      calls.push(args);
      return Promise.resolve(0);
    },
  } as unknown as EngineModule;
  const bridge = new WasmBridge(module);

  await bridge.loadRuntimeModel('/models/model.gguf', {
    placement: { gpu_layers: { count: 99 }, split_mode: 'row' },
    context: { n_ctx: 8192, flash_attention: 'enabled' },
    sampling: { samplers: ['top_k', 'top_p'], top_k: 32 },
    scheduler: {
      continuous_batching: true,
      policy: {
        mode: 'throughput_first',
        decode_token_reserve: 2,
        enable_adaptive_prefill_chunking: true,
      },
    },
  });

  assert.deepEqual(calls, [
    [
      '/models/model.gguf',
      JSON.stringify({
        placement: { gpu_layers: { count: 99 }, split_mode: 'row' },
        context: { n_ctx: 8192, flash_attention: 'enabled' },
        sampling: { samplers: ['top_k', 'top_p'], top_k: 32 },
        scheduler: {
          continuous_batching: true,
          policy: {
            mode: 'throughput_first',
            decode_token_reserve: 2,
            enable_adaptive_prefill_chunking: true,
          },
        },
      }),
    ],
  ]);
});

test('WasmBridge hashes blob streams without releasing the reader lock', async () => {
  const updateLengths: number[] = [];
  const module = createSha256TestModule(updateLengths);
  const released: string[] = [];
  let readCount = 0;
  const reader = {
    async read() {
      readCount += 1;
      await Promise.resolve();
      if (readCount === 1) {
        return { done: false, value: new Uint8Array([1, 2, 3]) };
      }
      return { done: true, value: undefined };
    },
    async cancel() {},
    releaseLock() {
      released.push('release');
      throw new TypeError('Releasing Default reader');
    },
  };
  const blob = {
    stream: () => ({
      getReader: () => reader,
    }),
  } as unknown as Blob;
  const bridge = new WasmBridge(module);

  assert.equal(await bridge.sha256Blob(blob), 'a'.repeat(64));
  assert.deepEqual(updateLengths, [3]);
  assert.deepEqual(released, []);
});
