import test from 'node:test';
import assert from 'node:assert/strict';
import { QueryError } from '../models/types.js';
import { WasmBridge, unwrapLifecycleResponse } from './wasm-bridge.js';
import type { EngineModule } from './engine-module.js';

function createSha256TestModule(updateLengths: number[] = []): EngineModule {
  const heapU8 = new Uint8Array(4096);
  return {
    FS: {
      analyzePath: () => ({ exists: false }),
      mkdir: () => {},
      writeFile: () => {},
      unlink: () => {},
      mount: () => {},
      unmount: () => {},
    },
    HEAP32: new Int32Array(128),
    HEAPF32: new Float32Array(128),
    HEAPF64: new Float64Array(128),
    HEAPU8: heapU8,
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
        heapU8.set(new TextEncoder().encode('a'.repeat(64)), 64);
        return 64;
      }
      if (ident === 'CE_FreeString' || ident === 'CE_Sha256Close') {
        return 0;
      }
      throw new Error(`Unexpected call: ${ident}`);
    },
    UTF8ToString: () => {
      throw new Error('UTF8ToString should not be used for owned strings.');
    },
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

test('WasmBridge copies completed embedding responses as f32 values', () => {
  const memory = new ArrayBuffer(4096);
  const heapF32 = new Float32Array(memory);
  const module: EngineModule = {
    FS: {
      analyzePath: () => ({ exists: false }),
      mkdir: () => {},
      writeFile: () => {},
      unlink: () => {},
      mount: () => {},
      unmount: () => {},
    },
    HEAP32: new Int32Array(memory),
    HEAPF32: heapF32,
    HEAPF64: new Float64Array(memory),
    HEAPU8: new Uint8Array(memory),
    _malloc: () => 64,
    _free: () => {},
    ccall: (ident: string, _returnType: string | null, _argTypes: string[], args: unknown[]) => {
      if (ident === 'CE_GetCompletedRequestStatus') {
        return 1;
      }
      if (ident === 'CE_GetCompletedRequestOutputKind') {
        return 2;
      }
      if (ident === 'CE_GetCompletedRequestEmbeddingLength') {
        return 2;
      }
      if (ident === 'CE_GetCompletedRequestEmbeddingPooling') {
        return 1;
      }
      if (ident === 'CE_GetCompletedRequestEmbeddingNormalized') {
        return 0;
      }
      if (ident === 'CE_CopyCompletedRequestEmbedding') {
        const ptr = args[1] as number;
        heapF32[ptr / 4] = 3;
        heapF32[ptr / 4 + 1] = 4;
        return 2;
      }
      if (ident === 'CE_GetCompletedRequestErrorSize') {
        return 0;
      }
      if (ident === 'CE_GetCompletedRequestRuntimeObservability') {
        return -1;
      }
      if (ident === 'CE_ConsumeCompletedRequest') {
        return 1;
      }
      throw new Error(`Unexpected call: ${ident}`);
    },
    UTF8ToString: () => '',
    addFunction: () => 0,
    removeFunction: () => {},
  };
  const bridge = new WasmBridge(module);

  const response = bridge.takeCompletedResponse(7);

  assert.deepEqual(response, {
    requestId: 7,
    completed: true,
    failed: false,
    cancelled: false,
    embedding: {
      values: [3, 4],
      pooling: 'mean',
      normalized: false,
    },
    errorMessage: null,
    observability: null,
  });
});

test('WasmBridge copies completed text responses by output kind', () => {
  const memory = new SharedArrayBuffer(4096);
  const heapU8 = new Uint8Array(memory);
  const module: EngineModule = {
    FS: {
      analyzePath: () => ({ exists: false }),
      mkdir: () => {},
      writeFile: () => {},
      unlink: () => {},
      mount: () => {},
      unmount: () => {},
    },
    HEAP32: new Int32Array(memory),
    HEAPF32: new Float32Array(memory),
    HEAPF64: new Float64Array(memory),
    HEAPU8: heapU8,
    _malloc: () => 64,
    _free: () => {},
    ccall: (ident: string, _returnType: string | null, _argTypes: string[], args: unknown[]) => {
      if (ident === 'CE_GetCompletedRequestStatus') {
        return 1;
      }
      if (ident === 'CE_GetCompletedRequestOutputKind') {
        return 1;
      }
      if (ident === 'CE_GetCompletedRequestOutputSize') {
        return 4;
      }
      if (ident === 'CE_CopyCompletedRequestOutput') {
        const ptr = args[1] as number;
        heapU8.set(new TextEncoder().encode('done'), ptr);
        return 4;
      }
      if (ident === 'CE_GetCompletedRequestErrorSize') {
        return 0;
      }
      if (ident === 'CE_GetCompletedRequestRuntimeObservability') {
        return -1;
      }
      if (ident === 'CE_ConsumeCompletedRequest') {
        return 1;
      }
      throw new Error(`Unexpected call: ${ident}`);
    },
    UTF8ToString: () => {
      throw new Error('UTF8ToString should not be used for copied text.');
    },
    addFunction: () => 0,
    removeFunction: () => {},
  };
  const bridge = new WasmBridge(module);

  const response = bridge.takeCompletedResponse(8);

  assert.deepEqual(response, {
    requestId: 8,
    completed: true,
    failed: false,
    cancelled: false,
    outputText: 'done',
    errorMessage: null,
    observability: null,
  });
});

test('unwrapLifecycleResponse preserves unsupported operation errors', () => {
  assert.throws(
    () => {
      unwrapLifecycleResponse(
        {
          ok: false,
          error: {
            code: 'UNSUPPORTED_OPERATION',
            message: 'unsupported operation chat: model has no chat template',
          },
        },
        'chat'
      );
    },
    (error) => error instanceof QueryError && error.code === 'UNSUPPORTED_OPERATION'
  );
});
