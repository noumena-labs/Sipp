import test from 'node:test';
import assert from 'node:assert/strict';
import { WasmBridge } from './wasm-bridge.js';
import type { EngineModule } from './engine-module.js';

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
