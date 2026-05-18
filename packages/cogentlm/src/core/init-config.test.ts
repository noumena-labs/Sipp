import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeInitConfig } from './init-config.js';

function normalizedRuntimeConfig(input: Parameters<typeof normalizeInitConfig>[0]) {
  return JSON.parse(normalizeInitConfig(input).runtimeConfigJson);
}

test('normalizeInitConfig emits typed Rust runtime JSON', () => {
  assert.deepEqual(normalizedRuntimeConfig(undefined), {});

  assert.deepEqual(
    normalizedRuntimeConfig({
      placement: {
        gpuLayers: 'all',
        splitMode: 'row',
        tensorSplit: [1, 2],
      },
      context: {
        nCtx: 8192,
        nThreads: 0,
        flashAttention: 'enabled',
        cacheTypeK: 'q8_0',
      },
      sampling: {
        samplers: ['top-k', 'top-p', 'temperature'],
        topK: 32,
        typicalP: 0.95,
        temperature: 0.7,
        backendSampling: true,
      },
      scheduler: {
        policy: 'throughput-first',
        decodeTokenReserve: 2,
        adaptivePrefillChunking: true,
      },
    }),
    {
      placement: {
        gpu_layers: 'all',
        split_mode: 'row',
        tensor_split: [1, 2],
      },
      context: {
        n_ctx: 8192,
        n_threads: 0,
        flash_attention: 'enabled',
        cache_type_k: 'q8_0',
      },
      sampling: {
        samplers: ['top_k', 'top_p', 'temperature'],
        top_k: 32,
        typical_p: 0.95,
        temperature: 0.7,
        backend_sampling: true,
      },
      scheduler: {
        policy: {
          mode: 'throughput_first',
          decode_token_reserve: 2,
          enable_adaptive_prefill_chunking: true,
        },
      },
    }
  );
});

test('normalizeInitConfig maps counted GPU layer offload for Rust serde', () => {
  assert.deepEqual(normalizedRuntimeConfig({ placement: { gpuLayers: 'auto' } }), {
    placement: { gpu_layers: 'auto' },
  });
  assert.deepEqual(normalizedRuntimeConfig({ placement: { gpuLayers: 99 } }), {
    placement: { gpu_layers: { count: 99 } },
  });
});

test('normalizeInitConfig rejects unsupported GPU layer values', () => {
  assert.throws(
    () => normalizeInitConfig({ placement: { gpuLayers: -1 } }),
    /"placement.gpuLayers" must be an integer >= 0/
  );
});

test('normalizeInitConfig forwards advanced sampler fields instead of rejecting them', () => {
  assert.deepEqual(
    normalizedRuntimeConfig({
      sampling: {
        xtcProbability: 0.1,
        drySequenceBreakers: ['\n', '###'],
        mirostat: 2,
        logitBias: [{ token: 42, bias: -1.5 }],
        ignoreEos: true,
      },
    }),
    {
      sampling: {
        xtc_probability: 0.1,
        dry_sequence_breakers: ['\n', '###'],
        mirostat: 2,
        logit_bias: [{ token: 42, bias: -1.5 }],
        ignore_eos: true,
      },
    }
  );
});
