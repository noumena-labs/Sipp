import native from '../../lib/node/router.js';
import {
  DEFAULT_CONTEXT,
  DEFAULT_SEED,
  DEFAULT_TEMPERATURE,
  gpuLayers,
  intEnv,
  numberEnv,
  printEmbedding,
  readLocalArgs,
} from './_support.mjs';

const { CogentClient, backendObservabilityJson, setLlamaLogQuiet } = native;
const { model, input } = readLocalArgs('embed', 'CogentClient embedding example input.');

setLlamaLogQuiet(true);
console.log(`backend_before_load=${backendObservabilityJson(true)}`);
const client = new CogentClient();
await client.add('default', {
  kind: 'local',
  modelPath: model,
  config: runtimeConfig({ embeddings: true }),
});
console.log(`backend_after_load=${backendObservabilityJson(true)}`);

// Embeddings use the same local endpoint. The runtime is loaded with
// `embeddings=true`, and the request asks for a normalized vector.
const result = await client.embed({
  input,
  local: {
    contextKey: 'node-embed-example',
    normalize: true,
  },
}).response;
printEmbedding(result);

function runtimeConfig({ embeddings, projectorPath = undefined }) {
  const multimodal = projectorPath == null ? {} : { projector_path: projectorPath };
  return {
    placement: { gpu_layers: gpuLayers() },
    context: {
      n_ctx: intEnv('COGENTLM_CONTEXT', DEFAULT_CONTEXT),
      n_threads: intEnv('COGENTLM_THREADS'),
      n_threads_batch: intEnv('COGENTLM_THREADS'),
      embeddings,
      pooling: embeddings ? 'mean' : undefined,
    },
    sampling: {
      temperature: numberEnv('COGENTLM_TEMPERATURE', DEFAULT_TEMPERATURE),
      seed: intEnv('COGENTLM_SEED', DEFAULT_SEED),
    },
    scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
    cache: { mode: 'live_slot_prefix' },
    multimodal,
    residency: { max_gpu_models_per_device: 1 },
    observability: { runtime_metrics: true },
  };
}
