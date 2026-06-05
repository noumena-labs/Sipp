import native from '../../lib/node/router.js';
import {
  DEFAULT_CONTEXT,
  DEFAULT_SEED,
  DEFAULT_TEMPERATURE,
  gpuLayers,
  intEnv,
  numberEnv,
  printEmbedding,
  readGatewayArgs,
  requiredEnv,
} from './_support.mjs';

const { CogentClient, setLlamaLogQuiet } = native;
const { model, alias, input } = readGatewayArgs(
  'gateway_embed',
  'CogentClient gateway embedding example input.',
);
setLlamaLogQuiet(true);
const client = new CogentClient();
const localEndpoint = await client.addLocal(
  'local',
  model,
  runtimeConfig({ embeddings: true }),
);
const gateway = {
  alias,
  baseUrl: requiredEnv('COGENTLM_GATEWAY_URL'),
  token: requiredEnv('COGENTLM_GATEWAY_TOKEN'),
};
const gatewayEndpoint = client.addRemote('gateway', gateway);

const local = await client.embed({
  endpoint: localEndpoint,
  input,
  local: {
    contextKey: 'node-gateway-embed-local',
    normalize: true,
  },
}).response;

const remote = await client.embed({
  endpoint: gatewayEndpoint,
  input,
}).response;

console.log('local:');
printEmbedding(local);
console.log('gateway:');
printEmbedding(remote);

function runtimeConfig({ embeddings, projectorPath = undefined }) {
  const multimodal = projectorPath == null ? {} : { projector_path: projectorPath };
  return {
    placement: { gpu_layers: gpuLayers() },
    context: {
      n_ctx: intEnv('COGENTLM_CONTEXT', DEFAULT_CONTEXT),
      n_threads: intEnv('COGENTLM_THREADS'),
      n_threads_batch: intEnv('COGENTLM_THREADS'),
      embeddings,
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
