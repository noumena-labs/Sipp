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

const { EndpointDescriptor, SippClient, setLlamaLogQuiet } = native;
const { model: modelPath, target, input } = readGatewayArgs(
  'gateway_embed',
  'SippClient gateway embedding example input.',
);
setLlamaLogQuiet(true);
const client = new SippClient();
const model = await client.models.add([modelPath]);
const localEndpoint = await client.add(
  'local',
  EndpointDescriptor.local(model.id, {
    runtime: runtimeConfig({ embeddings: true }),
  })
);
const gatewayEndpoint = await client.add('gateway', EndpointDescriptor.gateway({
  target,
  baseUrl: requiredEnv('SIPP_GATEWAY_URL'),
  authentication: {
    kind: 'bearer',
    value: requiredEnv('SIPP_GATEWAY_TOKEN'),
  },
}));

const local = await client.embed({
  endpoint: localEndpoint,
  input,
  local: {
    contextKey: 'node-gateway-embed-local',
    normalize: true,
  },
}).response;

const gateway = await client.embed({
  endpoint: gatewayEndpoint,
  input,
}).response;

console.log('local:');
printEmbedding(local);
console.log('gateway:');
printEmbedding(gateway);

function runtimeConfig({ embeddings }) {
  return {
    placement: { gpu_layers: gpuLayers() },
    context: {
      n_ctx: intEnv('SIPP_CONTEXT', DEFAULT_CONTEXT),
      n_threads: intEnv('SIPP_THREADS'),
      n_threads_batch: intEnv('SIPP_THREADS'),
      embeddings,
      pooling: embeddings ? 'mean' : undefined,
    },
    sampling: {
      temperature: numberEnv('SIPP_TEMPERATURE', DEFAULT_TEMPERATURE),
      seed: intEnv('SIPP_SEED', DEFAULT_SEED),
    },
    scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
    cache: { mode: 'live_slot_prefix' },
    residency: { max_gpu_models_per_device: 1 },
    observability: { runtime_metrics: true },
  };
}
