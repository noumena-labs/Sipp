import native from '../../lib/node/router.js';
import {
  DEFAULT_CONTEXT,
  DEFAULT_MAX_TOKENS,
  DEFAULT_SEED,
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
  gpuLayers,
  intEnv,
  numberEnv,
  printText,
  readGatewayArgs,
  requiredEnv,
} from './_support.mjs';

const { CogentClient, setLlamaLogQuiet } = native;
const { model, alias, input } = readGatewayArgs(
  'gateway_chat',
  'Explain gateway-backed inference in one sentence.',
);
setLlamaLogQuiet(true);
const client = new CogentClient();
const localEndpoint = await client.addLocal(
  'local',
  model,
  runtimeConfig({ embeddings: false }),
);
const gateway = {
  alias,
  baseUrl: requiredEnv('COGENTLM_GATEWAY_URL'),
  token: requiredEnv('COGENTLM_GATEWAY_TOKEN'),
};
const gatewayEndpoint = client.addRemote('gateway', gateway);

// Local and gateway chat use the same message and streaming shape.
const localRun = client.chat({
  endpoint: localEndpoint,
  messages: chatMessages(input),
  options: textOptions(),
  local: {
    contextKey: 'node-gateway-chat-local',
  },
  emitTokens: true,
});
const local = await collectStreamedText('local', localRun);

const gatewayRun = client.chat({
  endpoint: gatewayEndpoint,
  messages: chatMessages(input),
  options: textOptions(),
  emitTokens: true,
});
const remote = await collectStreamedText('gateway', gatewayRun);

console.log('local:');
printText(local);
console.log('gateway:');
printText(remote);

function chatMessages(prompt) {
  return [
    { role: 'system', content: 'Answer concisely.' },
    { role: 'user', content: prompt },
  ];
}

async function collectStreamedText(label, run) {
  let streamed = '';
  process.stdout.write(`${label}_stream=`);
  for await (const batch of run.tokens) {
    process.stdout.write(batch.text);
    streamed += batch.text;
  }
  process.stdout.write('\n');
  const result = await run.response;
  if (streamed !== result.text) {
    throw new Error('streamed token batches did not match final response text');
  }
  return result;
}

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

function textOptions() {
  return {
    maxTokens: intEnv('COGENTLM_MAX_TOKENS', DEFAULT_MAX_TOKENS),
    temperature: numberEnv('COGENTLM_TEMPERATURE', DEFAULT_TEMPERATURE),
    topP: numberEnv('COGENTLM_TOP_P', DEFAULT_TOP_P),
  };
}
