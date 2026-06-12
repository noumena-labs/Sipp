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
  readLocalArgs,
} from './_support.mjs';

const { SippClient, backendObservabilityJson, setLlamaLogQuiet } = native;
const { model, input } = readLocalArgs('chat', 'Explain the SippClient API in one sentence.');

setLlamaLogQuiet(true);
console.log(`backend_before_load=${backendObservabilityJson(true)}`);
const client = new SippClient();
await client.add('default', {
  kind: 'local',
  modelPath: model,
  config: runtimeConfig({ embeddings: false }),
});
console.log(`backend_after_load=${backendObservabilityJson(true)}`);

// `chat` sends role-tagged messages and can stream partial token batches.
const run = client.chat({
  messages: [
    { role: 'system', content: 'Answer concisely.' },
    { role: 'user', content: input },
  ],
  options: textOptions(),
  local: {
    contextKey: 'node-chat-example',
  },
  emitTokens: true,
});
let streamed = '';
for await (const batch of run.tokens) {
  process.stdout.write(batch.text);
  streamed += batch.text;
}
process.stdout.write('\n');
const result = await run.response;
if (streamed !== result.text) {
  throw new Error('streamed token batches did not match final response text');
}
printText(result);

function runtimeConfig({ embeddings, projectorPath = undefined }) {
  const multimodal = projectorPath == null ? {} : { projector_path: projectorPath };
  return {
    placement: { gpu_layers: gpuLayers() },
    context: {
      n_ctx: intEnv('SIPP_CONTEXT', DEFAULT_CONTEXT),
      n_threads: intEnv('SIPP_THREADS'),
      n_threads_batch: intEnv('SIPP_THREADS'),
      embeddings,
    },
    sampling: {
      temperature: numberEnv('SIPP_TEMPERATURE', DEFAULT_TEMPERATURE),
      seed: intEnv('SIPP_SEED', DEFAULT_SEED),
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
    maxTokens: intEnv('SIPP_MAX_TOKENS', DEFAULT_MAX_TOKENS),
    temperature: numberEnv('SIPP_TEMPERATURE', DEFAULT_TEMPERATURE),
    topP: numberEnv('SIPP_TOP_P', DEFAULT_TOP_P),
  };
}
