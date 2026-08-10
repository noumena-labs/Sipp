import { readFileSync } from 'node:fs';

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
  readVisionArgs,
} from './_support.mjs';

const {
  model: modelPath,
  projector: projectorPath,
  image,
  input,
} = readVisionArgs('Describe this image in one sentence.');
const {
  Endpoint,
  SippClient,
  backendObservabilityJson,
  setLlamaLogQuiet,
} = native;

setLlamaLogQuiet(true);
console.log(`backend_before_load=${backendObservabilityJson(true)}`);
const client = new SippClient();
const model = await client.models.add([modelPath, projectorPath]);
await client.add(
  'default',
  Endpoint.local(model, {
    runtime: runtimeConfig({ embeddings: false }),
  })
);
console.log(`backend_after_load=${backendObservabilityJson(true)}`);

// Multimodal chat uses the same chat API. Model metadata pairs the projector;
// image bytes are passed on the request.
const run = client.chat({
  messages: [
    { role: 'user', content: input },
  ],
  options: textOptions(),
  local: {
    contextKey: 'node-vision-chat-example',
    media: [readFileSync(image)],
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

function runtimeConfig({ embeddings }) {
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
