import { readFileSync } from 'node:fs';

import native from '../../lib/node/router.js';
import { gpuLayers, intEnv } from './_support.mjs';

const [modelPath, projectorPath, audioPath] = process.argv.slice(2);
if (!modelPath || !projectorPath || !audioPath) {
  throw new Error('usage: node examples/node/listen.mjs <model.gguf> <projector.gguf> <audio>');
}

const { Endpoint, SippClient, setLlamaLogQuiet } = native;
setLlamaLogQuiet(true);
const client = new SippClient();
const model = await client.models.add([modelPath, projectorPath]);
await client.add('asr', Endpoint.local(model, { runtime: runtimeConfig(8192) }));

const response = await client.listen({
  audio: readFileSync(audioPath),
  language: process.env.SIPP_LANGUAGE,
}).response;
console.log(response.text.trim());

function runtimeConfig(contextSize) {
  return {
    placement: { gpu_layers: gpuLayers() },
    context: {
      n_ctx: intEnv('SIPP_CONTEXT', contextSize),
      n_threads: intEnv('SIPP_THREADS'),
      n_threads_batch: intEnv('SIPP_THREADS'),
      warmup: false,
    },
  };
}
