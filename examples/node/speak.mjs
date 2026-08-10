import { readFileSync, writeFileSync } from 'node:fs';

import native from '../../lib/node/router.js';
import { gpuLayers, intEnv } from './_support.mjs';

const [modelPath, projectorPath, outputPath, ...words] = process.argv.slice(2);
if (!modelPath || !projectorPath || !outputPath) {
  throw new Error(
    'usage: node examples/node/speak.mjs <model.gguf> <projector.gguf> <output.wav> [text]'
  );
}

const { Endpoint, SippClient, setLlamaLogQuiet } = native;
setLlamaLogQuiet(true);
const client = new SippClient();
const model = await client.models.add([modelPath, projectorPath]);
await client.add('tts', Endpoint.local(model, { runtime: runtimeConfig(4096) }));

const response = await client.speak({
  text: words.join(' ') || 'Hello from Sipp.',
  language: process.env.SIPP_LANGUAGE,
  speakerAudio: process.env.SIPP_SPEAKER_AUDIO
    ? readFileSync(process.env.SIPP_SPEAKER_AUDIO)
    : undefined,
  maxDurationMs: intEnv('SIPP_MAX_DURATION_MS'),
}).response;
writeFileSync(outputPath, response.audio);
console.log(`wrote ${response.durationMs} ms at ${response.sampleRateHz} Hz to ${outputPath}`);

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
