import native from '../router.js';

const {
  CogentClient,
  backendObservabilityJson,
  setLlamaLogQuiet
} = native;

const args = process.argv.slice(2);
const positionalArgs = [];
let gpuLayers = process.env.COGENTLM_GPU_LAYERS == null ? undefined : Number(process.env.COGENTLM_GPU_LAYERS);

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--gpu-layers') {
    gpuLayers = Number(args[++i]);
  } else {
    positionalArgs.push(args[i]);
  }
}

const model = positionalArgs[0];
const prompt = positionalArgs[1] ?? 'Describe browser LLM inference.';

if (!model) {
  console.error('usage: node examples/node_smoke.mjs <model.gguf> [prompt] [--gpu-layers N]');
  process.exit(2);
}

setLlamaLogQuiet(true);
console.log(`backend_before_load=${backendObservabilityJson(true)}`);

const client = new CogentClient();

await client.loadEngine('default', model, {
  placement: {
    gpu_layers: gpuLayers == null ? undefined : { count: gpuLayers }
  },
  context: {
    n_ctx: 2048
  },
  sampling: {
    temperature: 0.7,
    seed: 42
  }
});
console.log(`backend_after_load=${backendObservabilityJson(true)}`);
const pieces = [];
const run = client.chat({
  messages: [{ role: 'user', content: prompt }],
  options: {
    maxTokens: Number(process.env.COGENTLM_MAX_TOKENS ?? 1024)
  },
  streamTokens: true
});
for await (const batch of run.tokens) {
  pieces.push(batch.text);
  process.stdout.write(batch.text);
}
const result = await run.response;
process.stdout.write('\n');
if (pieces.join('') !== result.text) {
  throw new Error('streamed token batches did not match final response text');
}
const stats = result.localStats;
if (stats == null) {
  throw new Error('local CogentClient response did not include localStats');
}
const decodeMs = stats.decodeMs;
const outputTokens = stats.outputTokens;

console.log(`endpoint=${JSON.stringify(result.endpoint)}`);
console.log(`finish_reason=${result.finishReason}`);
console.log(`stream_batches=${pieces.length}`);
console.log(`output=${result.text.trim()}`);
console.log(
  `metrics=ttft_ms:${stats.ttftMs} ` +
  `decode_ms:${decodeMs.toFixed(3)} ` +
  `output_tokens:${outputTokens} ` +
  `tps:${stats.tokensPerSecond}`
);
