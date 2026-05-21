import native from '../index.js';

const {
  ModelService,
  backendObservabilityJson,
  setLlamaLogQuiet
} = native;

const args = process.argv.slice(2);
const positionalArgs = [];
let gpuLayers = process.env.COGENTLM_GPU_LAYERS == null ? undefined : Number(process.env.COGENTLM_GPU_LAYERS);
let backend = process.env.COGENTLM_BACKEND ?? 'auto';
let modelStore = process.env.COGENTLM_MODEL_STORE ?? `${process.env.TEMP ?? process.env.TMP ?? '.'}/cogentlm-rs-model-store`;

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--gpu-layers') {
    gpuLayers = Number(args[++i]);
  } else if (args[i] === '--backend') {
    backend = args[++i];
  } else if (args[i] === '--model-store') {
    modelStore = args[++i];
  } else {
    positionalArgs.push(args[i]);
  }
}

const model = positionalArgs[0];
const prompt = positionalArgs[1] ?? 'Describe browser LLM inference.';

if (!model) {
  console.error('usage: node examples/phase4_node_smoke.mjs <model.gguf> [prompt] [--gpu-layers N] [--backend auto|cpu|cuda|metal|vulkan|webgpu] [--model-store PATH]');
  process.exit(2);
}

setLlamaLogQuiet(true);
console.log(`backend_before_load=${backendObservabilityJson(true)}`);

const engine = new ModelService(modelStore);

try {
  const loaded = await engine.loadPath(model, {
    backend,
    stats: 'basic',
    runtime: {
      placement: {
        gpuLayers: gpuLayers == null ? undefined : String(gpuLayers)
      },
      context: {
        nCtx: 2048
      },
      sampling: {
        temperature: 0.7,
        seed: 42
      }
    }
  });
  console.log(`loaded_model=${JSON.stringify(loaded.model)}`);
  console.log(`selected_backend=${JSON.stringify(loaded.backend)}`);
  console.log(`backend_after_load=${backendObservabilityJson(true)}`);
  console.log(`engine_state_after_load=${JSON.stringify(await engine.state())}`);
  const pieces = [];
  const result = await engine.chat(
    [{ role: 'user', content: prompt }],
    { maxTokens: Number(process.env.COGENTLM_MAX_TOKENS ?? 1024) },
    batch => {
      pieces.push(batch.text);
      process.stdout.write(batch.text);
    }
  );
  process.stdout.write('\n');
  if (pieces.join('') !== result.text) {
    throw new Error('streamed token batches did not match final response text');
  }
  const decodeMs = result.stats.decodeMs;
  const outputTokens = result.stats.outputTokens;

  console.log(`finish_reason=${result.finishReason}`);
  console.log(`stream_batches=${pieces.length}`);
  console.log(`output=${result.text.trim()}`);
  console.log(`engine_state_after_chat=${JSON.stringify(await engine.state())}`);
  const eventCounts = Object.create(null);
  for (const event of engine.drainEvents()) {
    eventCounts[event.type] = (eventCounts[event.type] ?? 0) + 1;
  }
  console.log(`engine_events=${JSON.stringify(eventCounts)}`);
  console.log(
    `metrics=ttft_ms:${result.stats.ttftMs} ` +
      `decode_ms:${decodeMs.toFixed(3)} ` +
      `output_tokens:${outputTokens} ` +
      `tps:${result.stats.tokensPerSecond}`
  );
} finally {
  await engine.close();
}
