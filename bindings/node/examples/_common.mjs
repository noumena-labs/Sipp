import native from '../router.js';

export const {
  CogentClient,
  backendObservabilityJson,
  setLlamaLogQuiet,
} = native;

export function readArgs(defaultInput) {
  const model = process.argv[2];
  if (!model) {
    console.error('usage: node examples/<query|chat|embed>.mjs <model.gguf> [input]');
    process.exit(2);
  }
  return {
    model,
    input: process.argv.slice(3).join(' ') || defaultInput,
  };
}

export async function loadClient(model, { embeddings = false } = {}) {
  setLlamaLogQuiet(true);
  console.log(`backend_before_load=${backendObservabilityJson(true)}`);
  const client = new CogentClient();
  await client.addLocal('default', model, runtimeConfig({ embeddings }));
  console.log(`backend_after_load=${backendObservabilityJson(true)}`);
  return client;
}

export function readRemoteArgs(defaultInput) {
  const alias = process.argv[2];
  if (!alias) {
    console.error('usage: node examples/remote_<query|chat|embed>.mjs <gateway-alias> [input]');
    process.exit(2);
  }
  return {
    alias,
    input: process.argv.slice(3).join(' ') || defaultInput,
  };
}

export function addGatewayRemote(client, alias) {
  return client.addRemote(alias, {
    alias,
    baseUrl: requiredEnv('COGENTLM_GATEWAY_URL'),
    token: requiredEnv('COGENTLM_GATEWAY_TOKEN'),
  });
}

export function textOptions() {
  return {
    maxTokens: intEnv('COGENTLM_MAX_TOKENS', 128),
    temperature: numberEnv('COGENTLM_TEMPERATURE', 0.7),
    topP: numberEnv('COGENTLM_TOP_P', 0.8),
  };
}

export async function printTextRun(run) {
  const result = await run.response;
  printText(result);
}

export function printText(result) {
  console.log(`endpoint=${JSON.stringify(result.endpoint)}`);
  console.log(`finish_reason=${result.finishReason}`);
  console.log(`text=${result.text.trim()}`);
  if (result.localStats) {
    console.log(
      `metrics=ttft_ms:${result.localStats.ttftMs} ` +
      `decode_ms:${result.localStats.decodeMs.toFixed(3)} ` +
      `output_tokens:${result.localStats.outputTokens} ` +
      `e2e_tps:${result.localStats.e2eTokensPerSecond} ` +
      `decode_tps:${result.localStats.decodeTokensPerSecond}`
    );
  }
}

export function printEmbedding(result) {
  const preview = result.values.slice(0, 8).map((value) => value.toFixed(6)).join(', ');
  console.log(`endpoint=${JSON.stringify(result.endpoint)}`);
  console.log(`dimensions=${result.values.length}`);
  console.log(`pooling=${result.pooling}`);
  console.log(`normalized=${result.normalized}`);
  console.log(`preview=[${preview}]`);
}

function runtimeConfig({ embeddings }) {
  return {
    placement: {
      gpu_layers: gpuLayers(),
    },
    context: {
      n_ctx: intEnv('COGENTLM_CONTEXT', 2048),
      n_threads: intEnv('COGENTLM_THREADS'),
      n_threads_batch: intEnv('COGENTLM_THREADS'),
      embeddings,
    },
    sampling: {
      temperature: numberEnv('COGENTLM_TEMPERATURE', 0.7),
      seed: intEnv('COGENTLM_SEED', 42),
    },
    scheduler: {
      continuous_batching: true,
      prefill_chunk_size: 0,
    },
    cache: {
      mode: 'live_slot_prefix',
    },
    multimodal: {},
    residency: {
      max_gpu_models_per_device: 1,
    },
    observability: {
      runtime_metrics: true,
    },
  };
}

function gpuLayers() {
  const value = process.env.COGENTLM_GPU_LAYERS;
  if (value === 'all' || value === 'auto') return value;
  return value == null ? undefined : { count: Number(value) };
}

function intEnv(name, fallback = undefined) {
  return process.env[name] == null ? fallback : Number.parseInt(process.env[name], 10);
}

function numberEnv(name, fallback = undefined) {
  return process.env[name] == null ? fallback : Number(process.env[name]);
}

function requiredEnv(name) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`${name} is required`);
  }
  return value;
}
