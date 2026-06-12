export const DEFAULT_MAX_TOKENS = 2048;
export const DEFAULT_TEMPERATURE = 0.7;
export const DEFAULT_TOP_P = 0.8;
export const DEFAULT_CONTEXT = 2048;
export const DEFAULT_SEED = 42;

export function readLocalArgs(command, defaultInput) {
  const model = process.argv[2];
  if (!model) {
    console.error(`usage: node examples/node/${command}.mjs <model.gguf> [input]`);
    process.exit(2);
  }
  return {
    model,
    input: process.argv.slice(3).join(' ') || defaultInput,
  };
}

export function readVisionArgs(defaultInput) {
  const model = process.argv[2];
  const projector = process.argv[3];
  const image = process.argv[4];
  if (!model || !projector || !image) {
    console.error(
      'usage: node examples/node/vision_chat.mjs <model.gguf> <projector.gguf> <image> [input]',
    );
    process.exit(2);
  }
  return {
    model,
    projector,
    image,
    input: process.argv.slice(5).join(' ') || defaultInput,
  };
}

export function readGatewayArgs(command, defaultInput) {
  const model = process.argv[2];
  const target = process.argv[3];
  if (!model || !target) {
    console.error(
      `usage: node examples/node/${command}.mjs <model.gguf> <gateway-target> [input]`,
    );
    process.exit(2);
  }
  return {
    model,
    target,
    input: process.argv.slice(4).join(' ') || defaultInput,
  };
}

export function requiredEnv(name) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`${name} is required`);
  }
  return value;
}

export function intEnv(name, fallback = undefined) {
  return process.env[name] == null ? fallback : Number.parseInt(process.env[name], 10);
}

export function numberEnv(name, fallback = undefined) {
  return process.env[name] == null ? fallback : Number(process.env[name]);
}

export function gpuLayers() {
  const value = process.env.COGENTLM_GPU_LAYERS;
  if (value === 'all' || value === 'auto') return value;
  return value == null ? undefined : { count: Number(value) };
}

export function printText(result) {
  console.log(`endpoint=${JSON.stringify(result.endpoint)}`);
  console.log(`finish_reason=${result.finishReason}`);
  console.log(`text=${result.text.trim()}`);
  if (result.localStats) {
    console.log(
      `metrics=ttft_ms:${formatOptionalMetric(result.localStats.ttftMs)} ` +
      `decode_ms:${formatOptionalMetric(result.localStats.decodeMs)} ` +
      `output_tokens:${result.localStats.outputTokens} ` +
      `e2e_tps:${formatOptionalMetric(result.localStats.e2eTokensPerSecond)} ` +
      `decode_tps:${formatOptionalMetric(result.localStats.decodeTokensPerSecond)}`
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

function formatOptionalMetric(value) {
  return typeof value === 'number' ? value.toFixed(3) : 'n/a';
}
