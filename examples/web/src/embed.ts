import {
  CogentClient,
  type BrowserEmbeddingRun,
  type NativeRuntimeConfig,
} from '@noumena-labs/cogentlm';
import {
  EXAMPLE_LOCAL_ENDPOINT,
  formatEmbeddingResult,
  readModelSource,
  readPrompt,
  renderLocalPage,
  reportError,
  write,
} from './common.js';

const elements = renderLocalPage('Local Embed', 'CogentClient embedding example input.', false);
const client = new CogentClient();
let modelLoaded = false;

elements.loadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const source = readModelSource(elements.modelInput, elements.modelFileInput);
  if (source == null) {
    write(elements.output, 'Enter a GGUF model URL, path, or file.');
    return;
  }

  try {
    write(elements.output, 'Loading model...');
    const info = await client.addLocal(source, { runtime: runtimeConfig() });
    modelLoaded = true;
    write(elements.output, `Loaded ${info.name}.`);
  } catch (error) {
    reportError(elements.output, error);
  }
});

elements.runForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!modelLoaded) {
    write(elements.output, 'Load a model before running embed.');
    return;
  }
  const input = readPrompt(elements.promptInput);
  if (input == null) {
    write(elements.output, 'Enter input.');
    return;
  }

  try {
    // Embeddings return a vector instead of generated text.
    const run = client.embed(input, {
      contextKey: 'web-embed-example',
      normalize: true,
    });
    await printEmbeddingRun(elements.output, run);
  } catch (error) {
    reportError(elements.output, error);
  }
});

function runtimeConfig(): NativeRuntimeConfig {
  return {
    context: { n_ctx: 2048 },
    scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
    cache: { mode: 'live_slot_prefix' },
    observability: { runtime_metrics: true },
  };
}

async function printEmbeddingRun(output: HTMLPreElement, run: BrowserEmbeddingRun): Promise<void> {
  const result = await run.response;
  write(output, formatEmbeddingResult(EXAMPLE_LOCAL_ENDPOINT, result));
}
