import {
  EndpointDescriptor,
  SippClient,
  type BrowserEmbeddingRun,
  type NativeRuntimeConfig,
} from '@noumena-labs/sipp';
import {
  EXAMPLE_LOCAL_ENDPOINT,
  formatEmbeddingResult,
  installModel,
  readPrompt,
  renderLocalPage,
  reportError,
  write,
} from './common.js';

const elements = renderLocalPage('Local Embed', 'SippClient embedding example input.', false);
const client = new SippClient();
let modelLoaded = false;

elements.loadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    write(elements.output, 'Loading model...');
    const model = await installModel(client, elements.modelUrlInput, elements.modelFileInput);
    if (model == null) {
      write(elements.output, 'Enter a GGUF model URL, path, or file.');
      return;
    }
    await client.add(
      EXAMPLE_LOCAL_ENDPOINT.id,
      EndpointDescriptor.local(model.id, { runtime: runtimeConfig() })
    );
    modelLoaded = true;
    write(elements.output, `Loaded ${model.name}.`);
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
    context: { n_ctx: 4096, embeddings: true, pooling: 'mean' },
    scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
    cache: { mode: 'live_slot_prefix' },
    observability: { runtime_metrics: true },
  };
}

async function printEmbeddingRun(output: HTMLPreElement, run: BrowserEmbeddingRun): Promise<void> {
  const result = await run.response;
  write(output, formatEmbeddingResult(EXAMPLE_LOCAL_ENDPOINT, result));
}
