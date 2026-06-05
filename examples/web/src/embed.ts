import {
  createClient,
  EXAMPLE_LOCAL_ENDPOINT,
  loadLocalModel,
  printEmbeddingRun,
  readModelSource,
  readPrompt,
  renderLocalPage,
  reportError,
  write,
} from './common.js';

const elements = renderLocalPage('Local Embed', 'CogentClient embedding smoke input.', false);
const client = createClient();
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
    const info = await loadLocalModel(client, source);
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
    const run = client.embed(input, {
      contextKey: 'web-embed-smoke',
      normalize: true,
    });
    await printEmbeddingRun(elements.output, EXAMPLE_LOCAL_ENDPOINT, run);
  } catch (error) {
    reportError(elements.output, error);
  }
});
