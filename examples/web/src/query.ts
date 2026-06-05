import {
  createClient,
  EXAMPLE_LOCAL_ENDPOINT,
  loadLocalModel,
  localTextRunOptions,
  readMaxTokens,
  readModelSource,
  readPrompt,
  renderLocalPage,
  reportError,
  streamTextRun,
  write,
} from './common.js';

const elements = renderLocalPage('Local Query', 'Write one sentence about local browser inference.', true);
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
    write(elements.output, 'Load a model before running a query.');
    return;
  }
  const prompt = readPrompt(elements.promptInput);
  if (prompt == null) {
    write(elements.output, 'Enter input.');
    return;
  }

  try {
    const run = client.query(
      prompt,
      localTextRunOptions('web-query-smoke', readMaxTokens(elements.maxTokensInput))
    );
    await streamTextRun(elements.output, EXAMPLE_LOCAL_ENDPOINT, run);
  } catch (error) {
    reportError(elements.output, error);
  }
});
