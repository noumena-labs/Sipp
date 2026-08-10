import {
  Endpoint,
  SippClient,
  type BrowserTextRun,
  type ChatMessage,
  type EndpointRef,
  type NativeRuntimeConfig,
} from '@noumena-labs/sipp';
import {
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
  EXAMPLE_LOCAL_ENDPOINT_ID,
  formatTextResult,
  readMaxTokens,
  addModel,
  readPrompt,
  renderLocalPage,
  reportError,
  write,
} from './common.js';

const elements = renderLocalPage('Local Chat', 'Explain the SippClient API in one sentence.', true);
const client = new SippClient();
let endpoint: EndpointRef | null = null;

elements.loadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    write(elements.output, 'Loading model...');
    const model = await addModel(client, elements.modelUrlInput, elements.modelFileInput);
    if (model == null) {
      write(elements.output, 'Enter a GGUF model URL, path, or file.');
      return;
    }
    endpoint = await client.add(
      EXAMPLE_LOCAL_ENDPOINT_ID,
      Endpoint.local(model, { runtime: runtimeConfig() })
    );
    write(elements.output, `Loaded ${model.name}.`);
  } catch (error) {
    reportError(elements.output, error);
  }
});

elements.runForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (endpoint == null) {
    write(elements.output, 'Load a model before running chat.');
    return;
  }
  const prompt = readPrompt(elements.promptInput);
  if (prompt == null) {
    write(elements.output, 'Enter input.');
    return;
  }

  try {
    // `chat` sends role-tagged messages and streams token batches as they arrive.
    const run = client.chat(chatMessages(prompt), {
      endpoint,
      emitTokens: true,
      maxTokens: readMaxTokens(elements.maxTokensInput),
      contextKey: 'web-chat-example',
      temperature: DEFAULT_TEMPERATURE,
      topP: DEFAULT_TOP_P,
    });
    await streamTextRun(elements.output, run);
  } catch (error) {
    reportError(elements.output, error);
  }
});

function runtimeConfig(): NativeRuntimeConfig {
  return {
    context: { n_ctx: 4096 },
    scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
    cache: { mode: 'live_slot_prefix' },
    observability: { runtime_metrics: true },
  };
}

function chatMessages(prompt: string): readonly ChatMessage[] {
  return [
    { role: 'system', content: 'Answer concisely.' },
    { role: 'user', content: prompt },
  ];
}

async function streamTextRun(
  output: HTMLPreElement,
  run: BrowserTextRun
): Promise<void> {
  write(output, '');
  let streamed = '';
  for await (const batch of run.tokens) {
    output.textContent += batch.text;
    streamed += batch.text;
  }
  const result = await run.response;
  if (streamed !== '' && streamed !== result.text) {
    throw new Error('streamed token batches did not match final response text');
  }
  write(output, formatTextResult(EXAMPLE_LOCAL_ENDPOINT_ID, result));
}
