import {
  CogentClient,
  type BrowserTextRun,
  type EndpointRef,
  type NativeRuntimeConfig,
} from '@noumena-labs/cogentlm';
import {
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
  EXAMPLE_LOCAL_ENDPOINT,
  formatTextResult,
  readMaxTokens,
  readModelSource,
  readPrompt,
  readRemoteGatewayConfig,
  renderGatewayLocalPage,
  reportError,
  write,
} from './common.js';

const elements = renderGatewayLocalPage('Compare browser-local and gateway-local inference.');
const localClient = new CogentClient();
let localModelLoaded = false;

elements.loadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const source = readModelSource(elements.modelInput, elements.modelFileInput);
  if (source == null) {
    write(elements.localOutput, 'Enter a GGUF model URL, path, or file.');
    return;
  }

  try {
    write(elements.localOutput, 'Loading browser model...');
    const info = await localClient.addLocal(source, { runtime: runtimeConfig() });
    localModelLoaded = true;
    write(elements.localOutput, `Loaded ${info.name}.`);
  } catch (error) {
    reportError(elements.localOutput, error);
  }
});

elements.runForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const prompt = readPrompt(elements.promptInput);
  if (prompt == null) {
    write(elements.localOutput, 'Enter input.');
    write(elements.gatewayOutput, 'Enter input.');
    return;
  }
  const gateway = readRemoteGatewayConfig({ ...elements, output: elements.gatewayOutput });
  if (gateway == null) return;

  const gatewayClient = new CogentClient();
  try {
    const gatewayEndpoint = gatewayClient.addRemote(gateway.alias, gateway);
    const maxTokens = readMaxTokens(elements.maxTokensInput);

    if (localModelLoaded) {
      const localRun = localClient.query(prompt, {
        emitTokens: true,
        maxTokens,
        session: 'web-gateway-local-browser',
        temperature: DEFAULT_TEMPERATURE,
        topP: DEFAULT_TOP_P,
      });
      await streamTextRun(elements.localOutput, EXAMPLE_LOCAL_ENDPOINT, localRun);
    } else {
      write(elements.localOutput, 'Load a browser model to run local browser inference.');
    }

    const gatewayRun = gatewayClient.query(prompt, {
      endpoint: gatewayEndpoint,
      emitTokens: true,
      maxTokens,
      temperature: DEFAULT_TEMPERATURE,
      topP: DEFAULT_TOP_P,
    });
    await streamTextRun(elements.gatewayOutput, gatewayEndpoint, gatewayRun);
  } catch (error) {
    reportError(elements.gatewayOutput, error);
  } finally {
    await gatewayClient.close();
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

async function streamTextRun(
  output: HTMLPreElement,
  endpoint: EndpointRef,
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
  write(output, formatTextResult(endpoint, result));
}
