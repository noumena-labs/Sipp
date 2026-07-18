import {
  EndpointDescriptor,
  SippClient,
  type BrowserTextRun,
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
  readGatewayConfig,
  renderGatewayLocalPage,
  reportError,
  write,
} from './common.js';

const elements = renderGatewayLocalPage('Compare browser-local and gateway-local inference.');
const localClient = new SippClient();
let localEndpoint: EndpointRef | null = null;

elements.loadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    write(elements.localOutput, 'Loading browser model...');
    const model = await addModel(
      localClient,
      elements.modelUrlInput,
      elements.modelFileInput
    );
    if (model == null) {
      write(elements.localOutput, 'Enter a GGUF model URL, path, or file.');
      return;
    }
    localEndpoint = await localClient.add(
      EXAMPLE_LOCAL_ENDPOINT_ID,
      EndpointDescriptor.local(model.id, { runtime: runtimeConfig() })
    );
    write(elements.localOutput, `Loaded ${model.name}.`);
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
  const gateway = readGatewayConfig({ ...elements, output: elements.gatewayOutput });
  if (gateway == null) return;

  const gatewayClient = new SippClient();
  try {
    const gatewayEndpoint = await gatewayClient.add(
      'gateway',
      EndpointDescriptor.gateway(gateway)
    );
    const maxTokens = readMaxTokens(elements.maxTokensInput);

    if (localEndpoint != null) {
      const localRun = localClient.query(prompt, {
        endpoint: localEndpoint,
        emitTokens: true,
        maxTokens,
        contextKey: 'web-gateway-local-browser',
        temperature: DEFAULT_TEMPERATURE,
        topP: DEFAULT_TOP_P,
      });
      await streamTextRun(elements.localOutput, EXAMPLE_LOCAL_ENDPOINT_ID, localRun);
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
    await streamTextRun(elements.gatewayOutput, 'gateway', gatewayRun);
  } catch (error) {
    reportError(elements.gatewayOutput, error);
  } finally {
    await gatewayClient.close();
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

async function streamTextRun(
  output: HTMLPreElement,
  endpointId: string,
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
  write(output, formatTextResult(endpointId, result));
}
