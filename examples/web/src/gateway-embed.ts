import {
  EndpointDescriptor,
  SippClient,
  type BrowserEmbeddingRun,
} from '@noumena-labs/sipp';
import {
  formatEmbeddingResult,
  readPrompt,
  readGatewayConfig,
  renderGatewayPage,
  reportError,
  write,
} from './common.js';

const elements = renderGatewayPage(
  'Gateway Embed',
  'SippClient gateway embedding example input.',
  false
);

elements.runForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const config = readGatewayConfig(elements);
  if (config == null) return;
  const input = readPrompt(elements.promptInput);
  if (input == null) {
    write(elements.output, 'Enter input.');
    return;
  }

  const client = new SippClient();
  try {
    const endpoint = await client.add('gateway', EndpointDescriptor.gateway(config));
    const run = client.embed(input, { endpoint });
    await printEmbeddingRun(elements.output, run);
  } catch (error) {
    reportError(elements.output, error);
  } finally {
    await client.close();
  }
});

async function printEmbeddingRun(
  output: HTMLPreElement,
  run: BrowserEmbeddingRun
): Promise<void> {
  const result = await run.response;
  write(output, formatEmbeddingResult('gateway', result));
}
