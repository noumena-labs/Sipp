import { CogentClient, type BrowserEmbeddingRun, type EndpointRef } from '@noumena-labs/cogentlm';
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
  'CogentClient gateway embedding example input.',
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

  const client = new CogentClient();
  try {
    const endpoint = await client.add('gateway', { kind: 'gateway', ...config });
    const run = client.embed(input, { endpoint });
    await printEmbeddingRun(elements.output, endpoint, run);
  } catch (error) {
    reportError(elements.output, error);
  } finally {
    await client.close();
  }
});

async function printEmbeddingRun(
  output: HTMLPreElement,
  endpoint: EndpointRef,
  run: BrowserEmbeddingRun
): Promise<void> {
  const result = await run.response;
  write(output, formatEmbeddingResult(endpoint, result));
}
