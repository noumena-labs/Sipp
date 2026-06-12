import { SippClient, type BrowserTextRun, type EndpointRef } from '@noumena-labs/sipp';
import {
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
  formatTextResult,
  readMaxTokens,
  readPrompt,
  readGatewayConfig,
  renderGatewayPage,
  reportError,
  write,
} from './common.js';

const elements = renderGatewayPage(
  'Gateway Query',
  'Write one sentence about gateway inference.',
  true
);

elements.runForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const config = readGatewayConfig(elements);
  if (config == null) return;
  const prompt = readPrompt(elements.promptInput);
  if (prompt == null) {
    write(elements.output, 'Enter input.');
    return;
  }

  const client = new SippClient();
  try {
    const endpoint = await client.add('gateway', { kind: 'gateway', ...config });
    const run = client.query(prompt, {
      endpoint,
      emitTokens: true,
      maxTokens: readMaxTokens(elements.maxTokensInput),
      temperature: DEFAULT_TEMPERATURE,
      topP: DEFAULT_TOP_P,
    });
    await streamTextRun(elements.output, endpoint, run);
  } catch (error) {
    reportError(elements.output, error);
  } finally {
    await client.close();
  }
});

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
