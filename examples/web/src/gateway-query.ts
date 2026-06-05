import { CogentClient, type BrowserTextRun, type EndpointRef } from '@noumena-labs/cogentlm';
import {
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
  formatTextResult,
  readMaxTokens,
  readPrompt,
  readRemoteGatewayConfig,
  renderRemoteGatewayPage,
  reportError,
  write,
} from './common.js';

const elements = renderRemoteGatewayPage('Gateway Query', 'Write one sentence about gateway inference.', true);

elements.runForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const config = readRemoteGatewayConfig(elements);
  if (config == null) return;
  const prompt = readPrompt(elements.promptInput);
  if (prompt == null) {
    write(elements.output, 'Enter input.');
    return;
  }

  const client = new CogentClient();
  try {
    // The app registers a remote endpoint from gateway URL, bearer token, and alias.
    const endpoint = client.addRemote(config.alias, config);
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
