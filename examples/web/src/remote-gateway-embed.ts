import {
  createClient,
  printEmbeddingRun,
  readPrompt,
  readRemoteGatewayConfig,
  renderRemoteGatewayPage,
  reportError,
  write,
} from './common.js';

const elements = renderRemoteGatewayPage(
  'Remote Gateway Embed',
  'CogentClient remote embedding smoke input.',
  false
);

elements.runForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const config = readRemoteGatewayConfig(elements);
  if (config == null) {
    return;
  }
  const input = readPrompt(elements.promptInput);
  if (input == null) {
    write(elements.output, 'Enter input.');
    return;
  }

  const client = createClient();
  try {
    const endpoint = client.addRemote(config.alias, config);
    const run = client.embed(input, {
      endpoint,
    });
    await printEmbeddingRun(elements.output, endpoint, run);
  } catch (error) {
    reportError(elements.output, error);
  } finally {
    await client.close();
  }
});
