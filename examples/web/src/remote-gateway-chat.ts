import {
  chatMessages,
  createClient,
  readMaxTokens,
  readPrompt,
  readRemoteGatewayConfig,
  renderRemoteGatewayPage,
  reportError,
  streamTextRun,
  textRunOptions,
  write,
} from './common.js';

const elements = renderRemoteGatewayPage('Remote Gateway Chat', 'Explain remote inference in one sentence.', true);

elements.runForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const config = readRemoteGatewayConfig(elements);
  if (config == null) {
    return;
  }
  const prompt = readPrompt(elements.promptInput);
  if (prompt == null) {
    write(elements.output, 'Enter input.');
    return;
  }

  const client = createClient();
  try {
    const endpoint = client.addRemote(config.alias, config);
    const run = client.chat(chatMessages(prompt), {
      endpoint,
      ...textRunOptions(readMaxTokens(elements.maxTokensInput)),
    });
    await streamTextRun(elements.output, endpoint, run);
  } catch (error) {
    reportError(elements.output, error);
  } finally {
    await client.close();
  }
});
