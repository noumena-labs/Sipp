import { loadClient, printText, readArgs, textOptions } from './_common.mjs';

const { model, input } = readArgs('Explain the CogentClient API in one sentence.');
const client = await loadClient(model);
const run = client.chat({
  messages: [
    { role: 'system', content: 'Answer concisely.' },
    { role: 'user', content: input },
  ],
  options: textOptions(),
  local: {
    contextKey: 'node-chat-smoke',
  },
  streamTokens: true,
});
let streamed = '';
for await (const batch of run.tokens) {
  process.stdout.write(batch.text);
  streamed += batch.text;
}
process.stdout.write('\n');
const result = await run.response;
if (streamed !== result.text) {
  throw new Error('streamed token batches did not match final response text');
}
printText(result);
