import {
  CogentClient,
  addOpenAiRemote,
  printText,
  readRemoteArgs,
  textOptions,
} from './_common.mjs';

const { model, input } = readRemoteArgs('Explain remote inference in one sentence.');
const client = new CogentClient();
const endpoint = addOpenAiRemote(client, model);
const run = client.chat({
  endpoint,
  messages: [
    { role: 'system', content: 'Answer concisely.' },
    { role: 'user', content: input },
  ],
  options: textOptions(),
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
