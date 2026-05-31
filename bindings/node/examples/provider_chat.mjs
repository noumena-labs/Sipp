import {
  loadOpenAiProviderClient,
  printText,
  providerEndpoint,
  readProviderArgs,
  textOptions,
} from './_common.mjs';

const { model, input } = readProviderArgs('Explain provider inference in one sentence.');
const client = loadOpenAiProviderClient(model);
const run = client.chat({
  endpoint: providerEndpoint(model),
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
