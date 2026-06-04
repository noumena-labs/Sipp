import { readFileSync } from 'node:fs';

import { loadClient, printText, readVisionArgs, textOptions } from './_common.mjs';

const { model, projector, image, input } = readVisionArgs('Describe this image in one sentence.');
const client = await loadClient(model, { projectorPath: projector });
const run = client.chat({
  messages: [
    { role: 'user', content: input },
  ],
  options: textOptions(),
  local: {
    contextKey: 'node-vision-chat-smoke',
    media: [readFileSync(image)],
  },
  emitTokens: true,
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
