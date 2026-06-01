import { loadClient, printTextRun, readArgs, textOptions } from './_common.mjs';

const { model, input } = readArgs('Write one sentence about local inference.');
const client = await loadClient(model);
await printTextRun(client.query({
  prompt: input,
  options: textOptions(),
  local: {
    contextKey: 'node-query-smoke',
  },
}));
