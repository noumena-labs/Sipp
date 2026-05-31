import {
  CogentClient,
  addOpenAiRemote,
  printTextRun,
  readRemoteArgs,
  textOptions,
} from './_common.mjs';

const { model, input } = readRemoteArgs('Write one sentence about remote inference.');
const client = new CogentClient();
const endpoint = addOpenAiRemote(client, model);
await printTextRun(client.query({
  endpoint,
  prompt: input,
  options: textOptions(),
}));
