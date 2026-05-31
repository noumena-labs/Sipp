import {
  loadOpenAiProviderClient,
  printTextRun,
  providerEndpoint,
  readProviderArgs,
  textOptions,
} from './_common.mjs';

const { model, input } = readProviderArgs('Write one sentence about provider inference.');
const client = loadOpenAiProviderClient(model);
await printTextRun(client.query({
  endpoint: providerEndpoint(model),
  prompt: input,
  options: textOptions(),
}));
