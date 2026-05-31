import {
  loadOpenAiProviderClient,
  printEmbedding,
  providerEndpoint,
  readProviderArgs,
} from './_common.mjs';

const { model, input } = readProviderArgs('CogentClient provider embedding smoke input.');
const client = loadOpenAiProviderClient(model);
const result = await client.embed({
  endpoint: providerEndpoint(model),
  input,
}).response;
printEmbedding(result);
