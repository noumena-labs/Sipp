import {
  CogentClient,
  addOpenAiRemote,
  printEmbedding,
  readRemoteArgs,
} from './_common.mjs';

const { model, input } = readRemoteArgs('CogentClient remote embedding smoke input.');
const client = new CogentClient();
const endpoint = addOpenAiRemote(client, model);
const result = await client.embed({
  endpoint,
  input,
}).response;
printEmbedding(result);
