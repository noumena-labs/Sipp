import {
  CogentClient,
  addGatewayRemote,
  printEmbedding,
  readRemoteArgs,
} from './_common.mjs';

const { alias, input } = readRemoteArgs('CogentClient remote embedding smoke input.');
const client = new CogentClient();
const endpoint = addGatewayRemote(client, alias);
const result = await client.embed({
  endpoint,
  input,
}).response;
printEmbedding(result);
