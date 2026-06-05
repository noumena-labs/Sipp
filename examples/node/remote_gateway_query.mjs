import {
  CogentClient,
  addGatewayRemote,
  printTextRun,
  readRemoteArgs,
  textOptions,
} from './_common.mjs';

const { alias, input } = readRemoteArgs('Write one sentence about remote inference.');
const client = new CogentClient();
const endpoint = addGatewayRemote(client, alias);
await printTextRun(client.query({
  endpoint,
  prompt: input,
  options: textOptions(),
}));
