import { loadClient, printEmbedding, readArgs } from './_common.mjs';

const { model, input } = readArgs('CogentClient embedding smoke input.');
const client = await loadClient(model, { embeddings: true });
const result = await client.embed({
  input,
  local: {
    contextKey: 'node-embed-smoke',
    normalize: true,
  },
}).response;
printEmbedding(result);
