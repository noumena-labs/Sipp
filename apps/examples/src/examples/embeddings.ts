import { Example } from './base-example';

function preview(values: readonly number[]): string {
  return values.slice(0, 8).map((value) => value.toFixed(4)).join(', ');
}

export const embeddingsExample: Example = {
  id: '06-embeddings',
  title: 'Embeddings',
  description: 'Runs vector extraction through the embed API.',
  run: async ({ log }) => {
    log('Embeddings example loaded. Send text to run client.embed().', 'system');
  },
  onUserInput: async ({ client, log, userInput }) => {
    log(userInput, 'user');

    try {
      const result = await client.embed(userInput, {
        contextKey: 'examples:embeddings',
        normalize: true,
      }).response;
      log(`dimensions: ${result.values.length}`, 'ai');
      log(`pooling: ${result.pooling}`, 'ai');
      log(`normalized: ${result.normalized}`, 'ai');
      log(`values: [${preview(result.values)}${result.values.length > 8 ? ', ...' : ''}]`, 'ai');
    } catch (err) {
      log(`Error: ${err}`, 'error');
    }
  },
};
