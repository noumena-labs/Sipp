import { Example } from './base-example';

function preview(values: readonly number[]): string {
  return values.slice(0, 8).map((value) => value.toFixed(4)).join(', ');
}

export const embeddingsExample: Example = {
  id: '06-embeddings',
  title: 'Embeddings',
  description: 'Runs vector extraction through the embed API.',
  run: async ({ log }) => {
    log('Embeddings example loaded. Send text to run engine.embed().', 'system');
  },
  onUserInput: async ({ engine, log, userInput }) => {
    log(userInput, 'user');

    try {
      const result = await engine.embed(userInput, {
        contextKey: 'examples:embeddings',
        normalize: true,
      });
      log(`dimensions: ${result.values.length}`, 'ai');
      log(`pooling: ${result.pooling}`, 'ai');
      log(`normalized: ${result.normalized}`, 'ai');
      log(`values: [${preview(result.values)}${result.values.length > 8 ? ', ...' : ''}]`, 'ai');
    } catch (err) {
      log(`Error: ${err}`, 'error');
    }
  },
};
