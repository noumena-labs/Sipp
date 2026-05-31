import type { CogentClient } from '@noumena-labs/cogentlm-browser';
import { Example } from './base-example';

const ENCODER_DECODER_PROMPT = 'translate English to German: The house is wonderful.';

function isEncoderDecoder(client: CogentClient): boolean {
  return client.currentLocal()?.capabilities?.modelClass === 'encoder_decoder';
}

export const queryExample: Example = {
  id: '05-query',
  title: 'Query',
  description: 'Runs a raw prompt through the query API.',
  run: async ({ client, log, inputElement }) => {
    if (isEncoderDecoder(client) && inputElement.value.trim().length === 0) {
      inputElement.value = ENCODER_DECODER_PROMPT;
    }
    log('Query example loaded. Send a prompt to run client.query().', 'system');
  },
  onUserInput: async ({ client, log, userInput }) => {
    log(userInput, 'user');

    let fullResponse = '';
    const responseEl = log('', 'ai');

    try {
      const run = client.query(userInput, {
        maxTokens: isEncoderDecoder(client) ? 32 : 64,
        session: 'examples:query',
        streamTokens: true,
      });

      for await (const batch of run.tokens) {
        fullResponse += batch.text;
        responseEl.innerText = fullResponse;
      }

      const result = await run.response;
      responseEl.innerText = result.text;
    } catch (err) {
      log(`Error: ${err}`, 'error');
    }
  },
};
