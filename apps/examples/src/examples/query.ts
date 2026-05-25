import type { CogentEngine } from '@noumena-labs/cogentlm';
import { Example } from './base-example';

const ENCODER_DECODER_PROMPT = 'translate English to German: The house is wonderful.';

function isEncoderDecoder(engine: CogentEngine): boolean {
  return engine.models.current()?.capabilities?.modelClass === 'encoder_decoder';
}

export const queryExample: Example = {
  id: '05-query',
  title: 'Query',
  description: 'Runs a raw prompt through the query API.',
  run: async ({ engine, log, inputElement }) => {
    if (isEncoderDecoder(engine) && inputElement.value.trim().length === 0) {
      inputElement.value = ENCODER_DECODER_PROMPT;
    }
    log('Query example loaded. Send a prompt to run engine.query().', 'system');
  },
  onUserInput: async ({ engine, log, userInput }) => {
    log(userInput, 'user');

    let fullResponse = '';
    const responseEl = log('', 'ai');

    try {
      await engine.query(userInput, {
        maxTokens: isEncoderDecoder(engine) ? 32 : 64,
        session: 'examples:query',
        onTokens: (batch) => {
          fullResponse += batch.text;
          responseEl.innerText = fullResponse;
        },
      });
    } catch (err) {
      log(`Error: ${err}`, 'error');
    }
  },
};
