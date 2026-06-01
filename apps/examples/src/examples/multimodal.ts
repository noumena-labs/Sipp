import type { ChatInput } from '@noumena-labs/cogentlm-browser';
import { Example } from './base-example';

export const multimodalExample: Example = {
  id: '02-multimodal',
  title: 'Multimodal Vision',
  description: 'Demonstrates how to pass image data to vision-language models.',
  run: async ({ log }) => {
    log('This example demonstrates vision capabilities.', 'system');
    log('To use this, ensure you loaded a multimodal model (LLM + Projector).', 'dim');
    log('Example usage in code:', 'dim');
    log(`
const run = client.chat({
  messages: [{ role: 'user', content: 'Describe this image.' }],
  media: [imageUint8Array]
});
const result = await run.response;
    `, 'dim');
  },
  onUserInput: async ({ client, log, userInput, media }) => {
    log(userInput, 'user');

    let fullResponse = '';
    const responseEl = log('', 'ai');

    try {
      // If media is present, use the multimodal structure; otherwise, standard chat
      const chatInput: ChatInput = (media && media.length > 0)
        ? { messages: [{ role: 'user', content: userInput }], media }
        : [{ role: 'user', content: userInput }];

      const run = client.chat(chatInput, {
        emitTokens: true,
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
  }
};
