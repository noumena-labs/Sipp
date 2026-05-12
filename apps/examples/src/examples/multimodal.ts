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
await engine.chat({
  messages: [{ role: 'user', content: 'Describe this image.' }],
  media: [imageUint8Array]
});
    `, 'dim');
  },
  onUserInput: async ({ engine, log, userInput, media }) => {
    log(userInput, 'user');

    let fullResponse = '';
    const responseEl = log('', 'ai');

    try {
      // If media is present, use the multimodal structure; otherwise, standard chat
      const chatInput = (media && media.length > 0)
        ? { messages: [{ role: 'user', content: userInput }], media }
        : [{ role: 'user', content: userInput }];

      await engine.chat(chatInput as any, {
        onToken: (tokens) => {
          for (const token of tokens) {
            fullResponse += token;
          }
          responseEl.innerText = fullResponse;
        }
      });
    } catch (err) {
      log(`Error: ${err}`, 'error');
    }
  }
};
