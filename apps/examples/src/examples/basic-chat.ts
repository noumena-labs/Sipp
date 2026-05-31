import { Example } from './base-example';

export const basicChatExample: Example = {
  id: '01-basic-chat',
  title: 'Basic Chat',
  description: 'A simple demonstration of chat with interactive token delivery.',
  run: async ({ log }) => {
    log('Example loaded. Type a message in the console to start chatting.', 'system');
  },
  onUserInput: async ({ client, log, userInput }) => {
    log(userInput, 'user');

    let fullResponse = '';
    const responseEl = log('', 'ai'); // Create persistent element for live tokens

    try {
      const run = client.chat([
        { role: 'user', content: userInput }
      ], {
        tokenDelivery: 'interactive',
      });

      for await (const batch of run.tokens) {
        fullResponse += batch.text;
        responseEl.innerText = fullResponse; // Update in real-time
      }

      const result = await run.response;
      responseEl.innerText = result.text;
    } catch (err) {
      log(`Error: ${err}`, 'error');
    }
  }
};
