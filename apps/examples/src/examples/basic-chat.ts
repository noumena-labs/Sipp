import { Example } from './base-example';

export const basicChatExample: Example = {
  id: '01-basic-chat',
  title: 'Basic Chat',
  description: 'A simple demonstration of the chat API with streaming support.',
  run: async ({ log }) => {
    log('Example loaded. Type a message in the console to start chatting.', 'system');
  },
  onUserInput: async ({ engine, log, userInput }) => {
    log(userInput, 'user');
    
    let fullResponse = '';
    const responseEl = log('', 'ai'); // Create persistent element for streaming
    
    try {
      await engine.chat([
        { role: 'user', content: userInput }
      ], {
        onToken: (token) => {
          fullResponse += token;
          responseEl.innerText = fullResponse; // Update in real-time
        }
      });
    } catch (err) {
      log(`Error: ${err}`, 'error');
    }
  }
};
