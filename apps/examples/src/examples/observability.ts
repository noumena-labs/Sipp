import { Example } from './base-example';

export const observabilityExample: Example = {
  id: '04-observability',
  title: 'Observability',
  description: 'Monitoring real-time performance metrics like tokens/sec and TTFT.',
  run: async ({ engine, log }) => {
    log('Subscribing to engine observability events...', 'system');
    
    engine.observability.subscribe((event) => {
      if (event.type === 'query-complete') {
        const metrics = event.snapshot.runtime;
        if (metrics) {
          log(`--- Performance Report ---`, 'dim');
          log(`Speed: ${metrics.tokensPerSecond.toFixed(2)} t/s`, 'ai');
          log(`TTFT: ${metrics.ttftMs.toFixed(0)}ms`, 'ai');
          log(`Prompt Eval: ${metrics.promptEvalMs.toFixed(0)}ms`, 'ai');
          log(`Prefix Cache Hits: ${metrics.prefixCacheHitCount}`, 'ai');
          log(`-------------------------`, 'dim');
        }
      }
    });
    
    log('Observability active. Send any chat message to see metrics after completion.', 'system');
  },
  onUserInput: async ({ engine, log, userInput }) => {
    log(userInput, 'user');
    
    let fullResponse = '';
    const responseEl = log('', 'ai'); // Create persistent element for streaming
    await engine.chat([{ role: 'user', content: userInput }], {
      onToken: (t) => {
        fullResponse += t;
        responseEl.innerText = fullResponse; // Update in real-time
      }
    });
  }
};
