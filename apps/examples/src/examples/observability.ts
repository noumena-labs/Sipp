import { Example } from './base-example';

function formatMetric(value: number | null | undefined, digits = 0) {
  return typeof value === 'number' ? value.toFixed(digits) : 'n/a';
}

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
          log(`Speed: ${formatMetric(metrics.tokensPerSecond, 2)} t/s`, 'ai');
          log(`TTFT: ${formatMetric(metrics.ttftMs)}ms`, 'ai');
          log(`Prefill: ${formatMetric(metrics.prefillMs)}ms`, 'ai');
          log(`Decode: ${formatMetric(metrics.decodeMs)}ms`, 'ai');
          log(`Cache Hits: ${metrics.cacheHits} tokens`, 'ai');
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
      onToken: (tokens) => {
        for (const token of tokens) {
          fullResponse += token;
        }
        responseEl.innerText = fullResponse; // Update in real-time
      }
    });
  }
};
