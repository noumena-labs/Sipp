# Node.js Package

The Node.js package target is `cogentlm-server`. It exposes the native
CogentLM client API to Node server processes. Applications own framework
routes, request validation, and deployment policy.

## Use It For

- Server-side local GGUF inference.
- Gateway-backed and provider-backed inference.
- Token streaming from Node processes.
- Backend selection for native bindings.

## Local GGUF Query

```ts
import { CogentClient } from 'cogentlm-server';

const client = new CogentClient();
await client.add('default', {
  kind: 'local',
  modelPath: process.argv[2],
  config: {
    context: { n_ctx: 2048 },
    scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
    cache: { mode: 'live_slot_prefix' },
    observability: { runtime_metrics: true },
  },
});

const run = client.query({
  prompt: 'Explain CogentLM in one sentence.',
  emitTokens: true,
  options: { maxTokens: 64, temperature: 0.7 },
  local: { contextKey: 'node-local' },
});

let streamed = '';
for await (const batch of run) {
  streamed += batch.text;
}
const response = await run.response;
console.log(streamed || response.text);
await client.close();
```

Set `COGENTLM_NODE_BACKEND=cpu|vulkan|cuda|metal` to choose a native backend.

## Gateway

Register a gateway endpoint when a Node process calls a separate CogentLM
gateway or provider-backed target. Gateway examples live under `examples/node`.

## Related Docs

- [Local Inference](../guides/local-inference.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Testing](../testing.md)
