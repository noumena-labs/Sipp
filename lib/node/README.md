# CogentLM Server for Node.js

`lib/node` is the Node.js package source for the public `cogentlm-server`
package. It exposes CogentLM's native client API to Node server processes for
local GGUF inference, gateway-backed inference, provider descriptors, and token
streaming.

Source builds use the workspace manifest in this directory. Public docs use the
`cogentlm-server` package target. Applications own framework routes and call
`client.query()`, `client.chat()`, or `client.embed()` inside those routes.

## Source Build

From the repository root:

```bash
cargo xtask build node --backend cpu
node examples/node/query.mjs <model.gguf> "Explain CogentLM."
```

Set `COGENTLM_NODE_BACKEND=cpu|vulkan|cuda|metal` to choose a native backend.

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

Gateway clients use `kind: 'gateway'` descriptors when a Node process calls a
separate CogentLM gateway.

## Learn More

- [Node.js package docs](../../docs/packages/node.md)
- [Local inference](../../docs/guides/local-inference.md)
- [Gateway and hybrid inference](../../docs/guides/gateway-hybrid.md)
- [Node examples](../../examples/node/README.md)
