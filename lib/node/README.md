# Sipp Server for Node.js

`lib/node` is the Node.js package source for the public `@sipp/sipp-server`
package. It exposes Sipp's native client API to Node server processes for
local GGUF inference, gateway-backed inference, provider descriptors, and token
streaming.

Source builds use the workspace manifest in this directory. Public docs use the
`@sipp/sipp-server` package target. Applications own framework routes and call
`client.query()`, `client.chat()`, or `client.embed()` inside those routes.

## Source Checkout

From the repository root, after `source ./setup.sh`:

```bash
sipp build node --backend cpu && node examples/node/query.mjs <model.gguf> "Explain Sipp."
```

`sipp` forwards to `cargo xtask`; use `cargo xtask ...` with the same arguments
if the launcher is not active.

Set `SIPP_NODE_BACKEND=cpu|vulkan|cuda|metal` to choose a native backend.

## Local GGUF Query

```ts
import { SippClient } from '@sipp/sipp-server';

const client = new SippClient();
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
  prompt: 'Explain Sipp in one sentence.',
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
separate Sipp gateway.

## Gateway Profile Helpers

Use the gateway profile helpers when a Node route should behave like a
first-party gateway endpoint for browser `kind: 'gateway'` clients:

```ts
import {
  SippClient,
  decodeGatewayQueryBody,
  gatewayErrorResponse,
  gatewayTextResponseBody,
  gatewayTextStreamResponse,
} from '@sipp/sipp-server';

export async function handleQuery(request: Request): Promise<Response> {
  try {
    const decoded = decodeGatewayQueryBody(await request.json());
    const client = new SippClient();
    const endpoint = await client.add('gateway', {
      kind: 'gateway',
      target: decoded.target,
      baseUrl: process.env.SIPP_GATEWAY_URL!,
      authentication: {
        kind: 'bearer',
        value: process.env.SIPP_GATEWAY_TOKEN!,
      },
    });
    const run = client.query({ ...decoded.request, endpoint });
    return decoded.stream
      ? gatewayTextStreamResponse(run)
      : Response.json(
          gatewayTextResponseBody(decoded.target, await run.response),
        );
  } catch (error) {
    const response = gatewayErrorResponse(error);
    return Response.json(response.body, response.init);
  }
}
```

`decodeGatewayChatBody()`, `decodeGatewayEmbedBody()`,
`gatewayTextResponseBody()`, and `gatewayEmbeddingResponseBody()` mirror the
first-party gateway JSON profile used by Sipp clients.

## Learn More

- [Node.js package docs](../../docs/packages/node.md)
- [Local inference](../../docs/guides/local-inference.md)
- [Gateway and hybrid inference](../../docs/guides/gateway-hybrid.md)
- [Node examples](../../examples/node/README.md)
