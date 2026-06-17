# Node.js Package

The Node.js package target is `@sipp/sipp-server`. It exposes the native
Sipp client API to Node server processes, route handlers, and framework
server functions. Applications own framework routes, request validation, auth,
and deployment policy.

See the [Library API Overview](../api) for the shared `add`, `query`,
`chat`, and `embed` contracts.

## Install

```bash
npm install @sipp/sipp-server
```

Use this package only in Node runtime code. Browser components should use
[`@sipp/sipp`](browser.md).

`@sipp/sipp-server` is a wrapper package. npm installs the matching optional
platform package for the current OS and CPU, and the runtime loader selects
the best packaged backend for that host.

## Use It For

- Server-side local GGUF inference.
- Gateway-backed and provider-backed inference from server code.
- Token streaming from Node processes.
- Framework route handlers in Node runtimes.
- Backend selection for native bindings.

## Local GGUF Query

```ts
import { SippClient } from '@sipp/sipp-server';

const client = new SippClient();
const endpoint = await client.add('default', {
  kind: 'local',
  modelPath: process.argv[2],
  config: {
    context: { n_ctx: 2048 },
    scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
    cache: { mode: 'live_slot_prefix' },
    observability: { runtime_metrics: true },
  },
});
const queryPrompt = [
  '<|system|>',
  'Answer concisely.',
  '<|user|>',
  'Explain Sipp in one sentence.',
  '<|assistant|>',
].join('\n');

const run = client.query({
  endpoint,
  // query: raw prompt; replace markers with the target model's template.
  prompt: queryPrompt,
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
```

Set `SIPP_NODE_BACKEND=cpu|vulkan|cuda|metal` to choose a native backend.
By default, macOS tries `metal` then `cpu`; Windows and Linux try `cuda`,
`vulkan`, then `cpu`.
See [Runtime Options](../reference/runtime-options.md) for local runtime config
groups and request option boundaries.

## Gateway Chat

```ts
function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: requiredEnv('SIPP_GATEWAY_TARGET'),
  baseUrl: requiredEnv('SIPP_GATEWAY_URL'),
  authentication: {
    kind: 'bearer',
    value: requiredEnv('SIPP_GATEWAY_TOKEN'),
  },
});
const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain gateway inference.' },
];
const run = client.chat({
  endpoint,
  messages,
  options: { maxTokens: 64 },
});
console.log((await run.response).text);
```

The application only needs the gateway URL, bearer token, and public target.
Provider credentials and local model paths stay in the gateway process.

## Direct Provider Chat

Use direct provider endpoints only in trusted server code. Keep the provider
key in the server environment; `OPENAI_API_KEY="<mock-openai-key>"` is only a
placeholder value in examples.

```ts
function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

const endpoint = await client.add('provider', {
  kind: 'provider',
  provider: 'openai',
  model: process.env.OPENAI_MODEL ?? 'gpt-5-mini',
  apiKey: requiredEnv('OPENAI_API_KEY'),
});
const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain provider inference.' },
];
const run = client.chat({
  endpoint,
  messages,
  options: { maxTokens: 64 },
});
console.log((await run.response).text);
```

Pass provider-only request fields through `providerOptions`. See
[Providers](../guides/providers.md) for the full provider/gateway split.

## Gateway Profile Helpers

Use the gateway profile helpers when a Node route should behave like a
first-party gateway endpoint for browser `kind: 'gateway'` clients. The helpers
decode `model`, `prompt`, `messages`, `input`, and snake_case generation
options, then format JSON or SSE responses. The route can execute the decoded
request against a provider, a local endpoint, or a separate gateway.

```ts
import {
  SippClient,
  decodeGatewayQueryBody,
  gatewayErrorResponse,
  gatewayTextResponseBody,
  gatewayTextStreamResponse,
} from '@sipp/sipp-server';

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

export async function handleQuery(request: Request): Promise<Response> {
  try {
    const decoded = decodeGatewayQueryBody(await request.json());
    const client = new SippClient();
    const endpoint = await client.add('provider', {
      kind: 'provider',
      provider: 'openai',
      model: decoded.target,
      apiKey: requiredEnv('OPENAI_API_KEY'),
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

Use `decodeGatewayChatBody()` and `decodeGatewayEmbedBody()` for `/v1/chat`
and `/v1/embed` compatible routes. Use `gatewayEmbeddingResponseBody()` for
finite embedding responses.

## Framework Routes

Use `@sipp/sipp-server` in server-only code such as Next.js App Router route
handlers with `runtime = 'nodejs'`, TanStack Start server functions, Express
routes, or background workers. Do not import it from browser bundles.

## Related Docs

- [Gateway Server](../gateway/server.md)
- [Next.js](frameworks/nextjs.md)
- [TanStack](frameworks/tanstack.md)
- [Local Inference](../guides/local-inference.md)
- [Providers](../guides/providers.md)
- [Runtime Options](../reference/runtime-options.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Maintainer source builds](../maintainers/source-builds.md)
