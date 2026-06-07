# Node.js Package

The Node.js package target is `cogentlm-server`. It exposes the native
CogentLM client API to Node server processes, route handlers, and framework
server functions. Applications own framework routes, request validation, auth,
and deployment policy.

## Install

```bash
npm install cogentlm-server
```

Use this package only in Node runtime code. Browser components should use
[`cogentlm`](browser.md).

## Use It For

- Server-side local GGUF inference.
- Gateway-backed and provider-backed inference from server code.
- Token streaming from Node processes.
- Framework route handlers in Node runtimes.
- Backend selection for native bindings.

## Local GGUF Query

```ts
import { CogentClient } from 'cogentlm-server';

const client = new CogentClient();
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

const run = client.query({
  endpoint,
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
```

Set `COGENTLM_NODE_BACKEND=cpu|vulkan|cuda|metal` to choose a native backend.

## Gateway Query

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
  target: requiredEnv('COGENTLM_GATEWAY_TARGET'),
  baseUrl: requiredEnv('COGENTLM_GATEWAY_URL'),
  authentication: {
    kind: 'bearer',
    value: requiredEnv('COGENTLM_GATEWAY_TOKEN'),
  },
});
const run = client.query({
  endpoint,
  prompt: 'Explain gateway inference.',
  options: { maxTokens: 64 },
});
console.log((await run.response).text);
```

The application only needs the gateway URL, bearer token, and public target.
Provider credentials and local model paths stay in the gateway process.

## Framework Routes

Use `cogentlm-server` in server-only code such as Next.js App Router route
handlers with `runtime = 'nodejs'`, TanStack Start server functions, Express
routes, or background workers. Do not import it from browser bundles.

## Related Docs

- [Gateway Server](gateway-server.md)
- [Next.js](frameworks/nextjs.md)
- [TanStack](frameworks/tanstack.md)
- [Local Inference](../guides/local-inference.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Maintainer source builds](../maintainers/source-builds.md)
