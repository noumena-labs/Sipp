# CogentLM Server for Node.js

## What this library is for

`@noumena-labs/cogentlm-server` exposes the native CogentLM client API to
Node.js server processes. Use it when a Node app needs local GGUF inference,
gateway-backed inference, direct provider adapters, token streaming, and native
backend selection from one package.

The package does not provide Express, Fastify, or other framework handlers.
Framework integrations should define their own routes and call
`client.query()`, `client.chat()`, or `client.embed()` inside those routes.

## Getting Started

Start a CogentLM gateway on `127.0.0.1:8787`, set `COGENTLM_GATEWAY_TOKEN`, and
run this from a module with top-level `await`:

```ts
import { CogentClient } from '@noumena-labs/cogentlm-server'
const client = new CogentClient()
const gateway = await client.add('gateway', { kind: 'gateway', target: 'local', baseUrl: 'http://127.0.0.1:8787', authentication: { kind: 'bearer', value: process.env.COGENTLM_GATEWAY_TOKEN } })
const run = client.query({ endpoint: gateway, prompt: 'Explain gateway inference in one sentence.', options: { maxTokens: 64 } })
console.log((await run.response).text)
```

`add(id, descriptor)` returns an endpoint reference. Pass that reference as
`endpoint` when a request should use a specific local, gateway, or provider
destination.

## Gateway And Hybrid Inference

Registering both local and gateway endpoints lets the application choose where
each request runs. The gateway process keeps provider credentials or remote
model placement details out of the Node process.

```ts
import { CogentClient, setLlamaLogQuiet } from '@noumena-labs/cogentlm-server'

setLlamaLogQuiet(true)

const client = new CogentClient()
const local = await client.add('local', {
  kind: 'local',
  modelPath: process.argv[2],
  config: {
    context: { n_ctx: 2048 },
    scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
    cache: { mode: 'live_slot_prefix' },
    observability: { runtime_metrics: true },
  },
})
const gateway = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: process.env.COGENTLM_GATEWAY_URL ?? 'http://127.0.0.1:8787',
  authentication: {
    kind: 'bearer',
    value: process.env.COGENTLM_GATEWAY_TOKEN,
  },
})

const prompt = process.argv[3] ?? 'Compare local and gateway inference.'
const localRun = client.query({
  endpoint: local,
  prompt,
  emitTokens: true,
  options: { maxTokens: 96, temperature: 0.7 },
  local: { contextKey: 'node-local' },
})
for await (const batch of localRun) process.stdout.write(batch.text)
const localResponse = await localRun.response

const gatewayResponse = await client.query({
  endpoint: gateway,
  prompt,
  options: { maxTokens: 96, temperature: 0.7 },
}).response

console.log('\nlocal:', localResponse.text)
console.log('gateway:', gatewayResponse.text)
```

Gateway descriptors use the first-party profile by default: `/v1/query`,
`/v1/chat`, and `/v1/embed`. Use `queryRoute`, `chatRoute`, `embedRoute`, and
`protocolOptions` only when your gateway app deliberately exposes a compatible
custom profile. Per-request gateway flags go in `endpointOptions`; provider-only
flags go in `providerOptions`.

Set `COGENTLM_NODE_BACKEND=cpu|vulkan|cuda|metal` to choose a staged native
backend. The `CogentTextRun` returned by `query` and `chat` exposes both a
final `response` promise and an async token iterator when `emitTokens` is true.
