# CogentLM Browser Package

## What this library is for

`lib/web` is the browser package source for `@noumena-labs/cogentlm`. It runs
local GGUF inference through the browser runtime and can also call CogentLM
gateway or direct provider endpoints through the same `CogentClient.add`
endpoint API.

Use this package in browser apps that need local model loading, WebGPU or CPU
execution, token streaming, OPFS model caching, gateway-backed inference, and a
single request API for `query`, `chat`, and `embed`.

## Getting Started

Start a CogentLM gateway on `127.0.0.1:8787`, store a short-lived token in
`sessionStorage`, and run this in a browser module:

```ts
import { CogentClient } from '@noumena-labs/cogentlm'
const client = new CogentClient()
const gateway = await client.add('gateway', { kind: 'gateway', target: 'local', baseUrl: 'http://127.0.0.1:8787', authentication: { kind: 'bearer', valueProvider: () => sessionStorage.getItem('cogentlm-token') ?? '' } })
const run = client.query('Explain gateway inference in one sentence.', { endpoint: gateway, maxTokens: 64 })
console.log((await run.response).text)
```

Browser gateway authentication can use a static `value` or a `valueProvider`.
Prefer a short-lived provider so long-lived secrets are not bundled into client
code.

## Gateway And Hybrid Inference

Hybrid browser inference can compare a browser-local GGUF model with a gateway
target. This example passes explicit endpoint references for both requests so
the routing choice is visible at each call site.

```ts
import { CogentClient, type BrowserTextRun, type EndpointRef } from '@noumena-labs/cogentlm'

async function runHybrid(modelUrl: string, prompt: string): Promise<void> {
  const client = new CogentClient()
  try {
    const local = await client.add('local', {
      kind: 'local',
      source: modelUrl,
      options: {
        runtime: {
          context: { n_ctx: 2048 },
          scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
          cache: { mode: 'live_slot_prefix' },
          observability: { runtime_metrics: true },
        },
      },
    })
    const gateway = await client.add('gateway', {
      kind: 'gateway',
      target: 'local',
      baseUrl: 'http://127.0.0.1:8787',
      authentication: {
        kind: 'bearer',
        valueProvider: () => sessionStorage.getItem('cogentlm-token') ?? '',
      },
    })

    const localRun = client.query(prompt, {
      endpoint: local,
      emitTokens: true,
      maxTokens: 96,
      session: 'web-local',
      temperature: 0.7,
    })
    const gatewayRun = client.query(prompt, {
      endpoint: gateway,
      emitTokens: true,
      maxTokens: 96,
      temperature: 0.7,
    })

    await printRun('local', local, localRun)
    await printRun('gateway', gateway, gatewayRun)
  } finally {
    await client.close()
  }
}

async function printRun(label: string, endpoint: EndpointRef, run: BrowserTextRun): Promise<void> {
  let streamed = ''
  for await (const batch of run.tokens) streamed += batch.text
  const response = await run.response
  console.log(label, endpoint, streamed || response.text)
}
```

Gateway descriptors use `routes.query`, `routes.chat`, and `routes.embed` only
when your application exposes non-default paths. `protocolOptions` are merged
into every first-party gateway request, while per-request `endpointOptions` are
merged only for one call.

Call `client.close()` when a browser page, worker, or component no longer needs
the runtime. This releases worker and local runtime resources. Gateway-only
clients should still be closed so the same lifecycle works across local,
gateway, and provider endpoints.
