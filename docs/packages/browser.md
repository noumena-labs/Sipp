# Browser Package

The browser package target is `cogentlm`. It exposes `CogentClient` for
browser-local GGUF inference, gateway calls, provider descriptors where
supported, token streaming, OPFS-backed model caching, and browser runtime
lifecycle management.

## Install

```bash
npm install cogentlm
```

Use this package in browser code. For server routes or Node services, use
[`cogentlm-server`](node.md).

## Use It For

- Browser-local text and vision inference.
- WebGPU or CPU execution through the browser runtime.
- OPFS-backed model caching.
- Gateway-backed query, chat, and embedding calls.
- Character and director helpers used by demos.

## Local GGUF Query

```ts
import { CogentClient } from 'cogentlm';

const client = new CogentClient();
const endpoint = await client.add('default', {
  kind: 'local',
  source: '/models/model.gguf',
  options: {
    runtime: {
      context: { n_ctx: 2048 },
      scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
      cache: { mode: 'live_slot_prefix' },
      observability: { runtime_metrics: true },
    },
  },
});

const run = client.query('Explain CogentLM in one sentence.', {
  endpoint,
  emitTokens: true,
  maxTokens: 64,
  session: 'browser-local',
});

let streamed = '';
for await (const batch of run.tokens) {
  streamed += batch.text;
}
const response = await run.response;
console.log(streamed || response.text);
await client.close();
```

## Gateway Query

Use gateway endpoints when a separate server owns model paths, provider
credentials, target policy, and metrics.

```ts
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: 'https://gateway.example.com',
  authentication: {
    kind: 'bearer',
    valueProvider: getShortLivedGatewayToken,
  },
});
const run = client.query('Explain gateway inference.', {
  endpoint,
  maxTokens: 64,
});
```

Browser apps should use short-lived gateway tokens or proxy through an
application server route. Do not ship provider credentials or long-lived
gateway tokens in browser bundles.

## WebGPU Engine

The browser runtime links CogentLM's Rust WASM ABI with llama.cpp and ggml
through Emscripten. It runs GGUF text and vision models with WebGPU when the
browser exposes a compatible adapter, and falls back to CPU execution for
compatible local workflows. OPFS-backed model caching keeps repeated browser
loads local after the first model fetch or file import.

The package resolves its packaged JavaScript and WASM runtime assets at
runtime. Override `moduleUrl` and `wasmUrl` only when your bundler or
deployment moves package assets. The pthread runtime requires
`SharedArrayBuffer` and cross-origin isolation.

<!--
Future benchmark graph placeholder:
- CogentLM browser WebGPU engine
- transformers.js WebGPU
- WebLLM WebGPU
Add the graph only with checked-in benchmark methodology, model names, browser
versions, hardware, and raw measurements.
-->

## Related Docs

- [Gateway Server](gateway-server.md)
- [Next.js](frameworks/nextjs.md)
- [TanStack](frameworks/tanstack.md)
- [React And Vite](frameworks/vite-react.md)
- [Browser Caching](../guides/browser-caching.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Examples And Demos](../examples-demos.md)
- [Maintainer source builds](../maintainers/source-builds.md)
