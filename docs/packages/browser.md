# Browser Package

The browser package target is `@sipp/sipp`. It exposes `SippClient` for
browser-local GGUF inference, gateway calls, provider descriptors where
supported, token streaming, OPFS-backed model caching, and browser runtime
lifecycle management.

See the [Library API Overview](../api/) for the shared `add`, `query`,
`chat`, and `embed` contracts.

## Install

```bash
npm install @sipp/sipp
```

Use this package in browser code. For server routes or Node services, use
[`@sipp/sipp-server`](node.md).

## Use It For

- Browser-local text and vision inference.
- WebGPU or CPU execution through the browser runtime.
- OPFS-backed model caching.
- Gateway-backed query, chat, and embedding calls.
- Character and director helpers used by demos.

## Local GGUF Chat

```ts
import { SippClient, type ChatMessage } from '@sipp/sipp';

const client = new SippClient();
const endpoint = await client.add('default', {
  kind: 'local',
  source: '/models/model.gguf',
  options: {
    backend: 'webgpu',
    runtime: {
      context: { n_ctx: 2048 },
    },
  },
});

const messages: readonly ChatMessage[] = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain Sipp in one sentence.' },
];

const run = client.chat(messages, {
  endpoint,
  emitTokens: true,
  maxTokens: 64,
  contextKey: 'browser-local',
});

let streamed = '';
for await (const batch of run.tokens) {
  streamed += batch.text;
}
const response = await run.response;
console.log(streamed || response.text);
await client.close();
```

Use `query` when the prompt is already rendered for the target model. See the
[API overview](../api#query---raw-prompt-text-generation) for the
`query`/`chat`/`embed` contracts.

## Gateway Chat

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
const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain gateway inference.' },
];

const run = client.chat(messages, {
  endpoint,
  maxTokens: 64,
});
```

Browser apps should use short-lived gateway tokens or proxy through an
application server route. Do not ship provider credentials or long-lived
gateway tokens in browser bundles.

## Browser Runtime Options

The browser runtime links Sipp's Rust WASM ABI with llama.cpp and ggml
through Emscripten. It runs GGUF text and vision models with WebGPU when the
browser exposes a compatible adapter, and falls back to CPU execution for
compatible local workflows. OPFS-backed model caching keeps repeated browser
loads local after the first model fetch or file import.

The package resolves its packaged JavaScript and WASM assets at runtime. Most
apps should not override asset URLs. Use `executionMode`, `wasmThreading`,
`browserCache`, and local endpoint `options.runtime` only when the application
needs explicit control over browser execution, storage, or local runtime
behavior.

See [Runtime Options](../reference/runtime-options.md) for `SippClient`
options, WebGPU/backend selection, worker mode, pthread requirements, and
local runtime config groups.

## Related Docs

- [Gateway](../gateway/README.md)
- [Next.js](frameworks/nextjs.md)
- [TanStack](frameworks/tanstack.md)
- [React And Vite](frameworks/vite-react.md)
- [Local Inference](../guides/local-inference.md)
- [Runtime Options](../reference/runtime-options.md)
- [Providers](../guides/providers.md)
- [Browser Caching](../guides/browser-caching.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Examples And Demos](../examples-demos.md)
- [Maintainer source builds](../maintainers/source-builds.md)
