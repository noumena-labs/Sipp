# CogentLM Browser Package

`lib/web` is the browser package source for the public `cogentlm` package. It
supports browser-local GGUF inference, gateway calls, streaming text, OPFS model
caching, and browser runtime lifecycle management through `CogentClient`.

Source builds use the workspace manifest in this directory. Public docs use the
`cogentlm` package target.

## Source Checkout

From the repository root, after `source ./setup.sh`:

```bash
clm build wasm && clm run examples serve browser
```

`clm` forwards to `cargo xtask`; use `cargo xtask ...` with the same arguments
if the launcher is not active.

## Local GGUF Query

```ts
import { CogentClient } from 'cogentlm';

const client = new CogentClient();
await client.add('default', {
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
  emitTokens: true,
  maxTokens: 64,
  session: 'web-local',
});

let streamed = '';
for await (const batch of run.tokens) {
  streamed += batch.text;
}
const response = await run.response;
console.log(streamed || response.text);
await client.close();
```

## WebGPU Engine

The browser runtime links the Rust WASM ABI with llama.cpp and ggml through
Emscripten. It runs GGUF text and vision models with WebGPU on compatible
browsers, falls back to CPU execution for compatible local workflows, and uses
OPFS-backed caching for repeated model loads.

<!--
Future benchmark graph placeholder:
- CogentLM browser WebGPU engine
- transformers.js WebGPU
- WebLLM WebGPU
Add the graph only with checked-in benchmark methodology, model names, browser
versions, hardware, and raw measurements.
-->

Gateway clients use the same endpoint API. Browser applications provide
short-lived gateway tokens at runtime.

## Learn More

- [Browser package docs](../../docs/packages/browser.md)
- [Browser caching](../../docs/guides/browser-caching.md)
- [Gateway and hybrid inference](../../docs/guides/gateway-hybrid.md)
- [Web examples](../../examples/web/README.md)
