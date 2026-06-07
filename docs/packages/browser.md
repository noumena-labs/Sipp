# Browser Package

The browser package target is `cogentlm`. It exposes `CogentClient` for
browser-local GGUF inference, gateway calls, provider descriptors where
supported, token streaming, and browser runtime lifecycle management.

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

## WebGPU Engine

The browser runtime links CogentLM's Rust WASM ABI with llama.cpp and ggml
through Emscripten. It runs GGUF text and vision models with WebGPU when the
browser exposes a compatible adapter, and falls back to CPU execution for
compatible local workflows. OPFS-backed model caching keeps repeated browser
loads local after the first model fetch or file import.

<!--
Future benchmark graph placeholder:
- CogentLM browser WebGPU engine
- transformers.js WebGPU
- WebLLM WebGPU
Add the graph only with checked-in benchmark methodology, model names, browser
versions, hardware, and raw measurements.
-->

## Gateway

Gateway endpoints use the same `CogentClient.add` endpoint model with a base
URL, target name, and authentication provider. Browser applications provide
short-lived gateway tokens at runtime.

## Related Docs

- [Browser Caching](../guides/browser-caching.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Examples And Demos](../examples-demos.md)
