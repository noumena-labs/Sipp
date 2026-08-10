# Browser Package

The browser package target is `@sipphq/sipp`. It exposes `SippClient` for
browser-local GGUF inference, gateway calls, provider endpoints where
supported, token streaming, an OPFS-backed model store, and browser runtime
lifecycle management.

See the [Library API Overview](../api/) for the shared `add`, `query`,
`chat`, `embed`, `listen`, and `speak` contracts.

## Install

```bash
npm install @sipphq/sipp
```

Use this package in browser code. For server routes or Node services, use
[`@sipphq/sipp-server`](node.md).

## Use It For

- Browser-local text and vision inference.
- Browser-local speech recognition and synthesis.
- WebGPU or CPU execution through the browser runtime.
- Persistent browser model storage in OPFS.
- Gateway-backed query, chat, and embedding calls.
- Character and director helpers used by demos.

## Local GGUF Chat

```ts
import { Endpoint, SippClient, type ChatMessage } from '@sipphq/sipp';

const client = new SippClient();
const model = await client.models.add(['/models/model.gguf']);
const endpoint = await client.add(
  'default',
  Endpoint.local(model, {
    backend: 'webgpu',
    runtime: {
      context: { n_ctx: 2048 },
    },
  })
);

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
`query`/`chat`/`embed` contracts and the speech operation reference.

## Local Speech

Load the main speech GGUF together with its matching projector. `listen()`
accepts WAV, MP3, or FLAC bytes and has an independent transcript token limit.
`speak()` returns a terminal mono PCM16 WAV payload.

```ts
const model = await client.models.add([
  '/models/qwen3-tts.gguf',
  '/models/mmproj-qwen3-tts.gguf',
]);
const endpoint = await client.add('tts', Endpoint.local(model));
const run = client.speak('Hello from Sipp.', {
  endpoint,
  language: 'en',
  maxDurationMs: 10_000,
});
const result = await run.response;
const url = URL.createObjectURL(new Blob([result.audio], { type: 'audio/wav' }));
```

The model's end-of-generation token completes synthesis successfully.
`maxDurationMs` is an optional safety limit: reaching it before end of
generation fails the request. Omit it to use the loaded model adapter's
generation default. The input text length does not determine this limit, and
the context size remains a model-state capacity rather than a speech-duration
control.

## Gateway Chat

Use gateway endpoints when a separate server owns model paths, provider
credentials, target policy, and metrics.

```ts
import { Endpoint } from '@sipphq/sipp';

const endpoint = await client.add('gateway', Endpoint.gateway({
  target: 'local',
  baseUrl: 'https://gateway.example.com',
  authentication: {
    kind: 'bearer',
    valueProvider: getShortLivedGatewayToken,
  },
}));
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
browser exposes the required adapter, or with CPU execution when CPU is the
selected backend. After the first URL fetch or `File` import, registered models
remain in OPFS and can be loaded again by model ID.

Browser-local runtimes always execute in a Worker. Activating another model
retires the current Worker before creating the next runtime. Always await
`client.close()` so Worker termination and browser resource cleanup complete.

Model metadata returned by `client.models` is readonly. `assetFingerprint`
identifies the installed asset revision. `capabilities` is `null` when that
model is not active; for the active model, use `capabilities.operations` to
check support for `query`, `chat`, `embed`, `listen`, and `speak`. Unavailable
optional capability values are represented by `null`.

Token streaming delivers `TokenBatch` values whose `stats` report `framesSent`,
`bytesSent`, and `batchesSent`. Per-batch drain timings are not reported; use
observability's `jsTokenDrainMs` and `jsTokenDrainCalls` for transport cost.

The package resolves its packaged JavaScript and WASM assets at runtime. Most
apps should not override asset URLs. Use `wasmThreading`, `browserCache`, and
local endpoint `options.runtime` only when the application
needs explicit control over browser execution, storage, or local runtime
behavior.

Packaged browser runtime assets use pthreads, so browser-local inference needs
`SharedArrayBuffer` and cross-origin isolation headers. Hosts that cannot serve
those headers must set `wasmThreading: 'single-thread'` and provide custom
single-thread `moduleUrl` and `wasmUrl` assets.

See [Runtime Options](../reference/runtime-options.md) for `SippClient`
options, WebGPU/backend selection, worker lifecycle, pthread requirements, and
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
