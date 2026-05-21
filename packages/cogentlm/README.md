# cogentlm

A high-performance, WebGPU-accelerated inference and vision runtime for executing Large Language Models (LLMs) and computer vision locally in the browser.

## Runtime Architecture Direction

The browser package exposes the unified runtime contract:
`EngineState`, `EngineStats`, `EngineEvent`, and `RequestResult`.
Browser inference is owned by the Rust browser engine linked into the
Emscripten/WebGPU artifact. Emscripten remains the browser platform/link layer
for llama.cpp, ggml-webgpu, mtmd, JSPI, WorkerFS, and WebGPU glue.

```ts
import { CogentEngine } from 'cogentlm';

const engine = await CogentEngine.create();

await engine.models.load('https://example.com/model.gguf');
const answer = await engine.chat([
  { role: 'user', content: 'Explain browser-hosted inference in one paragraph.' },
]);

console.log(answer.text);
await engine.close();
```

## Engine State And Events

Use `state()` for the current point-in-time engine view and
`subscribeEvents()` for structured runtime events:

```ts
const unsubscribe = engine.subscribeEvents((event) => {
  if (event.type === 'request-completed') {
    console.log(event.result.stats.tokensPerSecond);
  }
});

const result = await engine.query('Explain browser LLM inference.', {
  maxTokens: 64,
});

console.log(result.text);
console.log(engine.state().backend.selected);
unsubscribe();
```

Worker mode still uses `postMessage` for state/result/events. SharedArrayBuffer
is required for worker token streaming when `onTokens` is attached.

## Observability

Observability is opt-in. Use `"runtime"` for request/runtime metrics or `"profile"` when backend profiling is needed:

```ts
await engine.models.load('https://example.com/model.gguf', {
  observability: 'profile',
  runtime: { context: { n_ctx: 4096 } },
});

const unsubscribe = engine.observability.subscribe(({ type, snapshot }) => {
  if (type === 'query-complete') {
    console.log(`TPS: ${snapshot.runtime?.tokensPerSecond}, TTFT: ${snapshot.runtime?.ttftMs}ms`);
  }
});

const measured = await engine.chat([{ role: 'user', content: 'Measure this request.' }]);
console.log(measured.stats.tokensPerSecond);
unsubscribe();
```

`chat()` and `query()` return `RequestResult`, including final text and request
metrics. Metrics are also available from `engine.observability.current()`.

## Model Lifecycle

Use `engine.models` for model management:

```ts
const loaded = await engine.models.load({
  model: 'https://example.com/vision-model.gguf',
  projector: 'https://example.com/mmproj.gguf',
});

await engine.models.load(loaded.id);

console.log(engine.models.current());
console.log(await engine.models.list());

await engine.models.remove(loaded.id);
```

`engine.models.load(...)` handles first load, reload, model switching, local imports, remote downloads, shard arrays, and explicit model/projector assembly.

`ModelInfo.id` is the installed model id for the persisted base-model entry. If a model has already been validated with a projector, later `engine.models.load(id)` reuses that stored pairing automatically. Installed entries and pairings live in OPFS, so they survive tab refresh and browser restart for the same origin.

Large monolithic GGUF inputs use the browser cache policy only in the browser package: files at or below 2 GiB use the direct single-file path, while larger remote URLs and local `File` imports are split into llama.cpp-compatible OPFS shards by the Rust-backed browser ingest layer. This OPFS split path is not used by native Rust, Python, or Node.

Managed storage requires OPFS. If OPFS is unavailable, loading fails clearly instead of silently falling back to transient memory.

## Worker Mode

When worker execution is selected, the worker hosts the same high-level model service used by main-thread mode. The main thread talks to a worker model-service client, while low-level WASM, scheduling, cache, and runtime details stay internal.

## Query

Use `engine.chat(...)` for normal assistant-style interaction. Cogent reads the
loaded GGUF model's native chat template and renders your messages into the
model-specific prompt format before inference:

```ts
const reply = await engine.chat([
  { role: 'system', content: 'Be concise.' },
  { role: 'user', content: 'Summarize the current model.' },
]);
console.log(reply.text);
```

Use `engine.query(...)` only when you already have a complete raw prompt string.
Cogent does not apply a chat template in `query()`, so the prompt must already
include whatever control tokens, role markers, or assistant prefix your model
expects:

```ts
const text = await engine.query(
  '<|im_start|>user\nSummarize the current model.<|im_end|>\n<|im_start|>assistant\n'
);
console.log(text.text);

const vision = await engine.chat({
  messages: [{ role: 'user', content: 'What is in this image?' }],
  media: [imageBytes],
});
```

Streaming is available through `onTokens`:

```ts
const output = await engine.chat([{ role: 'user', content: 'Write a haiku.' }], {
  onTokens: (batch) => {
    console.log(batch.text);
  },
});
console.log(output.text);
```

## Public Exports

- `CogentEngine`
- `CogentEngineOptions`
- `ModelSource`
- `ModelLoadOptions`
- `ModelInfo`
- `EngineState`
- `EngineStats`
- `EngineEvent`
- `BackendInfo`
- `RequestResult`
- `RequestState`
- `RequestStats`
- `BrowserRuntimeSmokeResult`
- `BrowserGgufIngestSmokeResult`
- `ObservabilityMode`
- `EngineObservability`
- `ObservabilityEvent`
- `ObservabilityEventType`
- `ObservabilitySnapshot`
- `QueryObservation`
- `RuntimeObservation`
- `BackendProfileObservation`
- `ChatInput`
- `ChatMessage`
- `ChatOptions`
- `QueryInput`
- `QueryOptions`
- `TokenBatch`
- `StreamStats`
- `QueryError`

Custom-hosted runtime assets can be supplied with `CogentEngine.create({ moduleUrl, wasmUrl })`.
