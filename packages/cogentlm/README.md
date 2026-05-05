# cogentlm

A high-performance inference and vision runtime for running models locally in browser.

```ts
import { CogentEngine } from 'cogentlm';

const engine = await CogentEngine.create();

await engine.models.load('https://example.com/model.gguf');
const answer = await engine.chat([
  { role: 'user', content: 'Explain browser-hosted inference in one paragraph.' },
]);

console.log(answer);
await engine.close();
```

## Observability

Observability is opt-in. Use `"runtime"` for request/runtime metrics or `"profile"` when backend profiling is needed:

```ts
await engine.models.load('https://example.com/model.gguf', {
  observability: 'profile',
  runtime: { nCtx: 4096 },
});

const unsubscribe = engine.observability.subscribe((event) => {
  console.log(event.type, event.snapshot.state);
});

await engine.chat([{ role: 'user', content: 'Measure this request.' }]);
console.log(engine.observability.current().runtime);
unsubscribe();
```

`chat()` and `query()` still return only a string. Metrics are read from `engine.observability.current()` and lifecycle events are emitted only at load, query, error, and close boundaries.

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
```

Use `engine.query(...)` only when you already have a complete raw prompt string.
Cogent does not apply a chat template in `query()`, so the prompt must already
include whatever control tokens, role markers, or assistant prefix your model
expects:

```ts
const text = await engine.query(
  '<|im_start|>user\nSummarize the current model.<|im_end|>\n<|im_start|>assistant\n'
);

const vision = await engine.chat({
  messages: [{ role: 'user', content: 'What is in this image?' }],
  media: [imageBytes],
});
```

Streaming is available through `onToken`:

```ts
const output = await engine.chat([{ role: 'user', content: 'Write a haiku.' }], {
  maxTokens: 64,
  onToken: (token) => {
    console.log(token);
  },
});
```

## Public Exports

- `CogentEngine`
- `CogentEngineOptions`
- `ModelSource`
- `ModelLoadOptions`
- `ModelInfo`
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
- `QueryError`

Custom-hosted runtime assets can be supplied with `CogentEngine.create({ moduleUrl, wasmUrl })`.
