# @noumena-labs/cogentlm-browser

Browser-local GGUF inference with WebGPU, WebAssembly, worker execution, OPFS model storage, text generation, vision, and embeddings.

```ts
import { CogentEngine } from '@noumena-labs/cogentlm-browser';

const engine = await CogentEngine.create();
await engine.models.load('/models/model.gguf');

const result = await engine.chat([
  { role: 'user', content: 'Explain browser-local inference in one sentence.' },
]);

console.log(result.text);
await engine.close();
```

## Core API

```ts
const engine = await CogentEngine.create({
  executionMode: 'auto',       // 'auto' | 'worker' | 'main-thread'
  wasmThreading: 'single-thread',
});
```

- `engine.models.load(source, options?)`
- `engine.models.current()`
- `engine.models.list()`
- `engine.models.remove(id)`
- `engine.chat(input, options?)`
- `engine.query(input, options?)`
- `engine.embed(input, options?)`
- `engine.state()`
- `engine.subscribeEvents(listener)`
- `engine.observability.current()`
- `engine.observability.subscribe(listener)`
- `engine.close()`

## Model Loading

`models.load()` accepts remote URLs, local `File` objects, shard arrays, and explicit model/projector pairs.

```ts
await engine.models.load('/models/text-model.gguf');

await engine.models.load({
  model: '/models/vision-model.gguf',
  projector: '/models/mmproj.gguf',
});

await engine.models.load(fileFromInput, {
  backend: 'webgpu',
  runtime: { context: { n_ctx: 4096 } },
  onProgress: (progress) => console.log(progress.phase, progress.percent),
});
```

Loaded models are installed in OPFS. `ModelInfo.id` can be passed back to `models.load(id)` to reload a persisted model. Large monolithic GGUF files are split into OPFS shards before runtime load.

## Chat, Query, Embed

Use `chat()` when the loaded model has a GGUF chat template. It renders your messages with that template, then runs text generation.

```ts
const reply = await engine.chat([
  { role: 'system', content: 'Be concise.' },
  { role: 'user', content: 'Summarize this model.' },
]);
```

Use `query()` for raw text generation. CogentLM sends your prompt directly to the runtime. Use it for plain prompts, custom templates, and encoder-decoder models such as T5.

```ts
const translated = await engine.query('translate English to French: The house is warm.');
```

Use `embed()` for vector output. It returns `EmbeddingResult`, not `GenerationResult`, and it never streams tokens.

```ts
const embedding = await engine.embed('document text', {
  normalize: true,
  contextKey: 'search',
});

console.log(embedding.values, embedding.pooling, embedding.normalized);
```

Model-class behavior:

- Decoder-only text models: `chat()` if a chat template exists, otherwise `query()`.
- Encoder-decoder models: `query()` for source-to-target generation; `chat()` only if the model has a usable chat template.
- Encoder-only models: `embed()` only.
- Decoder-only embedding models: `embed()` only when loaded with embedding context.

`embed()` defaults to L2-normalized vectors. Normalization is ignored for rank-pooling reranker outputs.

## Streaming

```ts
const result = await engine.chat([{ role: 'user', content: 'Write a haiku.' }], {
  maxTokens: 64,
  onTokens: (batch) => {
    console.log(batch.text);
  },
});
```

Worker streaming uses `SharedArrayBuffer`, so the page must be cross-origin isolated when worker token streaming is enabled.

## State, Events, Observability

```ts
const unsubscribe = engine.subscribeEvents((event) => {
  if (event.type === 'request-completed') {
    console.log('done', event.requestId);
  }
});

console.log(engine.state().status);
unsubscribe();
```

Runtime metrics are opt-in:

```ts
await engine.models.load('/models/model.gguf', {
  observability: 'runtime',
});

const stop = engine.observability.subscribe(({ type, snapshot }) => {
  if (type === 'query-complete') {
    console.log(snapshot.runtime?.tokensPerSecond);
  }
});
```

Use `observability: 'profile'` when backend/device profiling is needed.

## Vision

Vision models require a compatible projector.

```ts
await engine.models.load({
  model: '/models/vision.gguf',
  projector: '/models/mmproj.gguf',
});

const result = await engine.chat({
  messages: [{ role: 'user', content: 'What is in this image?' }],
  media: [imageBytes],
});
```

## Character Runtime

```ts
import { createCharacterFromConfigUrl } from '@noumena-labs/cogentlm-browser/character';

const { character } = await createCharacterFromConfigUrl({
  configUrl: '/characters/aria/character.json',
  engine,
});

for await (const event of character.chat('Say hello.')) {
  if (event.kind === 'prose') console.log(event.text);
}
```

The character runtime owns persona prompting, action-cue grammar, streaming action parsing, and short sliding-window memory.

## Director Runtime

```ts
import { createDirectorFromConfigUrl } from '@noumena-labs/cogentlm-browser/director';

const { director } = await createDirectorFromConfigUrl({
  configUrl: '/directors/default/director.json',
  engine,
});

const result = await director.run('choose_action', {
  inputs: { state: { kind: 'data', value: worldState } },
  choices: [{ id: 'wait' }, { id: 'move' }],
});
```

The director runtime builds task prompts, constrains structured choices with grammar, parses output shapes, and returns plain result objects for the host app.

## Runtime Assets

The package ships browser wasm assets under `dist/wasm`. By default `CogentEngine.create()` resolves bundled assets. Custom-hosted assets can be supplied explicitly:

```ts
const engine = await CogentEngine.create({
  moduleUrl: '/cogentlm/cogentlm.js',
  wasmUrl: '/cogentlm/cogentlm.wasm',
});
```

## Build

```bash
bun run build
bun run build:release
bun run pack:validate
```

## Architecture

The Rust wasm engine owns model capabilities, scheduling, text generation, embeddings, state, and events. TypeScript owns browser integration: `fetch`, `File`, OPFS, workers, asset URLs, and the character/director helper APIs.
