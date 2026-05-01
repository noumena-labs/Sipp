# CogentLM monorepo

Monorepo for the published `cogentlm` package plus the local avatar,
benchmark, and simulation apps that exercise it.

## Workspace layout

- `packages/cogentlm`: publishable npm package and native/WebAssembly bridge
- `packages/cogentlm/third_party/llama.cpp`: pinned `llama.cpp` submodule
- `apps/avatar`: browser character harness with a VRM avatar
- `apps/benchmark`: browser benchmark harness
- `apps/simulation`: browser simulation and director example

## Clone

Clone with submodules so the vendored `llama.cpp` checkout is present from the start:

```bash
git clone --recurse-submodules <repo-url> cogentlm
cd cogentlm
```

If you already cloned the repo without submodules:

```bash
git submodule update --init --recursive
```

## Install

```bash
bun install
```

## Build the package

```bash
bun run build:package
```

Use the release build when you need the publishable package layout with browser and Bun wasm
artifacts:

```bash
bun run build:package:release
```

For a clean rebuild:

```bash
bun run rebuild:package
```

## Getting Started

Get started in a few lines of code:

```ts
import { CogentEngine } from 'cogent-engine';

const engine = await CogentEngine.create();

await engine.models.load('https://example.com/model.gguf');
const answer = await engine.chat([
  { role: 'user', content: 'Explain browser-hosted inference in one paragraph.' },
]);

console.log(answer);
await engine.close();
```

### Query

Use `engine.chat(...)` when you have chat messages and want Cogent to apply the
loaded model's native chat template:

```ts
const reply = await engine.chat([
  { role: 'system', content: 'Be concise.' },
  { role: 'user', content: 'Summarize the current model.' },
]);
```

Use `engine.query(...)` when you already have a raw prompt string:

```ts
const text = await engine.query('Summarize the current model.');

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
### Model Lifecycle

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