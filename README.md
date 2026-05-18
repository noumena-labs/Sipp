# CogentLM monorepo

Monorepo for the published `cogentlm` package plus the local avatar,
benchmark, and simulation apps that exercise it.

## Workspace layout

- `packages/cogentlm`: publishable npm package and native/WebAssembly bridge
- `packages/cogentlm-rs`: Rust runtime workspace for native and binding work
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
import { CogentEngine } from 'cogentlm';

const engine = await CogentEngine.create();

await engine.models.load('https://example.com/model.gguf');
const answer = await engine.chat([
  { role: 'user', content: 'Explain browser-hosted inference in one paragraph.' },
]);

console.log(answer.text);
await engine.close();
```

### Query

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

Streaming is available through `onTokens`:

```ts
const output = await engine.chat([{ role: 'user', content: 'Write a haiku.' }], {
  maxTokens: 64,
  onTokens: (batch) => {
    console.log(batch.text);
  },
});
console.log(output.text);
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
